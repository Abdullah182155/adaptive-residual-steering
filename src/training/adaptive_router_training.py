"""Offline contextual-bandit training for the adaptive subset router."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from src.routing.adaptive_subset_router import AdaptiveSubsetRouter, sample_route_actions
from src.training.adaptive_subset_training import _prepare


class _Probe:
    """Capture one unsteered backbone layer for prompt-only router features."""

    def __init__(self, layer: nn.Module) -> None:
        self.hidden: torch.Tensor | None = None
        self.handle = layer.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output) -> None:
        value = output[0] if isinstance(output, tuple) else output
        self.hidden = value.detach()

    def close(self) -> None:
        self.handle.remove()


def _completion_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_labels = labels[:, 1:]
    losses = F.cross_entropy(
        logits[:, :-1].float().transpose(1, 2),
        shifted_labels,
        ignore_index=-100,
        reduction="none",
    )
    valid = shifted_labels.ne(-100)
    return (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def _save_router_checkpoint(
    router, path: str | Path, probe_layer: int, best_val_nll: float,
    route_catalog: Sequence[Sequence[int]] = (),
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "format": "ars_adaptive_router_v1",
            "candidate_layers": list(router.candidate_layers),
            "probe_layer": int(probe_layer),
            "max_active_layers": router.max_active_layers,
            "alpha_values": list(router.alpha_values),
            "router": router.state_dict(),
            "best_validation_nll": float(best_val_nll),
            "route_catalog": [list(route) for route in route_catalog],
        },
        temporary,
    )
    temporary.replace(destination)


def _score_action(
    model, row, route, alpha: torch.Tensor, device, prompt_shield_mask=None
) -> torch.Tensor:
    model.set_routing_override("matrix", route)
    model.set_route_gains(alpha)
    model.set_prompt_shield_mask(prompt_shield_mask)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits = model(
            input_ids=row["input_ids"],
            attention_mask=row["attention_mask"],
            use_cache=False,
        ).logits
        nll = _completion_nll(logits, row["labels"])
    del logits
    return nll


@torch.no_grad()
def _evaluate_router_nll(
    model,
    router: AdaptiveSubsetRouter,
    rows: Sequence[dict[str, torch.Tensor]],
    probe: _Probe,
    device: torch.device,
    limit: int,
) -> float:
    router.eval()
    values: list[float] = []
    candidates = router.candidate_layers
    for item in rows[:limit]:
        row, extras = _prepare([item], device)
        model.set_routing_override("none", candidates)
        model.set_route_gains(None)
        model.set_prompt_shield_mask(None)
        model(input_ids=row["input_ids"], attention_mask=row["attention_mask"], use_cache=False)
        if probe.hidden is None:
            continue
        prompt_mask = row["labels"].eq(-100) & row["attention_mask"].bool()
        decision = router.choose(probe.hidden, prompt_mask)
        route = decision.mask
        nll = _score_action(
            model,
            row,
            route,
            decision.alpha,
            device,
            prompt_mask,
        )
        values.append(float(nll.mean()))
    model.set_routing_override("none", candidates)
    model.set_route_gains(None)
    model.set_prompt_shield_mask(None)
    return sum(values) / max(len(values), 1)


@torch.no_grad()
def _build_oracle_targets(
    model,
    rows: Sequence[dict[str, torch.Tensor]],
    route_catalog: Sequence[Sequence[int]],
    router: AdaptiveSubsetRouter,
    probe: _Probe,
    device: torch.device,
    *,
    route_cost: float,
    row_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score a pruned route catalog on verified completions to label prompts."""
    candidates = router.candidate_layers
    routes = [tuple(int(layer) for layer in route) for route in route_catalog]
    if () not in routes:
        routes.insert(0, ())
    if not routes or any(not set(route).issubset(candidates) or len(route) > 4 for route in routes):
        raise ValueError("route catalog contains an invalid candidate-layer subset")
    alpha_values = tuple(router.alpha_values)
    route_indices = list(range(min(len(rows), row_limit)))
    targets: list[dict[str, Any]] = []
    k_counts: Counter[int] = Counter()
    route_counts: Counter[str] = Counter()
    layer_counts: Counter[int] = Counter()
    alpha_counts: Counter[str] = Counter()

    forward_count = sum(1 if not route else len(alpha_values) for route in routes)
    progress = tqdm(
        total=len(route_indices) * (forward_count + 1),
        desc="Score route teacher",
        unit="forward",
        file=sys.stdout,
        mininterval=0.5,
    )
    for index in route_indices:
        row, extras = _prepare([rows[index]], device)
        model.set_routing_override("none", candidates)
        model.set_route_gains(None)
        model.set_prompt_shield_mask(None)
        probe.hidden = None
        model(input_ids=row["input_ids"], attention_mask=row["attention_mask"], use_cache=False)
        progress.update(1)
        if probe.hidden is None:
            progress.update(forward_count)
            del row, extras
            continue
        # Router inputs must contain prompt tokens only; positive completion
        # tokens are used to score routes, never as router features.
        prompt_mask = row["labels"].eq(-100) & row["attention_mask"].bool()
        pooled = router.raw_prompt_pool(probe.hidden, prompt_mask).squeeze(0).float().cpu()
        prompt_shield = extras.get("prompt_shield_mask")
        best_score = float("inf")
        best_route: tuple[int, ...] = ()
        best_alpha = 1.0
        for route in routes:
            alpha_candidates = (1.0,) if not route else alpha_values
            route_mask = torch.tensor(
                [[layer in route for layer in candidates]], dtype=torch.bool, device=device
            )
            for alpha in alpha_candidates:
                alpha_tensor = torch.tensor([alpha], dtype=torch.float32, device=device)
                nll = _score_action(model, row, route_mask, alpha_tensor, device, prompt_shield)
                progress.update(1)
                score = float(nll.mean()) + route_cost * len(route)
                if score < best_score:
                    best_score, best_route, best_alpha = score, route, alpha
        target = {
            "pooled_prompt": pooled,
            "route": best_route,
            "k": len(best_route),
            "alpha": best_alpha,
        }
        targets.append(target)
        k_counts[len(best_route)] += 1
        route_counts["+".join(map(str, best_route)) if best_route else "k0"] += 1
        layer_counts.update(best_route)
        alpha_counts[str(best_alpha)] += 1
        model.set_prompt_shield_mask(None)
        del row, extras
    progress.close()

    model.set_routing_override("none", candidates)
    model.set_route_gains(None)
    model.set_prompt_shield_mask(None)
    if not targets:
        raise ValueError("no route teacher labels could be produced")
    return targets, {
        "rows": len(targets),
        "k_counts": {str(k): k_counts[k] for k in range(5)},
        "route_counts": dict(route_counts),
        "layer_counts": {str(layer): layer_counts[layer] for layer in candidates},
        "alpha_counts": dict(alpha_counts),
        "catalog_routes": [list(route) for route in routes],
    }


def _balanced_binary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average positive and negative binary losses without class-count bias."""
    positive = target.bool()
    negative = ~positive
    terms = []
    if bool(positive.any()):
        terms.append(F.softplus(-logits[positive]).mean())
    if bool(negative.any()):
        terms.append(F.softplus(logits[negative]).mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0.0


def _train_router_from_oracle(
    router: AdaptiveSubsetRouter,
    targets: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, torch.Tensor]],
    model,
    probe: _Probe,
    device: torch.device,
    *,
    epochs: int,
    seed: int,
    save_path: str | Path,
    probe_layer: int,
    route_catalog: Sequence[Sequence[int]],
    validation_limit: int,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(router.parameters(), lr=5e-5, weight_decay=0.01)
    target_device = torch.device(device)
    history: list[dict[str, Any]] = []
    best_state = None
    best_val_nll = float("inf")
    k_counts: Counter[int] = Counter()
    route_counts: Counter[str] = Counter()
    layer_counts: Counter[int] = Counter()
    alpha_counts: Counter[str] = Counter()
    for target in targets:
        route = tuple(target["route"])
        k_counts[len(route)] += 1
        route_counts["+".join(map(str, route)) if route else "k0"] += 1
        layer_counts.update(route)
        alpha_counts[str(target["alpha"])] += 1

    for epoch in range(epochs):
        router.train()
        order = list(range(len(targets)))
        random.Random(seed + epoch).shuffle(order)
        progress = tqdm(order, desc=f"Router supervised epoch {epoch + 1}/{epochs}",
                        unit="example", file=sys.stdout, mininterval=0.5)
        losses = []
        for index in progress:
            target = targets[index]
            pooled = target["pooled_prompt"].to(target_device).unsqueeze(0)
            route = tuple(target["route"])
            target_mask = torch.tensor(
                [[layer in route for layer in router.candidate_layers]],
                dtype=torch.bool, device=target_device,
            )
            target_k = torch.tensor([len(route)], dtype=torch.long, device=target_device)
            alpha_index = min(
                range(len(router.alpha_values)),
                key=lambda i: abs(router.alpha_values[i] - float(target["alpha"])),
            )
            target_alpha = torch.tensor([alpha_index], dtype=torch.long, device=target_device)
            optimizer.zero_grad(set_to_none=True)
            utilities, synergy, k_logits, alpha_logits = router.encode_pooled(pooled)
            k_loss = F.cross_entropy(k_logits, target_k)
            layer_loss = _balanced_binary_loss(utilities, target_mask)
            if len(route) >= 2:
                synergy_target = target_mask.unsqueeze(2) & target_mask.unsqueeze(1)
                off_diagonal = ~torch.eye(
                    router.n_candidates, dtype=torch.bool, device=target_device
                ).unsqueeze(0)
                synergy_loss = _balanced_binary_loss(
                    synergy.masked_select(off_diagonal),
                    synergy_target.masked_select(off_diagonal),
                )
            else:
                synergy_loss = synergy.sum() * 0.0
            alpha_loss = (
                F.cross_entropy(alpha_logits, target_alpha)
                if route else alpha_logits.sum() * 0.0
            )
            loss = k_loss + layer_loss + 0.25 * synergy_loss + 0.5 * alpha_loss
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            progress.set_postfix(loss=f"{losses[-1]:.4f}", k=len(route))
        progress.close()
        val_nll = _evaluate_router_nll(
            model, router, validation_rows, probe, target_device, validation_limit
        )
        record = {
            "epoch": epoch + 1,
            "supervised_loss": sum(losses) / max(len(losses), 1),
            "validation_nll": val_nll,
            "optimizer_steps": len(losses),
            "teacher_k_counts": {str(k): k_counts[k] for k in range(5)},
            "teacher_route_counts": dict(route_counts),
            "teacher_layer_counts": {str(layer): layer_counts[layer] for layer in router.candidate_layers},
            "teacher_alpha_counts": dict(alpha_counts),
        }
        history.append(record)
        print(f"[adaptive-router] {json.dumps(record)}", flush=True)
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {name: value.detach().cpu().clone() for name, value in router.state_dict().items()}
            _save_router_checkpoint(router, save_path, probe_layer, best_val_nll, route_catalog)
    if best_state is not None:
        router.load_state_dict(best_state)
        _save_router_checkpoint(router, save_path, probe_layer, best_val_nll, route_catalog)
    return {
        "router": router,
        "history": history,
        "best_validation_nll": best_val_nll,
        "checkpoint": str(save_path),
        "teacher_labels": {"rows": len(targets), "k_counts": {str(k): k_counts[k] for k in range(5)},
                           "route_counts": dict(route_counts), "layer_counts": dict(layer_counts),
                           "alpha_counts": dict(alpha_counts)},
    }


def train_adaptive_router(
    model,
    train_rows: Sequence[dict[str, torch.Tensor]],
    validation_rows: Sequence[dict[str, torch.Tensor]],
    candidate_layers: Sequence[int],
    device: torch.device | str,
    *,
    epochs: int,
    save_path: str | Path,
    probe_layer: int,
    seed: int = 42,
    max_train_rows: int = 512,
    validation_limit: int = 64,
    route_cost: float = 0.01,
    entropy_weight: float = 0.01,
    route_catalog: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Learn routes from measured catalog rewards, with pairwise fallback support."""
    if not train_rows or not validation_rows or epochs < 1:
        raise ValueError("training/validation rows and positive epochs are required")
    if len(candidate_layers) < 4:
        raise ValueError("adaptive routing needs at least four candidate layers")
    if probe_layer in candidate_layers:
        raise ValueError("the router probe layer must remain outside the steering candidates")

    target_device = torch.device(device)
    oracle_alphas = (1.0, 1.5, 2.0) if route_catalog is not None else (1.0, 1.25, 1.5, 1.75, 2.0)
    router = AdaptiveSubsetRouter(
        candidate_layers=candidate_layers,
        hidden_dim=int(model.base_model.config.hidden_size),
        n_backbone_layers=len(model._layers),
        max_active_layers=4,
        alpha_values=oracle_alphas,
    ).to(target_device)
    for parameter in model.base_model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.steer_nets.parameters():
        parameter.requires_grad_(False)
    model.eval()
    probe = _Probe(model._layers[probe_layer])
    if route_catalog is not None:
        if max_train_rows < 1 or route_cost < 0.0:
            probe.close()
            raise ValueError("oracle router needs positive max_train_rows and non-negative route_cost")
        try:
            selected_indices = list(range(len(train_rows)))
            random.Random(seed).shuffle(selected_indices)
            selected_rows = [train_rows[index] for index in selected_indices[:max_train_rows]]
            targets, teacher_stats = _build_oracle_targets(
                model,
                selected_rows,
                route_catalog,
                router,
                probe,
                target_device,
                route_cost=route_cost,
                row_limit=max_train_rows,
            )
            result = _train_router_from_oracle(
                router,
                targets,
                validation_rows,
                model,
                probe,
                target_device,
                epochs=epochs,
                seed=seed,
                save_path=save_path,
                probe_layer=probe_layer,
                route_catalog=route_catalog,
                validation_limit=validation_limit,
            )
            result["teacher_data"] = teacher_stats
            report_path = Path(save_path).with_name("session3_router_training.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report = {
                key: value for key, value in result.items() if key != "router"
            }
            report.update({
                "route_cost": float(route_cost),
                "alpha_values": list(router.alpha_values),
                "route_catalog": [list(route) for route in route_catalog],
            })
            temporary = report_path.with_suffix(report_path.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            temporary.replace(report_path)
            result["training_report"] = str(report_path)
            return result
        finally:
            probe.close()
            model.set_prompt_shield_mask(None)
            model.set_route_gains(None)
            model.set_routing_override("none", candidate_layers)
    optimizer = torch.optim.AdamW(router.parameters(), lr=5e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=target_device.type == "cuda")
    rows = list(train_rows[:max_train_rows])
    history: list[dict[str, float]] = []
    best_state = None
    best_val_nll = float("inf")

    try:
        for epoch in range(epochs):
            router.train()
            order = list(range(len(rows)))
            torch_generator = torch.Generator().manual_seed(seed + epoch)
            order = torch.randperm(len(order), generator=torch_generator).tolist()
            progress = tqdm(
                order,
                desc=f"Adaptive router epoch {epoch + 1}/{epochs}",
                unit="example",
                file=sys.stdout,
                mininterval=0.5,
            )
            losses: list[float] = []
            k_counts: Counter[int] = Counter()
            layer_counts: Counter[int] = Counter()
            alpha_counts: Counter[float] = Counter()
            for index in progress:
                item = rows[index]
                row, _extras = _prepare([item], target_device)
                model.set_routing_override("none", candidate_layers)
                model.set_route_gains(None)
                model.set_prompt_shield_mask(None)
                with torch.no_grad(), torch.autocast(
                    device_type=target_device.type,
                    enabled=target_device.type == "cuda",
                ):
                    model(
                        input_ids=row["input_ids"],
                        attention_mask=row["attention_mask"],
                        use_cache=False,
                    )
                if probe.hidden is None:
                    continue

                prompt_mask = row["labels"].eq(-100) & row["attention_mask"].bool()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=target_device.type,
                    enabled=target_device.type == "cuda",
                ):
                    action_a, log_prob_a, entropy_a = sample_route_actions(
                        router, probe.hidden, prompt_mask
                    )
                    action_b, log_prob_b, entropy_b = sample_route_actions(
                        router, probe.hidden, prompt_mask
                    )
                for action in (action_a, action_b):
                    k_counts[int(action.k.item())] += 1
                    layer_counts.update(
                        layer
                        for layer, selected in zip(candidate_layers, action.mask[0].tolist())
                        if selected
                    )
                    alpha_counts[float(action.alpha.item())] += 1

                prompt_shield = _extras.get("prompt_shield_mask")
                nll_a = _score_action(
                    model, row, action_a.mask, action_a.alpha, target_device, prompt_shield
                )
                nll_b = _score_action(
                    model, row, action_b.mask, action_b.alpha, target_device, prompt_shield
                )
                reward_a = -nll_a - route_cost * action_a.k.float()
                reward_b = -nll_b - route_cost * action_b.k.float()
                advantage = (reward_a - reward_b).detach()
                policy_loss = -((log_prob_a - log_prob_b) * advantage).mean()
                entropy = 0.5 * (entropy_a.mean() + entropy_b.mean())
                loss = policy_loss - entropy_weight * entropy
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
                progress.set_postfix(loss=f"{losses[-1]:.4f}", mean_k=f"{(action_a.k.float().mean() + action_b.k.float().mean()) / 2:.2f}")
                model.set_prompt_shield_mask(None)
                del row, _extras, nll_a, nll_b, loss
            progress.close()

            val_nll = _evaluate_router_nll(
                model, router, validation_rows, probe, target_device, validation_limit
            )
            record = {
                "epoch": float(epoch + 1),
                "policy_loss": sum(losses) / max(len(losses), 1),
                "validation_nll": val_nll,
                "optimizer_steps": float(len(losses)),
                "k_counts": {str(k): k_counts[k] for k in range(5)},
                "layer_selection_counts": {
                    str(layer): layer_counts[layer] for layer in candidate_layers
                },
                "alpha_counts": {str(alpha): alpha_counts[alpha] for alpha in router.alpha_values},
            }
            history.append(record)
            print(f"[adaptive-router] {json.dumps(record)}", flush=True)
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_state = {name: value.detach().cpu().clone() for name, value in router.state_dict().items()}
                _save_router_checkpoint(router, save_path, probe_layer, best_val_nll)

        if best_state is not None:
            router.load_state_dict(best_state)
    finally:
        probe.close()
        model.set_prompt_shield_mask(None)
        model.set_route_gains(None)
        model.set_routing_override("none", candidate_layers)

    if best_state is not None:
        _save_router_checkpoint(router, save_path, probe_layer, best_val_nll)
    destination = Path(save_path)
    return {"router": router, "history": history, "best_validation_nll": best_val_nll, "checkpoint": str(destination)}


def load_adaptive_router(model, checkpoint_path: str | Path, device):
    """Load router weights and return a prompt-prefill callback for generation."""
    target_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=True)
    candidates = tuple(int(value) for value in checkpoint["candidate_layers"])
    if candidates != tuple(model.target_layers):
        raise ValueError("router candidate layers do not match the model's steering bank")
    router = AdaptiveSubsetRouter(
        candidate_layers=candidates,
        hidden_dim=int(model.base_model.config.hidden_size),
        n_backbone_layers=len(model._layers),
        max_active_layers=int(checkpoint["max_active_layers"]),
        alpha_values=checkpoint["alpha_values"],
    ).to(target_device)
    router.load_state_dict(checkpoint["router"])
    router.eval()
    probe = _Probe(model._layers[int(checkpoint["probe_layer"])])

    @torch.no_grad()
    def prefill(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        model.set_routing_override("none", candidates)
        model.set_route_gains(None)
        probe.hidden = None
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        if probe.hidden is None:
            raise RuntimeError("router probe did not capture prompt hidden states")
        decision = router.choose(probe.hidden, attention_mask.bool())
        model.set_routing_override("matrix", decision.mask)
        model.set_route_gains(decision.alpha)

    return router, prefill, probe
