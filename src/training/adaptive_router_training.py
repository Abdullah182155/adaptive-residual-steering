"""Offline contextual-bandit training for the adaptive subset router."""

from __future__ import annotations

import json
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


def _save_router_checkpoint(router, path: str | Path, probe_layer: int, best_val_nll: float) -> None:
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
        prompt_mask = extras["prompt_shield_mask"].bool()
        prompt_mask &= row["attention_mask"].bool()
        decision = router.choose(probe.hidden, prompt_mask)
        route = decision.mask
        nll = _score_action(
            model,
            row,
            route,
            decision.alpha,
            device,
            extras["prompt_shield_mask"].bool(),
        )
        values.append(float(nll.mean()))
    model.set_routing_override("none", candidates)
    model.set_route_gains(None)
    model.set_prompt_shield_mask(None)
    return sum(values) / max(len(values), 1)


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
) -> dict[str, Any]:
    """Learn per-question k, subset, and alpha from paired completion-NLL rewards."""
    if not train_rows or not validation_rows or epochs < 1:
        raise ValueError("training/validation rows and positive epochs are required")
    if len(candidate_layers) < 4:
        raise ValueError("adaptive routing needs at least four candidate layers")
    if probe_layer in candidate_layers:
        raise ValueError("the router probe layer must remain outside the steering candidates")

    target_device = torch.device(device)
    router = AdaptiveSubsetRouter(
        candidate_layers=candidate_layers,
        hidden_dim=int(model.base_model.config.hidden_size),
        n_backbone_layers=len(model._layers),
        max_active_layers=4,
    ).to(target_device)
    for parameter in model.base_model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.steer_nets.parameters():
        parameter.requires_grad_(False)
    model.eval()
    probe = _Probe(model._layers[probe_layer])
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

                prompt_mask = _extras["prompt_shield_mask"].bool()
                prompt_mask &= row["attention_mask"].bool()
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
