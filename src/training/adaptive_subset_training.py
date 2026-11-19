"""Training utilities for a frozen-backbone, variable-subset steering bank."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.training.route_curriculum import candidate_routes, select_routes_by_depth
from src.training.rft_trainer import _enable_checkpointing_without_dropout
from src.training.batching import prepare_batch as _prepare


def _nll_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
        reduction="sum",
    )


def _write_bank_checkpoint(
    model, cfg, candidate_layers, path: str | Path, epoch: int,
    curriculum: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "format": "ars_adaptive_steering_v1",
            "architecture": "steernet_v2",
            "candidate_layers": list(candidate_layers),
            "rank": int(cfg.lora_rank),
            "rho_max": float(cfg.rho_max),
            "norm_stat": str(cfg.steernet_norm_stat),
            "norm_refs": {
                str(layer): float(model.steer_nets[str(layer)].layer_norm_ref)
                for layer in candidate_layers
            },
            "steer_nets": {
                name: value.detach().cpu().clone()
                for name, value in model.steer_nets.state_dict().items()
            },
            "epoch": int(epoch),
            "phase1_lr": float(cfg.phase1_lr),
            "frozen_backbone": True,
            "curriculum": curriculum or {},
        },
        temporary,
    )
    temporary.replace(destination)


def train_adaptive_steering_bank(
    model,
    train_rows: Sequence[dict[str, torch.Tensor]],
    validation_rows: Sequence[dict[str, torch.Tensor]],
    cfg,
    device: torch.device | str,
    *,
    epochs: int,
    save_path: str | Path,
    seed: int = 42,
    grad_accum: int = 16,
    route_examples_per_layer: int = 256,
    screen_validation_rows: int = 24,
    max_pair_routes: int = 12,
    max_triple_routes: int = 8,
    max_quad_routes: int = 6,
) -> dict[str, Any]:
    """Curriculum-train singleton, paired, triple, and quad-layer routes.

    The Phi-2 backbone stays frozen. All singleton adapters receive equal
    exposure first. Validation NLL then screens cross-depth pairs, followed by
    promising triples and quads grown from retained routes. No-steering is
    scored as an explicit k=0 action but has no trainable adapter parameters.
    """
    if not train_rows or not validation_rows or epochs < 1 or grad_accum < 1:
        raise ValueError("training/validation rows, epochs, and grad_accum must be positive")
    if route_examples_per_layer < 1 or screen_validation_rows < 1:
        raise ValueError("route training and screening sample counts must be positive")
    if any(parameter.requires_grad for parameter in model.base_model.parameters()):
        raise ValueError("Phi-2 must remain frozen while training the steering bank")
    candidates = tuple(model.target_layers)
    if len(candidates) < 4:
        raise ValueError("adaptive steering requires at least four candidate layers")
    if abs(float(cfg.phase1_lr) - 1.5e-4) > 1e-12:
        raise ValueError("phase1_lr is locked at 1.5e-4")

    target_device = torch.device(device)
    if target_device.type == "cuda" and hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    parameters = [parameter for parameter in model.steer_nets.parameters()]
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=cfg.phase1_lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=target_device.type == "cuda"
    )
    checkpointed_modules = _enable_checkpointing_without_dropout(model.base_model)
    if target_device.type == "cuda" and checkpointed_modules == 0:
        raise RuntimeError("CUDA adaptive training requires active gradient checkpointing")

    screen_rows = list(validation_rows[:screen_validation_rows])
    if not screen_rows:
        raise ValueError("route screening needs at least one validation row")
    curriculum: dict[str, Any] = {
        "depth_bands": {
            "early": [layer for layer in candidates if layer < 14],
            "middle": [layer for layer in candidates if 14 <= layer < 22],
            "late": [layer for layer in candidates if layer >= 22],
        },
        "stages": {},
        "k0_validation_nll": None,
    }
    total_examples = 0

    def train_routes(routes: Sequence[tuple[int, ...]], stage: str, examples_per_route: int) -> None:
        nonlocal total_examples
        routes = [tuple(route) for route in routes if route]
        if not routes:
            return
        model.eval()
        checkpointed_modules = _enable_checkpointing_without_dropout(model.base_model)
        if target_device.type == "cuda" and checkpointed_modules == 0:
            raise RuntimeError("CUDA adaptive training requires active gradient checkpointing")
        progress = tqdm(
            total=len(routes) * min(examples_per_route, len(train_rows)) * epochs,
            desc=f"Adaptive bank {stage}", unit="example", file=sys.stdout, mininterval=0.5,
        )
        for route_index, route in enumerate(routes):
            seed_indices = list(range(len(train_rows)))
            random.Random(seed + route_index + len(route) * 1009).shuffle(seed_indices)
            selected_indices = seed_indices[: min(examples_per_route, len(train_rows))]
            mask = torch.tensor(
                [[layer in route for layer in candidates]], dtype=torch.bool, device=target_device
            )
            model.set_routing_override("matrix", mask)
            model.set_route_gains(torch.ones(1, device=target_device))
            for epoch in range(epochs):
                order = list(selected_indices)
                random.Random(seed + route_index * 100 + epoch).shuffle(order)
                optimizer.zero_grad(set_to_none=True)
                for start in range(0, len(order), grad_accum):
                    window = order[start : start + grad_accum]
                    tokens_in_window = sum(
                        int(train_rows[index]["labels"][1:].ne(-100).sum()) for index in window
                    )
                    if tokens_in_window <= 0:
                        progress.update(len(window))
                        continue
                    for index in window:
                        row, extras = _prepare([train_rows[index]], target_device)
                        model.set_prompt_shield_mask(extras.get("prompt_shield_mask"))
                        with torch.autocast(
                            device_type=target_device.type,
                            enabled=target_device.type == "cuda",
                        ):
                            logits = model(
                                input_ids=row["input_ids"],
                                attention_mask=row["attention_mask"],
                                use_cache=False,
                            ).logits
                            nll_sum = _nll_sum(logits, row["labels"])
                            loss = nll_sum / tokens_in_window
                        if loss.requires_grad:
                            scaler.scale(loss).backward()
                        del loss, nll_sum, logits, row, extras
                        progress.update(1)
                    if any(parameter.grad is not None for parameter in parameters):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(parameters, cfg.clip_grad)
                        scaler.step(optimizer)
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    model.set_prompt_shield_mask(None)
            total_examples += len(selected_indices) * epochs
        progress.close()
        model.set_routing_override("none", candidates)
        model.set_route_gains(None)
        model.set_prompt_shield_mask(None)

    @torch.no_grad()
    def score_routes(routes: Sequence[tuple[int, ...]], stage: str) -> list[tuple[tuple[int, ...], float]]:
        model.eval()
        scored: list[tuple[tuple[int, ...], float]] = []
        progress = tqdm(
            total=(len(routes) + int(() not in routes)) * len(screen_rows), desc=f"Screen routes {stage}",
            unit="route-example", file=sys.stdout, mininterval=0.5,
        )
        route_sets = [tuple(route) for route in routes]
        if () not in route_sets:
            route_sets = [(), *route_sets]
        for route in route_sets:
            mask = torch.tensor(
                [[layer in route for layer in candidates]], dtype=torch.bool, device=target_device
            )
            model.set_routing_override("matrix", mask)
            model.set_route_gains(torch.ones(1, device=target_device))
            nll_total = 0.0
            token_total = 0
            for item in screen_rows:
                row, extras = _prepare([item], target_device)
                model.set_prompt_shield_mask(extras.get("prompt_shield_mask"))
                with torch.autocast(
                    device_type=target_device.type,
                    enabled=target_device.type == "cuda",
                ):
                    logits = model(
                        input_ids=row["input_ids"],
                        attention_mask=row["attention_mask"],
                        use_cache=False,
                    ).logits
                    nll = _nll_sum(logits, row["labels"])
                token_count = int(row["labels"][:, 1:].ne(-100).sum())
                nll_total += float(nll.float())
                token_total += token_count
                del nll, logits, row, extras
                progress.update(1)
            scored.append((route, nll_total / max(token_total, 1)))
        progress.close()
        model.set_routing_override("none", candidates)
        model.set_route_gains(None)
        model.set_prompt_shield_mask(None)
        return scored

    def screen_and_select(
        stage: str, routes: Sequence[tuple[int, ...]], *, k: int, limit: int
    ) -> list[tuple[int, ...]]:
        scored = score_routes(routes, stage)
        nonempty = [(route, value) for route, value in scored if route]
        selected = select_routes_by_depth(
            nonempty, k=k, per_bucket=4, limit=limit
        )
        empty_score = next((value for route, value in scored if not route), None)
        if empty_score is not None:
            curriculum["k0_validation_nll"] = empty_score
        curriculum["stages"][stage] = {
            "candidates": [list(route) for route in routes],
            "selected": [list(route) for route in selected],
            "validation_token_mean_nll": {"+".join(map(str, route)): value for route, value in scored},
        }
        return selected

    model.train()
    singles = candidate_routes(candidates, 1)
    train_routes(singles, "singletons", route_examples_per_layer)
    selected_singles = screen_and_select("singletons", singles, k=1, limit=len(singles))
    _write_bank_checkpoint(model, cfg, candidates, save_path, 1, curriculum)

    pair_candidates = candidate_routes(candidates, 2)
    pairs = screen_and_select("pair_screen", pair_candidates, k=2, limit=max_pair_routes)
    trained_pairs = list(pairs)
    train_routes(pairs, "pairs", max(64, route_examples_per_layer * 2 // 3))
    pair_scores = score_routes(pairs, "pairs_after_joint_training")
    pairs = select_routes_by_depth(
        [(route, score) for route, score in pair_scores if route],
        k=2, per_bucket=4, limit=max_pair_routes,
    )
    curriculum["stages"]["pairs"] = {
        "candidates": [list(route) for route in trained_pairs],
        "selected": [list(route) for route in pairs],
        "validation_token_mean_nll": {"+".join(map(str, route)): value for route, value in pair_scores},
    }
    _write_bank_checkpoint(model, cfg, candidates, save_path, 2, curriculum)

    triple_candidates = candidate_routes(candidates, 3, selected_routes=pairs)
    triples = screen_and_select("triple_screen", triple_candidates, k=3, limit=max_triple_routes)
    trained_triples = list(triples)
    train_routes(triples, "triples", max(48, route_examples_per_layer // 2))
    triple_scores = score_routes(triples, "triples_after_joint_training")
    triples = select_routes_by_depth(
        [(route, score) for route, score in triple_scores if route],
        k=3, per_bucket=3, limit=max_triple_routes,
    )
    curriculum["stages"]["triples"] = {
        "candidates": [list(route) for route in trained_triples],
        "selected": [list(route) for route in triples],
        "validation_token_mean_nll": {"+".join(map(str, route)): value for route, value in triple_scores},
    }
    _write_bank_checkpoint(model, cfg, candidates, save_path, 3, curriculum)

    quad_candidates = candidate_routes(candidates, 4, selected_routes=triples)
    quads = screen_and_select("quad_screen", quad_candidates, k=4, limit=max_quad_routes)
    trained_quads = list(quads)
    train_routes(quads, "quads", max(32, route_examples_per_layer // 3))
    quad_scores = score_routes(quads, "quads_after_joint_training")
    quads = select_routes_by_depth(
        [(route, score) for route, score in quad_scores if route],
        k=4, per_bucket=2, limit=max_quad_routes,
    )
    curriculum["stages"]["quads"] = {
        "candidates": [list(route) for route in trained_quads],
        "selected": [list(route) for route in quads],
        "validation_token_mean_nll": {"+".join(map(str, route)): value for route, value in quad_scores},
    }
    curriculum["catalog"] = [[], *[list(route) for route in selected_singles],
                              *[list(route) for route in pairs],
                              *[list(route) for route in triples],
                              *[list(route) for route in quads]]
    _write_bank_checkpoint(model, cfg, candidates, save_path, epochs, curriculum)
    model.set_routing_override("none", candidates)
    model.set_route_gains(None)
    return {
        "checkpoint": str(save_path),
        "candidate_layers": list(candidates),
        "route_catalog": curriculum["catalog"],
        "curriculum": curriculum,
        "trained_examples": total_examples,
    }
