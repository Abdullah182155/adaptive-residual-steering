"""Training utilities for a frozen-backbone, variable-subset steering bank."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.training.rft_trainer import _enable_checkpointing_without_dropout
from src.training.trainer import _prepare


def _sample_subset(
    candidates: Sequence[int], k: int, rng: random.Random
) -> list[int]:
    """Sample a route while spreading multi-layer choices across model depth."""
    if k == 0:
        return []
    ordered = sorted(candidates)
    buckets = [ordered[i::3] for i in range(3)]
    selected: list[int] = []
    for bucket in buckets:
        if bucket and len(selected) < k and rng.random() < 0.7:
            selected.append(rng.choice(bucket))
    remaining = [layer for layer in ordered if layer not in selected]
    if len(selected) < k:
        selected.extend(rng.sample(remaining, k - len(selected)))
    return sorted(selected)


def _nll_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
        reduction="sum",
    )


def _write_bank_checkpoint(model, cfg, candidate_layers, path: str | Path, epoch: int) -> None:
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
        },
        temporary,
    )
    temporary.replace(destination)


def train_adaptive_steering_bank(
    model,
    train_rows: Sequence[dict[str, torch.Tensor]],
    cfg,
    device: torch.device | str,
    *,
    epochs: int,
    save_path: str | Path,
    seed: int = 42,
    grad_accum: int = 16,
) -> dict[str, Any]:
    """Train all candidate adapters on sampled 0–4-layer subsets and positive NLL.

    The backbone stays frozen. Each accumulation window samples a balanced
    cardinality schedule; adapters selected together receive gradients from the
    same completion objective, exposing pairwise and higher-order interactions.
    """
    if not train_rows or epochs < 1 or grad_accum < 1:
        raise ValueError("training rows, epochs, and grad_accum must be positive")
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
    rng = random.Random(seed)
    history: list[dict[str, float]] = []
    alpha_values = (1.0, 1.25, 1.5, 1.75, 2.0)
    max_k = min(4, len(candidates))

    for epoch in range(epochs):
        model.train()
        checkpointed_modules = _enable_checkpointing_without_dropout(model.base_model)
        if target_device.type == "cuda" and checkpointed_modules == 0:
            raise RuntimeError("CUDA adaptive training requires active gradient checkpointing")
        order = list(range(len(train_rows)))
        random.Random(seed + epoch).shuffle(order)
        progress = tqdm(
            range(len(order)),
            desc=f"Adaptive steering epoch {epoch + 1}/{epochs}",
            unit="example",
            file=sys.stdout,
            mininterval=0.5,
        )
        epoch_loss = 0.0
        seen = 0
        cardinality_counts: Counter[int] = Counter()
        layer_counts: Counter[int] = Counter()
        subset_counts: Counter[tuple[int, ...]] = Counter()
        alpha_counts: Counter[float] = Counter()
        optimizer.zero_grad(set_to_none=True)

        for window_start in range(0, len(order), grad_accum):
            window = order[window_start : window_start + grad_accum]
            token_count = sum(
                int(train_rows[index]["labels"][1:].ne(-100).sum()) for index in window
            )
            if token_count <= 0:
                progress.update(len(window))
                continue
            # A shuffled balanced cycle guarantees that all cardinalities are
            # represented across training, including the no-steering action.
            cardinalities = [index % (max_k + 1) for index in range(window_start, window_start + len(window))]
            rng.shuffle(cardinalities)
            for index, k in zip(window, cardinalities):
                subset = _sample_subset(candidates, k, rng)
                cardinality_counts[k] += 1
                layer_counts.update(subset)
                subset_counts[tuple(subset)] += 1
                route = torch.tensor(
                    [[layer in subset for layer in candidates]],
                    dtype=torch.bool,
                    device=target_device,
                )
                alpha = rng.choice(alpha_values) if k else 1.0
                alpha_counts[alpha] += 1
                row, extras = _prepare([train_rows[index]], target_device)
                model.set_routing_override("matrix", route)
                model.set_route_gains(torch.tensor([alpha], device=target_device))
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
                    loss = _nll_sum(logits, row["labels"]) / token_count
                if loss.requires_grad:
                    scaler.scale(loss).backward()
                epoch_loss += float(loss.detach())
                seen += 1
                del loss, logits, row, extras
                progress.update(1)

            if not any(parameter.grad is not None for parameter in parameters):
                optimizer.zero_grad(set_to_none=True)
                model.set_prompt_shield_mask(None)
                model.set_route_gains(None)
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, cfg.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            model.set_prompt_shield_mask(None)
            model.set_route_gains(None)

        progress.close()
        record = {
            "epoch": float(epoch + 1),
            "mean_token_nll": epoch_loss / max(seen, 1),
            "examples": float(seen),
            "k_counts": {str(k): cardinality_counts[k] for k in range(max_k + 1)},
            "layer_selection_counts": {str(layer): layer_counts[layer] for layer in candidates},
            "unique_subsets": len(subset_counts),
            "alpha_counts": {str(alpha): alpha_counts[alpha] for alpha in alpha_values},
        }
        history.append(record)
        _write_bank_checkpoint(model, cfg, candidates, save_path, epoch + 1)
        print(f"[adaptive-steering] {json.dumps(record)}", flush=True)

    model.set_routing_override("none", candidates)
    model.set_route_gains(None)
    return {"history": history, "checkpoint": str(save_path), "candidate_layers": list(candidates)}
