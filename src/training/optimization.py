"""Optimizer parameter grouping and learning-rate schedules shared by trainers."""

from __future__ import annotations

import math
from typing import Any

from src.configs.config import RSCConfig


def steering_parameter_groups(model: Any, cfg: RSCConfig) -> list[dict[str, Any]]:
    """Group trainable steering parameters with the configured decay policy."""
    decay, no_decay, up = [], [], []
    for name, parameter in model.steer_nets.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("up.weight"):
            up.append(parameter)
        elif name.endswith("mag_head.bias"):
            decay.append(parameter)
        elif parameter.ndim >= 2 and ".conv." not in name:
            decay.append(parameter)
        else:
            no_decay.append(parameter)

    has_mag_head = getattr(cfg, "steernet_use_mag_head", getattr(cfg, "use_mag_head", True))
    up_weight_decay = cfg.weight_decay if has_mag_head else 0.0
    return [
        {"params": decay, "lr": cfg.phase1_lr, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "lr": cfg.phase1_lr, "weight_decay": 0.0},
        {"params": up, "lr": cfg.phase1_lr, "weight_decay": up_weight_decay},
    ]


def warmup_cosine_lr(step: int, total: int, warmup: int, floor: float) -> float:
    """Return linear warmup followed by cosine decay to ``floor``."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(1, total - warmup)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
