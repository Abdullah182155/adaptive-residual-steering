"""Checkpoint metadata and persistence for fixed-route SteerNet runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.configs.config import RSCConfig


def checkpoint_metadata(cfg: RSCConfig) -> dict[str, Any]:
    return {
        "steernet_arch": cfg.steernet_arch,
        "route": list(cfg.fixed_layers),
        "rank": cfg.lora_rank,
        "rho_max": cfg.rho_max,
        "norm_stat": cfg.steernet_norm_stat,
        "steer_scale": cfg.steer_scale,
        "steer_kappa": cfg.steer_kappa,
        "k_scale_mode": cfg.k_scale_mode,
        "loss_mode": cfg.loss_mode,
    }

def save_steer_weights(
    model: torch.nn.Module, path: str | Path, cfg: RSCConfig
) -> None:
    """Persist Phase 1 adapter state with enough metadata to reject mismatches."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": checkpoint_metadata(cfg),
            "steer_nets": model.steer_nets.state_dict(),
        },
        path,
    )

def load_steer_weights(
    model: torch.nn.Module,
    path: str | Path,
    cfg: RSCConfig,
    device: torch.device | None = None,
) -> dict[str, Any]:
    state = torch.load(path, map_location=device or "cpu", weights_only=True)
    metadata = state["metadata"]
    if metadata.get("steernet_arch", "v5") != cfg.steernet_arch:
        raise ValueError("checkpoint SteerNet architecture does not match the active config")
    if metadata["route"] != list(cfg.fixed_layers):
        raise ValueError("checkpoint route does not match the active fixed route")
    if metadata.get("rank", cfg.lora_rank) != cfg.lora_rank:
        raise ValueError("checkpoint SteerNet rank does not match the active config")
    model.steer_nets.load_state_dict(state["steer_nets"])
    return metadata
