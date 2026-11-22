"""Checkpoint metadata and persistence for fixed-route SteerNet runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.configs.config import RSCConfig


def checkpoint_metadata(cfg: RSCConfig) -> dict[str, Any]:
    metadata = {
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
    if cfg.steernet_arch == "prompt_conditioned_residual":
        metadata.update(
            {
                "architecture_id": "ars.prompt_conditioned_residual",
                "architecture_version": "1.0.0",
                "steering_rank": cfg.lora_rank,
                "strength_conditioning": "masked_prompt_mean",
                "strength_max": cfg.prompt_strength_max,
                "strength_scope": "per_example_per_layer",
            }
        )
    elif cfg.steernet_arch == "v2":
        metadata.update(
            {
                "architecture_id": "ars.fixed_route_low_rank_residual",
                "architecture_version": "1.0.0",
                "steering_rank": cfg.lora_rank,
                "residual_cap_statistic": cfg.steernet_norm_stat,
                "legacy_architecture_label": "v2",
            }
        )
    else:
        metadata.update(
            {
                "architecture_id": "ars.legacy_feature_residual",
                "architecture_version": "1.0.0",
                "legacy_architecture_label": cfg.steernet_arch,
            }
        )
    return metadata

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
    stored_architecture = metadata.get("steernet_arch", "v5")
    if stored_architecture != cfg.steernet_arch:
        raise ValueError("checkpoint SteerNet architecture does not match the active config")
    if metadata["route"] != list(cfg.fixed_layers):
        raise ValueError("checkpoint route does not match the active fixed route")
    if metadata.get("rank", cfg.lora_rank) != cfg.lora_rank:
        raise ValueError("checkpoint SteerNet rank does not match the active config")
    if cfg.steernet_arch == "prompt_conditioned_residual":
        expected_metadata = {
            "architecture_id": "ars.prompt_conditioned_residual",
            "architecture_version": "1.0.0",
            "steering_rank": cfg.lora_rank,
            "strength_conditioning": "masked_prompt_mean",
            "strength_scope": "per_example_per_layer",
            "strength_max": cfg.prompt_strength_max,
        }
        mismatches = {
            name: (metadata.get(name), expected)
            for name, expected in expected_metadata.items()
            if metadata.get(name) != expected
        }
        if mismatches:
            raise ValueError(f"checkpoint steering contract mismatch: {mismatches}")
    model.steer_nets.load_state_dict(state["steer_nets"])
    return metadata
