"""Typed configuration and CLI override support for ARS Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


DEFAULT_SHOT_AWARE_SCALE: Dict[int, float] = {0: 1.0, 2: 0.50, 4: 0.35}


@dataclass
class RSCConfig:
    """Certified defaults for fixed-route SteerNet V5 training and deployment."""

    model_name: str = "microsoft/phi-2"
    candidate_layers: List[int] = field(default_factory=lambda: [13, 15, 18, 20, 22])
    fixed_layers: List[int] = field(default_factory=lambda: [13, 20, 22])
    steernet_arch: str = "v5"
    objective: str = "nll"

    lora_rank: int = 64
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    steer_kappa: float = 0.04
    steer_scale: float = 0.30
    k_scale_mode: str = "sqrt"
    layer_norms_calibration: str = "norms_calibration.json"
    prompt_shielding: bool = True
    shot_aware_deploy: bool = True
    fewshot_decode_only: bool = True
    guard_revert_to_k0: bool = True
    guard_relative_norm_cap: Optional[float] = None

    batch_size: int = 1
    grad_accum: int = 4
    clip_grad: float = 1.0
    weight_decay: float = 0.01
    phase1_epochs: int = 2
    phase1_lr: float = 1e-3
    n_samples: int = 7000
    max_tok_len: int = 384
    val_split: float = 0.08
    diversify_prompts: bool = False
    seed: int = 42

    checkpoint_dir: str = "checkpoints"
    lora_save_path: str = "checkpoints/steernet.pt"
    contract_path: str = "deployment_contract.json"
    record_path: str = "experiment_record.json"
    report_dir: str = "report"
    bench_shots: List[int] = field(default_factory=lambda: [0, 2, 4])
    generation_max_new_tokens: int = 256
    generation_temperature: float = 0.0

    def validate(self) -> None:
        """Reject configurations that violate the fixed-route Phase 1 contract."""
        if self.steernet_arch != "v5":
            raise ValueError("Phase 1 supports SteerNet V5 only")
        if self.objective != "nll":
            raise ValueError("Phase 1 supports token-weighted NLL only")
        if self.fixed_layers != [13, 20, 22]:
            raise ValueError("Phase 1 fixed route must be [13, 20, 22]")
        if self.k_scale_mode != "sqrt":
            raise ValueError("Phase 1 uses sqrt route scaling")


def decode_only_for_shot(shots: int, cfg: RSCConfig) -> bool:
    """Keep 0-shot prefill active; protect demonstrations in few-shot prompts."""
    return bool(cfg.fewshot_decode_only and shots > 0)


def shot_aware_scale_map(cfg: RSCConfig) -> Dict[int, float]:
    """Return deployment energy multipliers, or a neutral map when disabled."""
    return (
        dict(DEFAULT_SHOT_AWARE_SCALE)
        if cfg.shot_aware_deploy
        else {0: 1.0, 2: 1.0, 4: 1.0}
    )


def apply_overrides(cfg: RSCConfig, assignments: List[str]) -> RSCConfig:
    """Apply repeatable ``key=value`` CLI overrides with type-aware parsing."""
    known = {item.name: getattr(cfg, item.name) for item in fields(cfg)}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"override must be key=value: {assignment}")
        key, raw = assignment.split("=", 1)
        if key not in known:
            raise ValueError(f"unknown configuration key: {key}")
        current = known[key]
        if isinstance(current, bool):
            value: Any = raw.strip().lower() in {"1", "true", "yes", "on"}
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        elif isinstance(current, list):
            subtype = type(current[0]) if current else str
            value = [subtype(part.strip()) for part in raw.split(",") if part.strip()]
        else:
            value = raw
        setattr(cfg, key, value)
    cfg.validate()
    return cfg
