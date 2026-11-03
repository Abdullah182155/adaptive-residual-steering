"""Typed configuration and CLI override support for ARS Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


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
    amplitude_mode: str = "relative"
    rho_max: float = 0.04
    steer_penalty_weight: float = 0.05
    steer_penalty_margin: float = 1.0
    calc_weight_alpha: float = 1.0
    layer_norms_calibration: str = "norms_calibration.json"
    prompt_shielding: bool = True
    guard_revert_to_k0: bool = True
    guard_relative_norm_cap: Optional[float] = None

    batch_size: int = 2
    grad_accum: int = 8
    clip_grad: float = 1.0
    weight_decay: float = 0.01
    phase1_epochs: int = 2
    phase1_lr: float = 2e-4
    phase1_warmup_frac: float = 0.06
    phase1_min_lr_frac: float = 0.10
    log_every: int = 20
    n_samples: int = 7000
    max_tok_len: int = 1024
    val_split: float = 0.08
    diversify_prompts: bool = False
    seed: int = 42

    # V2 Typed Loss
    loss_mode: str = "typed"  # "typed" or "legacy"
    lambda_format: float = 0.05
    lambda_operator: float = 0.10
    lambda_operand: float = 0.20
    lambda_result: float = 0.35
    lambda_final: float = 0.30
    focal_gamma: float = 1.0
    kl_weight: float = 1.0

    # V2 Multi-Context Training Mixture
    train_shot_probs: List[float] = field(default_factory=lambda: [0.40, 0.15, 0.20, 0.10, 0.15])  # 0, 1, 2, 3, 4 shots
    train_use_preface: bool = True
    train_v5_demo_frac: float = 0.50
    shield_policy: str = "context"

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
        if self.amplitude_mode not in {"relative", "capped"}:
            raise ValueError("amplitude_mode must be relative or capped")
        if self.loss_mode not in {"typed", "legacy"}:
            raise ValueError("loss_mode must be 'typed' or 'legacy'")


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
