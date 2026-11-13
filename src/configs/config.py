"""Typed configuration and CLI override support for ARS Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


@dataclass
class RSCConfig:
    """Certified defaults for fixed-route SteerNet V5 training and deployment."""

    model_name: str = "microsoft/phi-2"
    candidate_layers: List[int] = field(default_factory=lambda: [8, 11, 13, 15, 17, 19, 20, 21, 22, 24])
    fixed_layers: List[int] = field(default_factory=lambda: [8, 11, 13, 15, 17, 19, 20, 21, 22, 24])
    steernet_arch: str = "v5"
    objective: str = "nll"

    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    steer_kappa: float = 0.15
    steer_scale: float = 0.30
    k_scale_mode: str = "sqrt"
    amplitude_mode: str = "relative"
    rho_max: float = 0.04  # V23: reduced from 0.10; 2% of hidden norm avoids disrupting correct chains
    mag_head_init_bias: float = -3.0  # sigmoid(-3)≈0.05; start low so network must earn magnitude
    calc_weight_alpha: float = 1.0
    layer_norms_calibration: str = "norms_calibration.json"
    prompt_shielding: bool = True
    steernet_init_gain: float = 1.0     # V22: direction is unit-normalized; gain only affects feature expressivity
    steernet_use_mag_head: bool = True   # V22: per-token magnitude via sigmoid(mag_head(x))
    steernet_use_soft_cap: bool = False  # disabled: rho_max is the single amplitude bound
    steernet_use_conv: bool = True       # causal depthwise convolution; old code had none
    guard_revert_to_k0: bool = True

    batch_size: int = 2
    grad_accum: int = 16
    clip_grad: float = 1.0
    weight_decay: float = 0.01
    phase1_epochs: int = 5
    phase1_lr: float = 1.5e-4
    phase1_warmup_frac: float = 0.08
    phase1_min_lr_frac: float = 0.10
    phase1_plateau_factor: float = 0.5
    phase1_plateau_patience: int = 3
    phase1_plateau_min_lr: float = 5e-6
    log_every: int = 20
    n_samples: int = 7473
    max_tok_len: int = 512
    val_split: float = 0.08
    diversify_prompts: bool = True
    seed: int = 42

    # S1 answer-region upweighting — focuses gradient on the answer portion
    s1_answer_weight: float = 2.0     # V20: upweight final answer region 2× (was 1.0)
    s1_answer_region_frac: float = 0.30

    # Phase 1C: Contrastive Residual Training
    contrastive_phase: bool = False
    contrastive_epochs: int = 3
    contrastive_lr: float = 5e-5
    contrastive_weight: float = 1.0
    contrastive_nll_weight: float = 0.3
    contrastive_gen_batch: int = 4
    contrastive_temperature: float = 0.0
    contrastive_n_samples: int = 1
    contrastive_frac: float = 1.0

    # Phase 1R: Self-Taught Reasoning (STaR)
    star_k: int = 8
    star_temperature: float = 0.7
    star_epochs: int = 2
    star_lr: float = 5e-5
    star_rounds: int = 1
    star_hint_k: int = 4
    star_gen_batch: int = 2

    # V2 Typed Loss
    loss_mode: str = "typed"  # "typed", "legacy", or "legacy_pure"
    typed_learned_weights: bool = True   # Kendall auto-weighting; False = manual lambdas
    lambda_format: float = 0.02
    lambda_operator: float = 0.10
    lambda_operand: float = 0.15
    lambda_result: float = 0.40
    lambda_final: float = 0.30
    lambda_prose: float = 0.03
    focal_gamma: float = 2.0
    kl_weight: float = 0.3
    kl_arithmetic_weight: float = 0.15     # V23: compromise (V21=0.30, V22=0.05); partial arithmetic anchor prevents plan disruption
    mag_l1_weight: float = 0.10            # V23: raised from 0.03; 10× stronger pressure forces mag_head to be selective
    steer_amplitude_k: Optional[int] = None

    # V2 Multi-Context Training Mixture
    train_shot_probs: List[float] = field(default_factory=lambda: [0.50, 0.15, 0.20, 0.10, 0.05])  # V23: balanced 0-4 shot (V22 had 90% 0-shot which killed few-shot generalization)
    train_use_preface: bool = True
    train_v5_demo_frac: float = 0.90
    shield_policy: str = "context"

    # On-policy training
    on_policy: bool = True
    on_policy_frac: float = 0.20            # keep from V22: reduced from 0.50
    on_policy_start_epoch: int = 1
    on_policy_gen_batch: int = 4
    on_policy_k: int = 2                  # V23: reduced from 4; 43% correct rate makes pass@4 wasteful
    on_policy_temperature: float = 0.5    # V23: reduced from 0.7; more deterministic → higher correct rate
    on_policy_interval: int = 1
    on_policy_gold_fallback: bool = False
    on_policy_verify_steps: bool = True   # V23: reject completions with provably wrong arithmetic equations
    on_policy_shot_probs: List[float] = field(default_factory=lambda: [0.50, 0.15, 0.20, 0.10, 0.05])  # V23: match training distribution

    # Phase 2 Router
    phase2_lr: float = 5e-5
    phase2_epochs: int = 2
    router_max_k: int = 8
    router_min_k: int = 2
    router_semantic_dim: int = 16
    router_d_model: int = 32
    router_n_heads: int = 2
    router_beta_entropy_k: float = 0.05
    router_beta_entropy_subset: float = 0.30
    router_balance_weight: float = 0.1
    router_warmup_epochs: int = 0
    router_diversity_weight: float = 0.05
    router_answer_nll_weight: float = 2.0
    use_gate: bool = False
    gate_dim: int = 64
    gate_dropout: float = 0.1
    gate_init_alpha: float = 0.25
    gate_use_global_context: bool = False
    gate_lr: float = 1e-4
    gate_epochs: int = 3               # V20: GRPO converges in 2-3 epochs (was 5)
    gate_l1_weight: float = 0.0        # V20: replaced by gate_entropy_weight (kept for compat)
    gate_rloo_weight: float = 1.0
    gate_contrastive_weight: float = 2.0  # V23: raised from 1.0; stronger type discrimination needed (FINAL=0.218 in V22)
    gate_contrastive_mode: str = "hinge"
    gate_use_task_nll: bool = False
    gate_antisat_weight: float = 0.25
    gate_antisat_threshold: float = 0.85
    gate_rloo_baseline: float = 0.0
    gate_finetune_steernet: bool = True
    gate_steernet_lr: float = 1e-5
    gate_save_path: str = "checkpoints/gate.pt"
    steernet_gated_save_path: str = "checkpoints/steernet_gated.pt"
    gate_gen_reward_every: int = 4     # every N optimizer steps: generate + score accuracy
    gate_gen_subset: int = 4            # V20: 4 examples for GRPO normalization (was 2)
    gate_gen_weight: float = 2.0        # weight of generation-based PG loss
    gate_gen_max_tokens: int = 150      # max new tokens for fast gate gen reward
    gate_entropy_weight: float = 0.1   # V20: binary entropy reg (crisp 0/1 decisions)
    gate_training_mode: str = "supervised"   # V22: "supervised" (per-token advantage) or "rloo" (legacy)
    gate_supervised_temperature: float = 5.0 # V22: sigmoid(advantage * temp) sharpness for gate targets
    lambda_cost: float = 0.0
    router_probe_layer: Optional[int] = None
    router_save_path: str = "checkpoints/router.pt"

    checkpoint_dir: str = "checkpoints"
    lora_save_path: str = "checkpoints/steernet.pt"
    contract_path: str = "deployment_contract.json"
    record_path: str = "experiment_record.json"
    report_dir: str = "report"
    bench_shots: List[int] = field(default_factory=lambda: [0, 2, 4])
    generation_max_new_tokens: int = 256
    generation_temperature: float = 0.0
    generation_repetition_penalty: float = 1.1    # 1.0 = disabled
    generation_no_repeat_ngram: int = 0           # disabled: prompt tokens cause false bans in 2-shot

    def validate(self) -> None:
        """Reject configurations that violate Phase 1 contracts."""
        if self.steernet_arch != "v5":
            raise ValueError("Phase 1 supports SteerNet V5 only")
        if self.objective != "nll":
            raise ValueError("Phase 1 supports token-weighted NLL only")
        if not self.fixed_layers:
            raise ValueError("fixed_layers cannot be empty")
        if not all(0 <= l < 32 for l in self.fixed_layers):
            raise ValueError("fixed_layers contains out-of-range indices for phi-2 (0–31)")
        if self.candidate_layers and not set(self.fixed_layers).issubset(set(self.candidate_layers)):
            raise ValueError("fixed_layers must be a subset of candidate_layers")
        if self.k_scale_mode != "sqrt":
            raise ValueError("Phase 1 uses sqrt route scaling")
        if self.amplitude_mode not in {"relative", "capped", "norm_capped"}:
            raise ValueError("amplitude_mode must be relative or capped")
        if self.loss_mode not in {"typed", "legacy", "legacy_pure"}:
            raise ValueError("loss_mode must be 'typed', 'legacy', or 'legacy_pure'")


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
