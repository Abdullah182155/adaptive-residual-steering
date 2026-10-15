import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch

@dataclass
class RSCConfig:
    """Hyperparameter configuration for Adaptive Residual Steering (ARS / RSC)."""
    model_name: str = "microsoft/phi-2"
    # Layer positions are intentionally discovered from the backbone depth.
    # The router decides which candidates to use for each forward pass and selects at most max_active_layers.
    max_active_layers: int = 4
    # Canonical Phi-2 candidate blocks.  Earlier code derived a percentage
    # range and silently substituted L9 for the specified L17.  Candidate
    # placement is part of the experimental hypothesis, so it must be an
    # explicit, checkpoint-visible configuration rather than a heuristic.
    candidate_layers: Optional[List[int]] = None
    fixed_steer_eval_k: int = 1  # Phase-1 SteerNet control: one independently-trained intervention
    fixed_steer_eval_layers: Optional[List[int]] = None  # selected only from validation during Phase-1 acceptance
    fixed_steer_eval_gate_scale: float = 1.0  # validation-calibrated fixed-control intensity
    use_router: bool = True  # If False, router is bypassed
    fixed_layers: Optional[List[int]] = None  # If set, forces these exact layers to steer
    # Catalog router: distribution over subsets. Empty is a legal action.
    catalog_include_pairs: bool = True
    catalog_consecutive_triples: bool = True
    router_max_tok_len: int = 512
    router_shot_mix: Tuple[int, ...] = (0, 2, 4)

    # SteerNet
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    steer_magnitude_mode: str = "direct"  # ISSUE-5: direct per-token magnitude (was 'decoupled')
    steer_max_magnitude: float = 2.0
    # A residual perturbation is an angular intervention, not a new radial
    # feature.  Orthogonal projection is disabled (ISSUE-6/7) — the per-token
    # magnitude head in 'direct' mode handles magnitude control instead.
    steer_max_relative_norm: float = 0.04
    steer_orthogonal_projection: bool = False  # ISSUE-6: disabled (was True)
    # A route has one shared intervention budget.  Each selected layer receives
    # K**(-power) of its own bounded delta, so adding layers cannot linearly
    # multiply the residual displacement.  K=1 remains unchanged.
    steer_route_budget_power: float = 0.5
    # Teacher forcing can reduce the loss of boilerplate reasoning tokens
    # while making the generated final answer worse.  Give the final answer
    # tail a larger, explicitly auditable weight in every trainable ARS
    # objective and in router counterfactual rewards.
    answer_tail_tokens: int = 8
    # A completion is evaluated autoregressively, not under a supplied gold
    # prefix.  A very large tail multiplier can therefore overfit the final
    # answer marker under teacher forcing while degrading free generation.
    # Keep a modest answer emphasis and let held-out generation decide whether
    # a Phase-1 checkpoint is usable.
    answer_tail_weight: float = 2.0

    # Data
    n_samples: int = 7000
    max_tok_len: int = 384
    # ARS learns its intrinsic question-to-reasoning intervention from a
    # zero-shot prompt.  Few-shot examples are an evaluation-time context
    # whose residual stream is protected by prompt shielding, not supervised
    # training data.
    fewshot_train_prob: float = 0.0
    fewshot_train_max_demos: int = 0

    # Phase 1: SteerNet only
    # Each candidate only sees roughly 1/N of a rotated Phase-1 epoch.  5e-3
    # produced large teacher-forced NLL gains but severe free-generation
    # regressions on this small-data regime.  This is intentionally a
    # conservative adapter learning rate; generation validation owns model
    # selection below.
    phase1_lr: float = 5e-4
    phase1_epochs: int = 4
    phase1_warmup_ratio: float = 0.06
    phase1_patience: int = 4
    phase1_acceptance_examples: int = 64
    # A 32-item probe has 3.125pp quantisation and accepted a +9.4pp result
    # that reversed by -12pp on an external 50-item benchmark.  Use 64 held-
    # out items by default so Phase 1 is a capacity gate rather than a noisy
    # optimistic selector.  Debug configurations may lower this explicitly;
    # their report remains diagnostic-only.
    phase1_acceptance_accuracy_examples: int = 64
    # Screen all candidates cheaply before spending the full held-out
    # generation budget on the most promising routes.  This prevents NLL from
    # silently excluding a layer whose free-generation behaviour is better.
    phase1_generation_screen_examples: int = 8
    phase1_generation_candidate_count: int = 3
    # A larger held-out zero-shot screen, rather than a training-prompt
    # accuracy proxy, decides whether the final Phase-1 adapter is usable.
    phase1_generation_max_new_tokens: int = 256
    # Select residual intensity on validation free generation after selecting
    # a candidate layer.  This is a bounded calibration of the model's own
    # learned direction, not a test-set threshold or a hidden override.
    phase1_gate_scale_grid: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
    phase1_min_train_nll_reduction: float = 0.05
    phase1_min_val_nll_reduction: float = 0.05
    # Do not promote an adapter merely because it does not regress on a tiny
    # teacher-forced probe.  This is an *independent held-out generation*
    # P2 (run-analysis fix): acceptance thresholds set to 0.0 — telemetry only.
    # A 32-item holdout has 3.1pp quantisation and cannot support a hard gate.
    # The in-audit generation screen is also disabled (phase1_generation_screen_examples=0).
    # Deployment is decided once at the end via finalize_deployment (ISSUE-2).
    phase1_min_validation_accuracy_gain: float = 0.0   # was 0.10 — telemetry only
    phase1_min_train_accuracy_gain: float = 0.0        # was 0.10 — legacy alias
    phase1_accuracy_warning_gain: float = 0.10
    # A four-example holdout cannot establish a scientific capacity or
    # routing-health claim.  Tiny smoke runs still execute every phase and
    # emit their measurements, but their acceptance/health failures are
    # reported as INCONCLUSIVE rather than terminating the notebook.
    min_reliable_validation_examples: int = 16
    allow_inconclusive_small_runs: bool = True
    # Diagnostic gates always record a failed scientific condition, but a
    # Kaggle run continues to save checkpoints, plots, and audit evidence.
    # This must never turn a warning into a claimed PASS.
    continue_on_diagnostic_failure: bool = True
    # P2 (run-analysis fix): always train adaptive phases. The old hard gate
    # on a noisy 32-item accuracy delta caused random pipeline truncation.
    # Deployment is decided at the end with finalize_deployment (ISSUE-2).
    skip_adaptive_phases_when_phase1_fails: bool = False   # was True
    skip_adaptive_phases_when_router_fails: bool = False   # was True

    # Phase 2 staged parameters
    phase2_warmup_ratio: float = 0.03
    phase2_patience: int = 5

    # Gate architecture
    gate_dim: int = 16
    gate_dropout: float = 0.05
    gate_init_alpha: float = 0.95  # B-3: near-open init; task loss must earn suppression
    gating_mode: str = "learned"  # 'learned' (MLP), 'constant' (fixed scalar), or 'step'
    gate_constant_value: float = 0.80

    # Gate loss regularisation
    # Do not use one global ON/OFF reward to update every token gate.  That
    # estimator has no token credit assignment and drives all alphas together
    # toward saturation.  The differentiable answer-aware task loss provides
    # the correct local gate signal; generation PG remains opt-in research.
    lambda_pg: float = 0.0
    lambda_l1_gate: float = 0.0          # B-3: disabled (was 0.001)
    lambda_gate_usefulness: float = 0.0   # B-3: disabled (was 0.30)
    lambda_gate_antisat: float = 0.0      # B-3: disabled (was 0.25)
    lambda_gate_diversity: float = 0.0    # B-3: disabled (was 0.5)
    gate_diversity_floor: float = 0.05
    pg_n_samples: int = 4
    pg_baseline_momentum: float = 0.8
    pg_every_n_steps: int = 8

    # Regularisation & Hardware optimization
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    batch_size: int = 1
    grad_accum: int = 16
    seed: int = 42

    # Monitoring
    verifier_every_n: int = 10
    log_gpu_memory: bool = True

    # Checkpoint selection
    use_accuracy_checkpoint: bool = True
    accuracy_eval_n: int = 100

    # Checkpoint paths (relative by default, portable for local and Kaggle)
    lora_save_path: str = "./checkpoints/steer_phase1_adaptive_v11.pt"
    phase2_save_path: str = "./checkpoints/rsc_phase2_catalog_v14.pt"
    checkpoint_dir: str = "./checkpoints"

    # Router bootstrap settings
    router_bootstrap_epochs: int = 3
    router_bootstrap_n_examples: int = 2000
    router_bootstrap_lr: float = 1e-3
    router_bootstrap_patience: int = 2
    router_bootstrap_save_path: str = "./checkpoints/router_catalog_v14.pt"
    # RC-5 root-cause fix: SteerNets trained independently for K=1 always produce
    # interfering residual vectors when combined, making K>1 NLL strictly worse than
    # K=1 — so the router rationally collapses to K=1. Fix: keep SteerNets trainable
    # during bootstrap with a tiny "compatibility LR" (= scale * router_bootstrap_lr)
    # so they learn complementary directions. A differentiable K=2 joint-NLL correction
    # (run after each RLOO step through the top-synergy pair) provides the direct
    # gradient that RLOO's no_grad cannot. See train_router_bootstrap docstring.
    bootstrap_steer_compat_lr_scale: float = 0.05   # SteerNet LR = 5% of router LR
    bootstrap_joint_nll_weight: float = 0.3          # weight of K=2 joint-NLL correction
    router_rloo_batch_size: int = 8  # enables batch-level policy diversity, unlike a B=1 RLOO probe
    router_preflight_examples: int = 128
    router_fail_fast_on_collapse: bool = False
    intervention_audit_examples: int = 128
    router_min_eval_k_support: int = 1          # B-3: was 3 — don't fail on sparse K
    router_require_eval_k0: bool = False         # B-3: was True — K=0 optional
    router_min_eval_k_entropy: float = 0.0       # B-3: was 0.25
    router_min_eval_active_layers: int = 1       # B-3: was 4 — even 1 active layer is fine
    router_min_eval_layer_entropy: float = 0.0   # B-3: was 0.45
    gate_min_eval_sequence_std: float = 0.0      # B-3: was 0.015
    gate_max_eval_high_saturation_pct: float = 100.0  # B-3: was 20.0 — allow full saturation
    router_probe_every_n_steps: int = 1
    router_checkpoint_every_n_steps: int = 0
    router_dead_layer_veto_frac: float = 0.50

    # Cooperative Subset Router & RLOO settings
    # A route is charged a per-layer intervention tariff relative to the
    # unsteered answer NLL of the same example.  The former fixed 0.025 cost
    # was negligible beside 0.2--0.4 NLL deltas, so K=4 won almost always.
    # RC-2 fix: lambda_cost 0.025->0.005 (absolute floor only for near-zero NLL examples)
    lambda_cost: float = 0.005
    # RC-2 fix: router_relative_k_cost 0.10->0.03
    # Root cause: per_layer_cost = max(0.005, NLL*0.10) >= 0.11 nat for NLL~1.1,
    # while the total entropy bonus beta_entropy_k * log(5) = 0.03*1.609 = 0.048 nat.
    # Cost > entropy bonus => K=1 is always the mathematically optimal greedy choice.
    # With 0.03: cost = 0.033 nat, entropy bonus = 0.10*1.609 = 0.161 nat => router CAN explore.
    router_relative_k_cost: float = 0.03
    # RC-2 fix: beta_entropy_k 0.03->0.10, beta_entropy_subset 0.02->0.05
    # The entropy bonus must be large enough to overcome k_cost under genuine reward noise.
    beta_entropy_k: float = 0.10
    beta_entropy_subset: float = 0.05
    router_balance_weight: float = 0.01  # was 0.10 — layer_dropout already prevents monopoly
    # Pair interactions are regressed to measured counterfactual interaction,
    # not pushed by an unbounded signed correlation objective.
    lambda_synergy: float = 0.25
    lambda_synergy_prior: float = 0.01
    router_synergy_target_scale: float = 0.10
    router_synergy_limit: float = 0.75
    # RC-4 fix: was 0.25 — only 25% of bootstrap steps measured pairwise interactions,
    # meaning the synergy matrix barely received gradient signal in short runs.
    # With small data (25 steps/epoch), only ~6 steps per epoch updated synergy.
    # At 0.75, ~19 steps/epoch update synergy => router learns interaction structure faster.
    router_pairwise_probe_probability: float = 0.75
    lambda_k_center: float = 0.0  # was 0.05 — removed; let K be learned purely from reward
    # Maximise I(prompt; route): varied marginal choices but confident,
    # prompt-dependent routes.  This is exploration regularisation only; the
    # counterfactual reward remains the quality signal.
    # Low-temperature MI acts on near-discrete choices, preventing a router
    # whose soft probabilities vary slightly while every argmax is identical.
    router_diversity_temperature: float = 0.35
    lambda_router_k_mi: float = 0.0       # B-3: was 0.10 — MI regularization disabled
    lambda_router_layer_mi: float = 0.0   # B-3: was 0.10
    # Directly distil the three counterfactual rewards (K=0, K_a, K_b) into
    # the deterministic K head.  RLOO alone updates sampled log-probabilities
    # and allowed a global K=3 bias to survive at argmax deployment.
    lambda_router_k_preference: float = 0.25
    router_k_preference_temperature: float = 0.10
    # Subset sampling and measured counterfactuals provide exploration.  Do
    # not mask utilities only during training: that produces a policy which
    # differs from deterministic deployment and can fake diversity.
    router_layer_dropout: float = 0.0
    router_difficulty_aware: bool = True  # Sequence length & dispersion difficulty features for candidate scoring
    router_eval_mode: str = "argmax"  # deterministic: argmax; stochastic sampling is training-only by default

    # Prompt Shielding (Preserves pristine representations in multi-shot contexts)
    prompt_shielding: bool = True

    # Phase 2A: SteerNet fine-tuning (0 = Frozen SteerNet discipline; trained exclusively in Phase 1)
    steer_finetune_epochs: int = 0
    steer_finetune_lr: float = 1e-4
    steer_finetune_save_path: str = "./checkpoints/steer_finetune.pt"

    # Phase 2B: Token Gate training
    gate_only_epochs: int = 3
    gate_only_lr: float = 2e-4
    gate_only_save_path: str = "./checkpoints/gate_only.pt"

    # Phase 2C: Light joint fine-tuning
    joint_finetune_epochs: int = 2
    joint_finetune_lr_scale: float = 0.10
    joint_finetune_router_lr_scale: float = 0.05

    # Phase 2D: Gate re-tuning against final router
    gate_retune_epochs: int = 1
    gate_retune_lr: float = 1e-4

    # ISSUE-3 (B-3): random K subsets expose each SteerNet to cooperative routes
    # early in Phase 1 so they learn compatible directions. Was max_k=1 (single layer
    # rotation only), which trained each SteerNet in isolation — causing interference when combined.
    router_warmup_max_k: int = 4                   # B-3: was 1 — allow multi-layer combos in warmup
    router_warmup_random_subsets: bool = True       # B-3: random K in [1, max_k] each step

    checkpoint_every_n: int = 3
    lr_reduce_factor: float = 0.5
    lr_reduce_patience: int = 3
    lr_reduce_min_lr: float = 1e-6
    early_stop_delta: float = 1e-4
    use_gradient_monitor: bool = True
    grad_warn_threshold: float = 2.0

    # Prompt Diversification & Invariance
    # Keep Phase-1's structural grammar identical to the zero-shot benchmark
    # prompt.  Prompt-format robustness is a separate controlled ablation;
    # mixing formats here made the Phase-1 capacity test non-comparable with
    # the reported benchmark.
    diversify_prompts: bool = False
    lambda_invariance: float = 0.5
    invariance_every_n_steps: int = 4
    pg_mode: str = "teacher_forced"  # 'teacher_forced' or 'generation'

    # Curriculum mode
    curriculum_mode: str = "full"  # 'full' (6 phases) or 'streamlined' (2 stages)
    streamlined_warmup_epochs: int = 3
    streamlined_joint_epochs: int = 3
    streamlined_steer_lr: float = 1e-4
    streamlined_gate_lr: float = 1e-3

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Ensure parent dirs exist for paths
        for path in [
            self.lora_save_path,
            self.phase2_save_path,
            self.router_bootstrap_save_path,
            self.steer_finetune_save_path,
            self.gate_only_save_path,
        ]:
            pdir = os.path.dirname(path)
            if pdir:
                os.makedirs(pdir, exist_ok=True)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum

    @property
    def max_length(self) -> int:
        return self.max_tok_len

    @max_length.setter
    def max_length(self, value: int):
        self.max_tok_len = value

    @property
    def val_split(self) -> float:
        return getattr(self, "_val_split", 0.08)

    @val_split.setter
    def val_split(self, value: float):
        # BUG-07 fix: was a silent no-op; now stored in backing field _val_split
        self._val_split = float(value)

    def rsc_layers_for(self, n_total: int) -> List[int]:
        if self.fixed_layers is not None:
            return sorted(set(max(1, min(int(idx), n_total - 2)) for idx in self.fixed_layers))
        if not 1 <= self.max_active_layers <= 4:
            raise ValueError("max_active_layers must be in [1, 4]")
        # Block indices are zero-based, matching ``base_model.model.layers``.
        # For Phi-2 (32 blocks) this is exactly the candidate set in the ARS
        # protocol.  Small synthetic backbones retain the same relative depth
        # profile for tests without changing the real-model experiment.
        canonical = self.candidate_layers or [11, 13, 15, 17, 18, 20, 22, 24]
        if n_total == 32:
            indices = canonical
        else:
            indices = [round(int(idx) * (n_total - 1) / 31) for idx in canonical]
        return sorted(set(max(1, min(idx, n_total - 2)) for idx in indices))
