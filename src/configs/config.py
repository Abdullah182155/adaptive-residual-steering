import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
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
    lora_dropout: float = 0.0   # Engine v2.1: 0.0 eliminates train vs eval/generation adapter dropout discrepancy
    steer_magnitude_mode: str = "direct"  # ISSUE-5: direct per-token magnitude (was 'decoupled')
    steer_max_magnitude: float = 2.0
    # A residual perturbation is an angular intervention, not a new radial
    # feature.  Orthogonal projection is disabled (ISSUE-6/7) — the per-token
    # magnitude head in 'direct' mode handles magnitude control instead.
    # Engine v2.1: envelope tightened from 0.04 to 0.02 once norm regularisation is active.
    steer_max_relative_norm: float = 0.02
    steer_orthogonal_projection: bool = False  # ISSUE-6: disabled (was True)
    # A route has one shared intervention budget.  Each selected layer receives
    # K**(-power) of its own bounded delta.  A square-root route budget keeps
    # K=2 corrections viable while still preventing linear growth in aggregate
    # intervention energy.
    steer_route_budget_power: float = 0.5
    # Cap-aware constrained SteerNet norm regularisation (Engine v2.1)
    lambda_steer_norm: float = 0.60         # Strongly penalise raw pre-cap ratio excess above q_hi
    steer_norm_q_hi: float = 0.50           # Keep Phase-1 residuals in the linear operating regime
    steer_norm_huber_delta: float = 0.10    # Transition delta for smooth Huber norm penalty
    phase1_max_eval_p95_norm_ratio: float = 1.0  # Legacy telemetry threshold
    # Raw q is measured before smooth saturation.  It is an informational
    # saturation diagnostic, never a Phase-1 promotion veto.  Promotion uses
    # the realized post-cap, gate-scaled ratio below instead.
    phase1_max_generation_p95_norm_ratio: float = 1.50
    phase1_max_effective_generation_p95_norm_ratio: float = 1.00
    # Teacher forcing can reduce the loss of boilerplate reasoning tokens
    # while making the generated final answer worse.
    answer_tail_tokens: int = 8
    # Engine v2.1: modest tail weight (1.10 down from 2.0) to preserve full multi-step CoT reasoning
    answer_tail_weight: float = 1.10
    # Engine v2.1: calculation_weighted preserves supervision on reasoning prose (1.0) while boosting math (1.35)
    steer_loss_target: str = "calculation_weighted"  # 'calculation_weighted', 'calculation', or 'all'
    calc_token_weight: float = 1.35          # Multiplier for calculation tokens when in 'calculation_weighted' mode
    router_reward_type: str = "execution"   # 'execution' (reward exact calculation tokens) or 'nll' (legacy)
    router_outcome_reward_weight: float = 1.0

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
    phase1_lr: float = 3e-4
    phase1_epochs: int = 2  # Engine v3 Stage A: capacity before pair co-adaptation
    phase1_warmup_ratio: float = 0.06
    phase1_patience: int = 4
    phase1_acceptance_examples: int = 64
    # A 32-item probe has 3.125pp quantisation and accepted a +9.4pp result
    # that reversed by -12pp on an external 50-item benchmark.  Use 64 held-
    # out items by default so Phase 1 is a capacity gate rather than a noisy
    # optimistic selector.  Debug configurations may lower this explicitly;
    # their report remains diagnostic-only.
    # Legacy compatibility field.  Engine v2.3 always uses the explicit,
    # disjoint ``phase1_generation_confirmation_examples`` partition below.
    phase1_acceptance_accuracy_examples: int = 0
    # Search each (layer, residual-scale) pair on this compact, fixed
    # calibration partition.  It detects CoT collapse without making a T4
    # preflight prohibitively slow.
    phase1_generation_screen_examples: int = 12
    # Successive halving evaluates every layer/scale on this small prefix, then
    # promotes only the top safe route from each physical depth band to the
    # full calibration split.  It prevents a 32-arm screen from consuming an
    # entire Kaggle session before co-adaptation can start.
    phase1_successive_halving_examples: int = 6
    # The strongest safe calibration pairs are measured on this disjoint
    # partition before promotion.  It is deliberately independent of the
    # calibration rows.
    phase1_generation_confirmation_examples: int = 16
    # Calibration rankings are noisy on compact generation probes.  Confirm a
    # bounded number of the strongest independently safe pairs on the same
    # disjoint rows before failing closed, rather than treating a one-item
    # fluctuation from the top-ranked pair as evidence against every route.
    phase1_max_confirmation_candidates: int = 3
    # Compact-screen noise is reported but cannot promote a fixed control.
    # Every fixed SteerNet needs a strictly positive disjoint confirmation
    # gain; calibration wins and one-question tolerances are diagnostic only.
    phase1_allow_confirmation_noise_tolerance: bool = False
    # A selected calibration winner must not be promoted solely because a
    # small confirmation split happened to be positive.  Require its pooled,
    # disjoint calibration-plus-confirmation result to be non-regressing.
    phase1_require_nonnegative_combined_gain: bool = True
    # During training, screen every epoch on a smaller fixed partition and
    # conservative scales only.  Full five-scale calibration happens once.
    phase1_epoch_generation_screen_examples: int = 6
    phase1_epoch_gate_scale_grid: Tuple[float, ...] = (0.2, 0.4)
    phase1_generation_candidate_count: int = 2  # Legacy compatibility field; joint scale search uses all pairs.
    phase1_min_completion_len_ratio: float = 0.85 # Engine v2.1: reject candidate if median CoT length collapses below 85% of baseline
    deployment_examples: int = 300                # Upper bound for final held-out paired CI decision rows
    # Run 8 requires at least 120 *questions* for the deployment decision.
    # Each question is evaluated at both 0- and 2-shot, and the bootstrap
    # resamples questions as clusters rather than treating those two outcomes
    # as independent observations.
    deployment_min_examples: int = 120
    # Confirmation must cover each shot condition claimed for deployment.
    deployment_eval_shots: Tuple[int, ...] = (0, 2)
    # A larger held-out zero-shot screen, rather than a training-prompt
    # accuracy proxy, decides whether the final Phase-1 adapter is usable.
    phase1_generation_max_new_tokens: int = 256
    # Select residual intensity on validation free generation after selecting
    # a candidate layer.  This is a bounded calibration of the model's own
    # learned direction, not a test-set threshold or a hidden override.
    # Fine lower scales are essential for early and middle residuals; late
    # layers may still use larger calibrated interventions when safe.
    phase1_gate_scale_grid: Tuple[float, ...] = (0.25, 0.35, 0.50, 0.70)
    phase1_min_train_nll_reduction: float = 0.05
    phase1_min_val_nll_reduction: float = 0.05
    # Promotion requires confirmation_gain > this floor, not merely a tie.
    # A zero floor therefore means one strictly positive binomial increment.
    phase1_min_validation_accuracy_gain: float = 0.0
    phase1_min_train_accuracy_gain: float = 0.0        # Legacy alias
    phase1_accuracy_warning_gain: float = 0.10
    # A four-example holdout cannot establish a scientific capacity or
    # routing-health claim.  Tiny smoke runs still execute every phase and
    # emit their measurements, but their acceptance/health failures are
    # reported as INCONCLUSIVE rather than terminating the notebook.
    min_reliable_validation_examples: int = 16
    allow_inconclusive_small_runs: bool = True
    # A decision run must fail closed: training an adaptive policy after its
    # fixed SteerNet or deterministic router has failed produces an artefact,
    # not valid ARS evidence.  Smoke tests may opt into diagnostic continuation
    # explicitly, but production defaults never spend Phase 2 budget this way.
    continue_on_diagnostic_failure: bool = False
    skip_adaptive_phases_when_phase1_fails: bool = True
    skip_adaptive_phases_when_router_fails: bool = True

    # Phase 2 staged parameters
    phase2_warmup_ratio: float = 0.03
    phase2_patience: int = 5

    # Gate architecture
    gate_dim: int = 16
    gate_dropout: float = 0.05
    gate_init_alpha: float = 0.95  # B-3: near-open init; task loss must earn suppression
    gating_mode: str = "learned"  # 'learned' (MLP), 'constant' (fixed scalar), or 'step'
    gate_constant_value: float = 0.80
    # Engine v3: for K>=2, token gates provide local strength while this
    # masked-softmax allocation shares the route-level intervention budget.
    factorized_gate_allocation: bool = True
    gate_allocation_temperature: float = 0.70
    lambda_gate_overlap: float = 0.10
    # Engine v4: a nominal route member is effective only when it owns a
    # material fraction of realised gate-scaled residual energy.
    effective_k_min_energy_fraction: float = 0.05
    router_min_eval_route_tuple_entropy: float = 0.0

    # Gate loss regularisation
    # Do not use one global ON/OFF reward to update every token gate.  That
    # estimator has no token credit assignment and drives all alphas together
    # toward saturation.  The differentiable answer-aware task loss provides
    # the correct local gate signal; generation PG remains opt-in research.
    lambda_pg: float = 0.0
    lambda_l1_gate: float = 0.05          # Engine v2: re-enabled L1 sparsity
    lambda_gate_budget: float = 0.50      # Engine v2: quadratic budget pulling mean alpha to target_sparsity
    gate_target_sparsity: float = 0.20     # Engine v2: target average gate activation
    lambda_gate_usefulness: float = 0.0   # B-3: disabled (was 0.30)
    lambda_gate_antisat: float = 0.0      # B-3: disabled (was 0.25)
    lambda_gate_diversity: float = 0.20   # Engine v2.2: require token-selective gates
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
    # Engine v3.3: allocation is the primary capacity signal.  ``reserved``
    # includes reusable allocator cache, so it triggers reclamation rather
    # than an immediate failure.  The reservation ceiling leaves ~1.1 GiB on
    # a 15 GiB T4 for allocator/driver headroom after cache reclamation.
    cuda_active_memory_limit_mb: int = 11500
    cuda_reserved_memory_limit_mb: int = 14200
    cuda_memory_min_free_mb: int = 512
    cuda_memory_watchdog_every_n_steps: int = 50
    cuda_memory_reduced_rloo_batch_size: int = 4
    pre_cap_telemetry_reservoir_size: int = 8192

    # Checkpoint selection
    use_accuracy_checkpoint: bool = True
    accuracy_eval_n: int = 100

    # Checkpoint paths (relative by default, portable for local and Kaggle)
    lora_save_path: str = "./checkpoints/steer_phase1_adaptive_v11.pt"
    phase2_save_path: str = "./checkpoints/rsc_phase2_catalog_v14.pt"
    checkpoint_dir: str = "./checkpoints"
    phase1_report_save_path: str = "./checkpoints/phase1_report.json"
    resume_phase1: bool = False
    # Full optimizer/scaler/scheduler/RNG state for a recoverable Phase-2
    # interruption.  This is intentionally separate from portable ARS weight
    # checkpoints, which remain tensor-only for evaluation compatibility.
    phase2_training_state_path: str = "./checkpoints/phase2_training_state.pt"
    resume_phase2: bool = False
    phase2_state_checkpoint_every_n_steps: int = 50

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
    # A small compatibility update lets independently trained K=1 residuals
    # learn complementary K>=2 directions on the router split.
    bootstrap_steer_compat_lr_scale: float = 0.05
    bootstrap_joint_nll_weight: float = 0.3          # weight of K=2 joint-NLL correction
    router_rloo_batch_size: int = 8  # enables batch-level policy diversity, unlike a B=1 RLOO probe
    router_preflight_examples: int = 128
    # Retained for backward-compatible notebook configurations.  Router
    # preflight always reports its verdict; train_rsc owns the clean
    # fail-closed return before expensive adaptive phases begin.
    router_fail_fast_on_collapse: bool = True
    intervention_audit_examples: int = 128
    router_min_eval_k_support: int = 2
    router_require_eval_k0: bool = False
    router_require_eval_k_ge2: bool = True
    router_min_eval_k_ge2_rate: float = 0.02
    router_min_eval_k_entropy: float = 0.20
    router_min_eval_active_layers: int = 2
    # Distinct depth bands in the *union across the deterministic preflight
    # batch*, not a requirement that every prompt selects every band.
    router_min_eval_active_depth_bands: int = 2
    router_min_eval_layer_entropy: float = 0.25
    gate_min_eval_sequence_std: float = 0.015
    gate_max_eval_high_saturation_pct: float = 35.0
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
    # Bootstrap starts with an unbiased utility estimate and linearly ramps
    # this tariff across epoch one.  Cap the tariff to a fraction of observed
    # positive gain so pricing cannot reverse a route that beats K=0.
    router_tariff_warmup_epochs: int = 1
    router_tariff_max_positive_gain_fraction: float = 0.50
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
    lambda_router_k_mi: float = 0.15       # Engine v2: re-enabled MI diversity for K
    lambda_router_layer_mi: float = 0.15   # Engine v2: re-enabled MI diversity for Layer
    # Directly distil the three counterfactual rewards (K=0, K_a, K_b) into
    # the deterministic K head.  RLOO alone updates sampled log-probabilities
    # and allowed a global K=3 bias to survive at argmax deployment.
    lambda_router_k_preference: float = 0.25
    router_k_preference_temperature: float = 0.10
    # Retained in checkpoint-visible configuration for compatibility.  Engine
    # v2.2 uses symmetric K-value distillation, so the neutral value is 0.0.
    router_k0_margin: float = 0.0
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

    # Engine v3 curriculum: after independent capacity is established, train
    # a balanced pool of safe K=1/K=2 routes before router bootstrap.  The
    # unified adaptive stage then replaces the former 2B->2C->2D sequence.
    phase1b_coadapt_epochs: int = 1
    phase1b_candidate_layers: int = 4
    # Require a scientifically meaningful co-adaptation pool before routing.
    # Two physical bands are sufficient when they establish a long-range pair;
    # requiring all three bands made valid early+late pools impossible.
    phase1b_min_candidate_layers: int = 2
    phase1b_min_depth_bands: int = 2
    phase1b_lr: float = 5e-5
    phase1b_joint_nll_weight: float = 1.0
    unified_adaptive_phase: bool = True
    unified_adaptive_epochs: int = 2
    gate_retune_lr: float = 1e-4

    # Phase 1 establishes independent, generation-safe directions.  K>=2
    # complementarity is learned later by the explicit Phase-1.5 correction,
    # rather than co-adapting adapters before their single-layer capacity is
    # known.
    router_warmup_max_k: int = 1
    router_warmup_random_subsets: bool = False

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
            self.phase2_training_state_path,
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


# The Kaggle runner must not re-specify these values cell-by-cell.  That
# previously allowed a notebook at the correct Git commit to silently run a
# different four-epoch, high-LR experiment.  Keep the decision-run contract in
# source control and validate it immediately before expensive model loading.
ENGINE_V32_KAGGLE_PROFILE_NAME = "engine_v3_2_kaggle_t4"
ENGINE_V32_KAGGLE_PROFILE: Dict[str, Any] = {
    "lora_rank": 16,
    "lora_alpha": 32.0,
    "steer_magnitude_mode": "direct",
    "steer_max_relative_norm": 0.02,
    "steer_orthogonal_projection": False,
    "n_samples": 3000,
    "max_tok_len": 512,
    "fewshot_train_prob": 0.0,
    "fewshot_train_max_demos": 0,
    "diversify_prompts": False,
    "router_bootstrap_n_examples": 500,
    "deployment_min_examples": 120,
    "phase1_lr": 3e-4,
    "phase1_epochs": 2,
    "phase1_acceptance_accuracy_examples": 0,
    "phase1_generation_screen_examples": 12,
    "phase1_successive_halving_examples": 6,
    "phase1_generation_confirmation_examples": 16,
    "phase1_max_confirmation_candidates": 3,
    "phase1_allow_confirmation_noise_tolerance": False,
    "phase1_min_validation_accuracy_gain": 0.0,
    "phase1_max_generation_p95_norm_ratio": 1.50,
    "phase1_max_effective_generation_p95_norm_ratio": 1.00,
    "phase1_gate_scale_grid": (0.25, 0.35, 0.50, 0.70),
    "phase1_epoch_generation_screen_examples": 6,
    "phase1_epoch_gate_scale_grid": (0.2, 0.4),
    "phase1b_min_candidate_layers": 2,
    "phase1b_min_depth_bands": 2,
    "router_bootstrap_epochs": 3,
    "steer_finetune_epochs": 0,
    "gate_only_epochs": 3,
    "joint_finetune_epochs": 1,
    "gate_retune_epochs": 1,
    "continue_on_diagnostic_failure": False,
    "skip_adaptive_phases_when_phase1_fails": True,
    "skip_adaptive_phases_when_router_fails": True,
    "router_fail_fast_on_collapse": True,
    "prompt_shielding": True,
    "lambda_cost": 0.005,
    "router_relative_k_cost": 0.03,
    "router_tariff_warmup_epochs": 1,
    "router_tariff_max_positive_gain_fraction": 0.50,
    "beta_entropy_k": 0.10,
    "lambda_synergy": 0.25,
    "lambda_synergy_prior": 0.01,
    "beta_entropy_subset": 0.05,
    "router_rloo_batch_size": 8,
    "router_diversity_temperature": 0.35,
    "lambda_router_k_mi": 0.15,
    "lambda_router_layer_mi": 0.15,
    "lambda_router_k_preference": 0.25,
    "router_k_preference_temperature": 0.10,
    "router_layer_dropout": 0.0,
    "answer_tail_tokens": 8,
    "answer_tail_weight": 1.10,
    "steer_loss_target": "calculation_weighted",
    "calc_token_weight": 1.35,
    "steer_route_budget_power": 0.5,
    "lambda_steer_norm": 0.60,
    "steer_norm_q_hi": 0.50,
    "router_warmup_max_k": 1,
    "router_warmup_random_subsets": False,
    "router_k0_margin": 0.0,
    "bootstrap_steer_compat_lr_scale": 0.05,
    "lambda_pg": 0.0,
    "gate_target_sparsity": 0.20,
    "lambda_gate_budget": 0.50,
    "lambda_l1_gate": 0.05,
    "lambda_gate_diversity": 0.20,
    "router_reward_type": "execution",
    "resume_phase1": False,
    "curriculum_mode": "full",
    "use_router": True,
}
ENGINE_V32_KAGGLE_EVALUATION: Dict[str, Any] = {
    "n_test": 100,
    "shots": (0, 2, 4, 8),
    "max_new_tokens": 512,
}


def make_engine_v32_kaggle_config() -> RSCConfig:
    """Return the committed, reproducible Engine v3.2 Kaggle T4 profile."""
    cfg = RSCConfig()
    for field, value in ENGINE_V32_KAGGLE_PROFILE.items():
        setattr(cfg, field, value)
    return cfg


def assert_engine_v32_kaggle_profile(cfg: RSCConfig, strict: bool = False) -> str:
    """Fingerprint a decision-run config; permit flexible edits unless strict=True."""
    drift = {
        field: {"expected": expected, "actual": getattr(cfg, field)}
        for field, expected in ENGINE_V32_KAGGLE_PROFILE.items()
        if getattr(cfg, field) != expected
    }
    if drift:
        msg = f"{ENGINE_V32_KAGGLE_PROFILE_NAME} custom overrides detected ({len(drift)} fields): {list(drift.keys())}"
        if strict:
            raise RuntimeError(msg)
        print(f"ℹ️  [Profile Notice] {msg}")
    effective_data = {
        field: getattr(cfg, field) for field in ENGINE_V32_KAGGLE_PROFILE.keys()
    }
    payload = json.dumps(
        {"name": ENGINE_V32_KAGGLE_PROFILE_NAME, "config": effective_data},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
