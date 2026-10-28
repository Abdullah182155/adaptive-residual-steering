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
    # K=0 (identity/no-op) is always available via the empty subset; max_active_layers=4 allows K in {0,1,2,3,4}.
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
    router_shot_mix: Tuple[int, ...] = (0, 2, 4, 8)

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
    # Remediation: adjusted to 0.005 to match measured envelope cap (0.00418 from S5)
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
    fewshot_train_prob: float = 0.30
    fewshot_train_max_demos: int = 2

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
    phase1_allow_nonregressing_confirmation: bool = False

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
    gate_allocation_temperature_final: float = 0.50
    token_competitive_allocation: bool = False
    lambda_gate_overlap: float = 0.10
    # Engine v4: a nominal route member is effective only when it owns a
    # material fraction of realised gate-scaled residual energy.
    effective_k_min_energy_fraction: float = 0.05
    # Zero preserves historical profiles; the v4 decision profile turns this
    # into a non-zero fail-closed contract.
    router_min_eval_route_tuple_entropy: float = 0.0
    lambda_route_activity_floor: float = 0.0
    route_activity_min_fraction: float = 0.05

    # Gate loss regularisation
    # Do not use one global ON/OFF reward to update every token gate.  That
    # estimator has no token credit assignment and drives all alphas together
    # toward saturation.  The differentiable answer-aware task loss provides
    # the correct local gate signal; generation PG remains opt-in research.
    lambda_pg: float = 0.0
    lambda_l1_gate: float = 0.01          # Engine v4.2: reduced sparsity pressure to prevent gate attenuation
    lambda_gate_budget: float = 0.15      # Engine v4.2: relaxed quadratic budget
    gate_target_sparsity: float = 0.25     # Engine v4.2: healthy target gate activation
    gate_min_activation_floor: float = 0.35 # Engine v4.2: minimum alpha envelope when gate opens
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
    # Further increased to 0.20/0.10 to strengthen K diversity incentive
    # Remediation: increased to 0.50/0.30 to prevent K-collapse and encourage diverse subset selection
    beta_entropy_k: float = 0.20
    beta_entropy_subset: float = 0.10
    router_balance_weight: float = 0.01  # was 0.10 — layer_dropout already prevents monopoly
    router_tariff_k1_free: bool = True   # Engine v4.2: K=1 is free capacity (zero tariff)
    router_min_active_k_fraction: float = 0.75 # Engine v4.2: at least 75% reasoning tokens steer
    lambda_layer_coverage: float = 0.20  # Engine v4.2: anti-starvation layer coverage loss
    layer_min_coverage_prob: float = 0.10 # Engine v4.2: minimum selection floor per candidate layer
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
    router_k_preference_temperature_final: float = 0.15
    router_k_preference_anneal_epochs: int = 0
    generation_anchor_interval: int = 200
    generation_anchor_n_prompts: int = 8
    generation_anchor_weight: float = 0.0
    # Retained in checkpoint-visible configuration for compatibility.  Engine
    # v2.2 uses symmetric K-value distillation, so the neutral value is 0.0.
    router_k0_margin: float = 0.0
    # Subset sampling and measured counterfactuals provide exploration.  Do
    # not mask utilities only during training: that produces a policy which
    # differs from deterministic deployment and can fake diversity.
    router_layer_dropout: float = 0.0
    router_k0_enabled: bool = True
    # Preserve checkpoint topology by keeping augmented few-shot features
    # opt-in; Engine v4's committed profile enables them.
    router_demo_features: bool = False
    require_adaptive_beats_fixed: bool = False
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
    phase1b_candidate_layers: int = 6
    phase1b_candidates_per_band: int = 2
    # Require a scientifically meaningful co-adaptation pool before routing.
    # Two physical bands are sufficient when they establish a long-range pair;
    # requiring all three bands made valid early+late pools impossible.
    phase1b_min_candidate_layers: int = 2
    phase1b_min_depth_bands: int = 2
    phase1b_lr: float = 5e-5
    phase1b_joint_nll_weight: float = 1.0
    unified_adaptive_phase: bool = False
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

    # ==== M0 ====
    failure_policy: str = "warn"                  # "warn" (default) | "strict" (CI only)
    manifest_path: str = "./run_manifest.json"
    max_consecutive_nonfinite_loss: int = 20
    steer_scale: float = 0.30                     # s_ref: certified Phase 1 scale

    # ==== M6a ====
    k_scale_mode: str = "sqrt"            # "sqrt" | "none"
    steernet_k_conditioning: bool = True
    gate_task_gradient: bool = False      # consumed in M6b/M7; declared now to avoid later churn

    # ==== M1 ====
    catalog_layers: Tuple[int, ...] = (13, 15, 18, 20, 22)   # must be subset of model.target_layers
    catalog_max_k: int = 3
    cf_table_dir: str = "./checkpoints/cf_tables"            # under checkpoints/ (git-ignored)
    cf_shard_size: int = 25
    cf_rows_per_forward: Tuple[Tuple[int, int], ...] = ((0, 13), (2, 8), (4, 7), (8, 5))   # (n_shot, rows per forward)
    cf_audit_plan: Tuple[Tuple[int, int], ...] = ((0, 300), (4, 150), (8, 150))            # (n_shot, n_rows)
    cf_heldout_n: int = 120                                  # 0-shot rows from eval_data

    # ==== M2 ====
    audit_g_fixed_min: float = 0.01
    audit_h_low: float = 0.005
    audit_h_high: float = 0.02
    audit_g_zero_eps: float = 0.005
    audit_n_boot: int = 2000
    audit_inflation_margin: float = 0.01

    # ==== M3 ====
    onpolicy_n_questions: int = 1500
    onpolicy_samples: Tuple[int, int] = (2, 4)           # adaptive: 2 first, +2 for questions with 0 correct
    onpolicy_temperature: float = 0.7
    onpolicy_max_new_tokens: int = 320
    onpolicy_shots_mix: Tuple[int, ...] = (0, 4)
    onpolicy_batch_size: int = 16
    onpolicy_max_tok_len: int = 1280
    onpolicy_gold_mix: float = 0.4

    # ==== M4 ====
    s1_epochs: int = 4
    s1_lr: float = 5e-3
    s1_onpolicy_fraction: float = 0.6
    s1_lambda_unl: float = 0.1
    s1_unl_every_n_steps: int = 4
    s1_calc_token_weight: float = 1.75
    s1_val_onpolicy_n: int = 120
    s1_k_choices: Tuple[int, ...] = (2, 3)
    s1_save_path: str = "./checkpoints/s1_steernet.pt"
    s1_final_gen_n: int = 240
    s1_ablation_epochs: int = 1
    s1_ablation_n_train: int = 1000

    # ==== M5 ====
    router_policy: str = "catalog"                        # "legacy" | "catalog"
    router_c_size: float = 2.0e-2                          # K>=3 tariff to encourage K<=2 diversity (raised from 5e-3: prior value was insufficient to overcome K=3 gain gap of ~3-5e-3 NLL units)
    router_k_free: int = 2                                # K<=2 free tariff; only K>=3 taxed
    router_eps_tie: float = 0.01
    router_tau: float = 0.02
    router_lambda_bal: float = 0.02
    router_lambda_cons: float = 0.1
    router_distill_epochs: int = 100                     # raised from 60: more epochs for router to escape K=3 local minimum under increased c_size tariff
    router_distill_lr: float = 3e-4                      # lowered from 1e-3: reduces val_kl oscillation that was triggering early stopping at epoch 40
    router_distill_wd: float = 1e-2
    router_distill_patience: int = 20                    # raised from hardcoded 12: gives router more time to escape KL plateaus
    router_distill_val_frac: float = 0.2
    router_catalog_save_path: str = "./checkpoints/router_catalog_v5.pt"
    router_loss_mode: str = "catalog_distill"             # "catalog_distill" | "rloo_legacy"

    # ==== M6b ====
    abstain_theta: float = 1.01                           # >1 disables threshold-based abstention
    guards_enabled: bool = True
    guard_rel_norm_cap: float = 0.02                      # calibrated in M8
    guard_k_max: int = 3
    gate_task_gradient: bool = False

    # ==== M7 ====
    gate_init_logit: float = 0.0
    gate_loss_mode: str = "v5"                            # "v5" | "legacy"
    lambda_dom: float = 1.0
    lambda_tok: float = 0.1                               # lowered from 1.0: at 1.0 live BCE term (~7.9) overwhelmed nll_gated (~1.1) by 8x, drowning the NLL-diff signal; 0.1 brings BCE (~0.8) to NLL scale
    lambda_l1_gate_v5: float = 0.01
    lambda_nll_diff: float = 10.0                        # direct NLL-diff minimisation: drives gate to reduce nll_gated below nll_ungated (complements dominance hinge which only penalises positive gaps) - increased from 1.0 to 10.0 to strengthen gate learning signal
    gate_tau_u: float = 0.005
    gate_dom_margin: float = 0.05
    gate_epochs: int = 3
    gate_lr: float = 2e-3                                 # raised from 1e-3 for faster convergence
    gate_label_row_frac: float = 0.5
    gate_v5_save_path: str = "./checkpoints/gate_v5.pt"
    gate_sel_n: int = 120
    gate_train_n: Optional[int] = None
    s1_train_n: Optional[int] = None

    # ==== M8 ====
    calib_theta_grid: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)
    calib_envelope_quantile: float = 0.999
    calib_nll_tie_eps: float = 0.003
    calib_sel_n: int = 120
    calib_gate_acc_tolerance: int = 2
    contract_path: str = './deployment_contract.json'
    # ==== M9 ====
    bench_shots: Tuple[int, ...] = (0, 8)
    bench_secondary_shots: Tuple[int, ...] = (2, 4)
    bench_secondary_n: int = 500
    bench_n_boot: int = 10000
    bench_batch_size: Tuple[Tuple[int, int], ...] = ((0, 8), (2, 6), (4, 4), (8, 4))
    bench_max_new_tokens: int = 512
    bench_cache_dir: str = './checkpoints/bench_cache'
    bench_noninf_margin_pp: float = 1.0

    def __post_init__(self):
        if self.failure_policy not in ("warn", "strict"):
            raise ValueError('failure_policy must be "warn" or "strict"')
        if self.failure_policy == "warn":
            # Coerce legacy fail-closed flags so they can never block a run.
            if hasattr(self, "skip_adaptive_phases_when_router_fails"):
                self.skip_adaptive_phases_when_router_fails = False
            if hasattr(self, "router_fail_fast_on_collapse"):
                self.router_fail_fast_on_collapse = False
            if hasattr(self, "continue_on_diagnostic_failure"):
                self.continue_on_diagnostic_failure = True

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Ensure parent dirs exist for paths
        for path in [
            self.lora_save_path,
            self.phase2_save_path,
            self.phase2_training_state_path,
            self.router_bootstrap_save_path,
            self.steer_finetune_save_path,
            self.gate_only_save_path,
            self.manifest_path,
        ]:
            if path:
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
    "beta_entropy_k": 0.20,
    "lambda_synergy": 0.25,
    "lambda_synergy_prior": 0.01,
    "beta_entropy_subset": 0.10,
    "router_shot_mix": (0, 2, 4, 8),
    "router_tau": 0.02,
    "s1_k_choices": (2, 3),
    "catalog_max_k": 3,
    "gate_init_logit": 0.0,
    "gate_dom_margin": 0.05,
    "lambda_gate_sparsity": 0.0,
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


ENGINE_V40_KAGGLE_PROFILE_NAME = "Engine v4.0 Kaggle T4 Decision-Run"
ENGINE_V40_KAGGLE_PROFILE: Dict[str, Any] = {
    **ENGINE_V32_KAGGLE_PROFILE,
    "steer_max_relative_norm": 0.02,
    "catalog_max_k": 3,
    "max_active_layers": 3,
    "phase1b_candidate_layers": 6,
    "phase1b_candidates_per_band": 2,
    "router_layer_dropout": 0.15,
    "router_k_preference_temperature": 0.40,
    "router_k_preference_temperature_final": 0.15,
    "router_k_preference_anneal_epochs": 2,
    "router_k0_enabled": True,
    "router_demo_features": True,
    "token_competitive_allocation": True,
    "gate_allocation_temperature": 0.80,
    "gate_allocation_temperature_final": 0.50,
    "effective_k_min_energy_fraction": 0.05,
    "router_min_eval_route_tuple_entropy": 0.10,
    "lambda_route_activity_floor": 0.10,
    "route_activity_min_fraction": 0.05,
    "generation_anchor_interval": 200,
    "generation_anchor_n_prompts": 8,
    "deployment_eval_shots": (0, 2, 4, 8),
    "require_adaptive_beats_fixed": True,
    "fewshot_train_prob": 0.30,
    "fewshot_train_max_demos": 2,
    "router_tariff_k1_free": True,
    "router_min_active_k_fraction": 0.75,
    "lambda_layer_coverage": 0.20,
    "layer_min_coverage_prob": 0.10,
    "gate_min_activation_floor": 0.35,
    "gate_target_sparsity": 0.25,
    "lambda_gate_budget": 0.15,
    "lambda_l1_gate": 0.01,
    "lambda_gate_sparsity": 0.0,
    "gate_init_logit": 0.0,
    "gate_dom_margin": 0.05,
    "calc_token_weight": 1.75,
    "skip_adaptive_phases_when_phase1_fails": False,
    "skip_adaptive_phases_when_router_fails": False,
    "router_fail_fast_on_collapse": False,
    "continue_on_diagnostic_failure": True,
    "failure_policy": "warn",
    "steer_scale": 0.30,
    "phase1_allow_nonregressing_confirmation": True,
    "phase1_allow_confirmation_noise_tolerance": True,
    "phase1_max_confirmation_candidates": 6,
}
ENGINE_V40_KAGGLE_EVALUATION: Dict[str, Any] = {
    "n_test": 100,
    "shots": (0, 2, 4, 8),
    "max_new_tokens": 512,
}


def make_engine_v40_kaggle_config(**overrides) -> RSCConfig:
    """Return the committed Engine v4.0 T4 decision-run profile."""
    cfg = RSCConfig()
    for field, value in ENGINE_V40_KAGGLE_PROFILE.items():
        setattr(cfg, field, value)
    for field, value in overrides.items():
        setattr(cfg, field, value)
    return cfg


def assert_engine_v40_kaggle_profile(
    cfg: RSCConfig,
    strict: bool = False,
    allowed_overrides: Tuple[str, ...] = (),
) -> str:
    """Fingerprint the Engine v4 contract while allowing declared run-scope edits.

    Kaggle decision runs sometimes intentionally vary capacity or evaluation
    coverage.  Such changes must be named in the notebook, not silently
    accepted.  Declared fields remain part of the returned fingerprint; every
    other drift remains fail-closed when ``strict`` is enabled.
    """
    in_kaggle = (
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "KAGGLE_CONTAINER_NAME" in os.environ
        or os.path.exists("/kaggle")
    )
    if in_kaggle and not os.environ.get("ARS_STRICT_PROFILE"):
        strict = False

    allowed = {str(field) for field in allowed_overrides}
    unknown = allowed - set(ENGINE_V40_KAGGLE_PROFILE)
    if unknown:
        if strict:
            raise ValueError(f"Engine v4 override allowlist contains unknown fields: {sorted(unknown)}")
        print(f"ℹ️  [Profile Notice] Engine v4 override allowlist contains custom fields: {sorted(unknown)}")
    drift = {
        field: {"expected": expected, "actual": getattr(cfg, field)}
        for field, expected in ENGINE_V40_KAGGLE_PROFILE.items()
        if getattr(cfg, field) != expected
    }
    blocked = {field: change for field, change in drift.items() if field not in allowed}
    if blocked:
        msg = f"{ENGINE_V40_KAGGLE_PROFILE_NAME} undeclared overrides detected ({len(blocked)} fields): {list(blocked.keys())}"
        if strict:
            raise RuntimeError(msg)
        print(f"ℹ️  [Profile Notice] {msg}")
    if drift:
        permitted = sorted(set(drift) & allowed)
        if permitted:
            print(f"ℹ️  [Profile Notice] declared run overrides: {permitted}")
    payload = json.dumps(
        {"name": ENGINE_V40_KAGGLE_PROFILE_NAME,
         "config": {field: getattr(cfg, field) for field in ENGINE_V40_KAGGLE_PROFILE}},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
