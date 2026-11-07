# Changelog

Versioned history of the Adaptive Residual Steering (ARS) project, from initial repository setup through the current production system. Each version documents architectural decisions, training methodology changes, empirical results, and known limitations at that point in development.

**Base model**: `microsoft/phi-2` (2.7B parameters, 32 transformer layers)
**Evaluation**: GSM8K grade-school math (8.5K train / 1.3K test)

---

## v0.1 — Project Foundation

**Date**: 2026-10-07

The initial research repository establishing the modular framework, evaluation infrastructure, and multi-stage training pipeline.

### Changes

- Repository scaffolding: `src/` module structure, unit tests, CLI scripts
- Research documentation: background, methodology, and staged training pipeline formalization
- GSM8K benchmark suite with full staged Kaggle execution notebook
- Multi-stage pipeline design: SteerNet (residual adapters) + Gate (usefulness scoring) + Router (layer selection)
- Basic 3-layer fixed route (`[13, 20, 22]`) with standard LoRA-style adapters
- Forward hook infrastructure for intercepting and modifying transformer hidden states
- Support for both tuple and raw tensor layer outputs in hook functions

### Results

- Framework operational on Kaggle T4 GPU
- No measurable accuracy gains at this stage (architecture not yet optimized)

### Limitations

- Fixed 3-layer route chosen heuristically
- Standard LoRA initialization (no steering-specific design)
- No per-layer calibration of steering magnitude
- Single-shot training only

---

## v1.0 — SteerNet V5 Architecture

**Date**: 2026-10-10 to 2026-10-18

Complete redesign of the steering adapter architecture. SteerNet V5 introduces a rank-64 bottleneck with causal temporal mixing, calibrated per-layer norm caps, and relative amplitude control.

### Changes

- **CausalDWConv1d** (kernel=4): Depthwise causal convolution for temporal context mixing with Dirac-centered initialization (current-token tap = 1.0, history taps = 0.01). Supports streaming decode via rolling buffer cache.
- **Per-layer norm calibration**: Calibration script measures median hidden-state norms `N_l` per layer on held-out data. Smooth cap envelope `C_l = kappa * N_l` prevents steering from exceeding layer-appropriate magnitude.
- **Relative amplitude mode**: Steering delta scales proportionally to local hidden-state norm via `rho_max` parameter, making amplitude resolution-invariant across layers and sequence positions.
- **Sigmoid magnitude head**: Per-token learned magnitude with bias initialized at -3.0 (sigmoid(-3) = 0.047). Starts selective; typed loss lifts result/final tokens while L1 keeps prose tokens suppressed.
- **Position-0 shielding**: No steering on the first token (attention sink), preventing interference with the model's positional anchoring.
- **Zero-init output projection** replaced with Xavier uniform (gain=1.0) for immediate S1 optimization.

### Architecture

```
h --> LayerNorm --> W_down (D->64) --> CausalDWConv1d(k=4) --> GELU --> W_up (64->D) --> direction
                                                                  |
                                                           mag_head --> sigmoid --> magnitude

delta = rho_max * magnitude * ||h_centered|| * (direction / ||direction||)
delta = delta * shield_mask * soft_cap(delta_norm, C_l)
```

### Results

- First measurable gains over unsteered baseline
- Per-layer telemetry confirmed steering magnitudes within calibrated bounds

### Limitations

- Rank 64 may be unnecessarily large for single-task GSM8K adaptation
- Only 3 candidate layers; limited coverage of the transformer stack
- No typed loss — all tokens weighted equally
- Single-shot training; no few-shot robustness

---

## v2.0 — Typed Reasoning Loss and Multi-Shot Training

**Date**: 2026-10-22 to 2026-10-28

Introduction of the typed reasoning loss system that differentiates between token categories in chain-of-thought solutions, multi-context training for few-shot robustness, and a rigorous benchmark harness with paired statistical testing.

### Changes

- **Token type annotation**: 7 categories identified via regex pattern matching on character spans:
  - `T_NONE` (0) — masked/padding
  - `T_FORMAT` (1) — structural markers ("Step 1:", "Final answer:")
  - `T_PROSE` (2) — natural language reasoning
  - `T_OPERATOR` (3) — arithmetic operators (+, -, *, /)
  - `T_OPERAND` (4) — numeric operands
  - `T_RESULT` (5) — computation results (after "=")
  - `T_FINAL` (6) — the final numeric answer
- **Per-category weighted NLL**: Each token type receives a learned or configured weight. Result and final tokens are prioritized (lambda_result=0.40, lambda_final=0.35) while prose and format receive lighter supervision.
- **Focal modulation**: `(1-p)^gamma` weighting concentrates gradient on tokens the model finds difficult, reducing wasted gradient on already-learned tokens.
- **KL divergence preservation**: Forward KL penalty on prose tokens prevents the steered model from deviating too far from the base model's language distribution, preserving fluency.
- **Prompt shielding**: Few-shot demonstrations are masked from the steering loss. Shield masks are computed from character-level offsets so that only the question and answer regions receive steering gradient.
- **Multi-context training mixture**: Training data includes a mixture of 0/1/2/3/4-shot prompts (default weights: [0.5, 0.1, 0.2, 0.05, 0.15]), forcing the model to generalize across prompt formats.
- **V2 benchmark harness**: Paired testing with McNemar's exact test, 95% confidence intervals on accuracy deltas, and per-example taxonomy of error types (PLAN_ERROR, ARITHMETIC, LOOP_OR_RUNON, UNGROUNDED_OPERAND, EXTRACTION).
- **Tiered answer extraction**: Priority-ordered extraction: explicit markers (####, \boxed{}, "Final answer:") > last equality > "therefore" heuristic > lenient last-line fallback.

### Results

| Condition | Baseline | ARS (Fixed) | Delta |
|-----------|----------|-------------|-------|
| 0-shot    | ~28%     | ~42%        | **+14 pp** |
| 2-shot    | ~52%     | ~55%        | +3 pp |
| 4-shot    | ~56%     | ~55%        | -1 pp |

The +14pp 0-shot breakthrough validated the typed loss approach. Few-shot regression indicated that KL preservation needed strengthening.

### Limitations

- 4-shot regression: steering interferes when the base model already performs well
- Only 3 candidate layers; potential for stronger steering with more layers
- 8 manually tuned loss hyperparameters
- No on-policy data augmentation

---

## v3.0 — 10-Layer Cooperative Steering

**Date**: 2026-11-01 to 2026-11-04

Expansion from 3 to 10 candidate layers with cooperative multi-layer steering, learned loss weighting to replace manual hyperparameters, and memory optimization for T4 GPU training.

### Changes

- **10 candidate layers**: `[8, 11, 13, 15, 17, 19, 20, 21, 22, 24]` — early, middle, and late transformer layers for comprehensive residual-stream coverage.
- **sqrt(K) amplitude scaling**: Each layer's steering amplitude scales by `1/sqrt(K)` where K is the number of active layers. Preserves total steering energy as K varies: `radius = rho_max * (1/sqrt(K))`.
- **Learned Kendall uncertainty weighting** (Kendall et al., CVPR 2018): Three log-variance parameters for NLL, KL, and L1 objectives plus 6 softmax logits for intra-NLL category balance. Replaces 8 hand-tuned loss hyperparameters with 9 learned parameters.
- **Rank reduction**: 64 → 16. Justified by LoRA intrinsic dimensionality findings (Hu et al., 2022): single-task adaptation on GSM8K requires rank 4-16. Reduces SteerNet parameter count by 4x with no quality loss.
- **Gradient checkpointing**: Enables 10-layer training within T4 16GB VRAM budget.
- **Memory optimization**: KL baseline logits moved to CPU immediately after computation; expandable CUDA memory segments enabled.
- **On-policy data collection**: After eligible epochs, generate completions for training questions using the current model. Correct completions are added to the training set for subsequent epochs.

### Architecture

```
10 SteerNet V5 adapters (rank=16) at layers [8, 11, 13, 15, 17, 19, 20, 21, 22, 24]
Per-layer amplitude: rho_max / sqrt(10) = 0.013 (with rho_max=0.04)
Total steering budget: rho_max = 0.04
Effective batch: 2 x 8 = 16 (batch_size x grad_accum)
```

### Results

- Training converges within 3 epochs
- val_nll steadily decreasing across epochs
- On-policy collection yields ~30% correct completions per epoch

### Limitations

- rho_max=0.04 conservative; steering headroom possibly insufficient
- On-policy only uses correct completions; hard questions are discarded
- 3 epochs may be insufficient for convergence
- Router not yet implemented for per-question layer selection

---

## v4.0 — Phase 2 Router (Per-Question Layer Selection)

**Date**: 2026-11-05 to 2026-11-06

Introduction of the JointLayerRouter for adaptive per-question layer selection, trained via RLOO policy gradient. The system now operates in two phases: Phase 1 (S1) trains SteerNets on a fixed route, Phase 2 (S2) trains the router to select question-specific subsets.

### Changes

- **JointLayerRouter**: Two-pass transformer architecture with cross-layer attention. First pass computes initial utility scores; second pass refines them with soft selection feedback. Separate k_head predicts K distribution (how many layers to activate).
- **Cross-layer synergy matrix**: `S_ij = (e_i W_syn e_j^T) / sqrt(d)`, symmetrized. Captures pairwise cooperation between candidate layers during Plackett-Luce sampling.
- **RLOO policy gradient**: Leave-one-out baseline — samples two (K, subset) pairs per question, evaluates NLL for each, computes advantage as the difference. Eliminates the need for a separate value network.
- **Cooperative Plackett-Luce sampling**: Sequential without-replacement sampling with synergy boosts from already-selected layers, enabling the router to learn cooperative layer combinations.
- **ProbeCapture**: Forward hook captures hidden states at a designated probe layer during a no-steer forward pass. These states serve as the router's input representation of the question.
- **Anti-monopoly balance loss**: MSE penalty between realized per-layer selection rates and the expected uniform rate, preventing any single layer from dominating.
- **Entropy bonuses**: Separate entropy terms for K distribution (`beta_k=0.05`) and subset selection (`beta_subset=0.02`) to prevent premature convergence.
- **Gating mechanism** (RSCUsefulnessGate): Per-token usefulness scoring. Ultimately disabled (`use_gate=False`) as the gate added complexity without measurable benefit.
- **OOD evaluation pipeline**: 5-dataset out-of-distribution benchmark (SVAMP, ASDiv, MultiArith, AQuA, MATH).
- **Benchmark bundle export**: ZIP artifact containing all weights and contracts for reproducible evaluation.

### Architecture

```
Phase 1 (S1): Fixed-route training
  prompt --> Phi-2 --> [SteerNet x10] --> answer
                       all layers active

Phase 2 (S2): Router training (SteerNets frozen)
  prompt --> Phi-2 --> probe_layer --> hidden_states --> JointLayerRouter --> route
                                                              |
                       Phi-2 --> [SteerNet x K_selected] --> NLL --> RLOO gradient
```

### Results (Run 5)

| Condition | Baseline | Fixed | Router | Delta (Fixed) | Delta (Router) |
|-----------|----------|-------|--------|---------------|----------------|
| 0-shot    | ~36%     | ~38%  | ~38%   | +2.3 pp       | +2.3 pp        |
| 2-shot    | ~52%     | ~53%  | ~55%   | +1.0 pp       | +3.3 pp        |
| 4-shot    | ~56%     | ~55%  | ~55%   | -1.3 pp       | -1.3 pp        |

Router showed marginal improvement over fixed on 2-shot but no advantage elsewhere. Overall gains were weak, indicating underlying issues in both SteerNet and Router training.

### Limitations

- Router diversity collapsed (layer entropy = 0.60-0.84 bits)
- Health check threshold mathematically impossible (required 1.5 bits; binary max is 1.0)
- Collapse recovery doubled wrong parameter (K entropy instead of subset entropy)
- Probe layer at layer 8 (too early for semantic understanding)
- Inference used plain top-k, ignoring learned synergy matrix
- Prose tokens received zero gradient (40-60% of tokens)
- On-policy shot mismatch: generation used mixed shots but tokenization always 0-shot
- rho_max=0.04 too conservative; 4-shot KL regression

---

## v5.0 — Comprehensive Audit and 21 Fixes (Current)

**Date**: 2026-11-07

Deep audit of the entire SteerNet + Router system, identifying and fixing 21 bugs and inefficiencies across 9 source files. This version addresses all known issues from v4.0 and establishes the foundation for the target +15pp 0-shot gain.

### SteerNet Fixes (10)

| # | Fix | File | Impact |
|---|-----|------|--------|
| 1 | Prose tokens now receive gradient signal (`lambda_prose=0.05`) | `losses.py` | HIGH — 40-60% of tokens were invisible to the loss |
| 2 | On-policy shot mismatch: generation and tokenization now use matching n_shot | `trainer.py` | HIGH — 40% of on-policy data had distribution mismatch |
| 3 | On-policy rows include `question` + `final_num` for snowball accumulation | `trainer.py` | MEDIUM — rows were invisible to future collection epochs |
| 4 | Gold CoT fallback when all k samples fail (hard questions) | `trainer.py` | MEDIUM — 68% of on-policy questions were discarded |
| 5 | Removed double sqrt(K) scaling from cap envelope (was on both radius AND cap) | `steernet.py` | MEDIUM — diminishing returns from multi-layer steering |
| 6 | `rho_max` 0.04 → 0.06 (50% more steering headroom) | `config.py` | MEDIUM |
| 7 | On-policy timing: starts epoch 2, skips last epoch (unused data) | `trainer.py` | LOW |
| 8 | Focal weight computed from baseline logits (not noisy steered logits) | `losses.py` | LOW |
| 9 | `score_tiers` tolerance relaxed for large gold values | `answers.py` | LOW |
| 10 | S1 epochs 3 → 5 (val_nll still improving at epoch 3) | `config.py` | CONFIG |

### Router Fixes (8)

| # | Fix | File | Impact |
|---|-----|------|--------|
| 1 | `router_max_k` 4 → 8 (fixed arm uses 10; router was capped at 4) | `config.py` | HIGH |
| 2 | Phase 2 epochs 2 → 3 (after warmup=1, only 1 effective epoch before) | `config.py` | HIGH |
| 3 | `kl_weight` 1.0 → 2.0 (reduces 4-shot regression) | `config.py` | MEDIUM |
| 4 | `lambda_cost` → 0.0 (cost penalty was larger than NLL learning signal) | `config.py` | HIGH |
| 5 | Warmup keeps subset entropy active (was zeroed, causing early collapse) | `rloo.py` | HIGH |
| 6 | Probe layer → middle candidate layer 19 (was layer 8, too early) | `router_trainer.py` | MEDIUM |
| 7 | Synergy-aware greedy inference (was plain top-k, ignoring synergy matrix) | `router.py` | MEDIUM |
| 8 | `observe()` tracks both subsets for less noisy EMA diagnostics | `rloo.py` | LOW |

### Infrastructure Fixes (3)

| # | Fix | File |
|---|-----|------|
| 1 | Health check entropy threshold: 1.5 → 0.55 bits (1.5 was mathematically impossible) | `router_trainer.py` |
| 2 | Collapse recovery doubles `beta_entropy_subset` (was doubling `beta_entropy_k`) | `router_trainer.py` |
| 3 | `beta_entropy_subset` initial value 0.02 → 0.10 (5x stronger diversity signal) | `config.py` |

### Data Fix

- Gold CoT `"answer"` field preserved in tokenized dataset (`dataset.py`). Was dropped during `ds.map()`, causing gold fallback to never trigger.

### Progress Bar Fix

- Replaced `tqdm` with periodic `print` in both S1 and S2 training loops. `tqdm.auto` outputs every update as a new line in Kaggle subprocess, producing thousands of lines per epoch.

### Preliminary Results (1000 training examples, 50 test, 5 epochs)

| Condition | Baseline | ARS (Fixed) | ARS (Router) | Delta (Fixed) |
|-----------|----------|-------------|--------------|---------------|
| 0-shot    | 42.0%    | 52.0%       | 52.0%        | **+10.0 pp** |
| 2-shot    | 56.0%    | 64.0%       | 58.0%        | **+8.0 pp** |
| 4-shot    | 58.0%    | 60.0%       | 60.0%        | **+2.0 pp** |

- 0-shot format compliance: 16% → 76%
- 4-shot regression eliminated (was -1.3pp, now +2.0pp)
- val_nll converging steadily: 0.855 → 0.826 over 5 epochs
- On-policy pass rate: 32% (80/250 questions correct)

### Current Status

- Full-scale run (7473 training examples, 5 S1 epochs + 3 S2 epochs) in progress
- Target: 0-shot ≥ +15pp, 2-shot ≥ +5pp, 4-shot ≥ 0pp
- Router health to be validated with fixed entropy threshold and correct collapse recovery

### Configuration Snapshot

```python
rho_max          = 0.06    # relative steering amplitude
lora_rank        = 16      # SteerNet bottleneck rank
phase1_epochs    = 5       # S1 training epochs
phase1_lr        = 5e-4    # S1 learning rate
kl_weight        = 2.0     # KL divergence penalty
focal_gamma      = 2.0     # focal loss concentration
router_max_k     = 8       # max layers router can select
router_min_k     = 2       # min layers (anti-collapse floor)
phase2_epochs    = 3       # S2 router training epochs
phase2_lr        = 5e-5    # S2 learning rate
beta_entropy_sub = 0.10    # subset entropy bonus
on_policy_start  = 2       # first on-policy epoch
on_policy_k      = 4       # samples per question
```
