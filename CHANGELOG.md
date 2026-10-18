# Changelog

All notable changes to the **Adaptive Residual Steering (ARS)** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.10.0] - 2026-10-18

### 🛡️ Engine v2.3: Generation-Safe Phase 1 Selection
- **Conservative residual regime (`src/configs/config.py`):** Increased Phase-1 norm pressure to `lambda_steer_norm = 0.60`, lowered the Huber onset to `q_hi = 0.50`, and reduced training to three epochs at `3e-4`.
- **Joint layer/intensity search (`src/training/trainer.py`):** Replaced alpha=1-only screening with a compact search over every `(layer, alpha)` pair on eight calibration questions, followed by a 16-question disjoint confirmation of the selected pair.
- **Generation-aware checkpoint selection:** Added per-epoch six-question screens at alpha `{0.2, 0.4}`; Phase 1 now restores the best generation-safe checkpoint rather than the lowest teacher-forced NLL checkpoint.
- **Independent capacity before cooperation:** Restored deterministic K=1 rotation in Phase 1. K>=2 complementarity remains the explicit responsibility of the Phase-1.5 compatibility correction.
- **Raw norm safety telemetry (`src/steering/steernet.py`):** Added reset-scoped pre-cap q95 measurement and rejects calibration or confirmation routes whose q95 exceeds the hard safety ceiling.
- **Audit-route precedence (`src/models/wrapper.py`):** Explicit routing overrides now take precedence over Phase-1 warmup; epoch generation screens also suspend and restore warmup defensively.

---

## [1.9.0] - 2026-10-18

### 🚀 Engine v2.2: Unblocking Router K-Policy & Multi-Layer Cooperation
- **Symmetric K Distillation (`src/routing/rloo.py`):**
  - Removed `k0_hinge = relu(logsumexp(z_{1..4}) - z_0 + margin)` which mathematically forced $P(K=0) \ge 51.25\%$ and caused 100% argmax K=0 collapse in Run 7.
  - Removed artificial baseline margin `adj_reward_0 = reward_0 + k0_margin`. Restored symmetric counterfactual distillation against natural layer costs.
- **Multi-Layer Cooperation Restoration (`src/configs/config.py`):**
  - Re-enabled compatibility learning rate during router bootstrap (`bootstrap_steer_compat_lr_scale = 0.05`) so SteerNets learn non-interfering, complementary residual directions for $K \ge 2$.
  - Adjusted cumulative route budget power from $1.0$ to $0.5$ (`steer_route_budget_power = 0.5`), keeping $K=2$ residual intensity viable ($\approx 0.71$ scale per layer).
  - Re-enabled token gate diversity regularization (`lambda_gate_diversity = 0.20`).
- **Telemetry & Acceptance Correction (`src/training/trainer.py`):**
  - Resolved false positive generation gain (+43.8pp) when `phase1_acceptance_accuracy_examples = 0` by comparing candidate screen accuracy strictly against matched `base_screen_acc`.
- **Notebook Alignment (`notebooks/kaggle_ars_runner.ipynb`):**
  - Removed duplicate overwrite `CFG.steer_loss_target = "calculation"` in Cell 3, strictly preserving `calculation_weighted` to protect multi-step reasoning prose from premature truncation.

---

## [1.8.0] - 2026-10-17

### 🎯 Empirical Milestone: Run 7 Benchmark (GSM8K, Microsoft Phi-2)
- **First Statistically Significant SteerNet Gain:**
  - **Fixed SteerNet (L22, $\alpha=1.0$):** Reached **62.00%** vs Baseline **56.00%** on 0-shot GSM8K (**$\Delta = +6.00$ pp, 95% Bootstrap CI: [+1.00, +12.00] pp**).
  - Statistically significant improvement ($p < 0.05$) without altering backbone weights.
- **Pre-Cap Huber Norm Regularization (Engine v2.1):**
  - Solved the Run 5 pre-cap norm explosion ($7.39\times$ mean, $26.4\times$ max) using Huber penalty on excess ratio above $q_{\text{hi}}=0.75$, stabilizing pre-cap ratios at $1.35\times - 1.45\times$.
  - Preserved reasoning chain length at 91.5% of baseline (164.8 tokens vs 180.0), eliminating premature EOS truncation.
- **Micro-Batch Computation Graph Isolation (`src/models/wrapper.py`):**
  - Isolated active pre-cap records per micro-batch to prevent PyTorch backward graph leakage during gradient accumulation across varying layer subsets.
- Full details archived in `experiments/runs/exp_20261017_210351_ars_phi2_kaggle/experiment_record.json`.

---

## [1.7.0] - 2026-10-17

### 🚀 Engine v2: Outcome-Supervised ARS & Mathematical Execution Alignment
Directly attacks the empirical ML gaps observed in Runs 1–4 (Router collapse, Token-gate saturation, Few-shot degradation) by replacing teacher-forced conversational prose objectives with exact calculation execution supervision:
- **Calculation Token Classification & Masking (`src/data/dataset.py`):**
  - Implemented cached token classifier `_is_calc_token_id` and exported `calc_mask` across all tokenized datasets. Separates arithmetic/calculation tokens (`0-9`, `+ - * / = $ % #`) from conversational English prose ("Step", "Indras has", "Together they have"), preventing SteerNets from taking prose format shortcuts.
- **Outcome Execution Reward in Router Policy (`src/routing/rloo.py`):**
  - Implemented `per_example_execution_reward` rewarding exact token prediction on calculation steps and answer tail ($\text{Reward} \in [-1.0, +1.0]$) minus NLL penalty.
  - Wired `compute_router_rloo_loss` to evaluate counterfactual routes via `counterfactual_score` when `router_reward_type="execution"`.
  - Established the **$K=0$ Competitive Advantage**: When Baseline solves the calculation correctly ($R_0 \approx +1.0$), steered routes receive a per-layer cost penalty, driving $K \to 0$ on easy problems and training the router to intervene strictly where the unsteered baseline fails.
- **Calculation-Targeted SteerNet Loss (`src/training/trainer.py`):**
  - Updated `_answer_aware_loss` and `_eval_val_nll` to support `calc_mask` and `calc_target_mode="calculation"`, focusing adapter gradients on numerical calculations.
  - Updated `_trim_batch` with variadic support `*extras` to trim `calc_mask` seamlessly without breaking existing callers.
- **Selective Gate Budget Sparsity & Router MI Diversity (`src/configs/config.py`, `src/training/losses.py`):**
  - Added `compute_gate_budget_loss(model, target_sparsity=0.20)` penalizing mean alpha deviation from target sparsity ($0.20$).
  - Configured `gate_target_sparsity=0.20`, `lambda_gate_budget=0.50`, and re-enabled L1 sparsity `lambda_l1_gate=0.05`.
  - Re-enabled low-temperature mutual information exploration (`lambda_router_k_mi=0.15`, `lambda_router_layer_mi=0.15`) to eliminate argmax router collapse.
- **Comprehensive Unit Testing:**
  - Added `TestEngineV2OutcomeSupervision` in `tests/test_modules.py` covering token classification, execution rewards, calculation-mode NLL, quadratic budget loss, and variadic batch trimming.
  - All **102 / 102 unit tests passing (100% OK)**.

---

## [1.6.0] - 2026-10-16

### 🎯 Forensic Review & Diagnostic Hardening (Run 4: n=100 GSM8K, Microsoft Phi-2)
Comprehensive diagnostic hardening and forensic roadmap execution based on Claude's deep review of Run 4 (`360067e`):
- **WP1: True Whole-Evaluation Pre-Cap Norm Saturation Telemetry (R2):**
  - Replaced single micro-batch `_last_pre_cap_ratio` snapshot with GPU tensor accumulation (`_pre_cap_sum`, `_pre_cap_n`, `_pre_cap_max`) across all forward passes in `RSCSteerNet.forward`.
  - Added `reset_pre_cap_stats()` on `RSCSteerNet` and exported `pre_cap_report(model, reset=True)` in `src/steering/` to measure genuine token-level pre-cap perturbation inflation without per-token CPU host sync.
- **WP2: Dual-Extractor Transparency & Reasoning Diagnostics (R4, R5, R7):**
  - Added lenient answer extraction tracking in `evaluate_sample_answer` (`src/evaluation/multi_dataset.py`) and recorded `lenient_correct` and `accuracy_lenient` in benchmark summaries (`src/evaluation/multi_benchmark_runner.py`).
  - Added `format_dual_extractor_table` displaying strict vs lenient comparisons side-by-side to directly assess scoring asymmetry.
  - Implemented `chain_stats(all_details)` and `format_chain_stats_table` to evaluate the reasoning chain compression hypothesis (mean steps, character lengths, tokens, and accuracy per condition/shot).
  - Implemented `alpha_jitter_floor(...)` in `src/evaluation/deployment.py` to quantify the empirical greedy decode / fp16 flip noise floor across tiny alpha perturbations ($\alpha \in \{1.0, 0.999, 0.998\}$).
  - Created standalone diagnostic script [`scripts/rescore_run.py`](file:///d:/projects/Adaptive%20Residual%20Steering%20(ARS)/scripts/rescore_run.py) for immediate offline E0 analysis.
- **WP3: Telemetry Hardening, Labeling & Notebook Integration (R6):**
  - Clamped Shannon entropy to non-negative floats in `route_entropy_by_shot` (`max(0.0, float(h))`), eliminating negative zero (`-0.0`) formatting artifacts.
  - Guarded route collapse warning for deployed baseline mode (`top_route == ()`), and explicitly labeled deployed condition as `Full ARS (Deployed=Baseline)` in paired benchmark tables.
  - Added per-shot (0-shot and 2-shot) paired bootstrap CIs alongside pooled confirmation delta in `train_rsc` (`src/training/trainer.py`).
  - Updated `notebooks/kaggle_ars_runner.ipynb`:
    - Wired `per_item_correct=benchmark_output.get("per_item_correct")` to `export_latex_summary_table`, guaranteeing non-significant cells ($lo \le 0$) are never bolded in research tables.
    - Added `ars_raw` logging to experiment registry and multishot scaling plots.
    - Clamped K entropy in diagnostics dashboard to positive zero.
- **WP4: Architecture Verification & Test Coverage:**
  - Expanded unit test suite in `tests/test_modules.py` with `TestRun4HardeningAndDiagnostics` covering pre-cap accumulation, route entropy clamping, chain dynamics, dual extractor formatting, alpha jitter floor, and significance-based LaTeX bolding.
  - Regenerated architecture diagram drawio and publication PNG artifacts; synchronized `.arch_hash` cache.

### 🧪 Verified
- Full pytest test suite expanded to **97 / 97 tests passing (100% OK)** in 82.1s.
- Architecture diagram synchronization and hash integrity verified via `scripts/check_architecture_diagrams.py`.

---

## [1.5.0] - 2026-10-16

### 🎯 Forensic Review & Diagnostic Hardening (Run 3: n=200 GSM8K, Microsoft Phi-2)
Comprehensive stabilization and diagnostic hardening based on Claude's deep forensic review of Run 3 (`0142a47` → `73a8de9`):
- **Safety Net Verification:** Confirmed `finalize_deployment` correctly triggered `BASELINE` fallback when empirical evidence showed 0-shot regression (-14.5 pp) and route collapse ($K=2$ `{L11, L13}`).
- **Resolved P0 Artifacts & Arm Mismatches:**
  1. **Arm-Matched Intensity (R5):** Set `phase1_gate_scale_grid = (1.0,)` and bypassed noisy generation sweep, guaranteeing Fixed SteerNet and Router+SteerNet evaluate at the identical $\alpha=1.0$ intensity.
  2. **Reserved Confirmation Split (R6):** Bifurcated `eval_data` into `eval_sel` (for early stopping, audits, and selection) and strictly reserved `confirm_rows` (evaluated pooled across 0-shot and 2-shot) to prevent validation contamination.
  3. **Raw ARS Condition & Paired 3-Way Reporting (R6, R8):** Added `ars_raw` condition to multi-benchmark runner to expose raw unmasked ARS performance alongside safe deployment. Implemented paired bootstrap comparisons across all control arms (Base vs Fixed, Fixed vs Router, Router vs ARS, Base vs ARS) and gated LaTeX table bolding to statistically significant improvements only ($lo > 0$ and $diff > 0$).
  4. **Lenient Fallback Extraction & Rescoring (R7):** Added fallback extraction for plain concluding statements (`Together, they have 13 letters.`) and exported `rescore_answers` / `extract_numeric_answer_lenient` for zero-compute post-hoc analysis.
  5. **Pre-Cap Norm Saturation Telemetry (R4):** Replaced synchronous `.item()` clipping statistics in `RSCSteerNet` with non-blocking tensor tracking (`_last_pre_cap_ratio`), exposing pre-cap perturbation inflation without CPU sync overhead during autoregressive generation.
  6. **Data Leakage Elimination (R10):** Set `bootstrap_steer_compat_lr_scale = 0.0` in Phase 1.5 to guarantee SteerNet parameters remain strictly frozen on router data rows; purged dead configuration overrides (`router_balance_weight`, `lambda_k_center`) in `kaggle_ars_runner.ipynb`.
  7. **Per-Shot Route Entropy Telemetry (R2):** Implemented `route_entropy_by_shot` measuring empirical Shannon entropy of selected routes per shot, emitting an explicit `ROUTE-COLLAPSED` warning if route entropy reaches 0.0 bits.

### 🧪 Verified
- Full pytest test suite expanded to **91 / 91 tests passing (100% OK)** in 112.4s.
- Architecture diagram synchronization and hash integrity verified via `scripts/check_architecture_diagrams.py`.

---

## [1.4.0] - 2026-10-16

### 🎯 Empirical Run: `exp_20261015_211528_ars_phi2_kaggle`
Full multi-stage ARS evaluation on GSM8K (Microsoft Phi-2 2.7B, Kaggle 2× Tesla T4, 2,500 train examples, 50 test problems per condition):
- **Headline Benchmark Matrix (n=50):**
  - **0-Shot:** Full ARS **60.00%** vs Baseline **52.00%** (+8.00% absolute gain; 95% CI: [-11.0, +27.0]%).
  - **2-Shot:** Full ARS **62.00%** vs Baseline **60.00%** (+2.00% absolute gain; 95% CI: [-17.0, +21.0]%).
  - **8-Shot:** Full ARS **58.00%** vs Baseline **58.00%** (+0.00% gain / parity; 95% CI: [-19.0, +19.0]%).
- **Deep Diagnostic Revelations:**
  1. **Format Proxy Effect:** SteerNet gain is concentrated at 0-shot where no exemplars are present. With 2 exemplars, baseline accuracy increases from 52% to 60%, matching SteerNet and demonstrating that perturbations acted as implicit format cues ("Step 1:") rather than genuine reasoning enhancements.
  2. **Token Gate Saturation:** Gate values saturated to $\alpha \approx 1.0$ (mean $\alpha = 0.9999-1.0$, std $< 0.0002$), rendering the dynamic token gate an identity multiplier without token discrimination.
  3. **Norm Cap Clamping:** Perturbations pinned to the 4% hard cap (`mean_raw_relative_norm` $\approx 0.04$) across 99.5% of tokens, causing gradient zeroing at the boundary.
  4. **Router Lookup Collapse:** Router collapsed to a fixed shot-count lookup table (0-shot: $K=2$ `[L11, L20]`; 2-shot/8-shot: $K=1$ `[L20]`).
- Full report archived in [`experiments/runs/exp_20261015_211528_ars_phi2_kaggle/EXPERIMENT_REPORT.md`](file:///d:/projects/Adaptive%20Residual%20Steering%20(ARS)/experiments/runs/exp_20261015_211528_ars_phi2_kaggle/EXPERIMENT_REPORT.md).

### 🛠️ Architecture & Pipeline Hardening (Commits: `0200b7a`, `214227e`, `eb58bd8`)
- **Smooth Monotone Norm Saturation (`src/steering/steernet.py`, `src/models/wrapper.py`):** Replaced hard 4% clamp with smooth monotone saturation envelope $f(\delta) = \text{cap} \cdot \frac{\delta}{\sqrt{\|\delta\|^2 + \text{cap}^2}}$ ensuring non-zero gradients across all perturbation magnitudes; re-zeroed `mag_head.bias`.
- **Micro-Batch Padding Trimming (`src/training/trainer.py`):** Wired `_trim_batch` at all forward/backward stacking sites, dynamically trimming sequences to actual token lengths and doubling training throughput (~2x compute savings).
- **Steer/Router Data Splitting (`src/training/trainer.py`):** Wired `split_steer_router(train_data, 0.2)` across all training phases (Phases 1, 1.5, 2A, 2B, 2C, 2D) to isolate SteerNet and Router optimization, eliminating data leakage.
- **Production Deployment Finalization (`src/evaluation/deployment.py`, `src/training/trainer.py`):** Wired `finalize_deployment` on held-out confirmation rows using a 95% paired bootstrap confidence interval ($\text{lo\_ci} > 0$), writing `deployment_mode` to model checkpoint manifests.
- **Audit Fast-Path (`src/configs/config.py`):** Set `phase1_acceptance_accuracy_examples = 0` and `phase1_generation_screen_examples = 0`, bypassing 65 minutes of noisy preflight generation audits.
- **Unified Generation Verifier (`src/evaluation/verifier.py`, `src/evaluation/answers.py`):** Standardized `epoch_end_accuracy` to use `AnswerStoppingCriteria`, symmetric text-level extraction, continuation trimming, and expanded stop markers (`_STOP_MARKERS`) to eliminate hallucinated follow-up tails.
- **Paired Bootstrap Benchmark Reporting (`src/evaluation/multi_benchmark_runner.py`):** Integrated `format_benchmark_cell` to report empirical accuracy alongside paired $\Delta$ 95% bootstrap intervals and statistical significance asterisks (`*`).
- **Format Control Diagnostic (`scripts/format_control.py`):** Resolved constructor import (`Phi2WithRSC`) and dataset loading signatures to enable direct testing of format-isolated steering.

### 🧪 Verified
- Unit test suite expanded to **86 / 86 tests passing (100% OK)** in 55.4 seconds.
- Architecture diagram synchronization and hash integrity verified via `scripts/check_architecture_diagrams.py`.

---

## [1.3.0] - 2026-10-11

### 🎯 Empirical Run: `exp_20261011_093749_ars_phi2_kaggle`
Full multi-stage ARS evaluation on GSM8K (Phi-2, 5,000 train examples, 300 test problems per condition):
- **2-Shot Breakthrough:** Full ARS Pipeline achieved **48.67%** vs Frozen Baseline **43.33%** (+5.34% gain).
- **Diagnostics:** $K=0$ probe contamination completely eradicated (0.00% unsteered no-ops).
- **Audit Findings:** Uncovered root-cause architectural bottlenecks: double-dampening via `k_scale = sqrt(K)`, policy-suppressing entropy regularizers ($\beta=0.30$), and hardcoded MSE gate targets. Full report archived in [`experiments/runs/exp_20261011_093749_ars_phi2_kaggle/EXPERIMENT_REPORT.md`](file:///d:/projects/Adaptive%20Residual%20Steering%20(ARS)/experiments/runs/exp_20261011_093749_ars_phi2_kaggle/EXPERIMENT_REPORT.md).

### 🛠️ Core Root-Cause Refactoring (Commits: `491bf8d`, `a9781f7`)
- **SteerNet Signal Restoration:** Removed double-dampening `k_scale` division; `max_relative_norm = 0.15` remains the single sufficient norm envelope.
- **Joint Gradient Flow:** Added `_joint_grad_mode` enabling task-loss gradients through gate activations ($\alpha$) during Phase 2C joint fine-tuning.
- **Router Policy Calibration:** Slashed $\beta_{\text{entropy\_k}}$ (0.30 $\to$ 0.03) and $\text{balance\_weight}$ (0.10 $\to$ 0.01); removed `lambda_k_center` to unleash natural data-driven $K$-budget decisions.
- **Margin-Based Gate Supervision:** Replaced fixed MSE targets (0.85/0.05/0.40) with structural margin constraints ($\alpha_{\text{math}} > \alpha_{\text{non-math}} + 0.15$).

---

## [1.2.0] - 2026-10-10

### 🎯 Overview
Architectural overhaul of ARS core components (Semantic Token Gate 2.0, Bounded Relative Perturbation SteerNet 2.0, and Difficulty-Aware Layer Router with Layer Dropout) addressing empirical shortcomings uncovered during controlled ablations on Microsoft Phi-2 (GSM8K).

### 🧠 Architectural Innovations & Upgrades
- **Semantic Token Gate 2.0 (`src/gating/gate.py`, `src/training/losses.py`):**
  - **Dead Gradient Elimination:** Resolved root cause of flatline gate telemetry ($\alpha \equiv 0.800$ across all token categories) where zero-initialization of `self.fc2.weight` caused an identically zero gradient $\frac{\partial \mathcal{L}}{\partial W_{\text{fc1}}} = 0$. Replaced with Gaussian initialization $\mathcal{N}(0, 0.02^2)$.
  - **Causal Depthwise 1D Convolution:** Integrated causal temporal depthwise convolution ($k=3$, `groups=gate_dim`) with identity Dirac initialization, capturing preceding token context (e.g. `Step`, `=`, numbers) without breaking causal autoregression.
  - **Dynamic Range Span Penalty:** Enhanced `compute_gate_diversity_loss` to penalize narrow dynamic range spans ($\max(\alpha) - \min(\alpha) < 0.15$) alongside token variance.
- **SteerNet 2.0 — Bounded Relative Perturbation (`src/steering/steernet.py`, `src/configs/config.py`):**
  - **Relative Norm Envelope:** Enforces $\|\Delta h\| \le \beta_{\max} \|h\|$ ($\beta_{\max} = 0.15$) via smooth scaling factor $\min(1.0, \frac{\beta_{\max} \|h\|}{\|\Delta h\| + \epsilon})$, mathematically preventing unconstrained residual explosion and base representation collapse (which caused the -3.0% regression in ablation Arm 1).
  - **Orthogonal Steering Projection:** Added configurable orthogonal complement projection $\Delta h_{\perp} = \Delta h - \frac{\langle \Delta h, h \rangle}{\|h\|^2 + \epsilon} h$ to rotate representation features rather than altering base radial magnitude.
- **Joint Layer Router 2.0 — Difficulty-Aware with Layer Dropout (`src/routing/router.py`, `src/configs/config.py`):**
  - **Difficulty Conditioning:** Enriched router scalar features with sequence reasoning length ($\log(1 + T)/6.24$) and hidden state token dispersion to adapt layer selection and $K$-budget to problem complexity.
  - **Stochastic Layer Dropout:** Introduced stochastic layer dropout during training (`router_layer_dropout = 0.15`), masking individual candidate layer logits to break co-dependent clique stacking and force each layer to learn robust independent steering utility.

### 🧪 Verified
- Unit test suite expanded to **45 / 45 tests passing (100% OK)** in 28.4 seconds.
- Architecture diagram synchronization and hash integrity verified via `scripts/check_architecture_diagrams.py`.

---

## [1.1.0] - 2026-10-10

### 🎯 Overview
Research audit, bug fixes, architecture simplification hooks, and controlled experiment parameters on branch `audit/research-simplification-and-controlled-experiments`.

### 🐛 Fixed
- **$K$-Distribution Visualization Bug (`src/visualization/plots.py`):**
  - Resolved filter condition `if int(k) > 0` at line 228 that omitted $K=0$ from the realized active layers bar chart.
  - Updated to `if int(k) >= 0` to accurately display the 27.2% unsteered fallback (no-op) frequency observed during GSM8K evaluation.
  - Added unit test coverage in `tests/test_visualization.py`.

### ✨ Added
- **SteerNet Direction vs. Magnitude Decoupling (Priority 1 / EXP-01):**
  - Added `steer_magnitude_mode` (`"direct"` vs. `"decoupled"`) in `src/configs/config.py` and `src/steering/steernet.py`.
  - In `"decoupled"` mode, SteerNet normalizes the output correction vector $\hat{v} = \frac{\Delta h}{\|\Delta h\| + \epsilon}$ and scales it with a bounded learnable log-magnitude parameter to eliminate norm drift.
- **Fixed Layer Routing Override (Priority 2 / EXP-03):**
  - Added `fixed_layers` (e.g. `[11, 15, 20]`) and `use_router` flag in `RSCConfig` and `Phi2WithRSC`.
  - Allows isolating the empirical necessity of dynamic routing versus static deep-layer allocations.
- **Token Gate Constant Mode & Saturation Profiler (Priority 3 / EXP-05):**
  - Added `gating_mode` (`"learned"` vs. `"constant"`) and `gate_constant_value` in `src/gating/gate.py`.
  - Added `get_saturation_metrics()` to compute token saturation frequencies ($\alpha < 0.05$ and $\alpha > 0.95$).
- **Streamlined 2-Stage Training Curriculum (Priority 5 / EXP-06):**
  - Added `train_streamlined_ars` in `src/training/trainer.py` to enable training in 2 consolidated stages (Warmup + Joint Gated Optimization) instead of 6 phases, cutting training wall-clock time by $>50\%$.
- **CLI Flags for Controlled Ablation Experiments (`scripts/run_experiment.py`):**
  - Exposed `--steer_magnitude_mode`, `--fixed_layers`, `--gating_mode`, and `--curriculum_mode`.
- **Audit & Simplification Test Suite (`tests/test_audit_simplification.py`):**
  - Added 5 unit tests validating decoupled SteerNet, constant gating, gate saturation profiling, and fixed-layer routing.

### 🧪 Verified
- Full test suite passed: **42 / 42 tests passing (100% OK)** in ~37 seconds.
- Architecture diagram synchronization and hash integrity verified via `scripts/check_architecture_diagrams.py`.

### 📊 Empirical Audit Findings (Kaggle Ablation Run)
- **Controlled Scorecard (n=100 test problems, 0-Shot & 2-Shot):**
  - **Baseline:** 41.0% (0-shot), 55.0% (2-shot).
  - **SteerNet Only:** 38.0% (0-shot, -3.0% regression), 55.0% (2-shot). Demonstrates unconstrained residual perturbation without gating directly harms base representations.
  - **Fixed Triplet `[11, 15, 20]` + Gate:** 43.0% (0-shot, +2.0%), **56.0% (2-shot, +1.0% over Full ARS)**. Highest accuracy achieved without any router complexity.
  - **Full ARS System:** 43.0% (0-shot, +2.0%), 55.0% (2-shot, +0.0%).
- **Architectural Verdict:**
  1. **Router Redundancy:** The dynamic layer router and RLOO training do not yield measurable gains over fixed semantic layer allocation (`[11, 15, 20]`).
  2. **Gate Criticality:** Token gating is indispensable to prevent the -3.0% collapse of pure SteerNet.
  3. **Gate Saturation:** Token gate weights stagnated at initialization $\alpha = 0.800$ across all token categories, proving that the gate acted primarily as a static attenuator rather than a token-selective discriminator.

---

## [1.0.0] - 2026-10-10

### 🎯 Initial Release & Kaggle Multi-Shot Validation
- Complete multi-stage ARS pipeline on Microsoft Phi-2 (2.7B) with GSM8K CoT evaluation.
- Validated multi-shot robustness across 0, 2, 4, and 8 shots:
  - Baseline: 39.5% (0-shot) $\to$ 30.5% (8-shot) (context drift degradation).
  - SteerNet: 43.0% (0-shot) $\to$ 53.0% (8-shot).
  - Full ARS: 51.5% (0-shot) $\to$ 54.0% (8-shot) (+23.5% gain over baseline at 8-shot).
- Implementation of Prompt Shielding, Cooperative Layer Router with Plackett-Luce ranking, and Token Usefulness Gate.
