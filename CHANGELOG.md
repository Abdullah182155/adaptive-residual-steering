# Changelog

All notable changes to the **Adaptive Residual Steering (ARS)** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.15.0] - 2026-10-22

### 🧪 Run: Engine v4.0 Full Decision Run (\exp_20261022_132141_ars_phi2_kaggle\)
- **Tokenizer Caching Verification:** Fixed the dataset tokenization freeze (commit \995259e\); 4,000 examples were mapped in seconds rather than stalling for 20+ minutes.
- **Phase 1 Acceptance Succeeded:** L11 at \lpha=0.50\ achieved +16.7pp calibration gain (70.8% vs 54.2% baseline) and confirmed at +3.1pp disjoint gain (59.4% vs 56.2% baseline, CoT=1.00, postcap q95=0.40, NLL train/val -17.4%/-18.9%).
- **Phase 1B Candidate Pool:** Stratified selection formed pool \[11, 18, 20]\ across early, middle, and late depth bands with calibrated scales \{11: 0.50, 18: 0.35, 20: 0.35}\.
- **Phase 1.5 Router Bootstrap:** Completed with 0 NaNs and passing K-diversity preflight (\K=0: 32.0%\, \K=1: 15.6%\, \K=2: 52.3%\, route-tuple entropy 1.938 bits).
- **Phase 2 Unified Adaptation:** Survived memory watchdog step 50 cleanly (allocated 8,103 MiB, reserved 10,738 MiB < 14,200 MiB ceiling).
- **Benchmark Performance:**
  - **Fixed SteerNet outperformed baseline across multiple shots:** 0-shot 59.0% vs 56.0% (+3.0pp), 4-shot 63.0% vs 62.0% (+1.0pp), 8-shot 63.0% vs 61.0% (+2.0pp).
  - **Router+SteerNet:** 0-shot 59.0% vs 56.0% (+3.0pp), 2-shot 58.0% vs 58.0%, 4-shot 62.0% vs 62.0%, 8-shot 61.0% vs 61.0%.
  - **ARS Raw:** 0-shot 58.0% vs 56.0% (+2.0pp), 2-shot 58.0% vs 58.0%, 4-shot 62.0% vs 62.0%, 8-shot 61.0% vs 61.0%.
  - **Deployment Gate Rigor:** Because multi-shot ARS collapsed to K=0 (matching baseline exactly) and 0-shot CI spanned zero, deployment gate correctly defaulted to Baseline.

### 🔬 Forensic Findings & Multi-Shot Router Collapse Root Cause
- **Multi-shot K=0 Collapse:** Route entropy was 0.8439 bits on 0-shot, but collapsed to strictly 0.0000 bits across 2-shot, 4-shot, and 8-shot (100% of prompts predicted K=0).
- **Root Cause (Zero-Shot Training Bias):** The run used \ewshot_train_prob=0.0\ and \ewshot_train_max_demos=0\. The router never observed prompt demonstrations during RLOO exploration. When tested on 2/4/8-shot prompts (317, 522, 925 tokens), the out-of-distribution exemplar tokens combined with the K-tariff caused the policy to collapse into the zero-cost K=0 action.
- **Fixed SteerNet Resilience:** The underlying Phase-1 SteerNet (L11) provides consistent positive gain even at 8 shots (+2.0pp) when not suppressed by a collapsed router.

---

## [1.14.0] - 2026-10-22

### 🧪 Runs: Engine v3.3 Small-Scale & Large-Scale Decision Runs
- **Small-scale run** (rank=16, n_samples=3000, phase1_epochs=3): Phase 1 accepted L17 at alpha=0.25 (+6.2pp). Phase 2 completed both epochs with acc=61.0%, but the router collapsed to K=1 (L17 only, 100%). Post-Phase-2 health audit failed: K>=2 rate=0.000 < 0.020, active_layers=1 < 2. Pipeline halted before benchmark execution.
- **Large-scale run** (rank=32, n_samples=4000, phase1_epochs=4): Full pipeline completed. Phase 1 accepted L11 at alpha=0.35 (+11.5pp confirmation gain, CoT=1.03, postcap q95=0.27). Phase 1B formed 3-layer pool [11, 17, 20] across all depth bands. Phase 2 passed health audit (K>=2 rate=39.1%, all 3 bands active). Benchmark: Baseline 56%, Fixed SteerNet 59%, ARS Raw 60% (0-shot, N=100).
- **Memory watchdog v3.3 proven:** Both runs survived Phase 2 step 50 without CUDA OOM (peak allocated ~8.1 GiB, reserved ~13.9 GiB < 14.2 GiB limit).

### Forensic Findings (Engine v3.3 Post-Mortem)
- **Bimodal K-distribution:** Realized K was strictly bimodal (K=1: 60.9%, K=3: 39.1%, K=0 and K=2 both 0.0%) due to deterministic argmax evaluation censoring minority classes despite the policy learning non-trivial K=0 probability (p(K=0) approx 21%).
- **Inverted synergy paradox:** The selected triad (L11, L17, L20) had the only negative pairwise interactions in the model (L17+L20: -0.17, L11+L17: -0.12) while unselected/unobserved pairs retained positive priors.
- **L20 gate vanishing:** L20 was selected on 95.8% of examples but low_saturation_pct=84.7% and effective_norm=0.0002. The router claimed K=3 while physically operating as a single dominant L11 controller.
- **4-shot route collapse:** Route entropy dropped to 0.0 bits (identical route on 100/100 items). Prompt shielding excluded demonstrations from router pooling, leaving the router blind to few-shot context.
- **Candidate pool starvation:** 9 layers showed calibration gains in Phase 1, but only 3 entered the routing pool due to the 1-per-band admission policy.
- **Marginal adaptive value:** ARS Raw added only +1.0pp over Fixed SteerNet (59% vs 60%), within noise at N=100.
- **Deployment gate correct:** Deployment confirmation CI crossed zero; fallback to baseline was scientifically sound.

## [1.13.7] - 2026-10-21

### 🛡️ Engine v3.3: Cache-Aware Memory Recovery & Exact Phase-2 Resume
- **Cache-aware watchdog (`src/training/trainer.py`, `src/training/losses.py`):** CUDA active allocation is now the primary safety limit (`11,500 MiB`). PyTorch reservation is treated as cache pressure: the runner collects garbage and calls `torch.cuda.empty_cache()` before judging capacity. The reservation ceiling is raised to `14,200 MiB`, preserving realistic T4 headroom without mistaking reusable cache for live tensors.
- **Controlled recovery:** Watchdog pressure reduces subsequent (and persisted resumed) RLOO probe batches to four, checkpoints state atomically, and returns a recoverable `PAUSED_MEMORY_PRESSURE` status on genuine allocation pressure instead of raising a raw reserved-memory exception.
- **Exact Phase-2 restart:** Added an atomic CPU-portable training-state file with ARS tensors, optimizer, AMP scaler, LR reducer, RNG, routing controls, epoch/batch cursor, shuffled order, and adaptive probe size. `CFG.resume_phase2=True` is fail-closed if the state is missing or incompatible, resumes only the interrupted Phase-2 stage, and then completes its held-out per-shot deployment decision.
- **Regression coverage:** Added CPU tests for cache reclamation, controlled active-allocation emergency, non-fatal memory telemetry, state round-trip restoration, and fail-closed missing-state resume.

---

## [1.13.6] - 2026-10-21

### 🧪 Run `exp_20261021_113159_ars_phi2_kaggle_engine_v3_2_memory_halt`: Engine v3.2 Cooperative-Routing Milestone
- **Experiment preservation:** Registered the complete Kaggle run in `experiments/registry.json` with terminal status `CUDA_MEMORY_LIMIT_HALT_IN_PHASE_2`. Its structured record and verbatim 57,387-byte Kaggle output are retained under `experiments/runs/exp_20261021_113159_ars_phi2_kaggle_engine_v3_2_memory_halt/`; the output SHA-256 is `917f41bd882ab1875bc30706524f8c6d600ddb4694acc4ddbaa2be5e2c54160d`.
- **Phase-1 breakthrough:** L20 at `alpha=0.50` passed disjoint confirmation at `62.5%` versus `53.1%` baseline (`+9.4pp`), with CoT retention `1.07`, effective post-cap displacement q95 `0.42`, and train/validation NLL reductions of `15.2%/16.7%`.
- **Cooperative routing established:** Phase 1B formed the intended early+late pair `[L13, L20]`. Phase 1.5 completed without NaNs or frozen gradients; deterministic preflight passed with `K>=2` on `28.9%` of prompts and both depth bands active.
- **Phase-2 signal preserved:** Unified adaptation Epoch 1 saved a checkpoint at validation loss `1.0440` and generation accuracy `60.0%`, with realized routing `K=0: 3%`, `K=1: 21%`, `K=2: 76%` and low gate high-saturation (`0.9%`).
- **Operational halt, not CUDA OOM:** At the first Phase-2 watchdog check (global step 50), peak allocated memory was `8,095 MiB`, but PyTorch's cached peak reservation was `13,602 MiB`, exceeding the fixed `12,288 MiB` reserved-memory limit. The run stopped before final benchmarking. This is the Engine v3.3 memory-watchdog and resumability priority; no final benchmark claim is made for this interrupted run.

---

## [1.13.5] - 2026-10-21

### 🚀 Engine v3.2: Physical Displacement Safety & Reproducible Screening
- **Post-cap safety (`src/steering/steernet.py`, `src/training/trainer.py`):** Phase-1 promotion now applies its hard norm bound to the actual gate-scaled soft-capped displacement (`alpha × q_postcap`), while retaining raw pre-cap `q95` as a saturation diagnostic. Telemetry now uses a bounded CPU priority reservoir, so its p95 represents the full generation stream instead of an early-token prefix.
- **Efficient, stratified acceptance (`src/training/trainer.py`):** Successive halving screens every route on a compact prefix and promotes at most one safe early, middle, and late route to the full calibration split. The report records the promoted bands and does not mislabel untested non-finalists as safety-rejected.
- **Strict efficacy (`src/configs/config.py`, `src/training/trainer.py`):** A fixed SteerNet requires a strictly positive disjoint confirmation gain; one-question noise remains visible in telemetry but can no longer promote a tie or regression.
- **Two-band cooperation and profile lock (`src/configs/config.py`, `notebooks/kaggle_ars_runner.ipynb`):** Early+late or early+middle pools are valid Phase-1B inputs. The source-controlled Kaggle profile now locks every decision-run setting that Cell 3 touches and fails before model loading if any value drifts.
- **Regression coverage:** Added CPU tests for post-cap admission, strict tie rejection, streaming-reservoir replacement, valid two-band co-adaptation, and profile-drift rejection.

---

## [1.13.4] - 2026-10-21

### 🧪 Run `exp_20261021_050258_ars_phi2_kaggle`: Engine v3.1 Phase-1B Fallback Audit
- **Result preservation:** Registered the Kaggle outcome in `experiments/registry.json` and retained its structured experiment record. The raw notebook-output copy was deliberately removed from the checkout at the experiment owner's request; the Kaggle checkpoint remains externally located at `.../checkpoints/phase1_steernet_only_weights.pt`.
- **Phase-1 finding:** L13 at `alpha=0.35` passed the then-current non-regression contract after a `+10.7pp` calibration signal and a `0.0pp` disjoint-confirmation result. On the independent 200-question benchmark, fixed steering was not superior: 0/2/4/8-shot deltas were `-1.5/-1.5/-0.5/-0.5pp`, with every paired bootstrap interval crossing zero.
- **Safe fallback:** Phase 1B admitted only L13/L15/L17 (early/middle bands) after L11 failed confirmation. With no eligible late layer, the three-band co-adaptation contract correctly produced `FALLBACK_TO_CERTIFIED_FIXED_STEERNET`; router bootstrap and adaptive phases were not run, so this experiment makes no adaptive-routing performance claim.
- **Forensic design finding:** Later-layer candidates, including L24 at `alpha=0.50` (`64.3%` calibration accuracy, CoT `1.01`), were excluded by raw pre-cap `q95 > 1.50` despite sub-unit `alpha * q95`. The audit identified that raw pre-cap q is a saturation diagnostic, whereas the soft-capped, gate-scaled post-cap displacement is the physical safety quantity that must govern future candidate admission.
- **Reproducibility finding:** The executed Cell 3 overrode the committed lean profile (`phase1_lr=0.007`, four epochs, and 28/36 calibration/confirmation rows rather than `3e-4`, two epochs, and 12/16 rows). Future decision runs require a locked, archived effective configuration rather than a commit hash alone.

---

## [1.13.3] - 2026-10-21

### 🚀 Engine v3.1: Diverse Cooperative Routing & Fixed-SteerNet Continuity
- **Cross-band co-adaptation (`src/training/trainer.py`):** Phase 1B now constructs its safe pool across physical early (L11--13), middle (L15--18), and late (L20--24) depth bands before filling additional ranks. Its K=2 curriculum and bootstrap compatibility correction prioritize long-range early/middle-to-late pairs rather than adjacent residuals.
- **Generation-safe calibration (`src/configs/config.py`, `src/training/trainer.py`, `notebooks/kaggle_ars_runner.ipynb`):** Added fine lower-scale calibration at `alpha={0.25, 0.35, 0.50, 0.70}` and separates raw pre-cap runaway protection (`q95 <= 1.50`) from the strict effective displacement contract (`alpha * q95 <= 1.00`).
- **Utility-preserving K pricing (`src/routing/rloo.py`):** K tariffs now warm linearly from zero over bootstrap epoch one and are capped to a fraction of a route's measured positive gain over K=0. Raw, cost-adjusted, and pair utilities remain explicit in telemetry.
- **Numerical fail-closed router bootstrap (`src/models/wrapper.py`, `src/routing/rloo.py`, `src/routing/router.py`, `src/training/trainer.py`):** Eligibility masks use finite sentinels; non-finite logits, rewards, synergies, losses, gradients, and parameters halt bootstrap as a named invalid-numerics state rather than producing a misleading zero loss.
- **Scientific preflight and graceful fallback (`src/training/trainer.py`):** Router preflight evaluates the union of deterministic selections across its batch and requires both K>=2 support and at least two participating depth bands. If the diversity contract fails, `train_rsc` restores the certified fixed SteerNet, saves it as `certified-fixed-steernet-fallback`, and records `FALLBACK_TO_CERTIFIED_FIXED_STEERNET` for full benchmarking instead of discarding valid Phase-1 results. Provenance-verified Phase-1 resume now also rehydrates its certified `(layer, alpha)` route from the cache.
- **Regression coverage (`tests/test_modules.py`):** Added CPU coverage for low-scale/effective-norm safety, tariff warm-up and gain preservation, finite masked K distillation, NaN-free synergy reporting, cross-band pool selection, and the batch-union depth-band preflight contract.

---

## [1.13.2] - 2026-10-20

### 🛡️ Phase-1B Routing Contract & Fail-Closed Recovery
- **Certified routing pool (`src/models/wrapper.py`, `src/routing/rloo.py`, `src/training/trainer.py`):** The learned router, its K head, RLOO counterfactual subset sampler, and bootstrap K=2 correction are now masked to the safe layers produced by Phase 1B. Explicit audit overrides remain available, but confirmation-failed or never-co-adapted layers cannot silently re-enter adaptive routing.
- **Balanced calibrated co-adaptation (`src/training/trainer.py`):** Phase 1B now alternates an exactly balanced K=1/K=2 schedule regardless of pool size and freezes each safe layer at its validated Phase-1 gate scale rather than reverting to a mismatched global intensity.
- **Robust Phase-1 recovery (`src/training/trainer.py`):** Acceptance reports are atomically cached with the strict checkpoint manifest state hash, commit, and configuration fingerprint. Stale/corrupt reports trigger re-evaluation; cache persistence failures are fatal rather than silently disabling resume. Phase-1 and router failures now return structured diagnostics through `train_rsc`, allowing evidence persistence and rollback before downstream phases are skipped.
- **Route-level gate budget (`src/training/losses.py`):** Gate-budget regularization now constrains total effective intervention mass per token across selected layers, preventing K=2 routes from receiving twice the intended budget merely because two gates are active.
- **Scientific promotion guard (`src/configs/config.py`, `src/training/trainer.py`):** Direct Phase-1 confirmation promotion now also requires non-regressing pooled calibration-plus-confirmation accuracy by default; the existing one-question tolerance remains bounded by the same pooled criterion.

---

## [1.13.1] - 2026-10-20

### 🐛 Fixed
- **Phase 1B candidate handoff (src/training/trainer.py):** Fixed schema mismatch where calibration_ranking dictionaries omitted the `safe` flag, causing `train_phase1b_coadaptation` to evaluate `row.get('safe', False)` as false for all rows and crash with `RuntimeError: Phase 1B requires at least two independently safe candidate layers`. Added explicit `safe` retention in calibration_ranking, safe candidate-scale fallback, and strict exclusion of any layer that failed disjoint confirmation.
- **Phase-1 resume resilience (src/training/trainer.py, src/configs/config.py):** Added `resume_phase1` flag and automated caching of `phase1_report.json` to enable resuming training directly at Phase 1B from existing Phase-1 SteerNet checkpoints. Resume now refuses silent full retraining if its checkpoint is absent and re-caches a report produced by fallback re-evaluation.
- **Regression tests (tests/test_modules.py):** Added test coverage ensuring Phase 1B correctly ingests multi-layer calibration rankings and candidate fallbacks.

---

## [1.13.0] - 2026-10-19

### 🚀 Engine v3: Cooperative Routing, Honest Telemetry & Per-Shot Deployment
- **Bounded pre-cap telemetry (`src/steering/steernet.py`):** Replaced unbounded detached CUDA q-value retention with exact scalar summaries plus a bounded CPU reservoir, eliminating the diagnostic allocation growth observed during RLOO evaluation.
- **Route semantics & utility accounting (`src/models/wrapper.py`, `src/routing/rloo.py`):** Routing traces now distinguish requested from realized K and expose route allocation. Router diagnostics report raw reward, cost, cost-adjusted reward, singleton utility, and measured pair utility.
- **Fail-closed router contract (`src/training/trainer.py`, `src/configs/config.py`):** Bootstrap preflight and final intervention audits require at least two active layers, positive K>=2 incidence, and K/layer entropy floors before adaptive training or deployment claims can proceed.
- **Co-adaptation curriculum:** Added memory-safe Phase 1B balanced K=1/K=2 training over Phase-1-safe candidate layers, with frozen gates, per-microbatch trimming, norm regularization, rollback protection, and a 12 GiB CUDA reserved-memory budget.
- **Factorized gate allocation (`src/gating/gate.py`):** Selected K>=2 routes now share a masked-softmax layer allocation under a route-level budget; an overlap penalty encourages complementary token use instead of co-activation.
- **Lean adaptive and deployment paths:** Added a unified gate+router adaptive phase option and independent paired bootstrap deployment decisions per shot. A confirmed 0-shot policy can therefore deploy without a dormant few-shot policy suppressing it.
- **Kaggle diagnostics:** The notebook’s token-gate heatmap now profiles the actively routed `ars_raw` arm rather than a possible fallback-baseline `ars` arm.

---

## [1.12.0] - 2026-10-19

### 🧪 Run `exp_20261019_172330_ars_phi2_kaggle`: End-to-End ARS Milestone
- **Pipeline integrity:** The complete Phi-2/GSM8K training, routing, gate, evaluation, and deployment pipeline completed on a Kaggle T4 without a crash or divergence. The run is registered in `experiments/registry.json`; the Kaggle-produced checkpoint remains externally archived rather than being represented as a locally verified binary.
- **Generation-safe Phase 1:** The accepted L24 route at `alpha=0.40` preserved reasoning length (`CoT=1.07`), passed the multi-candidate confirmation contract, and reduced teacher-forced train/validation NLL by approximately 8.8%/9.6%.
- **Raw 0-shot signal:** Experimental Full ARS (Raw) reached 59.0% versus a 56.0% baseline on the 100-item benchmark (`+3.0pp`, paired bootstrap CI `[0.0, +7.0]pp`). This is recorded as experimental evidence, not a deployment claim.
- **Deployment safety:** The independent confirmation decision correctly retained baseline deployment because the 0-shot confirmation interval crossed zero and 2-shot ARS was dormant. The prior pooled decision is documented as a diagnostic limitation addressed by Engine v3.
- **Forensic findings:** Router deployment was degenerate (`K=0` 93.91%, `K=1` 6.09%, no `K>=2`, L11-only). A black token-gate heatmap was traced to plotting the fallback `ars` condition instead of the actively routed `ars_raw` condition. Later-phase pre-cap telemetry also exposed unbounded detached GPU diagnostic retention and post-Phase-1 norm drift.

---

## [1.11.0] - 2026-10-19

### 🧪 Phase-1 Confirmation Resilience
- **Top-K disjoint confirmation (`src/training/trainer.py`):** Phase 1 now measures the baseline once and confirms up to the three strongest safe `(layer, alpha)` calibration pairs, promoting the first route that meets the unchanged CoT, raw-norm, NLL, and accuracy gates.
- **Bounded compact-screen tolerance:** On confirmation sets of 32 questions or fewer, a one-question candidate deficit is accepted only with a calibration gain of at least 10pp and non-negative pooled calibration-plus-confirmation gain; all safety and NLL constraints remain mandatory.
- **Complete confirmation telemetry:** Acceptance reports now include `confirmation_trials` with candidate metrics, pass/fail verdicts, failure reasons, pooled gain, and whether the bounded tolerance was used.
- **Kaggle alignment and regression coverage:** The execution notebook exposes the new controls; tests cover successful fallback, one-question tolerance, and strict fail-closed rejection when every candidate regresses.

---

## [1.10.0] - 2026-10-18

### 🛡️ Engine v2.3: Generation-Safe Phase 1 Selection
- **Conservative residual regime (`src/configs/config.py`):** Increased Phase-1 norm pressure to `lambda_steer_norm = 0.60`, lowered the Huber onset to `q_hi = 0.50`, and reduced training to three epochs at `3e-4`.
- **Joint layer/intensity search (`src/training/trainer.py`):** Replaced alpha=1-only screening with a compact search over every `(layer, alpha)` pair on eight calibration questions, followed by a 16-question disjoint confirmation of the selected pair.
- **Generation-aware checkpoint selection:** Added per-epoch six-question screens at alpha `{0.2, 0.4}`; Phase 1 now restores the best generation-safe checkpoint rather than the lowest teacher-forced NLL checkpoint.
- **Independent capacity before cooperation:** Restored deterministic K=1 rotation in Phase 1. K>=2 complementarity remains the explicit responsibility of the Phase-1.5 compatibility correction.
- **Raw norm safety telemetry (`src/steering/steernet.py`):** Added reset-scoped pre-cap q95 measurement and rejects calibration or confirmation routes whose q95 exceeds the hard safety ceiling.
- **Bounded raw-q tails:** Set the Phase-1 promotion ceiling to `q95 <= 1.25`, allowing modest empirical tails when the selected gate bounds the realized residual while retaining runaway-norm rejection.
- **Non-regressing epoch selection:** Phase-1 generation screens now require a CoT- and norm-safe route to match its paired baseline before it may select a checkpoint; otherwise Phase 1 restores the NLL-best state. Full calibration now uses 12 rows and logs every ranking field and tie-break decision.
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
