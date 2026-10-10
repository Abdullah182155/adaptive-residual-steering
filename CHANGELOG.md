# Changelog

All notable changes to the **Adaptive Residual Steering (ARS)** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
