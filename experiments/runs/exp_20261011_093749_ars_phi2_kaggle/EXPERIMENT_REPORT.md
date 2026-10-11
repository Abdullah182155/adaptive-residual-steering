# Experiment Report: `exp_20261011_093749_ars_phi2_kaggle`

**Date:** 2026-10-11  
**Platform:** Kaggle GPU (NVIDIA T4)  
**Model Backbone:** Microsoft Phi-2 (2.7B)  
**Task / Dataset:** GSM8K CoT (5,000 train examples, 300 test problems per shot)  
**Git Baseline Commit:** `30c37e0` (Post 9-bug audit fixes)

---

## 1. Executive Summary

This experiment evaluated the ARS multi-stage pipeline across **0-shot, 2-shot, and 4-shot** reasoning on GSM8K using fair evaluation protocol across Frozen Baseline, SteerNet Only, and Full ARS Pipeline.

### Key Benchmark Results

| Protocol | Frozen Baseline | SteerNet Only | Full ARS Pipeline | Gain vs Base ($\Delta$) | Answer Present Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0-Shot** | 42.00% | 42.67% | **41.67%** | -0.33% | 100.0% |
| **2-Shot** | 43.33% | 46.00% | **48.67%** | **+5.34%** | 97.67% |
| **4-Shot** | 42.67% | 47.33% | **42.33%** | -0.34% | 87.67% |

---

## 2. Training Telemetry Summary

- **Phase 1 (SteerNet Warmup):** Best Val Loss = `0.7042` (Convergence from 0.8302)
- **Phase 1.5 (Router Bootstrap):** Best Probe Loss = `0.1180` (Win Rate vs Base = 100.0%)
- **Phase 2A (SteerNet Fine-tune):** Val = `0.7045`, Held-out Accuracy = `48.0%`
- **Phase 2B (Gate-Only Optimization):** Val = `0.8348`, Held-out Accuracy = `45.0%`
- **Phase 2C (Light Joint Fine-tune):** Val = `0.8423`, Held-out Accuracy = `45.0%`, Mean $K$ = `2.55`
- **Phase 2D (Gate Retune):** Val = `0.8399`, Held-out Accuracy = `45.0%`

---

## 3. Router Diagnostics & Emergent Patterns

### A. Layer Selection Rates (EMA)
The router distributed interventions across all 8 target layers without dead layers:
- **L9:** 35.81%
- **L11:** 29.24%
- **L13:** 30.84%
- **L15:** 28.49%
- **L18:** 28.95%
- **L20:** 34.73%
- **L22:** 29.96%
- **L24:** 32.22%

### B. Realized Active Layers ($K$)
- **$K=0$:** 0.00% (Clean — zero contamination from override probes)
- **$K=1$:** 23.15%
- **$K=2$:** 26.63%
- **$K=3$:** 25.80%
- **$K=4$:** 24.42%

---

## 4. Empirical Diagnoses & Root-Cause Findings

1. **Uniform $K$-Distribution Artifact:**
   - $K \in \{1, 2, 3, 4\}$ were evenly distributed at ~25% each.
   - **Diagnosis:** Regularizer term $\beta_{\text{entropy\_k}} = 0.30$ and `router_balance_weight = 0.20` were ~10× stronger than the RLOO policy gradient reward signal, artificially forcing a uniform distribution.

2. **SteerNet 0-Shot Ineffectiveness:**
   - SteerNet only gained +0.67% at 0-shot.
   - **Diagnosis:** Division by `k_scale = sqrt(K)` in `Phi2WithRSC` double-dampened perturbations down to ~6.9% of $\|h\|$, rendering SteerNet too weak without exemplar guidance.

3. **4-Shot Degradation (87.67% Answer Rate):**
   - At 4-shot, answer extraction fell from 97.67% to 87.67%.
   - **Diagnosis:** Hardcoded MSE targets in `compute_contrastive_gate_loss` (0.85/0.05/0.40) and absence of joint task gradient through the gate caused format token corruption under longer few-shot contexts.

---

## 5. Next Steps / Actions Taken
These empirical findings motivated the core refactoring in Commit `491bf8d` & `a9781f7`:
1. Removal of `k_scale` division to restore 12-15% steering impact.
2. Introduction of `_joint_grad_mode` for joint gate learning.
3. Slashing $\beta_{\text{entropy\_k}}$ to 0.03 to allow learned problem-complexity adaptation.
4. Structural margin-based contrastive loss replacing arbitrary MSE targets.
