# Experiment Report: `exp_20261011_134907_ars_phi2_kaggle`

**Date:** 2026-10-11  
**Platform:** Kaggle GPU (NVIDIA T4)  
**Model Backbone:** Microsoft Phi-2 (2.7B)  
**Task / Dataset:** GSM8K CoT (2,000 train examples, 30 test problems per shot)  
**Configuration Highlights:** Decoupled SteerNet ($r=16, \alpha=32$), $k\_scale$ removed, $\beta_{\text{entropy\_k}}=0.03$.  
**Git Baseline Commit:** `491bf8d`

---

## 1. Executive Summary

This experiment evaluated the ARS pipeline after the initial removal of $k\_scale = \sqrt{K}$ and the 10x reduction of entropy regularization. It served as a critical diagnostic milestone by revealing two fundamental underlying issues that were previously hidden under excessive dampening.

### Benchmark Results

| Protocol | Frozen Baseline | SteerNet Only | Full ARS Pipeline | Gain vs Base ($\Delta$) | Answer Present Rate (ARS) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0-Shot** | 40.00% (12/30) | 23.33% (7/30) | **43.33% (13/30)** | **+3.33%** | 100.0% |
| **2-Shot** | 46.67% (14/30) | 40.00% (12/30) | **40.00% (12/30)** | -6.67% | 96.67% |
| **4-Shot** | 43.33% (13/30) | 26.67% (8/30) | **36.67% (11/30)** | -6.66% | 80.00% |
| **8-Shot** | 36.67% (11/30) | 40.00% (12/30) | **33.33% (10/30)** | -3.34% | 73.33% |

---

## 2. Router Diagnostics & Emergent Patterns

### A. Realized Active Layers ($K$)
- **$K=0$:** 0.00%
- **$K=1$:** 1.05%
- **$K=2$:** 14.67%
- **$K=3$:** 47.72%
- **$K=4$:** 36.56%

*Finding:* Slashing $\beta_{\text{entropy\_k}}$ to 0.03 completely broke the uniform 25% collapse. The router organically identified that GSM8K reasoning predominantly requires $K \in \{3, 4\}$. However, $K=0$ was structurally absent because `k_head` was fixed to $K \ge 1$.

### B. Layer Selection Rates (EMA)
- **L9:** 42.47% | **L11:** 40.64% | **L13:** 38.92% | **L15:** 40.07%
- **L18:** 43.05% | **L20:** 46.92% | **L22:** 37.01% | **L24:** 45.03%

*Finding:* Selection frequencies remained clustered between 37% and 47% due to the artificial MSE `balance_loss` in RLOO forcing all layers toward $\text{mean}(K)/8 \approx 40\%$.

---

## 3. Key Findings & Triggered Root-Cause Solutions

1. **SteerNet Only Collapse (23.33% in 0-Shot):**
   - *Cause:* Removing $k\_scale$ with `steer_max_relative_norm = 0.15` allowed 8 unscaled layers to cumulatively alter the hidden state norm by $>100\%$, degrading backbone representations.
   - *Fix:* Calibrated `steer_max_relative_norm = 0.08` to bound total stacked intervention $\le 30\%$.

2. **Multi-Shot Generation Degradation (Ans Rate down to 73.33% in 8-shot):**
   - *Cause:* `RSCUsefulnessGate` bypassed its causal temporal convolution during autoregressive decoding (`T == 1`), creating a severe training-vs-generation representation mismatch.
   - *Fix:* Built a streaming state cache inside the gate's causal conv to preserve temporal context during step-by-step generation.

3. **Absence of $K=0$ Option:**
   - *Cause:* `k_head` only predicted 4 classes mapped to $\{1..4\}$.
   - *Fix:* Expanded `k_head` to 5 classes mapping to $\{0..4\}$, enabling authentic unsteered baseline execution when problems require zero intervention.
