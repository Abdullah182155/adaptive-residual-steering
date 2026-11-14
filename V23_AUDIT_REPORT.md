# ARS V23 Audit Report & V24 Remediation Plan

Generated: 2026-11-14

---

## Executive Summary

V23 made significant progress on gate training (+4.0pp gate benefit at 0-shot vs near-zero in V22), confirmed the balanced shot distribution landed correctly, and demonstrated a gate that now correctly discriminates token types. However, the core SteerNet regression problem persists: arithmetic errors remain elevated (+22 at 0-shot) and mag_head is still uniformly saturated. The root cause has been misdiagnosed: **the FINAL and RESULT gate targets were inverted** — V23 was pushing the gate to OPEN on FINAL tokens when it should CLOSE.

---

## 1. What V23 Fixed ✅

| Fix | Evidence |
|-----|----------|
| Balanced shot distribution landed | Log: `Shots mixture: [0.5, 0.15, 0.2, 0.1, 0.05]` ✅ |
| Gate contrastive is stronger | Gate benefit: +4.0pp (0-shot), +3.7pp (2-shot) vs ~0 in V22 |
| FORMAT gate alpha improved | FORMAT: 0.286 (V23) ← 0.347 (V22). Hinge 0.05 working |
| PROSE gate alpha on target | PROSE: 0.643 (target 0.65) ✅ |
| OPERATOR gate alpha on target | OPERATOR: 0.849 (target 0.85) ✅ |
| Gate entropy is high | std=0.263, selectivity (math−format)=0.33 → gate is active |
| Attribute fix (`use_mag_head`) | Experiment completed without crash ✅ |

---

## 2. What V23 Did NOT Fix ❌

### 2a. mag_head Still Uniformly Saturated

```
mean=0.0500  std=0.0000  range=0.0000  (both 0-shot and 2-shot)
```

Every question hits exactly 5% of hidden norm at maximum. `std=0.0` means every token of every question gets the SAME magnitude cap. The L1 multiplier changes (PROSE=1.5, FORMAT=4.0) are having no visible effect on the per-question maximum because the maximum is determined by **math tokens (OPERAND/OPERATOR)**, which still have LOW L1 multipliers (0.1-0.2). These saturate freely.

The magnitude diagnostic is measuring **per-question maximum** (not mean per token type). As long as any token hits saturation, the question maximum = rho_max/√K = 0.05.

### 2b. FINAL Gate Target Was Wrong Direction

V23 applied `2.0 × F.relu(0.85 − a[final_mask])` → tried to **push FINAL alpha UP to 0.85**.

But the supervised loss trains on actual advantage: for FINAL tokens, steering causes arithmetic errors, so advantage is NEGATIVE → supervised loss pushes FINAL alpha toward **0**.

Result: supervised loss (which dominates) won the tug-of-war. FINAL alpha went **DOWN** from 0.218 (V22) → 0.176 (V23), not up. The contrastive hinge was fighting the correct learning signal.

### 2c. RESULT Gate Also Needs Reconsideration

RESULT: 0.686 — gate is partially suppressing RESULT tokens. Contrastive hinge pushes RESULT to 0.85, but supervised loss pushes it lower (arithmetic errors on = tokens). Two-sided tension → settled at 0.686.

The semantics: RESULT tokens are the **number after "="**. Steering at max amplitude on these tokens can confuse the specific digit outputs. Gate should moderate here, not maximize.

### 2d. Arithmetic Regression Persists

```
ARITHMETIC errors: +22 (0-shot), +5 (2-shot)
```

Exactly the same magnitude as V22. Reducing rho_max 0.10→0.04 did not help because:
1. The per-question max is still 5% (rho_max/√K) → same steering amplitude
2. RESULT/FINAL tokens still have high mag_head output (low L1 multiplier 0.1)
3. Gate FINAL suppression (0.176) partially compensates but gate is downstream of SteerNet

---

## 3. Root Cause Analysis

### Why the Gate FINAL alpha decreased despite V23 2× push

```
Supervised loss: trains alpha = sigmoid(advantage × T)
  For FINAL tokens: advantage = NLL_base − NLL_gate < 0 (steering hurts)
  → sigmoid(negative × 5.0) → 0.0
  → Strong gradient pushing alpha → 0

Contrastive loss (V23): 2.0 × relu(0.85 − alpha)
  → Gradient pushing alpha → 0.85

Net: supervised is stronger (more data, clearer signal)
Result: FINAL went 0.218 → 0.176 (even lower: contrastive created instability)
```

**The V23 contrastive fix for FINAL was fighting the right answer.**

### Why Arithmetic Errors Persist Despite rho_max Reduction

The SteerNet direction vector at RESULT/FINAL positions encodes a "generic answer structure" learned from the training distribution. When steering amplitude is high on these tokens, the model is nudged away from the specific numerical answer it computed and toward a "typical answer" in the training distribution.

The gate cannot fully compensate: even with FINAL alpha=0.176, there is still 17.6% steering on FINAL tokens. With saturated magnitude (5% of hidden), this causes measurable arithmetic errors.

**The fix requires both sides:**
1. **SteerNet**: Suppress magnitude on RESULT/FINAL tokens (high L1 multiplier → near-zero magnitude)
2. **Gate**: Align contrastive target with supervised signal (push FINAL alpha toward 0, not 0.85)

---

## 4. V24 Remediation Plan

### Fix A: Invert FINAL and RESULT contrastive gate targets (`gate_trainer.py`)

| Token | V23 Target | V24 Target | Rationale |
|-------|-----------|-----------|-----------|
| OPERAND | 0.85 (push up) | 0.85 (keep) | Steering helps operand recall |
| OPERATOR | 0.85 (push up) | 0.85 (keep) | Steering helps operation selection |
| RESULT | 0.85 (push up) | Unconstrained | Let supervised loss find the right level |
| FINAL | 0.85 (push up) × 2 | 0.0 (push DOWN) × 2 | Steering is harmful on final answer |
| FORMAT | 0.05 (suppress) | 0.05 (keep) | Working well |
| PROSE | 0.65 (moderate) | 0.65 (keep) | Working well |

This aligns the contrastive hinge with the supervised advantage signal on FINAL tokens.

### Fix B: Invert mag_head L1 multipliers for RESULT/FINAL (`trainer.py`)

| Token | V23 L1 Mult | V24 L1 Mult | Rationale |
|-------|------------|------------|-----------|
| OPERAND | 0.2 | 0.0 | Let NLL maximize magnitude freely |
| OPERATOR | 0.2 | 0.0 | Let NLL maximize magnitude freely |
| RESULT | 0.1 | 3.0 | Suppress steering at = results |
| FINAL | 0.1 | 5.0 | Suppress steering at final answer number |
| PROSE | 1.5 | 1.5 | Keep |
| FORMAT | 4.0 | 4.0 | Keep |

Net effect: OPERAND/OPERATOR tokens saturate freely (high amplitude, good for plan/approach). RESULT/FINAL tokens get near-zero amplitude (prevents arithmetic corruption).

The per-question `std` will still be ~0 (OPERAND/OPERATOR always saturate to 5%), but that's now the DESIRED behavior: plan tokens at max, answer tokens at zero.

### Fix C: Increase gate_epochs from 3 → 5 (`config.py`)

Gate val_nll drop was only 0.0033 over 3 epochs. It needs more time, especially after the FINAL target change which requires unlearning V23's wrong gradient direction.

### Fix D: Notebook update with V24 gate_epochs variable

---

## 5. What to Expect from V24

| Metric | V23 Actual | V24 Expected |
|--------|-----------|--------------|
| 0-shot lenient | −3.0pp | +0 to +3pp |
| 2-shot lenient | −3.0pp | +1 to +4pp |
| ARITHMETIC errors (0-shot) | +22 | ≤ +5 |
| PLAN_ERROR (0-shot) | −16 | −15 to −20 |
| Gate FINAL alpha | 0.176 | < 0.05 |
| Gate RESULT alpha | 0.686 | 0.3–0.5 (supervised free) |
| mag_head std | 0.000 | 0.000 (now intentional) |
| Gate benefit (0-shot) | +4.0pp | +4 to +7pp |

> [!IMPORTANT]
> The core V24 hypothesis: the SteerNet helps with PLAN (approach selection), and the gate suppresses the harm on RESULT/FINAL. Together, we keep the PLAN_ERROR improvement while eliminating the ARITHMETIC regression.

> [!NOTE]
> The mag_head saturation "warning" will still fire (std=0.0), but it now means "OPERAND/OPERATOR saturate at max amplitude" which is the **correct** behavior. RESULT/FINAL will have near-zero magnitude (suppressed by high L1), so the saturation is selective.
