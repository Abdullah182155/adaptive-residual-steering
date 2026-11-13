# V22 Deep Audit Report — Adaptive Residual Steering (ARS)

> [!IMPORTANT]
> This report covers the V21 audit predictions, V22 actual results, root-cause analysis of discrepancies, and the V23 fix plan. Be brutally honest: some V22 changes made things worse. This report says why and what to do about it.

---

## Executive Summary

### V21 → V22 Scorecard

| Metric | V21 Actual | V22 Predicted | V22 Actual | Verdict |
|--------|-----------|---------------|------------|---------|
| 0-shot lenient delta | +3.3pp | +4–7pp | **+0.7pp** | ❌ Worse than V21 |
| 2-shot lenient delta | +1.0pp | +2–4pp | **−1.3pp** | ❌ Regression |
| 0-shot format rate delta | unknown | — | **+92.0pp** | ✅ Unexpected win |
| mag_head std | 0.0000 | > 0.05 | **0.0000** | ❌ Still saturated |
| Gate alpha spread | flat ≈0.25 | differentiated | **std=0.23** | ✅ Gate working |
| Gate benefit vs fixed | worse | +0.5–1.0pp | **+1.3pp (0s), −1.0pp (2s)** | ⚠️ Mixed |
| ARITHMETIC errors | +6 worse | ≈0 | **+26 worse (0s)** | ❌ Major regression |
| PLAN_ERROR | −12 | ≈−10 | **−18 (0s)** | ✅ Better than predicted |

### What V22 Proved and Disproved

**Proved:**
- Enabling `use_mag_head=True` fixes plan errors — plan errors reduced 23%, format compliance went from 5.7% → 97.7%
- The gate supervision redesign works: gate genuinely differentiates token types (operator 0.85, final 0.22)
- Gate alpha variance (std=0.23) vs V21's flat alpha≈0.25 shows the supervised advantage approach is fundamentally sound

**Disproved:**
- `use_mag_head=True` did NOT fix magnitude saturation — `std=0.0000` again, `mean=0.0500 = rho_max/sqrt(K)` exactly
- Reducing `kl_arithmetic_weight` and increasing `rho_max` caused +26 ARITHMETIC regressions at 0-shot
- 90% 0-shot training crushed 2-shot performance (−1.3pp)
- Net lenient improvement worse than V21 because format gain doesn't add answers — baseline was already extracting answers via lenient heuristics

**Three new root causes confirmed:**
- **RC4**: mag_head saturates to 1.0 for ALL tokens — NLL gradient (weight≈1.0) dominates L1 penalty (weight=0.01) by 100×
- **RC5**: rho_max=0.10 is 2.5× too high — uniform 5% perturbation disrupts correct computation paths
- **RC6**: 90% 0-shot training kills few-shot generalization; on-policy correct rate fell from 69–77% (V21) to 43–53% (V22)

---

## 1. SteerNet Architecture (V22 Update)

### 1.1 mag_head Saturation — New Root Cause Analysis

**What we see**: `mean=0.0500, std=0.0000` for all 300 questions, both shot counts. Every token in every question receives `magnitude ≈ 1.0`.

**Why magnitude=0.0500 exactly**: In `relative` amplitude mode with `use_mag_head=True`:
```
rel_max ≈ rho_max/sqrt(K) × magnitude × (centered_norm/‖h‖)
```
For high-dimensional Phi-2 hidden states (d=2560), `centered_norm/‖h‖ ≈ 1.0` (nearly constant by law of large numbers). So:
```
rel_max ≈ 0.10/sqrt(4) × 1.0 × 1.0 = 0.0500
```
The constant `0.0500` is the mathematical fingerprint of `magnitude=1.0` everywhere.

**Why mag_head saturates**: Gradient flow through magnitude:
- **NLL loss** on RESULT/FINAL → pushes magnitude UP (more steering = lower NLL)
- **NLL loss** on PROSE/FORMAT → also pushes magnitude UP (steering teaches format)
- **L1 prose penalty** → pushes magnitude DOWN on prose only, `weight=0.01`
- **Net**: NLL gradient (implicit weight≈1.0) overwhelms L1 (weight=0.01) by **100×**

The mag_head starts at bias=−3.0 (output≈0.047), receives NLL-driven gradient to increase, saturates to 1.0 within 1 epoch.

**The fundamental flaw**: With a frozen backbone, ANY steering direction that shifts token distributions away from the baseline reduces STEERED NLL (measured against gold format-compliant labels). Format compliance alone generates strong NLL improvement — this is a broad signal that pushes magnitude to 1.0 for all tokens, not just reasoning-critical ones.

### 1.2 rho_max Caused Arithmetic Regressions

**V22**: `rho_max=0.10`, effective `rel_max=5%` of hidden norm. This is 2.5× higher than V21's effective magnitude.

The 26 extra ARITHMETIC errors at 0-shot show a clear pattern — the uniform high-magnitude steering is injecting coherent but wrong computation steps:

```
James: 3 sprints × 60m × 3 times/week
  Gold: 540 meters/week
  Fixed: 540 × 7 = 3780  ← injected extra "per 7 days" multiplication
```
```
Downloads: months 1→2→3 with 30% reduction on month 3
  Gold: 60 + 180 + 126 = 366
  Fixed: 60 + 108 = 168   ← skipped first month, used wrong percent
```

These are not random errors — the steering direction encodes "do more computation" (learned from multi-step training data), which sometimes adds spurious steps or reorganizes the computation plan.

### 1.3 V23 Architecture Fixes

**Fix A1 — Global magnitude L1 (critical code change)**

Replace the prose-only L1 block in `trainer.py` (lines 558–574) with:

```python
# V23: global magnitude L1 on ALL tokens + extra weight on non-math
for sn in model.steer_nets.values():
    if sn.last_pre_cap_q is None:
        continue
    q = sn.last_pre_cap_q[:, :, 0]  # (B, T), range [0,1]
    active = q[q > 0]
    if active.numel() == 0:
        continue
    # Global penalty on all tokens — counteracts NLL-driven saturation
    global_pen = active.mean()
    mag_penalty = mag_penalty + global_pen
    # 2× penalty on format/prose (tokens that shouldn't be steered)
    if tt is not None:
        T_use = min(q.shape[1], tt.shape[1])
        q_2d = sn.last_pre_cap_q[:, :T_use, 0]
        tt_2d = tt[:, :T_use]
        non_math = (tt_2d == T_PROSE) | (tt_2d == T_FORMAT)
        q_nm = q_2d[non_math & (q_2d > 0)]
        if q_nm.numel() > 0:
            mag_penalty = mag_penalty + q_nm.mean()
    n_nets += 1
```

Config: `mag_l1_weight: float = 0.10` (was 0.01 — 10× increase).

**Fix A2 — Reduce rho_max to 0.04**

```python
rho_max: float = 0.04  # was 0.10 in V22; reduces to 2% of hidden norm
```

This reduces the worst-case disruption from 5% → 2% of hidden norm. With a working mag_head (non-uniform), only tokens that need steering will reach the 2% ceiling.

---

## 2. Training Pipeline (V22 Update)

### 2.1 90% 0-Shot Training is Destroying Few-Shot Generalization

**V22 config**: `train_shot_probs=[0.9, 0.05, 0.05, 0.0, 0.0]`

Consequences:
1. Model almost never sees few-shot exemplar formatting during training → can't generalize at 2-shot eval
2. On-policy generation is 90% 0-shot → harder problems → lower correct rate (43–53% vs 69–77%)
3. 2-shot evaluation at test time is essentially out-of-distribution

**Fix B1 — Restore balanced shot mixture**

```python
train_shot_probs: List[float] = field(
    default_factory=lambda: [0.50, 0.15, 0.20, 0.10, 0.05]
)  # was [0.9, 0.05, 0.05, 0.0, 0.0] in V22
```

Also match on-policy sampling to training distribution (avoid the 60/20/20 vs 90/5/5 mismatch seen in V22).

### 2.2 KL Arithmetic Weight Reduction Backfired

ARITHMETIC regressions went from +6 (V21) to +26 (V22). The `kl_arithmetic_weight` reduction removed the KL anchor on operator/operand tokens, allowing the steering to shift arithmetic computation distributions in arbitrary directions.

The V22 results show: plan error reduction (−18) proves we can fix reasoning WITHOUT touching arithmetic. The arithmetic regressions prove we went too far in removing the anchor.

**Fix B2 — Restore kl_arithmetic_weight to 0.15 (compromise)**

```python
kl_arithmetic_weight: float = 0.15  # was 0.30 (V21), tried 0.05 (V22)
```

0.15 allows some arithmetic distribution shift (improving cases like 10× ARITHMETIC wins at 2-shot) while preventing catastrophic reorganization.

### 2.3 New Problem: NLL vs Generation Gap

V22: `result_top1=0.950` (95% token prediction accuracy) but `gen_acc=34%`. This 61pp gap means the model is excellent at teacher-forced NLL but poor at generation. On-policy self-improvement is supposed to close this gap, but V22's on-policy correct rate (43–53%) is lower than V21's (69–77%).

The NLL objective is training the model to predict gold tokens given gold context. Generation requires compounding correct decisions without gold context. This gap is inherent, but it's larger in V22 because rho_max=0.10 disrupts coherent generation chains.

---

## 3. On-Policy Training (V22 Update)

### 3.1 Correct Rate Collapse

| Run | Epoch 1 | Epoch 3 |
|-----|---------|---------|
| V21 | 69.2% | 76.9% |
| V22 | 43.3% | 53.2% |

The ~26pp drop is caused by: (1) higher rho_max breaking chains that would otherwise be correct, (2) 90% 0-shot being a harder distribution for self-improvement.

### 3.2 V23 On-Policy Fixes

**Fix C1 — Reduce on_policy_k to 2, temperature to 0.5**

```python
on_policy_k: int = 2             # was 4; faster + more signal per question
on_policy_temperature: float = 0.5  # was 0.7; more deterministic sampling
```

With a correct rate of 43%, pass@4 wastes 3 of 4 slots. Pass@2 at lower temperature is more efficient.

**Fix C2 — Add arithmetic step verification filter**

Before accepting an on-policy completion, verify arithmetic equations:

```python
def _verify_arithmetic_steps(completion: str, tol: float = 0.02) -> bool:
    """Reject completions with provably wrong intermediate arithmetic."""
    import re
    for m in re.finditer(
        r'([\d\s\+\-\*\/\.\(\)×÷]+)\s*=\s*(-?[\d\.]+)', completion
    ):
        try:
            lhs = eval(
                m.group(1).replace('×','*').replace('÷','/').strip(),
                {"__builtins__": {}}
            )
            rhs = float(m.group(2))
            if abs(lhs - rhs) > tol * max(1.0, abs(rhs)):
                return False
        except Exception:
            pass
    return True
```

Apply in `collect_onpolicy_data` before `new_rows.append(row)`:
```python
if scored["lenient"] and _verify_arithmetic_steps(comp):
    new_rows.append(row)
```

This directly prevents trajectories like `540 × 7 = 3780` from entering the training data.

---

## 4. Gate Design (V22 Update)

### 4.1 V22 Gate: What Actually Worked

The gate redesign is a **genuine mechanistic success**:

| Token Type | Alpha | Target | Status |
|-----------|-------|--------|--------|
| OPERATOR | 0.849 | 0.85 | ✅ On target |
| OPERAND | 0.758 | 0.85 | ⚠️ Close |
| RESULT | 0.687 | 0.85 | ⚠️ Low |
| **FINAL** | **0.218** | **0.85** | **❌ Far below** |
| FORMAT | 0.347 | 0.15 | ⚠️ 2× target |
| PROSE | 0.652 | 0.65 | ✅ On target |

The gate std=0.23 (vs V21's flat 0.25) and operator=0.85 show real differentiation. Keep this architecture.

### 4.2 Why FINAL Tokens Get Low Alpha (and Why This is Actually Correct)

FINAL tokens (the numeric answer itself: `20`, `366`, `540`) have alpha=0.218. The gate is correctly responding to V22's reality: when the steered model computes wrong answers (+26 arithmetic regressions), NLL on FINAL tokens goes UP when steering is applied. The gate learns "suppress steering on FINAL tokens" — which is the right response to bad SteerNet outputs.

**The fix is not in the gate — it's in SteerNet.** Once Fix A1+A2 reduces arithmetic regressions, the gate will naturally give higher alpha to FINAL tokens because steering will improve (not worsen) the final number.

### 4.3 V23 Gate Fixes

**Fix D1 — Keep supervised advantage architecture (don't revert)**

The differentiation is working. Continue with this approach.

**Fix D2 — Higher contrastive weight for FINAL tokens**

```python
# In gate_trainer.py, _contrastive_gate_loss():
final_mask = v & (t == T_FINAL)
if math_mask.any():
    loss = loss + F.relu(0.85 - a[math_mask]).mean()
if final_mask.any():  # NEW: extra weight on FINAL
    loss = loss + 2.0 * F.relu(0.85 - a[final_mask]).mean()
```

**Fix D3 — Tighten FORMAT hinge to 0.05**

```python
if fmt_mask.any():
    loss = loss + F.relu(a[fmt_mask] - 0.05).mean()  # was 0.15
```

---

## 5. Data & Evaluation (V22 Update)

### 5.1 The Baseline Scoring Problem

```
0-shot: Baseline strict=4.0%  lenient=49.3%
        Fixed    strict=49.3% lenient=50.0%
```

The lenient baseline (49.3%) extracts correct answers from unformatted Phi-2 output using heuristics (last `= N`, last line, etc.). The steered model outputs proper format (97.7%) but only gains +0.7pp lenient. This is because:

1. Baseline was already getting 49.3% correct answers via lenient extraction
2. SteerNet teaches format without proportionally improving answer quality
3. The +45.3pp strict gain is a formatting success, not an accuracy success

**Key implication**: For V23 with high format compliance, **strict accuracy IS the right primary metric**. Lenient is dominated by baseline heuristics and understates the fixed model's format improvement. Report both, but lead with strict.

### 5.2 Statistical Significance

```
0-shot: 47W / 45L / 208T → p=0.9170 (NOT significant)
2-shot: 29W / 33L / 238T → p=0.7035 (NOT significant)
```

**Fix E1 — Use full 1319-question test set for V23**

Reduces CI from ±5.6pp to ±2.7pp. At +3pp improvement, this gives p≈0.03 (significant).

---

## 6. V23 Complete Config Diff

```python
# src/configs/config.py — V23 diff from V22

# Architecture
rho_max: float = 0.04              # was 0.10 — reduce aggressive steering

# Training
train_shot_probs: List[float] = field(
    default_factory=lambda: [0.50, 0.15, 0.20, 0.10, 0.05]
)                                   # was [0.9, 0.05, 0.05, 0.0, 0.0]
kl_arithmetic_weight: float = 0.15  # was 0.05 — partial arithmetic anchor
mag_l1_weight: float = 0.10         # was 0.01 — 10× stronger magnitude pressure

# On-policy
on_policy_k: int = 2                # was 4
on_policy_temperature: float = 0.5  # was 0.7

# Gate (if running S2B)
gate_contrastive_weight: float = 2.0  # was 1.0 — stronger type discrimination
```

Plus code change in `trainer.py` (Fix A1: global mag L1) and `gate_trainer.py` (Fix D2: FINAL hinge weight, Fix D3: FORMAT hinge=0.05).

---

## 7. Expected Impact Model (V23)

### 7.1 Per-Fix Estimated Impact

| Fix | 0-shot Strict | 0-shot Lenient | 2-shot Lenient |
|-----|--------------|---------------|---------------|
| rho_max=0.04 | +2–3pp | +1–2pp | +1pp |
| Balanced shots | +0.5pp | +0.5pp | +2–3pp |
| kl_arith=0.15 | +1pp | +0.5pp | +0.5pp |
| Global mag L1 | +1–2pp | +0.5–1pp | +0.5pp |
| Gate FINAL fix | +0.5–1pp | +0–0.5pp | +0.5pp |

### 7.2 Combined V23 Estimates (over Phi-2 baseline)

| Scenario | 0-shot Strict | 0-shot Lenient | 2-shot Lenient |
|----------|--------------|---------------|---------------|
| Phi-2 baseline | 4.0% | 49.3% | 54.0% |
| V22 actual | 49.3% | 50.0% | 52.7% |
| V23 conservative | 52–55% | 51–53% | 56–58% |
| V23 central | 55–60% | 53–55% | 57–60% |

The central V23 estimate gives +5–10pp strict at 0-shot (from V22's 49.3%) and +3–5pp lenient at 2-shot (from V22's 52.7%). These are meaningful, statistically significant improvements on the 1319-question test set.

### 7.3 Why +15pp Remains Unrealistic

For the record: +15pp lenient would require ≈63% accuracy. Getting there requires:
- Zero additional arithmetic regressions (currently +26 at 0-shot; must eliminate AND recover −26)
- Solving plan errors to near-zero (need −60 more from current 60)

This is not achievable with a frozen backbone at any rho_max. The backbone's internal computation paths are the ceiling.

**Most realistic path to +15pp**: Fine-tune Phi-2 backbone (LoRA rank 4) jointly with SteerNet, or switch to Phi-3-mini (3.8B) which has significantly better baseline math reasoning (≈69% GSM8K baseline vs Phi-2's ≈57%).

---

## 8. Synthesis: What V22 Taught Us

### V22 Surprising Successes
1. **Format compliance +92pp at 0-shot**: Proves `relative` mode + `use_mag_head=True` can teach format with no explicit format loss
2. **Plan error −18 at 0-shot**: The steering direction is finding useful reasoning patterns
3. **Gate differentiation std=0.23**: Supervised advantage gate architecture is sound

### V22 Clear Failures
1. **Magnitude still saturated**: 100× gradient imbalance (NLL vs L1) is the root cause, not architecture
2. **+26 ARITHMETIC regressions**: rho_max=0.10 at saturated magnitude is catastrophically disruptive
3. **90% 0-shot hurt generalization**: Simple distribution mismatch with large impact

### The Core Tension (Still Unresolved)
SteerNet is simultaneously trying to:
- **Teach format** (always steer → uniform high magnitude)
- **Teach selective reasoning** (selective steer → discriminative magnitude)

These are in conflict. V23 resolution: use global mag L1 at weight=0.10 to force selectivity. With strong downward pressure on ALL tokens, only tokens where NLL improvement exceeds the L1 penalty cost (≈0.10) will develop high magnitude. Format tokens (already improving format with any steering) may still get moderate magnitude; arithmetic tokens (need precise correct steps) must earn high magnitude via specific NLL improvement.
