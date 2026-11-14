# ARS V24 Audit Report & V25 Fix

Generated: 2026-11-14

---

## Executive Summary

V24 confirmed the FINAL/RESULT suppression hypothesis: arithmetic errors dropped from +22 → +7 at 0-shot, and 2-shot arithmetic IMPROVED by −7 (first time we've seen that). However a side-effect emerged: `rho_max=0.04` is now **actually** being applied (2% amplitude), which killed 0-shot format compliance (48% vs 97.7%). The fix is simple: raise `rho_max` to 0.08 while keeping all V24 suppression logic intact.

---

## 1. What Changed V23 → V24 (and what actually happened)

### V22/V23 had rho_max=0.10 accidentally — V24 is the first real 0.04 run

| Version | rho_max (config) | Actual amplitude | Evidence |
|---------|-----------------|-----------------|----------|
| V22 | 0.10 | 5.0% (`= 0.10/√4`) | mean=0.0500 |
| V23 | 0.04 (config) | **5.0% — same as V22** | mean=0.0500 (notebook still had RHO_MAX=0.10) |
| V24 | 0.04 | **2.0% (`= 0.04/√4`)** | mean=0.0199 ✅ first real V23 value |

The V23 notebook update landed correctly in V24. For the first time, `rho_max=0.04` was truly applied.

---

## 2. V24 Results vs Prior Versions

| Metric | V22 | V23 | V24 | Trend |
|--------|-----|-----|-----|-------|
| 0-shot lenient Δ | +0.7pp | −3.0pp | **+0.0pp** | ✅ recovering |
| 2-shot lenient Δ | −1.3pp | −3.0pp | **−0.7pp** | ✅ recovering |
| 0-shot ARITHMETIC Δ | +26 | +22 | **+7** | ✅ big improvement |
| 2-shot ARITHMETIC Δ | +5 | +5 | **−7** | ✅ first improvement! |
| 0-shot format rate | 97.7% | 97.7% | **48.0%** | ❌ collapsed |
| 0-shot max_tokens hits | 4 | 4 | **102** | ❌ truncation problem |

### Key wins in V24

- 0-shot CORRECT count: **+7** (baseline 141 → fixed 148) — first net-positive correct count
- 0-shot NO_ANSWER: −9 (completely eliminated)
- 0-shot LOOP_OR_RUNON: −6 (dramatically reduced)
- 2-shot ARITHMETIC: **−7** — first time steering actually improved arithmetic in 2-shot
- 2-shot EAA: **+1.5%** — arithmetic equation accuracy improved

### The format rate collapse

The reduced amplitude (2%) means the model is no longer being pushed hard enough toward "Final answer:" format at 0-shot. 102/300 questions hit `max_new_tokens` (300 tokens) without concluding — the model reasons verbosely but doesn't reach a formatted answer.

In 2-shot this doesn't matter (demonstrated format in context). In 0-shot, the model needs the steering to adopt the "Final answer: X" convention.

---

## 3. Root Cause: Format Compliance at 0-shot Requires ~4-5% Amplitude

| Amplitude | Format mechanism | Outcome |
|-----------|-----------------|---------|
| 5% (V22/V23 actual) | Strong steering at PROSE/OPERAND tokens guides to structured conclusion | 97.7% format, but ARITHMETIC corrupted |
| 2% (V24, rho_max=0.04) | Too gentle to override unstructured 0-shot baseline tendency | 48% format, 102 truncations |
| 4% (V25, rho_max=0.08) | Should be sufficient without over-steering result tokens | Expected ~75-85% format |

---

## 4. V25 Fix: rho_max 0.04 → 0.08

**The only change.** All V24 suppression logic is preserved:
- FINAL L1 multiplier: **5.0** (suppress answer token steering → prevents arithmetic corruption)
- RESULT L1 multiplier: **3.0** (suppress intermediate result token steering)
- OPERAND/OPERATOR L1 multiplier: **0.0** (saturate freely → plan/approach guidance)
- Gate FINAL target: **0.0** (suppress gate on answer tokens, aligned with supervised loss)

With rho_max=0.08 and K=4 layers: `0.08/√4 = 0.04` = 4% amplitude on saturating tokens.

> [!IMPORTANT]
> RESULT/FINAL token magnitudes are suppressed by L1 multipliers (3.0 and 5.0) **independent of rho_max**. The L1 penalty operates on `sigmoid(mag_head)` before the rho_max scaling, so RESULT/FINAL will stay near-zero even at higher rho_max. The arithmetic gains from V24 should be preserved.

---

## 5. Expected V25 Outcomes

| Metric | V24 Actual | V25 Expected |
|--------|-----------|-------------|
| 0-shot format rate | 48% | 75–85% |
| 0-shot max_tokens hits | 102 | 20–40 |
| 0-shot lenient Δ | +0.0pp | +2 to +4pp |
| 2-shot lenient Δ | −0.7pp | 0 to +2pp |
| 0-shot ARITHMETIC Δ | +7 | +8 to +15 (slight rise vs V24, but far below V22's +22) |
| 2-shot ARITHMETIC Δ | −7 | −5 to −3 (maintained improvement) |
| Amplitude (rel_max) | 2.0% | ~4.0% |

> [!NOTE]
> The gate was NEUTRAL in V24 (not helping/hurting). At higher amplitude (4%), the gate has more signal to selectively suppress/allow, so gate benefit may return.
