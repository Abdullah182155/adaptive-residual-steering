"""Paired statistics for per-item correctness vectors."""

import math
from typing import Dict, Sequence

import numpy as np


def mcnemar_exact(control: Sequence[bool], treat: Sequence[bool]) -> Dict[str, float]:
    """Exact two-sided McNemar test.  b = treatment-only correct, c = control-only correct."""
    c0 = np.asarray(control, dtype=bool)
    t0 = np.asarray(treat,   dtype=bool)
    if c0.shape != t0.shape:
        raise ValueError("control and treatment must have the same length")
    b = int((t0 & ~c0).sum())
    c = int((c0 & ~t0).sum())
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p": 1.0}
    k    = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    p    = min(1.0, (2 * tail) / (2 ** n))  # exact integer arithmetic: no underflow for large n
    return {"b": b, "c": c, "n_discordant": n, "p": float(p)}


def paired_bootstrap_ci(
    control: Sequence[bool],
    treat:   Sequence[bool],
    n_boot:  int   = 10000,
    alpha:   float = 0.05,
    seed:    int   = 0,
    chunk:   int   = 500,
) -> Dict[str, float]:
    """Percentile CI for Acc_treat - Acc_control (percentage points) by paired resampling."""
    c0 = np.asarray(control, dtype=bool)
    t0 = np.asarray(treat,   dtype=bool)
    if c0.shape != t0.shape:
        raise ValueError("control and treatment must have the same length")
    n = c0.size
    if n == 0:
        return {"delta_pp": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n_boot": n_boot, "alpha": alpha}
    d      = t0.astype(np.float64) - c0.astype(np.float64)
    rng    = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for s in range(0, n_boot, chunk):
        m             = min(chunk, n_boot - s)
        idx           = rng.integers(0, n, size=(m, n))
        deltas[s:s+m] = d[idx].mean(axis=1) * 100.0
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "delta_pp": float(d.mean() * 100.0),
        "ci_lo":    float(lo),
        "ci_hi":    float(hi),
        "n_boot":   int(n_boot),
        "alpha":    alpha,
    }


def holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    items   = sorted(pvals.items(), key=lambda kv: kv[1])
    m       = len(items)
    adj:    Dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        running   = max(running, min(1.0, (m - rank) * p))
        adj[name] = running
    return {k: adj[k] for k in pvals}


def compare_to_control(
    control:           Sequence[bool],
    treatments:        Dict[str, Sequence[bool]],
    n_boot:            int   = 10000,
    alpha:             float = 0.05,
    seed:              int   = 0,
    noninf_margin_pp:  float = 1.0,
) -> Dict[str, Dict[str, float]]:
    ctrl = np.asarray(control, dtype=bool)
    out: Dict[str, Dict[str, float]] = {}
    for i, (name, t) in enumerate(treatments.items()):
        t  = np.asarray(t, dtype=bool)
        mc = mcnemar_exact(ctrl, t)
        ci = paired_bootstrap_ci(ctrl, t, n_boot=n_boot, alpha=alpha, seed=seed + i)
        out[name] = {
            "delta_pp": ci["delta_pp"],
            "ci_lo":    ci["ci_lo"],
            "ci_hi":    ci["ci_hi"],
            "b":        mc["b"],
            "c":        mc["c"],
            "p":        mc["p"],
        }
    holm = holm_bonferroni({k: v["p"] for k, v in out.items()})
    for k, v in out.items():
        v["p_holm"]     = holm[k]
        v["significant"] = bool(v["p_holm"] < alpha)
        v["noninferior"] = bool(v["ci_lo"] > -noninf_margin_pp)
    return out
