from __future__ import annotations
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.analysis.cf_table import CFTable
from src.configs.config import RSCConfig
from src.utils.failure_policy import FatalContractError, RunManifest, get_manifest

_NEAR_TIE_EPS = 0.01
_CROSSFIT_REPEATS = 20


@dataclass
class ShotAudit:
    n: int
    g_fixed_star: float
    s_fixed_star: Tuple[int, ...]
    g_fixed_ci: Tuple[float, float]
    g_fixed_crossfit: float             # 2-fold cross-fitted (choose on fold A, score on fold B), mean of 20 random splits
    g_oracle: float
    g_oracle_ci: Tuple[float, float]
    headroom_H: float
    headroom_ci: Tuple[float, float]   # bootstrap re-selects S_fixed* inside each resample
    frac_empty_best: float
    frac_best_fixed_hurts: float
    single_layer_gain: Dict[int, float]
    best_gain_by_k: Dict[int, float]
    oracle_choice_entropy_bits: float
    mean_near_tie_set_size: float       # |E_n| at eps=0.01
    k0_diagnosis: str                   # "rational_abstention" | "feature_failure_possible" | "signal_present"


@dataclass
class AuditResult:
    per_shot: Dict[int, ShotAudit]
    overall: ShotAudit
    heldout: Optional[ShotAudit]
    flags: List[str]
    decision_1: str
    decision_2_inputs: Dict[str, float]


class _View:
    """Minimal row-slice of a CFTable exposing only what the audit needs."""

    def __init__(self, gain: np.ndarray, catalog: Sequence[Tuple[int, ...]]):
        self._gain = gain
        self.catalog = list(catalog)

    def gain(self) -> np.ndarray:
        return self._gain


def _catalog_of(table, C: int) -> List[Tuple[int, ...]]:
    cat = getattr(table, "catalog", None)
    if cat is not None and hasattr(cat, "subsets"):
        cat = cat.subsets
    if cat is None or len(cat) != C:
        return [(i,) for i in range(C)]
    return [tuple(int(x) for x in s) for s in cat]


def _bootstrap(gain: np.ndarray, n_boot: int, rng: np.random.Generator):
    """Example-level bootstrap via multinomial counts: mean_resample = counts @ x / N.

    S_fixed* is re-selected inside every resample (max over columns of resampled means).
    """
    N, C = gain.shape
    oracle_row = gain.max(axis=1)
    g_f, g_o = [], []
    done, chunk = 0, 500
    p = np.full(N, 1.0 / N)
    while done < n_boot:
        b = min(chunk, n_boot - done)
        counts = rng.multinomial(N, p, size=b).astype(np.float64)
        g_f.append((counts @ gain / N).max(axis=1))
        g_o.append(counts @ oracle_row / N)
        done += b
    g_f = np.concatenate(g_f)
    g_o = np.concatenate(g_o)
    h = g_o - g_f

    def ci(x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return float(lo), float(hi)

    return ci(g_f), ci(g_o), ci(h)


def _crossfit(gain: np.ndarray, rng: np.random.Generator) -> float:
    N = gain.shape[0]
    if N < 2:
        return float(gain.mean(axis=0).max())
    vals = []
    for _ in range(_CROSSFIT_REPEATS):
        perm = rng.permutation(N)
        a, b = perm[: N // 2], perm[N // 2:]
        ga, gb = gain[a].mean(axis=0), gain[b].mean(axis=0)
        s_ab = gb[int(np.argmax(ga))]
        s_ba = ga[int(np.argmax(gb))]
        vals.append(0.5 * (s_ab + s_ba))
    return float(np.mean(vals))


def compute_shot_audit(table_slice: CFTable, cfg: RSCConfig, *, seed: int = 0) -> ShotAudit:
    """Computes all statistical metrics for a single shot or overall table slice.
    
    Quantities:
        gain = table_slice.gain()  # (N, C)
        G_fixed* = max_c mean_n gain[n, c]
        G_oracle = mean_n max_c gain[n, c]
        H = G_oracle - G_fixed*
        CIs by vectorized example-level bootstrap (audit_n_boot resamples).
        Cross-fitted G_fixed: 2-fold split, repeated 20 times.
        k0_diagnosis:
            G_fixed* < eps and H < eps -> "rational_abstention"
            G_fixed* < eps and H >= 0.01 -> "feature_failure_possible"
            else -> "signal_present"
    """
    gain = np.asarray(table_slice.gain(), dtype=np.float64)
    if gain.ndim != 2 or gain.shape[0] < 1 or gain.shape[1] < 1:
        raise FatalContractError(f"signal audit requires a non-empty (N, C) gain matrix, got {gain.shape}", code="E_SIGNAL_AUDIT")
    N, C = gain.shape
    catalog = _catalog_of(table_slice, C)
    rng = np.random.default_rng(seed)

    col_mean = gain.mean(axis=0)
    c_star = int(np.argmax(col_mean))
    g_fixed = float(col_mean[c_star])
    row_best = gain.max(axis=1)
    g_oracle = float(row_best.mean())
    H = g_oracle - g_fixed

    g_fixed_ci, g_oracle_ci, h_ci = _bootstrap(gain, int(cfg.audit_n_boot), rng)
    crossfit = _crossfit(gain, rng)

    # Empty-subset-best: ties count; if no empty column exists, empty gain is 0 by definition.
    empty_cols = [i for i, s in enumerate(catalog) if len(s) == 0]
    if empty_cols:
        empty_best = gain[:, empty_cols].max(axis=1) >= row_best - 1e-12
    else:
        empty_best = row_best <= 1e-12
    frac_empty_best = float(empty_best.mean())
    frac_hurts = float((gain[:, c_star] < 0).mean())

    single = {s[0]: float(col_mean[i]) for i, s in enumerate(catalog) if len(s) == 1}
    by_k: Dict[int, float] = {}
    for i, s in enumerate(catalog):
        k = len(s)
        by_k[k] = max(by_k.get(k, -np.inf), float(col_mean[i]))

    choice = np.argmax(gain, axis=1)
    pc = np.bincount(choice, minlength=C) / N
    nz = pc[pc > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    near_tie = float((gain >= (row_best[:, None] - _NEAR_TIE_EPS)).sum(axis=1).mean())

    eps = float(cfg.audit_g_zero_eps)
    if g_fixed < eps:
        diag = "rational_abstention" if H < eps else "feature_failure_possible"
    else:
        diag = "signal_present"

    return ShotAudit(
        n=int(N), g_fixed_star=g_fixed, s_fixed_star=tuple(catalog[c_star]),
        g_fixed_ci=g_fixed_ci, g_fixed_crossfit=crossfit,
        g_oracle=g_oracle, g_oracle_ci=g_oracle_ci,
        headroom_H=float(H), headroom_ci=h_ci,
        frac_empty_best=frac_empty_best, frac_best_fixed_hurts=frac_hurts,
        single_layer_gain=single, best_gain_by_k=by_k,
        oracle_choice_entropy_bits=entropy, mean_near_tie_set_size=near_tie,
        k0_diagnosis=diag,
    )


def _slice(table, mask: np.ndarray) -> _View:
    gain = np.asarray(table.gain(), dtype=np.float64)[mask]
    return _View(gain, _catalog_of(table, gain.shape[1]))


def compute_signal_audit(table: CFTable, cfg: RSCConfig, *, seed: int = 0) -> AuditResult:
    """Computes shot-specific, overall, and heldout audits, sets advisory flags,
    and determines decision_1 branch:
        BRANCH_A: SWITCH_TRAINING_TARGET -> M3, M4, rebuild table
        BRANCH_B: PROCEED -> M4 (layer-dropout), rebuild table
    """
    manifest = get_manifest()
    gain = np.asarray(table.gain())
    if gain.ndim != 2 or gain.shape[0] < 1:
        raise FatalContractError(f"signal audit requires a non-empty (N, C) gain matrix, got {gain.shape}", code="E_SIGNAL_AUDIT")
    N = gain.shape[0]

    shots = getattr(table, "shots", None)
    if shots is None:
        shots = getattr(table, "n_shot", np.zeros(N, dtype=int))
    shots = np.asarray(shots)

    heldout = getattr(table, "heldout", None)
    if heldout is None and hasattr(table, "split"):
        heldout = np.asarray(table.split) == "heldout"
    heldout = np.zeros(N, dtype=bool) if heldout is None else np.asarray(heldout, dtype=bool)

    if shots.shape[0] != N or heldout.shape[0] != N:
        raise FatalContractError("shots/heldout length does not match gain rows", code="E_SIGNAL_AUDIT")

    train = ~heldout
    if not train.any():  # degenerate: audit everything rather than raise
        train = np.ones(N, dtype=bool)

    per_shot: Dict[int, ShotAudit] = {}
    for s in sorted(set(int(x) for x in shots[train])):
        per_shot[s] = compute_shot_audit(_slice(table, train & (shots == s)), cfg, seed=seed + s)
    overall = compute_shot_audit(_slice(table, train), cfg, seed=seed)
    held = compute_shot_audit(_slice(table, heldout), cfg, seed=seed + 7919) if heldout.any() else None

    flags: List[str] = []

    def emit(flag: str, warn: bool = False):
        flags.append(flag)
        if warn:
            manifest.warn(flag, f"Signal audit warning: {flag}")
        else:
            manifest.advisory(flag, True, f"Signal audit flag: {flag}")

    if 0 in per_shot and per_shot[0].g_fixed_star < cfg.audit_g_fixed_min:
        emit("SWITCH_TRAINING_TARGET")

    hs = [a.headroom_H for a in per_shot.values()] or [overall.headroom_H]
    if any(h >= cfg.audit_h_high for h in hs):
        emit("ROUTING_HEADROOM_OK")
    elif all(h < cfg.audit_h_low for h in hs):
        emit("ROUTING_NO_HEADROOM")
    else:
        emit("ROUTING_HEADROOM_INTERMEDIATE")

    for s, a in per_shot.items():
        if s in (4, 8) and a.g_fixed_star < cfg.audit_g_zero_eps:
            emit(f"MULTISHOT_NONINFERIORITY_ONLY[{s}]")

    if held is not None and 0 in per_shot:
        held0_mask = heldout & (shots == 0)
        held0 = compute_shot_audit(_slice(table, held0_mask), cfg, seed=seed + 1) if held0_mask.any() else held
        if per_shot[0].g_fixed_star - held0.g_fixed_star > cfg.audit_inflation_margin:
            emit("W_INSAMPLE_INFLATION", warn=True)

    if "SWITCH_TRAINING_TARGET" in flags:
        decision_1 = "BRANCH_A: SWITCH_TRAINING_TARGET -> M3, M4, rebuild table"
    else:
        decision_1 = "BRANCH_B: PROCEED -> M4 (layer-dropout), rebuild table"

    ref = per_shot.get(0, overall)
    d2 = {
        "g_fixed_star_0shot": float(ref.g_fixed_star),
        "g_oracle_0shot": float(ref.g_oracle),
        "headroom_H_0shot": float(ref.headroom_H),
        "headroom_H_max": float(max(hs)),
        "frac_empty_best_overall": float(overall.frac_empty_best),
        "mean_near_tie_set_size_overall": float(overall.mean_near_tie_set_size),
        "oracle_choice_entropy_bits_overall": float(overall.oracle_choice_entropy_bits),
    }
    return AuditResult(per_shot=per_shot, overall=overall, heldout=held, flags=flags,
                       decision_1=decision_1, decision_2_inputs=d2)


def write_audit(result: AuditResult, manifest: RunManifest, out_json: str, out_md: str) -> None:
    """Writes JSON structured dump and generates Markdown summary table AUDIT_REPORT_S0.md."""
    for p in (out_json, out_md):
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(asdict(result), f, indent=2, default=float)

    def row(name: str, a: ShotAudit) -> str:
        return (f"| {name} | {a.n} | {a.g_fixed_star:.4f} [{a.g_fixed_ci[0]:.4f}, {a.g_fixed_ci[1]:.4f}] "
                f"| {a.g_fixed_crossfit:.4f} | {a.g_oracle:.4f} | {a.headroom_H:.4f} "
                f"[{a.headroom_ci[0]:.4f}, {a.headroom_ci[1]:.4f}] | {a.frac_empty_best:.3f} "
                f"| {a.frac_best_fixed_hurts:.3f} | {a.mean_near_tie_set_size:.2f} | {a.k0_diagnosis} |")

    lines = [
        "# AUDIT_REPORT_S0 — Signal Audit",
        "",
        f"**Decision 1:** {result.decision_1}",
        "",
        "**Flags:** " + (", ".join(result.flags) if result.flags else "none"),
        "",
        "| Slice | N | G_fixed* [95% CI] | G_fixed crossfit | G_oracle | H [95% CI] "
        "| frac empty best | frac fixed hurts | mean |E_n| | k0 diagnosis |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s, a in sorted(result.per_shot.items()):
        lines.append(row(f"{s}-shot", a))
    lines.append(row("overall", result.overall))
    if result.heldout is not None:
        lines.append(row("heldout", result.heldout))
    lines += ["", "## Decision-2 inputs", ""]
    lines += [f"- `{k}`: {v:.5f}" for k, v in result.decision_2_inputs.items()]
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    try:
        manifest.advisory("signal_audit_files", {"json": out_json, "md": out_md})
    except Exception:
        pass
