"""Deployment decision utilities for ARS.

ISSUE-2 fix: deployment decision is made once, at the end of training,
using a paired bootstrap CI on held-out examples. The model is only
deployed in ARS mode if the lower CI bound is strictly positive.

A-2 (run-analysis fix): adds paired_delta_ci for use in benchmark
reporting — replaces raw percentages with delta + 95% CI.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Any, List, Callable

import numpy as np

logger = logging.getLogger(__name__)


def paired_delta_ci(
    flags_a: Sequence[int | float],
    flags_b: Sequence[int | float],
    n_boot: int = 4000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap a paired accuracy delta, preserving optional question clusters.

    Args:
        flags_a: Per-item correctness for arm A.  A one-dimensional sequence
            represents ordinary paired items.  A two-dimensional sequence of
            shape ``(questions, views)`` represents repeated views of each
            question (for example 0- and 2-shot); bootstrap samples questions,
            never the correlated views independently.
        flags_b: Matching correctness outcomes for arm B.
        n_boot:  Number of bootstrap resamples.
        seed:    Random seed for reproducibility.

    Returns:
        (delta, lo_ci, hi_ci) where delta = mean(b) - mean(a).
        CI is two-sided 95% bootstrap percentile interval.
        Positive lo_ci means arm B is significantly better.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(flags_a, dtype=float)
    b = np.asarray(flags_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired_delta_ci: shape(a)={a.shape} != shape(b)={b.shape}")
    if a.ndim not in {1, 2}:
        raise ValueError(f"paired_delta_ci expects 1D items or 2D question clusters, got shape {a.shape}")
    if a.size == 0 or a.shape[0] == 0:
        return 0.0, -1.0, 1.0

    d_obs = float(b.mean() - a.mean())
    # For clustered input, average views within each question before
    # resampling.  This keeps the point estimate equal to the pooled mean but
    # preserves the within-question dependence in the uncertainty estimate.
    diffs = b - a
    if a.ndim == 2:
        if a.shape[1] == 0:
            return 0.0, -1.0, 1.0
        diffs = diffs.mean(axis=1)
    boot_means = np.array([
        rng.choice(diffs, size=len(diffs), replace=True).mean()
        for _ in range(n_boot)
    ])
    # Percentile CI
    lo, hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    return d_obs, lo, hi


def format_benchmark_cell(
    flags_a: Sequence[int | float],
    flags_b: Sequence[int | float],
    n_boot: int = 4000,
) -> str:
    """Return a human-readable benchmark cell: 'acc_b% (Δ +X.X, CI [lo, hi])'.

    Marks the cell with * if the 95% CI is entirely positive (significant improvement).
    """
    b = np.asarray(flags_b, dtype=float)
    d, lo, hi = paired_delta_ci(flags_a, flags_b, n_boot=n_boot)
    sig = "*" if lo > 0 else ""
    return (
        f"{b.mean() * 100:.1f}%{sig} "
        f"(Δ {d * 100:+.1f}pp, CI [{lo * 100:+.1f}, {hi * 100:+.1f}])"
    )


def finalize_deployment(
    per_item_baseline: Sequence[int | float],
    per_item_ars: Sequence[int | float],
    checkpoint_path: str,
    n_boot: int = 4000,
    min_n: int = 200,
) -> str:
    """Decide deployment mode using a paired bootstrap CI on held-out examples.

    Args:
        per_item_baseline: Per-item correctness for the baseline arm.
        per_item_ars:      Per-item correctness for the ARS arm.
        checkpoint_path:   Path to the manifest JSON to update.
        n_boot:            Bootstrap resamples for the CI.
        min_n:             Minimum examples required for a valid decision.

    Returns:
        'ars' if ARS is significantly better (lo_ci > 0), else 'baseline'.
        Writes deployment_mode to the manifest JSON file.
    """
    baseline = np.asarray(per_item_baseline, dtype=float)
    ars = np.asarray(per_item_ars, dtype=float)
    if baseline.shape != ars.shape:
        raise ValueError(
            "finalize_deployment: baseline and ARS outcomes must have matching shapes, "
            f"got {baseline.shape} and {ars.shape}"
        )
    if baseline.ndim not in {1, 2}:
        raise ValueError("finalize_deployment expects 1D paired items or 2D question clusters")
    n = int(baseline.shape[0])
    if n < min_n:
        logger.warning(
            "finalize_deployment: only %d examples (< %d). "
            "Decision is unreliable; defaulting to baseline.",
            n, min_n,
        )
        mode = "baseline"
    else:
        d, lo, hi = paired_delta_ci(baseline, ars, n_boot=n_boot)
        mode = "ars" if lo > 0 else "baseline"
        logger.info(
            "finalize_deployment: Δ=%.3f CI=[%.3f, %.3f] n=%d → mode=%s",
            d, lo, hi, n, mode,
        )

    # Persist decision into the manifest
    manifest_path = checkpoint_path + ".manifest.json"
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            manifest = {}
        manifest["deployment_mode"] = mode
        manifest["deployment_confirmation_questions"] = n
        manifest["deployment_confirmation_views"] = int(baseline.shape[1]) if baseline.ndim == 2 else 1
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Updated manifest %s → deployment_mode=%s", manifest_path, mode)
    else:
        logger.warning("Manifest not found at %s; deployment_mode not persisted.", manifest_path)

    return mode


def finalize_deployment_by_shot(
    outcomes_by_shot: Dict[int | str, Tuple[Sequence[int | float], Sequence[int | float]]],
    checkpoint_path: str,
    n_boot: int = 4000,
    min_n: int = 200,
    fixed_outcomes_by_shot: Optional[Dict[int | str, Sequence[int | float]]] = None,
    require_adaptive_beats_fixed: bool = False,
) -> Dict[str, str]:
    """Make independent paired deployment decisions for each shot condition.

    A dormant few-shot policy must not suppress an independently validated
    zero-shot policy through a pooled confidence interval.  Each condition is
    still evaluated only on held-out paired questions and uses the same strict
    positive-lower-bound criterion as ``finalize_deployment``.
    """
    decisions: Dict[str, str] = {}
    evidence: Dict[str, Dict[str, float | int | str]] = {}
    for shot, (baseline, ars) in outcomes_by_shot.items():
        key = str(shot)
        base = np.asarray(baseline, dtype=float)
        candidate = np.asarray(ars, dtype=float)
        if base.shape != candidate.shape:
            raise ValueError(f"shot {shot}: baseline and ARS shapes differ: {base.shape} vs {candidate.shape}")
        n = int(base.shape[0]) if base.ndim else 0
        if n < min_n:
            mode, delta, lo, hi = "baseline", 0.0, -1.0, 1.0
        else:
            delta, lo, hi = paired_delta_ci(base, candidate, n_boot=n_boot)
            mode = "ars" if lo > 0.0 else "baseline"
        row: Dict[str, float | int | str] = {"mode": mode, "n": n, "delta": delta, "ci_low": lo, "ci_high": hi}
        if require_adaptive_beats_fixed and fixed_outcomes_by_shot is not None and shot in fixed_outcomes_by_shot:
            promotion = adaptive_vs_fixed_promotion(
                fixed_outcomes_by_shot[shot], candidate, require_adaptive_beats_fixed=True, n_boot=n_boot,
            )
            row.update({f"adaptive_vs_fixed_{name}": value for name, value in promotion.items()})
            if mode == "ars" and promotion["mode"] != "ars":
                mode = "fixed_steernet"
                row["mode"] = mode
        decisions[key] = mode
        evidence[key] = row

    manifest_path = checkpoint_path + ".manifest.json"
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            manifest = {}
        manifest["deployment_modes_by_shot"] = decisions
        manifest["deployment_evidence_by_shot"] = evidence
        # Legacy consumers still read one field.  It is ARS only when every
        # explicitly evaluated shot was validated; callers should use the map.
        manifest["deployment_mode"] = "ars" if decisions and all(v == "ars" for v in decisions.values()) else "baseline"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    return decisions


def adaptive_vs_fixed_promotion(
    fixed_flags: Sequence[int | float],
    adaptive_flags: Sequence[int | float],
    *,
    require_adaptive_beats_fixed: bool = False,
    n_boot: int = 4000,
) -> Dict[str, float | str]:
    """Return the evidence-backed winner between adaptive and fixed steering.

    A positive point estimate is insufficient for an adaptive deployment
    claim: when requested, the paired lower confidence bound against the
    already-certified fixed controller must be strictly positive.
    """
    delta, lower, upper = paired_delta_ci(fixed_flags, adaptive_flags, n_boot=n_boot)
    mode = "ars" if (not require_adaptive_beats_fixed or lower > 0.0) else "fixed_steernet"
    return {"mode": mode, "delta": delta, "ci_low": lower, "ci_high": upper}


def alpha_jitter_floor(
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    rows: Optional[Sequence[Dict[str, Any]]] = None,
    device: Optional[Any] = None,
    flags_fn: Optional[Callable[[float], Sequence[int | float]]] = None,
    alphas: Sequence[float] = (1.0, 0.999, 0.998),
    dataset_name: str = "gsm8k",
    n_shot: int = 0,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Measure the greedy decoding flip rate (noise floor) induced by tiny alpha perturbations.

    Router+SteerNet (alpha=1.0) and ARS raw (alpha≈0.999) differ by only ~0.1% in scale.
    Any difference in accuracy between them that is smaller than the jitter flip rate
    is indistinguishable from greedy decode / fp16 chaos rather than genuine reasoning.

    Args:
        model: ARS model wrapper (optional if flags_fn is provided).
        tokenizer: Tokenizer for text generation.
        rows: Dataset rows to evaluate.
        device: Torch device.
        flags_fn: Callable(alpha) -> Sequence[0/1] for testing or custom evaluation.
        alphas: Sequence of alpha values to evaluate (default: (1.0, 0.999, 0.998)).
        dataset_name: Dataset name for prompt building and evaluation.
        n_shot: Number of exemplar shots.
        max_new_tokens: Maximum tokens to generate per sample.

    Returns:
        Dict containing:
            - "per_alpha_acc": {alpha: float_accuracy}
            - "flip_rates": {(ref_alpha, target_alpha): float_flip_rate}
            - "max_flip_rate": float
            - "noise_floor_pp": float (max flip rate in percentage points)
    """
    per_alpha_flags: Dict[float, List[int]] = {}

    if flags_fn is not None:
        for a in alphas:
            per_alpha_flags[a] = [int(x) for x in flags_fn(a)]
    elif model is not None and tokenizer is not None and rows is not None:
        from src.evaluation.verifier import generate_response
        from src.evaluation.multi_dataset import build_benchmark_prompt, evaluate_sample_answer

        ref_override = getattr(model, "_routing_override", None)
        try:
            if hasattr(model, "set_routing_override"):
                model.set_routing_override(None)
            for a in alphas:
                if hasattr(model, "set_gate_freeze"):
                    model.set_gate_freeze(True, float(a))
                flags = []
                for sample in rows:
                    prompt = build_benchmark_prompt(dataset_name, sample["question"], n_shot=n_shot)
                    resp = generate_response(model, tokenizer, prompt, device, max_new_tokens=max_new_tokens)
                    ev = evaluate_sample_answer(dataset_name, resp, sample["gold"])
                    flags.append(int(ev["correct"]))
                per_alpha_flags[a] = flags
        finally:
            if hasattr(model, "set_routing_override"):
                model.set_routing_override(ref_override)
            if hasattr(model, "set_gate_freeze"):
                model.set_gate_freeze(False)
    else:
        raise ValueError("alpha_jitter_floor requires either flags_fn or (model, tokenizer, rows).")

    ref_alpha = alphas[0]
    ref_flags = np.asarray(per_alpha_flags.get(ref_alpha, []), dtype=int)
    flip_rates: Dict[Tuple[float, float], float] = {}
    per_alpha_acc: Dict[float, float] = {}

    for a, flags in per_alpha_flags.items():
        arr = np.asarray(flags, dtype=float)
        per_alpha_acc[a] = float(arr.mean()) if len(arr) > 0 else 0.0
        if a != ref_alpha and len(ref_flags) == len(arr) and len(arr) > 0:
            flips = float((ref_flags != arr).mean())
            flip_rates[(ref_alpha, a)] = round(flips, 4)

    max_flip = max(flip_rates.values()) if flip_rates else 0.0
    return {
        "per_alpha_acc": per_alpha_acc,
        "flip_rates": flip_rates,
        "max_flip_rate": max_flip,
        "noise_floor_pp": round(max_flip * 100.0, 2),
    }
