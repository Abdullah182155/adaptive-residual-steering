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
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def paired_delta_ci(
    flags_a: Sequence[int | float],
    flags_b: Sequence[int | float],
    n_boot: int = 4000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap 95% CI for mean(flags_b) - mean(flags_a), paired by item.

    Args:
        flags_a: Per-item correctness for arm A (0/1 ints or floats).
        flags_b: Per-item correctness for arm B (0/1 ints or floats).
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
    if len(a) != len(b):
        raise ValueError(f"paired_delta_ci: len(a)={len(a)} != len(b)={len(b)}")
    if len(a) == 0:
        return 0.0, -1.0, 1.0

    d_obs = float(b.mean() - a.mean())
    diffs = b - a  # per-item differences
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
    n = len(per_item_baseline)
    if n < min_n:
        logger.warning(
            "finalize_deployment: only %d examples (< %d). "
            "Decision is unreliable; defaulting to baseline.",
            n, min_n,
        )
        mode = "baseline"
    else:
        d, lo, hi = paired_delta_ci(per_item_baseline, per_item_ars, n_boot=n_boot)
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
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Updated manifest %s → deployment_mode=%s", manifest_path, mode)
    else:
        logger.warning("Manifest not found at %s; deployment_mode not persisted.", manifest_path)

    return mode
