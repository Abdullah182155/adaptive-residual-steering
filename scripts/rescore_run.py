"""Standalone offline rescoring script for ARS experimental runs (E0 Diagnostic).

Evaluates whether steering performance drops are an artifact of extractor strictness
or reflect real degradation in mathematical reasoning. Rescores all items using both
the canonical strict extractor and the lenient concluding-statement extractor,
computes paired bootstrap 95% CIs, and analyzes chain-of-thought length/step compression.

Usage:
    python scripts/rescore_run.py --file path/to/all_details.json
    python scripts/rescore_run.py --file experiments/runs/.../experiment_record.json
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.evaluation.answers import extract_numeric_answer
from src.evaluation.multi_dataset import extract_choice_answer
from src.evaluation.deployment import paired_delta_ci
from src.evaluation.multi_benchmark_runner import (
    chain_stats,
    format_chain_stats_table,
    format_dual_extractor_table,
)

logger = logging.getLogger(__name__)


def load_details(filepath: str) -> List[Dict[str, Any]]:
    """Load detailed results list from JSON file or experiment record."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if "all_details" in data:
            return data["all_details"]
        if "detailed_results" in data:
            return data["detailed_results"]
        if "artifacts" in data and isinstance(data["artifacts"], dict):
            for k, v in data["artifacts"].items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) and "response" in v[0]:
                    return v
    raise ValueError(f"Could not find list of detailed results in {filepath}")


def rescore_items(items: List[Dict[str, Any]], tolerance: float = 0.01) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, int], List[int]], Dict[Tuple[str, str, int], List[int]]]:
    """Rescore each response under both strict and lenient extractors."""
    rescored = []
    per_item_strict: Dict[Tuple[str, str, int], List[int]] = {}
    per_item_lenient: Dict[Tuple[str, str, int], List[int]] = {}

    for item in items:
        resp = item.get("response", "")
        gold = item.get("gold")
        ds = item.get("dataset", "gsm8k").lower().replace("-", "_")
        cond = item.get("condition", "unknown")
        shot = int(item.get("n_shot", 0))

        is_mc = ds in {"arc", "ai2_arc"}
        if is_mc:
            s_ext = extract_choice_answer(resp)
            l_ext = s_ext
            gold_str = str(gold).strip().upper()
            digit_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
            norm_gold = digit_to_letter.get(gold_str, gold_str)
            s_corr = (s_ext is not None and s_ext == norm_gold)
            l_corr = s_corr
        else:
            s_ext = extract_numeric_answer(resp, lenient=False)
            l_ext = extract_numeric_answer(resp, lenient=True)
            try:
                gold_num = float(gold) if gold is not None else None
            except (ValueError, TypeError):
                gold_num = None
            s_corr = (s_ext is not None and gold_num is not None and abs(s_ext - gold_num) <= tolerance)
            l_corr = (l_ext is not None and gold_num is not None and abs(l_ext - gold_num) <= tolerance)

        new_item = dict(item)
        new_item["extracted"] = s_ext
        new_item["lenient_extracted"] = l_ext
        new_item["correct"] = bool(s_corr)
        new_item["lenient_correct"] = bool(l_corr)
        new_item["chars"] = len(resp)
        rescored.append(new_item)

        per_item_strict.setdefault((ds, cond, shot), []).append(int(s_corr))
        per_item_lenient.setdefault((ds, cond, shot), []).append(int(l_corr))

    return rescored, per_item_strict, per_item_lenient


def run_rescore_analysis(filepath: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Execute complete E0 rescoring analysis and print diagnostic report."""
    print("=" * 90)
    print(f"  ARS OFFLINE RESCORING & EXTRACTION DIAGNOSTIC (E0)")
    print(f"  Target File: {filepath}")
    print("=" * 90)

    items = load_details(filepath)
    print(f"Loaded {len(items)} items from {filepath}.")

    rescored, p_strict, p_lenient = rescore_items(items)

    # 1. Dual Extractor Comparison Table
    print("\n" + "-" * 90)
    print("  1. DUAL EXTRACTOR COMPARISON (STRICT vs LENIENT)")
    print("-" * 90)
    print(format_dual_extractor_table(p_strict, p_lenient))

    # 2. Reasoning Chain Dynamics
    print("\n" + "-" * 90)
    print("  2. REASONING CHAIN DYNAMICS & COMPRESSION DIAGNOSTICS")
    print("-" * 90)
    stats = chain_stats(rescored)
    print(format_chain_stats_table(stats))

    # 3. Paired Bootstrap Comparisons under Both Extractors
    print("\n" + "-" * 90)
    print("  3. PAIRED DELTA VS BASELINE (95% Bootstrap CI)")
    print("-" * 90)
    datasets = sorted(list(set(k[0] for k in p_strict.keys())))
    shots = sorted(list(set(k[2] for k in p_strict.keys())))
    conditions = sorted(list(set(k[1] for k in p_strict.keys())))

    for ds in datasets:
        for shot in shots:
            if (ds, "baseline", shot) not in p_strict:
                continue
            b_s = p_strict[(ds, "baseline", shot)]
            b_l = p_lenient[(ds, "baseline", shot)]
            print(f"\n--- {ds.upper()} ({shot}-Shot) ---")
            for cond in conditions:
                if cond == "baseline" or (ds, cond, shot) not in p_strict:
                    continue
                c_s = p_strict[(ds, cond, shot)]
                c_l = p_lenient[(ds, cond, shot)]
                d_s, lo_s, hi_s = paired_delta_ci(b_s, c_s)
                d_l, lo_l, hi_l = paired_delta_ci(b_l, c_l)
                print(f"  {cond:<18} Strict : Δ = {d_s*100:+.2f}pp, 95% CI = [{lo_s*100:+.2f}, {hi_s*100:+.2f}]pp")
                print(f"  {cond:<18} Lenient: Δ = {d_l*100:+.2f}pp, 95% CI = [{lo_l*100:+.2f}, {hi_l*100:+.2f}]pp")

    # 4. Diagnostic Verdict
    print("\n" + "=" * 90)
    print("  4. DIAGNOSTIC VERDICT (E0)")
    print("=" * 90)
    # Check if lenient scoring changes ranking or flips delta signs
    has_sign_flip = False
    for ds in datasets:
        for shot in shots:
            if (ds, "baseline", shot) not in p_strict:
                continue
            b_s = p_strict[(ds, "baseline", shot)]
            b_l = p_lenient[(ds, "baseline", shot)]
            for cond in conditions:
                if cond == "baseline" or (ds, cond, shot) not in p_strict:
                    continue
                d_s, _, _ = paired_delta_ci(b_s, p_strict[(ds, cond, shot)])
                d_l, _, _ = paired_delta_ci(b_l, p_lenient[(ds, cond, shot)])
                if (d_s < 0 and d_l > 0) or (d_s > 0 and d_l < 0):
                    has_sign_flip = True
                    print(f"  ⚠️ Sign flip detected for {cond} ({shot}-shot): strict Δ={d_s*100:+.2f}pp -> lenient Δ={d_l*100:+.2f}pp")

    if has_sign_flip:
        print("  VERDICT: Extractor strictness explains observed performance discrepancies!")
        print("  Previous harm was at least partly an extraction artifact.")
    else:
        print("  VERDICT: Ranking and Δ signs are robust across strict and lenient extractors.")
        print("  Performance deficit is NOT an extraction artifact; steering does not improve underlying reasoning.")
    print("=" * 90)

    result = {
        "items_count": len(rescored),
        "chain_stats": {f"{k[0]}_{k[1]}": v for k, v in stats.items()},
        "has_sign_flip": has_sign_flip,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rescored, f, indent=2)
        print(f"Saved rescored details to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Offline rescoring & dual-extractor diagnostic (E0).")
    parser.add_argument("--file", type=str, default="", help="Path to all_details.json or experiment record.")
    parser.add_argument("--out", type=str, default="", help="Optional output path for rescored items.")
    args = parser.parse_args()

    target_file = args.file
    if not target_file or not os.path.exists(target_file):
        # Look for default experiment records or search
        candidates = [
            "experiments/all_details.json",
            "experiments/runs/exp_20261015_211528_ars_phi2_kaggle/experiment_record.json",
        ]
        found = [c for c in candidates if os.path.exists(c)]
        if found:
            target_file = found[0]
        else:
            print(f"No input file specified and default candidates not found. Specify --file <path>.")
            sys.exit(1)

    try:
        run_rescore_analysis(target_file, args.out or None)
    except Exception as e:
        logger.exception("Error during rescore analysis: %s", e)
        print(f"Error during rescore analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
