import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
import torch

from src.evaluation.multi_dataset import (
    EvalBenchmarkConfig,
    build_benchmark_prompt,
    evaluate_sample_answer,
    load_benchmark_dataset,
)
from src.evaluation.verifier import generate_response

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Single Model Condition Evaluator
# ==============================================================================

def evaluate_model_condition(
    model,
    tokenizer,
    device: torch.device,
    dataset_name: str,
    samples: List[Dict[str, Any]],
    n_shot: int = 0,
    condition: str = "ars",
    max_new_tokens: int = 512,
    display_examples: int = 2,
    progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Evaluate a condition under one reproducible intervention contract.

    ``fixed_steernet`` is the pure SteerNet control: a legal fixed-K route,
    never the invalid all-eight stress test.  ``steernet`` keeps the learned
    router but pins alpha=1, so it is explicitly a Router+SteerNet control.
    ARS vs Router+SteerNet then measures the Gate contribution under
    identical routing; ``ars`` enables both learned router and learned gate.
    """
    cond = condition.lower()
    if cond not in {"baseline", "fixed_steernet", "steernet", "ars", "ars_raw"}:
        raise ValueError(
            f"Unknown condition '{condition}'. Must be baseline, fixed_steernet, steernet, ars, or ars_raw."
        )

    # Benchmark numbers must not vary because the router samples K.  Preserve
    # an explicitly requested mode so callers can still run a stochastic
    # robustness study intentionally.
    previous_mode = getattr(model, "_router_inference_mode", None)
    if hasattr(model, "set_router_inference_mode"):
        model.set_router_inference_mode("argmax")

    # Set condition routing and gate configuration
    if cond == "baseline":
        if hasattr(model, "set_routing_override"):
            model.set_routing_override("none")
        if hasattr(model, "set_gate_freeze"):
            model.set_gate_freeze(False)
    elif cond == "fixed_steernet":
        if hasattr(model, "set_routing_override"):
            fixed_layers = list(getattr(getattr(model, "cfg", None), "fixed_steer_eval_layers", None) or [])
            if not fixed_layers and hasattr(model, "_eligible_routing_layers") and model._eligible_routing_layers:
                fixed_layers = list(model._eligible_routing_layers)
            if fixed_layers:
                model.set_routing_override("only", fixed_layers)
            else:
                model.set_routing_override("balanced")
        if hasattr(model, "set_gate_freeze"):
            cfg = getattr(model, "cfg", None)
            scales = getattr(model, "_certified_gate_scales", None)
            if scales:
                model.set_gate_freeze(True, scales)
            else:
                model.set_gate_freeze(True, float(getattr(cfg, "fixed_steer_eval_gate_scale", 1.0)))
    elif cond == "steernet":
        if hasattr(model, "set_routing_override"):
            model.set_routing_override(None)
        if hasattr(model, "set_gate_freeze"):
            model.set_gate_freeze(True, 1.0)
    elif cond == "ars":
        shot_mode = None
        if hasattr(model, "set_routing_override"):
            per_shot = getattr(model, "deployment_modes_by_shot", {})
            shot_mode = per_shot.get(str(n_shot), per_shot.get(n_shot)) if isinstance(per_shot, dict) else None
            disabled = (shot_mode == "baseline") if shot_mode is not None else getattr(model, "deployment_disabled", False)
            if shot_mode == "fixed_steernet":
                fixed_layers = list(getattr(getattr(model, "cfg", None), "fixed_steer_eval_layers", None) or [])
                if not fixed_layers and hasattr(model, "_eligible_routing_layers") and model._eligible_routing_layers:
                    fixed_layers = list(model._eligible_routing_layers)
                model.set_routing_override("only", fixed_layers if fixed_layers else None)
            else:
                model.set_routing_override("none" if disabled else None)
        if hasattr(model, "set_gate_freeze"):
            if shot_mode == "fixed_steernet":
                cfg = getattr(model, "cfg", None)
                scales = getattr(model, "_certified_gate_scales", None)
                if scales:
                    model.set_gate_freeze(True, scales)
                else:
                    model.set_gate_freeze(True, float(getattr(cfg, "fixed_steer_eval_gate_scale", 1.0)))
            else:
                model.set_gate_freeze(False)
    elif cond == "ars_raw":
        # Unconditionally evaluate the learned ARS model without deployment safety fallback
        if hasattr(model, "set_routing_override"):
            model.set_routing_override(None)
        if hasattr(model, "set_gate_freeze"):
            model.set_gate_freeze(False)

    correct_count = 0
    lenient_correct_count = 0
    total_time = 0.0
    total_tokens = 0
    prompt_token_counts = []
    detailed_results = []
    n_samples = len(samples)

    try:
        for idx, sample in enumerate(samples):
            q = sample["question"]
            gold = sample["gold"]

            prompt = build_benchmark_prompt(dataset_name, q, n_shot=n_shot)
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False)) if tokenizer else 0
            prompt_token_counts.append(prompt_tokens)
            t0 = time.time()
            collect_trace = idx < max(0, int(display_examples))
            collect_route = cond in {"steernet", "ars", "ars_raw"}
            resp = generate_response(
                model, tokenizer, prompt, device, max_new_tokens=max_new_tokens,
                collect_trace=collect_trace,
                collect_routing_trace=collect_route,
            )
            elapsed = time.time() - t0
            total_time += elapsed

            eval_res = evaluate_sample_answer(dataset_name, resp, gold)
            is_corr = eval_res["correct"]
            if is_corr:
                correct_count += 1
            is_lenient_corr = eval_res.get("lenient_correct", is_corr)
            if is_lenient_corr:
                lenient_correct_count += 1

            n_tok = len(tokenizer.encode(resp, add_special_tokens=False)) if tokenizer else 0
            total_tokens += n_tok

            # Extract chosen layer route for telemetry
            r_trace = (
                getattr(model, "last_generation_trace", {}).get("routing", {})
                if (collect_trace or collect_route) else {}
            )
            route_tuple = tuple(sorted(
                int(layer) for layer, trace in r_trace.items()
                if (trace.get("selected", [False])[0] if isinstance(trace.get("selected"), list) else trace.get("selected"))
            ))

            item = {
                "idx": idx,
                "dataset": dataset_name,
                "condition": cond,
                "n_shot": n_shot,
                "question": q,
                "gold": gold,
                "response": resp,
                "extracted": eval_res["extracted"],
                "lenient_extracted": eval_res.get("lenient_extracted", eval_res["extracted"]),
                "correct": is_corr,
                "lenient_correct": is_lenient_corr,
                "has_answer": eval_res["has_answer"],
                "steps": eval_res["steps"],
                "chars": len(resp),
                "time_sec": elapsed,
                "tokens": n_tok,
                "prompt_tokens": prompt_tokens,
                "route": route_tuple,
                "routing_trace": r_trace if collect_trace else {},
                "gate_trace": (
                    getattr(model, "last_generation_trace", {}).get("gate", {}) if collect_trace else {}
                ),
                "generated_token_ids": (
                    getattr(model, "last_generation_trace", {}).get("generated_token_ids", []) if collect_trace else []
                ),
            }
            detailed_results.append(item)

            if idx < display_examples:
                icon = "✅" if is_corr else "❌"
                selected_layers = sorted(
                    f"L{layer}" for layer, trace in item["routing_trace"].items()
                    if trace.get("selected", [False])[0]
                )
                desired_k = next(
                    (trace.get("desired_k", [0])[0] for trace in item["routing_trace"].values()),
                    0,
                )
                print(
                    f"\n[QUALITATIVE {idx + 1}/{display_examples} | {dataset_name.upper()} | "
                    f"{cond.upper()} | {n_shot}-shot] {icon}\n"
                    f"Question: {q}\nGold: {gold}\nPrediction: {eval_res['extracted']}\n"
                    f"Route: K={desired_k}, layers={selected_layers or ['none']}\n"
                    f"Generated CoT:\n{resp}\n"
                )

            if progress_callback:
                progress_callback(idx + 1, n_samples, item)

            # Periodic memory reclamation to prevent silent Linux OOM-killer crashes during long multi-hour runs
            if (idx + 1) % 25 == 0:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        # Reset model overrides to clean default state
        if hasattr(model, "set_routing_override"):
            model.set_routing_override(None)
        if hasattr(model, "set_gate_freeze"):
            model.set_gate_freeze(False)
        if previous_mode is not None and hasattr(model, "set_router_inference_mode"):
            model.set_router_inference_mode(previous_mode)

    accuracy = 100.0 * correct_count / max(1, n_samples)
    accuracy_lenient = 100.0 * lenient_correct_count / max(1, n_samples)
    ans_present = 100.0 * sum(r["has_answer"] for r in detailed_results) / max(1, n_samples)
    step_cov = 100.0 * sum(r["steps"] > 0 for r in detailed_results) / max(1, n_samples)
    tok_per_sec = total_tokens / max(1e-5, total_time)

    summary = {
        "dataset": dataset_name,
        "condition": cond,
        "n_shot": n_shot,
        "n_samples": n_samples,
        "correct": correct_count,
        "accuracy": round(accuracy, 2),
        "lenient_correct": lenient_correct_count,
        "accuracy_lenient": round(accuracy_lenient, 2),
        "answer_present": round(ans_present, 2),
        "step_coverage": round(step_cov, 2),
        "avg_time_sec": round(total_time / max(1, n_samples), 3),
        "tokens_per_sec": round(tok_per_sec, 2),
        "total_time_sec": round(total_time, 2),
        "mean_prompt_tokens": round(sum(prompt_token_counts) / max(1, n_samples), 1),
        "max_prompt_tokens": max(prompt_token_counts, default=0),
    }

    return {"summary": summary, "detailed_results": detailed_results}


def route_entropy_by_shot(all_details: List[Dict[str, Any]], condition: str = "steernet") -> Dict[int, float]:
    """Calculate empirical Shannon entropy (bits) of chosen routes per shot for a condition (R2)."""
    import math
    from collections import Counter

    shot_routes: Dict[int, List[Any]] = {}
    for item in all_details:
        if item.get("condition") != condition:
            continue
        shot = item.get("n_shot", 0)
        route = item.get("route")
        if not route:
            # Fallback to extracting from routing_trace if route not directly populated
            r_trace = item.get("routing_trace") or {}
            if r_trace:
                route = tuple(sorted(
                    int(l) for l, tr in r_trace.items()
                    if (tr.get("selected", [False])[0] if isinstance(tr.get("selected"), list) else tr.get("selected"))
                ))
        if route is not None:
            shot_routes.setdefault(shot, []).append(route)

    entropies = {}
    for shot, routes in shot_routes.items():
        if not routes:
            continue
        counts = Counter(routes)
        n = len(routes)
        h = -sum((cnt / n) * math.log2(cnt / n) for cnt in counts.values())
        entropies[shot] = round(max(0.0, float(h)), 4)
        if h <= 0.0:
            top_route = routes[0] if routes else ()
            if top_route == () or condition == "baseline":
                # Empty route indicates baseline passthrough (e.g. ARS deployed mode fallback)
                pass
            else:
                print(f"  ⚠️ ROUTE-COLLAPSED: Router selected identical route {top_route} across 100% of {n} items ({shot}-shot, {condition}).")
    return entropies


def chain_stats(all_details: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, float]]:
    """Compute reasoning chain diagnostics across conditions and shots (R5 / R7).

    Tests whether SteerNets compress reasoning (fewer steps/tokens) or cause premature stopping.
    Returns per-(condition, shot) statistics:
      - mean_steps: average reasoning steps
      - mean_chars: average response character length
      - mean_tokens: average generated tokens
      - accuracy_strict: strict extraction accuracy (%)
      - accuracy_lenient: lenient extraction accuracy (%)
      - count: number of samples
    """
    import numpy as np
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for item in all_details:
        cond = item.get("condition", "unknown")
        shot = item.get("n_shot", 0)
        groups.setdefault((cond, shot), []).append(item)

    stats: Dict[Tuple[str, int], Dict[str, float]] = {}
    for (cond, shot), items in groups.items():
        n = len(items)
        if n == 0:
            continue
        m_steps = float(np.mean([it.get("steps", 0) for it in items]))
        m_chars = float(np.mean([it.get("chars", len(it.get("response", ""))) for it in items]))
        m_toks = float(np.mean([it.get("tokens", 0) for it in items]))
        acc_s = 100.0 * float(np.mean([int(it.get("correct", False)) for it in items]))
        acc_l = 100.0 * float(np.mean([int(it.get("lenient_correct", it.get("correct", False))) for it in items]))

        stats[(cond, shot)] = {
            "mean_steps": round(m_steps, 2),
            "mean_chars": round(m_chars, 1),
            "mean_tokens": round(m_toks, 1),
            "accuracy_strict": round(acc_s, 2),
            "accuracy_lenient": round(acc_l, 2),
            "count": n,
        }
    return stats


def format_chain_stats_table(stats: Dict[Tuple[str, int], Dict[str, float]]) -> str:
    """Format chain_stats dictionary into a human-readable table (R5)."""
    hdr = f"{'Condition':<18} | {'Shots':<8} | {'Strict Acc':<12} | {'Lenient Acc':<12} | {'Mean Steps':<11} | {'Mean Chars':<11} | {'Mean Tokens':<11}"
    lines = ["=" * len(hdr), hdr, "-" * len(hdr)]
    for (cond, shot) in sorted(stats.keys(), key=lambda x: (x[1], x[0])):
        st = stats[(cond, shot)]
        row = (
            f"{cond:<18} | {f'{shot}-shot':<8} | {st['accuracy_strict']:>9.1f}% | {st['accuracy_lenient']:>9.1f}% | "
            f"{st['mean_steps']:>10.2f} | {st['mean_chars']:>10.1f} | {st['mean_tokens']:>10.1f}"
        )
        lines.append(row)
    lines.append("=" * len(hdr))
    return "\n".join(lines)


def format_dual_extractor_table(
    per_item_strict: Dict[Tuple[str, str, int], List[int]],
    per_item_lenient: Dict[Tuple[str, str, int], List[int]],
) -> str:
    """Format side-by-side comparison of strict vs lenient accuracy per condition and shot (R4)."""
    import numpy as np
    keys = sorted(
        list(set(per_item_strict.keys()).intersection(set(per_item_lenient.keys()))),
        key=lambda x: (x[0], x[2], x[1]),
    )
    hdr = f"{'Dataset':<10} | {'Condition':<16} | {'Shots':<6} | {'Strict Acc':<12} | {'Lenient Acc':<12} | {'Extractor Δ':<12}"
    lines = ["=" * len(hdr), hdr, "-" * len(hdr)]
    for (ds, cond, shot) in keys:
        s_arr = np.asarray(per_item_strict[(ds, cond, shot)], dtype=float)
        l_arr = np.asarray(per_item_lenient[(ds, cond, shot)], dtype=float)
        s_acc = float(s_arr.mean() * 100.0) if len(s_arr) > 0 else 0.0
        l_acc = float(l_arr.mean() * 100.0) if len(l_arr) > 0 else 0.0
        diff = l_acc - s_acc
        row = f"{ds.upper():<10} | {cond:<16} | {f'{shot}-shot':<6} | {s_acc:>10.2f}% | {l_acc:>10.2f}% | {diff:>+10.2f}pp"
        lines.append(row)
    lines.append("=" * len(hdr))
    return "\n".join(lines)


# ==============================================================================
# 2. Fair Multi-Shot Protocol Runner across All Benchmarks
# ==============================================================================

def run_fair_multishot_benchmark(
    model,
    tokenizer,
    device: torch.device,
    config: Optional[EvalBenchmarkConfig] = None,
    conditions: Tuple[str, ...] = ("baseline", "steernet", "ars"),
    datasets: Optional[List[str]] = None,
    shot_counts: Optional[List[int]] = None,
    experiment_run: Optional[Any] = None,
    verbose: bool = True,
    qualitative_examples: int = 5,
) -> Dict[str, Any]:
    """Executes fair multi-shot evaluation protocol spanning datasets and conditions.

    Returns structured output compatible with Experiment Registry and plotting engine:
    {
        "benchmark_results": {
            "gsm8k": {
                "baseline": {"0_shot": 37.0, "2_shot": 40.0, ...},
                "steernet": {"0_shot": 41.0, ...},
                "ars": {"0_shot": 47.0, "2_shot": 53.33, ...}
            }, ...
        },
        "metrics_summary": [ ... list of individual summary dicts ... ],
        "all_details": [ ... all per-sample items ... ]
    }
    """
    cfg = config or EvalBenchmarkConfig()
    if hasattr(model, "set_prompt_shielding"):
        model.set_prompt_shielding(cfg.prompt_shielding)
    # Controls do not contribute learned-router telemetry, so this one reset
    # gives the final dashboard the aggregate of every ARS sample and shot.
    if hasattr(model, "reset_routing_diagnostics"):
        model.reset_routing_diagnostics()
    eval_datasets = datasets or cfg.datasets
    eval_shots = shot_counts or cfg.shot_counts

    benchmark_tree: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    metrics_summary: List[Dict[str, Any]] = []
    all_details: List[Dict[str, Any]] = []
    per_item_correct: Dict[Tuple[str, str, int], List[int]] = {}
    per_item_lenient: Dict[Tuple[str, str, int], List[int]] = {}

    for ds_name in eval_datasets:
        norm_ds = ds_name.lower().replace("-", "_")
        benchmark_tree[norm_ds] = {c: {} for c in conditions}
        n_samples = cfg.samples_per_dataset.get(norm_ds, 100)

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"  Benchmark Suite: {norm_ds.upper()} (Loading {n_samples} test samples)")
            print(f"{'=' * 80}")

        # Load samples once per dataset to guarantee absolute fairness across conditions
        samples = load_benchmark_dataset(norm_ds, n_samples=n_samples, seed=cfg.seed)

        for shot in eval_shots:
            shot_key = f"{shot}_shot"
            for cond in conditions:
                if verbose:
                    print(f"\n>>> Running {norm_ds.upper()} | Condition: {cond.upper()} | {shot}-Shot")

                res = evaluate_model_condition(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    dataset_name=norm_ds,
                    samples=samples,
                    n_shot=shot,
                    condition=cond,
                    max_new_tokens=cfg.max_new_tokens,
                    display_examples=qualitative_examples,
                )

                summ = res["summary"]
                acc = summ["accuracy"]
                benchmark_tree[norm_ds][cond][shot_key] = acc
                metrics_summary.append(summ)
                all_details.extend(res["detailed_results"])
                per_item = [int(item["correct"]) for item in res["detailed_results"]]
                per_item_len = [int(item["lenient_correct"]) for item in res["detailed_results"]]
                per_item_correct[(norm_ds, cond, shot)] = per_item
                per_item_lenient[(norm_ds, cond, shot)] = per_item_len

                if verbose:
                    print(
                        f"    Result: {acc:.2f}% ({summ['correct']}/{summ['n_samples']}) "
                        f"| {summ['tokens_per_sec']} tok/s | Ans: {summ['answer_present']}% "
                        f"| Prompt≤{summ['max_prompt_tokens']} tok"
                    )

                # Sync directly with experiment run if attached
                if experiment_run is not None:
                    experiment_run.log_benchmark_result(
                        stage_name=cond,
                        accuracy=acc,
                        n_shot=shot,
                        dataset=norm_ds,
                        tokens_per_sec=summ["tokens_per_sec"],
                        extra_metrics={
                            "answer_present": summ["answer_present"],
                            "step_coverage": summ["step_coverage"],
                            "avg_time_sec": summ["avg_time_sec"],
                        },
                    )

    if verbose:
        print("\n" + format_benchmark_summary_table(benchmark_tree, per_item_correct=per_item_correct))

        # R4: Dual Extractor Comparison (Strict vs Lenient)
        print("\n" + "=" * 90)
        print("  DUAL EXTRACTOR COMPARISON: STRICT vs LENIENT ACCURACY (R4)")
        print("=" * 90)
        print(format_dual_extractor_table(per_item_correct, per_item_lenient))

        # R5 / R7: Reasoning Chain Dynamics Table
        print("\n" + "=" * 90)
        print("  REASONING CHAIN DYNAMICS & COMPRESSION DIAGNOSTICS (R5)")
        print("=" * 90)
        c_stats = chain_stats(all_details)
        print(format_chain_stats_table(c_stats))

        # R8: Print 3-way paired comparisons with 95% bootstrap CI
        from src.evaluation.deployment import paired_delta_ci
        print("\n" + "=" * 90)
        print("  PAIRED BENCHMARK COMPARISONS (95% Bootstrap CI)")
        print("=" * 90)
        is_deployed_baseline = getattr(model, "deployment_disabled", False)
        ars_label = "Full ARS (Deployed=Baseline)" if is_deployed_baseline else "Full ARS"
        for ds_name in eval_datasets:
            norm_ds = ds_name.lower().replace("-", "_")
            for shot in eval_shots:
                print(f"\n--- {norm_ds.upper()} ({shot}-Shot) ---")
                pairs = [
                    ("Fixed SteerNet vs Baseline", "baseline", "fixed_steernet"),
                    ("Router+SteerNet vs Fixed SteerNet", "fixed_steernet", "steernet"),
                    (f"{ars_label} vs Router+SteerNet", "steernet", "ars"),
                    (f"{ars_label} vs Baseline", "baseline", "ars"),
                ]
                if (norm_ds, "ars_raw", shot) in per_item_correct:
                    pairs.append(("ARS (Raw) vs Router+SteerNet", "steernet", "ars_raw"))
                    pairs.append(("ARS (Raw) vs Baseline", "baseline", "ars_raw"))

                for label, c_a, c_b in pairs:
                    if (norm_ds, c_a, shot) in per_item_correct and (norm_ds, c_b, shot) in per_item_correct:
                        fa = per_item_correct[(norm_ds, c_a, shot)]
                        fb = per_item_correct[(norm_ds, c_b, shot)]
                        d, lo, hi = paired_delta_ci(fa, fb)
                        sig = " *SIGNIFICANT*" if lo > 0 else ""
                        print(f"  {label:<35}: Δ = {d*100:+.2f}pp, 95% CI = [{lo*100:+.2f}, {hi*100:+.2f}]pp (n={len(fa)}){sig}")

        # R2: Calculate route entropy per shot and check for route collapse
        print("\n" + "-" * 90)
        print("  ROUTER POLICY DIVERSITY TELEMETRY")
        print("-" * 90)
        for cond_check in ("steernet", "ars", "ars_raw"):
            if cond_check in conditions:
                entropies = route_entropy_by_shot(all_details, condition=cond_check)
                if entropies:
                    print(f"  Route entropy by shot ({cond_check}): {entropies} bits")

    return {
        "benchmark_results": benchmark_tree,
        "metrics_summary": metrics_summary,
        "all_details": all_details,
        "per_item_correct": per_item_correct,
        "per_item_lenient": per_item_lenient,
        "chain_stats": chain_stats(all_details),
    }


# ==============================================================================
# 3. Formatting & Publication Output Helpers
# ==============================================================================

def format_benchmark_summary_table(
    benchmark_results: Dict[str, Any],
    per_item_correct: Optional[Dict[Tuple[str, str, int], List[int]]] = None,
) -> str:
    """Format all controls without mislabelling Router+SteerNet as SteerNet-only.

    When per_item_correct is provided, computes paired bootstrap 95% CIs
    against baseline and marks significant gains with *.
    """
    has_ars_raw = any("ars_raw" in conds for conds in benchmark_results.values())
    hdr = f"{'Dataset':<12} | {'Shots':<6} | {'Baseline':<10} | {'Fixed SteerNet':<15} | {'Router+SteerNet':<17} | {'Full ARS':<35} | {'ARS Δ':<16}"
    if has_ars_raw:
        hdr += f" | {'Full ARS (Raw)':<35}"
    table_width = len(hdr)

    lines = [
        "=" * table_width,
        hdr,
        "-" * table_width,
    ]

    for ds_name, conds in benchmark_results.items():
        base_dict = conds.get("baseline", {})
        fixed_dict = conds.get("fixed_steernet", {})
        steer_dict = conds.get("steernet", {})
        ars_dict = conds.get("ars", {})
        ars_raw_dict = conds.get("ars_raw", {})

        all_shots = sorted(
            list(set(list(base_dict.keys()) + list(fixed_dict.keys()) + list(steer_dict.keys()) + list(ars_dict.keys()) + list(ars_raw_dict.keys()))),
            key=lambda s: int(s.split("_")[0]) if "_" in s and s.split("_")[0].isdigit() else 0,
        )

        for shot_k in all_shots:
            shot_num = shot_k.split("_")[0]
            shot_int = int(shot_num) if shot_num.isdigit() else 0
            b_val = base_dict.get(shot_k)
            f_val = fixed_dict.get(shot_k)
            s_val = steer_dict.get(shot_k)
            a_val = ars_dict.get(shot_k)
            ar_val = ars_raw_dict.get(shot_k)

            b_str = f"{b_val:.2f}%" if b_val is not None else "--"
            f_str = f"{f_val:.2f}%" if f_val is not None else "--"
            s_str = f"{s_val:.2f}%" if s_val is not None else "--"
            a_str = f"{a_val:.2f}%" if a_val is not None else "--"
            ar_str = f"{ar_val:.2f}%" if ar_val is not None else "--"

            if per_item_correct is not None and (ds_name, "baseline", shot_int) in per_item_correct:
                base_flags = per_item_correct[(ds_name, "baseline", shot_int)]
                from src.evaluation.deployment import format_benchmark_cell, paired_delta_ci
                if a_val is not None and (ds_name, "ars", shot_int) in per_item_correct:
                    ars_flags = per_item_correct[(ds_name, "ars", shot_int)]
                    a_str = format_benchmark_cell(base_flags, ars_flags)
                    d, lo, hi = paired_delta_ci(base_flags, ars_flags)
                    sig = "*" if lo > 0 else ""
                    gain_str = f"{d * 100:+.1f}pp{sig} [{lo * 100:+.1f}, {hi * 100:+.1f}]"
                elif a_val is not None and b_val is not None:
                    diff = a_val - b_val
                    gain_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
                else:
                    gain_str = "--"

                if ar_val is not None and (ds_name, "ars_raw", shot_int) in per_item_correct:
                    ars_raw_flags = per_item_correct[(ds_name, "ars_raw", shot_int)]
                    ar_str = format_benchmark_cell(base_flags, ars_raw_flags)
            else:
                if a_val is not None and b_val is not None:
                    diff = a_val - b_val
                    gain_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
                else:
                    gain_str = "--"

            row_line = (
                f"{ds_name.upper():<12} | {shot_num + '-shot':<6} | {b_str:<10} | {f_str:<15} | {s_str:<17} | {a_str:<35} | {gain_str:<16}"
            )
            if has_ars_raw:
                row_line += f" | {ar_str:<35}"
            lines.append(row_line)
        lines.append("-" * table_width)

    lines.append("=" * table_width)
    return "\n".join(lines)


def export_latex_summary_table(
    benchmark_results: Dict[str, Any],
    caption: str = "Multi-dataset reasoning benchmark results across few-shot evaluations.",
    label: str = "tab:ars_multishot_benchmarks",
    per_item_correct: Optional[Dict[Tuple[str, str, int], List[int]]] = None,
) -> str:
    """Generates a publication-grade LaTeX booktabs table for research papers."""
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Shots} & \textbf{Frozen Baseline} & \textbf{Fixed SteerNet} & \textbf{Router+SteerNet} & \textbf{Full ARS} & \textbf{$\Delta$ Gain} \\",
        r"\midrule",
    ]

    for ds_name, conds in benchmark_results.items():
        base_dict = conds.get("baseline", {})
        fixed_dict = conds.get("fixed_steernet", {})
        steer_dict = conds.get("steernet", {})
        ars_dict = conds.get("ars", {})

        all_shots = sorted(
            list(set(list(base_dict.keys()) + list(fixed_dict.keys()) + list(steer_dict.keys()) + list(ars_dict.keys()))),
            key=lambda s: int(s.split("_")[0]) if "_" in s and s.split("_")[0].isdigit() else 0,
        )

        for i, shot_k in enumerate(all_shots):
            ds_col = f"\\textbf{{{ds_name.upper()}}}" if i == 0 else ""
            shot_num = shot_k.split("_")[0] + "-shot"
            shot_int = int(shot_k.split("_")[0]) if shot_k.split("_")[0].isdigit() else 0
            b_val = base_dict.get(shot_k)
            f_val = fixed_dict.get(shot_k)
            s_val = steer_dict.get(shot_k)
            a_val = ars_dict.get(shot_k)

            row_values = {
                "baseline": b_val,
                "fixed_steernet": f_val,
                "steernet": s_val,
                "ars": a_val,
            }
            observed = [value for value in row_values.values() if value is not None]
            best_value = max(observed) if observed else None

            def render(cond_name, value):
                if value is None:
                    return "--"
                text = f"{value:.2f}\\%"
                # When per_item_correct is provided, only bold if improvement over baseline is statistically significant (lo > 0)
                if per_item_correct is not None:
                    if (
                        cond_name in {"ars", "steernet", "fixed_steernet"}
                        and (ds_name, "baseline", shot_int) in per_item_correct
                        and (ds_name, cond_name, shot_int) in per_item_correct
                    ):
                        from src.evaluation.deployment import paired_delta_ci
                        base_flags = per_item_correct[(ds_name, "baseline", shot_int)]
                        cond_flags = per_item_correct[(ds_name, cond_name, shot_int)]
                        _, lo, _ = paired_delta_ci(base_flags, cond_flags)
                        if lo > 0 and (b_val is None or value > b_val):
                            return f"\\textbf{{{text}}}"
                        return text
                    # If per_item_correct is provided for the run, do not bold unverified conditions
                    return text

                # Fallback: A publication table must never make a method look best merely by aggregate tie. Never bold non-gain ties.
                if value == best_value and (b_val is None or best_value > b_val):
                    return f"\\textbf{{{text}}}"
                return text

            b_str = render("baseline", b_val)
            f_str = render("fixed_steernet", f_val)
            s_str = render("steernet", s_val)
            a_str = render("ars", a_val)

            if per_item_correct is not None and (ds_name, "baseline", shot_int) in per_item_correct and (ds_name, "ars", shot_int) in per_item_correct:
                from src.evaluation.deployment import paired_delta_ci
                base_flags = per_item_correct[(ds_name, "baseline", shot_int)]
                ars_flags = per_item_correct[(ds_name, "ars", shot_int)]
                d, lo, hi = paired_delta_ci(base_flags, ars_flags)
                diff = a_val - b_val if (a_val is not None and b_val is not None) else d * 100
                diff_text = f"+{diff:.2f}\\%" if diff >= 0 else f"{diff:.2f}\\%"
                gain_str = f"\\textbf{{{diff_text}}}" if (lo > 0 and diff > 0) else diff_text
            elif a_val is not None and b_val is not None:
                diff = a_val - b_val
                diff_text = f"+{diff:.2f}\\%" if diff >= 0 else f"{diff:.2f}\\%"
                gain_str = f"\\textbf{{{diff_text}}}" if diff > 0 else diff_text
            else:
                gain_str = "--"

            latex_lines.append(f"{ds_col} & {shot_num} & {b_str} & {f_str} & {s_str} & {a_str} & {gain_str} \\\\")
        latex_lines.append(r"\midrule")

    # Remove the last midrule and replace with bottomrule
    if latex_lines[-1] == r"\midrule":
        latex_lines.pop()

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table*}",
    ])

    return "\n".join(latex_lines)

