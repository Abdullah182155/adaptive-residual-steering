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
    """Evaluates a model under one of three fair experimental conditions:
    1. 'baseline': Unsteered backbone (routing override 'none')
    2. 'steernet': SteerNet only on all layers with gate forced to 1.0 (override 'all', gate freeze True)
    3. 'ars': Full ARS framework with learned dynamic routing and gating (override None, gate freeze False)
    """
    cond = condition.lower()
    if cond not in {"baseline", "steernet", "ars"}:
        raise ValueError(f"Unknown condition '{condition}'. Must be 'baseline', 'steernet', or 'ars'.")

    # Set condition routing and gate configuration
    if cond == "baseline":
        if hasattr(model, "set_routing_override"):
            model.set_routing_override("none")
        if hasattr(model, "set_gate_freeze"):
            model.set_gate_freeze(False)
    elif cond == "steernet":
        if hasattr(model, "set_routing_override"):
            model.set_routing_override("all")
        if hasattr(model, "set_gate_freeze"):
            model.set_gate_freeze(True, 1.0)
    elif cond == "ars":
        if hasattr(model, "set_routing_override"):
            model.set_routing_override(None)
        if hasattr(model, "set_gate_freeze"):
            model.set_gate_freeze(False)

    correct_count = 0
    total_time = 0.0
    total_tokens = 0
    detailed_results = []
    n_samples = len(samples)

    try:
        for idx, sample in enumerate(samples):
            q = sample["question"]
            gold = sample["gold"]

            prompt = build_benchmark_prompt(dataset_name, q, n_shot=n_shot)
            t0 = time.time()
            resp = generate_response(model, tokenizer, prompt, device, max_new_tokens=max_new_tokens)
            elapsed = time.time() - t0
            total_time += elapsed

            eval_res = evaluate_sample_answer(dataset_name, resp, gold)
            is_corr = eval_res["correct"]
            if is_corr:
                correct_count += 1

            n_tok = len(tokenizer.encode(resp, add_special_tokens=False)) if tokenizer else 0
            total_tokens += n_tok

            item = {
                "idx": idx,
                "dataset": dataset_name,
                "condition": cond,
                "n_shot": n_shot,
                "question": q,
                "gold": gold,
                "response": resp,
                "extracted": eval_res["extracted"],
                "correct": is_corr,
                "has_answer": eval_res["has_answer"],
                "steps": eval_res["steps"],
                "time_sec": elapsed,
                "tokens": n_tok,
            }
            detailed_results.append(item)

            if idx < display_examples:
                icon = "✅" if is_corr else "❌"
                logger.info(
                    "[%s|%s|%d-shot] [%d/%d] %s Gold: %s | Got: %s (%.2fs)",
                    dataset_name.upper(), cond, n_shot, idx + 1, n_samples, icon, gold, eval_res["extracted"], elapsed,
                )

            if progress_callback:
                progress_callback(idx + 1, n_samples, item)

    finally:
        # Reset model overrides to clean default state
        if hasattr(model, "set_routing_override"):
            model.set_routing_override(None)
        if hasattr(model, "set_gate_freeze"):
            model.set_gate_freeze(False)

    accuracy = 100.0 * correct_count / max(1, n_samples)
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
        "answer_present": round(ans_present, 2),
        "step_coverage": round(step_cov, 2),
        "avg_time_sec": round(total_time / max(1, n_samples), 3),
        "tokens_per_sec": round(tok_per_sec, 2),
        "total_time_sec": round(total_time, 2),
    }

    return {"summary": summary, "detailed_results": detailed_results}


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
    eval_datasets = datasets or cfg.datasets
    eval_shots = shot_counts or cfg.shot_counts

    benchmark_tree: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    metrics_summary: List[Dict[str, Any]] = []
    all_details: List[Dict[str, Any]] = []

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
                )

                summ = res["summary"]
                acc = summ["accuracy"]
                benchmark_tree[norm_ds][cond][shot_key] = acc
                metrics_summary.append(summ)
                all_details.extend(res["detailed_results"])

                if verbose:
                    print(
                        f"    Result: {acc:.2f}% ({summ['correct']}/{summ['n_samples']}) "
                        f"| {summ['tokens_per_sec']} tok/s | Ans: {summ['answer_present']}%"
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
        print("\n" + format_benchmark_summary_table(benchmark_tree))

    return {
        "benchmark_results": benchmark_tree,
        "metrics_summary": metrics_summary,
        "all_details": all_details,
    }


# ==============================================================================
# 3. Formatting & Publication Output Helpers
# ==============================================================================

def format_benchmark_summary_table(benchmark_results: Dict[str, Any]) -> str:
    """Formats an ASCII comparison table matching Baseline vs SteerNet vs Full ARS."""
    lines = [
        "=" * 92,
        f"{'Dataset':<14} | {'Shots':<6} | {'Baseline':<12} | {'SteerNet Only':<15} | {'Full ARS':<15} | {'Gain (vs Base)':<14}",
        "-" * 92,
    ]

    for ds_name, conds in benchmark_results.items():
        base_dict = conds.get("baseline", {})
        steer_dict = conds.get("steernet", {})
        ars_dict = conds.get("ars", {})

        all_shots = sorted(
            list(set(list(base_dict.keys()) + list(steer_dict.keys()) + list(ars_dict.keys()))),
            key=lambda s: int(s.split("_")[0]) if "_" in s and s.split("_")[0].isdigit() else 0,
        )

        for shot_k in all_shots:
            shot_num = shot_k.split("_")[0]
            b_val = base_dict.get(shot_k)
            s_val = steer_dict.get(shot_k)
            a_val = ars_dict.get(shot_k)

            b_str = f"{b_val:.2f}%" if b_val is not None else "--"
            s_str = f"{s_val:.2f}%" if s_val is not None else "--"
            a_str = f"{a_val:.2f}%" if a_val is not None else "--"

            if a_val is not None and b_val is not None:
                diff = a_val - b_val
                gain_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
            else:
                gain_str = "--"

            lines.append(
                f"{ds_name.upper():<14} | {shot_num + '-shot':<6} | {b_str:<12} | {s_str:<15} | {a_str:<15} | {gain_str:<14}"
            )
        lines.append("-" * 92)

    lines.append("=" * 92)
    return "\n".join(lines)


def export_latex_summary_table(
    benchmark_results: Dict[str, Any],
    caption: str = "Multi-dataset reasoning benchmark results across few-shot evaluations.",
    label: str = "tab:ars_multishot_benchmarks",
) -> str:
    """Generates a publication-grade LaTeX booktabs table for research papers."""
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Shots} & \textbf{Frozen Baseline} & \textbf{SteerNet Only} & \textbf{Full ARS Pipeline} & \textbf{$\Delta$ Gain} \\",
        r"\midrule",
    ]

    for ds_name, conds in benchmark_results.items():
        base_dict = conds.get("baseline", {})
        steer_dict = conds.get("steernet", {})
        ars_dict = conds.get("ars", {})

        all_shots = sorted(
            list(set(list(base_dict.keys()) + list(steer_dict.keys()) + list(ars_dict.keys()))),
            key=lambda s: int(s.split("_")[0]) if "_" in s and s.split("_")[0].isdigit() else 0,
        )

        for i, shot_k in enumerate(all_shots):
            ds_col = f"\\textbf{{{ds_name.upper()}}}" if i == 0 else ""
            shot_num = shot_k.split("_")[0] + "-shot"
            b_val = base_dict.get(shot_k)
            s_val = steer_dict.get(shot_k)
            a_val = ars_dict.get(shot_k)

            b_str = f"{b_val:.2f}\\%" if b_val is not None else "--"
            s_str = f"{s_val:.2f}\\%" if s_val is not None else "--"
            a_str = f"\\textbf{{{a_val:.2f}\\%}}" if a_val is not None else "--"

            if a_val is not None and b_val is not None:
                diff = a_val - b_val
                gain_str = f"\\textbf{{+{diff:.2f}\\%}}" if diff >= 0 else f"{diff:.2f}\\%"
            else:
                gain_str = "--"

            latex_lines.append(f"{ds_col} & {shot_num} & {b_str} & {s_str} & {a_str} & {gain_str} \\\\")
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
