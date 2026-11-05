"""Lightweight OOD evaluation runner for ARS.

Runs baseline vs steered generation on out-of-distribution datasets,
supporting both numeric (open-ended math) and MCQ answer types.
Uses batched generation for speed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm.auto import tqdm

from src.data.prompts import build_prompt_unified
from src.evaluation.answers import cut_continuation, extract_numeric_answer, score_tiers
from src.evaluation.chain import audit_chain, paired_bootstrap_ci, mcnemar_one_sided
from src.evaluation.ood import (
    extract_mcq_answer,
    format_mcq_question,
    load_ood_rows,
)


@torch.no_grad()
def _generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 256,
) -> List[str]:
    """Generate text for a batch of prompts."""
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in enc.items()}
    model.set_prompt_shield_mask(None)
    tokens = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )
    prompt_lens = inputs["attention_mask"].sum(dim=1)
    outputs = []
    for i in range(len(prompts)):
        plen = int(prompt_lens[i])
        text = tokenizer.decode(tokens[i, plen:], skip_special_tokens=True)
        outputs.append(text)
    return outputs


def _build_ood_prompt(row: Dict[str, Any], answer_type: str) -> str:
    """Build a 0-shot prompt for OOD evaluation with preface."""
    q = row["question"]
    if answer_type == "mcq":
        choices = row.get("choices", [])
        q = format_mcq_question(q, choices)
        return (
            f"Question: {q}\n"
            f"Answer: Let's think step by step, then give the answer letter.\n"
        )
    return build_prompt_unified(q, n_shot=0)


def _score_row(
    text: str, gold: str, answer_type: str
) -> Dict[str, Any]:
    """Score a single generation against gold."""
    text = cut_continuation(text)
    if answer_type == "mcq":
        pred = extract_mcq_answer(text)
        correct = pred is not None and pred.upper() == gold.upper()
        return {
            "correct": correct,
            "pred": pred,
            "text": text,
        }

    gold_num = float(gold.replace(",", "").replace("$", "").strip())
    scored = score_tiers(text, gold_num)
    return {
        "correct": scored["lenient"],
        "pred": scored.get("pred_lenient"),
        "text": text,
        **scored,
    }


def run_ood_eval(
    model,
    tokenizer,
    device: torch.device,
    dataset_name: str,
    n_samples: int = 100,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    fixed_layers: Optional[List[int]] = None,
    report_dir: str = "report",
) -> Dict[str, Any]:
    """Run baseline vs steered evaluation on one OOD dataset.

    Returns a results dict with per-arm accuracy and paired statistics.
    """
    rows = load_ood_rows(dataset_name, limit=n_samples)
    answer_type = rows[0].get("answer_type", "numeric") if rows else "numeric"

    if fixed_layers is None:
        fixed_layers = [8, 11, 13, 15, 17, 19, 20, 21, 22, 24]

    results: Dict[str, Any] = {
        "dataset": dataset_name,
        "n": len(rows),
        "answer_type": answer_type,
    }

    all_prompts = [_build_ood_prompt(r, answer_type) for r in rows]
    arm_results: Dict[str, List[Dict[str, Any]]] = {}

    for arm in ["baseline", "fixed"]:
        if arm == "baseline":
            model.set_routing_override("none", fixed_layers)
            model.set_intervention_gain(1.0)
        else:
            model.set_routing_override("only", fixed_layers)
            model.set_intervention_gain(1.0)

        scored_rows: List[Dict[str, Any]] = []
        n_batches = (len(rows) + batch_size - 1) // batch_size
        progress = tqdm(range(n_batches), desc=f"{dataset_name} {arm}", leave=False)

        for bi in progress:
            start = bi * batch_size
            end = min(start + batch_size, len(rows))
            batch_prompts = all_prompts[start:end]
            batch_rows = rows[start:end]

            texts = _generate_batch(
                model, tokenizer, batch_prompts, device, max_new_tokens
            )

            for row, text in zip(batch_rows, texts):
                scored = _score_row(text, row["answer"], answer_type)
                scored["question"] = row["question"]
                scored["gold"] = row["answer"]
                scored_rows.append(scored)

            n_correct = sum(r["correct"] for r in scored_rows)
            progress.set_postfix(acc=f"{100*n_correct/len(scored_rows):.1f}%")

        arm_results[arm] = scored_rows
        acc = 100.0 * sum(r["correct"] for r in scored_rows) / max(len(scored_rows), 1)
        results[arm] = {"accuracy": acc, "n": len(scored_rows)}

    if "baseline" in arm_results and "fixed" in arm_results:
        b_flags = [r["correct"] for r in arm_results["baseline"]]
        f_flags = [r["correct"] for r in arm_results["fixed"]]
        wins = sum(f and not b for b, f in zip(b_flags, f_flags))
        losses = sum(b and not f for b, f in zip(b_flags, f_flags))
        delta_pp, ci_lo, ci_hi = paired_bootstrap_ci(b_flags, f_flags, b=5000, seed=0)
        p_val = mcnemar_one_sided(wins, losses)
        results["paired"] = {
            "delta_pp": delta_pp,
            "ci_95": [ci_lo, ci_hi],
            "wins": wins,
            "losses": losses,
            "mcnemar_p": p_val,
        }

    out_dir = Path(report_dir) / "ood"
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm, scored_rows in arm_results.items():
        out_path = out_dir / f"{dataset_name}_{arm}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in scored_rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    return results


def run_all_ood(
    model,
    tokenizer,
    device: torch.device,
    datasets_config: Dict[str, int],
    max_new_tokens: int = 256,
    batch_size: int = 8,
    fixed_layers: Optional[List[int]] = None,
    report_dir: str = "report",
) -> Dict[str, Any]:
    """Run OOD evaluation on multiple datasets.

    Args:
        datasets_config: dict mapping dataset_name → n_samples,
            e.g. {"svamp": 100, "arc": 100, "math500": 100}
        batch_size: number of prompts per forward pass (default 8)

    Returns combined results dict.
    """
    all_results = {}
    for dname, n in datasets_config.items():
        print(f"\n{'='*60}")
        print(f"OOD Evaluation: {dname} (n={n})")
        print(f"{'='*60}")
        all_results[dname] = run_ood_eval(
            model, tokenizer, device, dname, n,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            fixed_layers=fixed_layers,
            report_dir=report_dir,
        )

    print_ood_summary(all_results)

    out_path = Path(report_dir) / "ood" / "ood_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return all_results


def print_ood_summary(results: Dict[str, Any]) -> None:
    """Print a formatted summary table of OOD results."""
    print(f"\n{'='*70}")
    print("OOD EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'Type':<8} {'N':>5} {'Baseline':>10} {'Steered':>10} {'Delta':>8} {'p-val':>8}")
    print(f"{'-'*70}")
    for dname, res in results.items():
        b_acc = res.get("baseline", {}).get("accuracy", 0)
        f_acc = res.get("fixed", {}).get("accuracy", 0)
        delta = f_acc - b_acc
        p_val = res.get("paired", {}).get("mcnemar_p", 1.0)
        p_str = f"{p_val:.4f}" if p_val >= 0.0001 else f"{p_val:.2e}"
        atype = res.get("answer_type", "?")
        n = res.get("n", 0)
        print(
            f"{dname:<15} {atype:<8} {n:>5} {b_acc:>9.1f}% {f_acc:>9.1f}% {delta:>+7.1f}pp {p_str:>8}"
        )
    print(f"{'='*70}")
