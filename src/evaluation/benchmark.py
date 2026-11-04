"""Paired GSM8K evaluation for baseline and fixed-route ARS."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

from src.data.prompts import build_prompt_with_shield
from src.evaluation.answers import extract_numeric_answer, score_tiers, should_stop_on_repeat
from src.evaluation.chain import (
    audit_chain,
    classify_outcome,
    paired_bootstrap_ci,
    stop_pattern,
)


STOP_RE = re.compile(
    r"(?:Final answer|The answer is|So the result is|####)\s*[:$-]?\s*\$?[-+]?[\d,]+(?:\.\d+)?[^\n]*\n"
)


class AnswerStop(StoppingCriteria):
    """Stop greedy decoding after an answer marker or repeated continuation."""

    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.reason: Optional[str] = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0, self.prompt_length :], skip_special_tokens=True)
        m = STOP_RE.search(text)
        if m:
            self.reason = "answer_stop"
            return True
        if should_stop_on_repeat(text):
            self.reason = "repeat_stop"
            return True
        return False


def _gold(value) -> float:
    return float(str(value).replace(",", "").replace("$", "").strip())


@torch.no_grad()
def _generate(
    model,
    tokenizer,
    question: str,
    n_shot: int,
    device: torch.device,
    max_new_tokens: int,
    policy: str = "context",
    **prompt_kw,
) -> Tuple[str, Dict[str, Any]]:
    prompt, enc, shield = build_prompt_with_shield(tokenizer, question, n_shot=n_shot, policy=policy, **prompt_kw)
    inputs = {k: v.to(device) for k, v in enc.items()}
    if bool(shield.any()):
        model.set_prompt_shield_mask(shield.unsqueeze(0).to(device))
    else:
        model.set_prompt_shield_mask(None)
    stop = AnswerStop(tokenizer, inputs["input_ids"].shape[1])
    try:
        tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            stopping_criteria=StoppingCriteriaList([stop]),
        )
    finally:
        model.set_prompt_shield_mask(None)
    prompt_tokens = inputs["input_ids"].shape[1]
    n_new = tokens.shape[1] - prompt_tokens
    text = tokenizer.decode(tokens[0, prompt_tokens:], skip_special_tokens=True)
    reason = stop.reason or ("max_new_tokens" if n_new >= max_new_tokens else "eos")
    return text, {"stop_reason": reason, "n_new_tokens": n_new}


def _score(text: str, gold: float) -> Dict[str, bool]:
    return score_tiers(text, gold)


def _summary(results: List[Dict[str, object]]) -> Dict[str, float]:
    total = len(results)
    strict = sum(item["strict"] for item in results)
    marker = sum(item["marker"] for item in results)
    lenient = sum(item["lenient"] for item in results)
    guard_trips = sum(item.get("guard_trip", False) for item in results)
    nonfinite = sum(item.get("nonfinite", False) for item in results)
    return {
        "n": total,
        "strict_accuracy": 100.0 * strict / max(total, 1),
        "canonical_accuracy": 100.0 * strict / max(total, 1),
        "marker_accuracy": 100.0 * marker / max(total, 1),
        "format_rate": sum(item["format_ok"] for item in results) / max(total, 1),
        "lenient_accuracy": 100.0 * lenient / max(total, 1),
        "guard_trip_rate": guard_trips / max(total, 1),
        "nonfinite_count": nonfinite,
    }


def _paired_counts(
    baseline: List[Dict[str, object]], fixed: List[Dict[str, object]]
) -> Dict[str, Any]:
    wins = losses = ties = 0
    for control, treatment in zip(baseline, fixed):
        if treatment["lenient"] and not control["lenient"]:
            wins += 1
        elif control["lenient"] and not treatment["lenient"]:
            losses += 1
        else:
            ties += 1
    b_flags = [bool(item["lenient"]) for item in baseline]
    f_flags = [bool(item["lenient"]) for item in fixed]
    delta_pp, ci_lo, ci_hi = paired_bootstrap_ci(b_flags, f_flags, b=10000, seed=0)
    return {
        "fixed_wins": wins,
        "fixed_losses": losses,
        "ties": ties,
        "delta_pp": delta_pp,
        "ci95_pp": [ci_lo, ci_hi],
    }


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def run_benchmark(
    model,
    tokenizer,
    rows: Iterable[dict],
    cfg,
    device: torch.device,
    shots: Sequence[int],
    arms: Sequence[str],
    contract: Dict[str, object] | None = None,
    suite: str = "gsm8k",
) -> Dict[str, object]:
    """Run greedy paired baseline/fixed evaluation and return a serializable record."""
    rows = list(rows)
    record: Dict[str, object] = {
        "model": cfg.model_name,
        "route": list(cfg.fixed_layers),
        "suite": suite,
        "shots": {},
    }
    shield_policy = getattr(cfg, "shield_policy", "context")
    for n_shot in shots:
        arm_results: Dict[str, List[Dict[str, object]]] = {}
        for arm in arms:
            if arm not in {"baseline", "fixed", "identity", "random"}:
                raise ValueError(f"unsupported benchmark arm: {arm}")

            if arm == "baseline":
                model.set_routing_override("none", cfg.fixed_layers)
                if hasattr(model, "set_intervention_gain"):
                    model.set_intervention_gain(1.0)
                if hasattr(model, "set_control"):
                    model.set_control(None)
            elif arm == "identity":
                model.set_routing_override("only", cfg.fixed_layers)
                if hasattr(model, "set_intervention_gain"):
                    model.set_intervention_gain(0.0)
                if hasattr(model, "set_control"):
                    model.set_control(None)
            elif arm == "fixed":
                model.set_routing_override("only", cfg.fixed_layers)
                if hasattr(model, "set_intervention_gain"):
                    model.set_intervention_gain(1.0)
                if hasattr(model, "set_control"):
                    model.set_control(None)
            elif arm == "random":
                model.set_routing_override("only", cfg.fixed_layers)
                if hasattr(model, "set_intervention_gain"):
                    model.set_intervention_gain(1.0)
                if hasattr(model, "set_control"):
                    model.set_control("random")

            results = []
            progress = tqdm(rows, desc=f"{suite} {n_shot}-shot {arm}", leave=False)
            for row_idx, row in enumerate(progress):
                if hasattr(model, "set_control") and arm == "random":
                    model.set_control("random", seed=row_idx)

                text, meta = _generate(
                    model,
                    tokenizer,
                    row["question"],
                    n_shot,
                    device,
                    cfg.generation_max_new_tokens,
                    policy=shield_policy,
                    question_idx=row_idx,
                )
                guard = dict(model.guard_state)
                cap = (
                    None
                    if contract is None
                    else contract.get("envelope", {}).get("relative_norm_cap")
                )
                guard_trip = cap is not None and float(
                    guard.get("rel_max", 0.0)
                ) > float(cap)
                if (
                    arm in {"fixed", "random"}
                    and cfg.guard_revert_to_k0
                    and (guard_trip or guard.get("nonfinite", False))
                ):
                    model.set_routing_override("none", cfg.fixed_layers)
                    text, meta = _generate(
                        model,
                        tokenizer,
                        row["question"],
                        n_shot,
                        device,
                        cfg.generation_max_new_tokens,
                        policy=shield_policy,
                        question_idx=row_idx,
                    )
                    model.set_routing_override("only", cfg.fixed_layers)

                gold_num = _gold(row["answer"])
                scored = _score(text, gold_num)
                pred_marker = extract_numeric_answer(text)
                pred_lenient = extract_numeric_answer(text, lenient=True)
                audit = audit_chain(text, row["question"])
                results.append(
                    {
                        **scored,
                        "text": text,
                        "pred_marker": pred_marker,
                        "pred_lenient": pred_lenient,
                        "stop_reason": meta["stop_reason"],
                        "n_new_tokens": meta["n_new_tokens"],
                        "rel_max": float(guard.get("rel_max", 0.0)),
                        "guard_trip": bool(guard_trip),
                        "nonfinite": bool(guard.get("nonfinite", False)),
                        "car": audit["car"],
                        "eaa": audit["eaa"],
                    }
                )
                progress.set_postfix(correct=sum(item["lenient"] for item in results))
            arm_results[arm] = results

            # Save individual arm generation file: report/generations/{suite}_{n_shot}shot_{arm}.jsonl
            generation_dir = Path(cfg.report_dir) / "generations"
            generation_dir.mkdir(parents=True, exist_ok=True)
            arm_output_path = generation_dir / f"{suite}_{n_shot}shot_{arm}.jsonl"
            arm_records = [
                {
                    "idx": i,
                    "question": row["question"],
                    "gold": row["answer"],
                    **res,
                }
                for i, (row, res) in enumerate(zip(rows, results))
            ]
            arm_output_path.write_text(
                "".join(json.dumps(item) + "\n" for item in arm_records),
                encoding="utf-8",
            )

        shot_record: Dict[str, object] = {
            arm: _summary(values) for arm, values in arm_results.items()
        }
        if "baseline" in arm_results and "fixed" in arm_results:
            paired_stats = _paired_counts(arm_results["baseline"], arm_results["fixed"])
            shot_record["paired"] = paired_stats
            shot_record["paired"]["mcnemar_p_value"] = mcnemar_exact(
                paired_stats["fixed_wins"], paired_stats["fixed_losses"]
            )
            identical = sum(
                baseline["text"] == fixed["text"]
                for baseline, fixed in zip(arm_results["baseline"], arm_results["fixed"])
            )
            shot_record["paired"]["identical_text_rate"] = identical / max(len(rows), 1)
            shot_record["paired"]["null_intervention_warning"] = (
                shot_record["paired"]["identical_text_rate"] > 0.95
            )
            # Add failure classification taxonomy for discordant questions
            taxonomy_counts: Dict[str, int] = {}
            for row, b_res, f_res in zip(rows, arm_results["baseline"], arm_results["fixed"]):
                cls_tag = classify_outcome(b_res, f_res, _gold(row["answer"]), question=row["question"])
                taxonomy_counts[cls_tag] = taxonomy_counts.get(cls_tag, 0) + 1
            shot_record["paired"]["taxonomy"] = taxonomy_counts

            # Save paired generation file for diagnostics and legacy tools: report/generations/{suite}_{n_shot}shot.jsonl
            paired_rows = [
                {"question": row["question"], "gold": row["answer"], "baseline": b, "fixed": f}
                for row, b, f in zip(rows, arm_results["baseline"], arm_results["fixed"])
            ]
            paired_output_path = generation_dir / f"{suite}_{n_shot}shot.jsonl"
            paired_output_path.write_text(
                "".join(json.dumps(item) + "\n" for item in paired_rows),
                encoding="utf-8",
            )

        record["shots"][str(n_shot)] = shot_record
    return record


def write_record(record: Dict[str, object], path: str) -> None:
    Path(path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def write_report(record: Dict[str, object], report_dir: str) -> Path:
    destination = Path(report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    suite = record.get("suite", "gsm8k")
    lines = [
        f"# ARS Phase 1 Benchmark - {suite.upper()}",
        "",
        "| Shots | Arm | Strict | Marker | Lenient | Format Rate | Guard trips | Non-finite |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for shots, result in record["shots"].items():
        for arm in ("baseline", "identity", "fixed", "random"):
            if arm not in result:
                continue
            summary = result[arm]
            strict_acc = summary.get("strict_accuracy", summary.get("canonical_accuracy", 0.0))
            marker_acc = summary.get("marker_accuracy", summary.get("lenient_accuracy", 0.0))
            lenient_acc = summary.get("lenient_accuracy", 0.0)
            fmt_rate = summary.get("format_rate", 0.0) * 100.0
            lines.append(
                f"| {shots} | {arm} | {strict_acc:.2f}% | {marker_acc:.2f}% | "
                f"{lenient_acc:.2f}% | {fmt_rate:.1f}% | {summary['guard_trip_rate']:.3f} | {summary['nonfinite_count']} |"
            )
        if "paired" in result:
            paired = result["paired"]
            p_val = paired.get("mcnemar_p_value", 1.0)
            p_str = f"{p_val:.4f}" if p_val >= 0.0001 else f"{p_val:.2e}"
            ident = paired.get("identical_text_rate", 0.0) * 100.0
            delta_pp = paired.get("delta_pp", 0.0)
            ci = paired.get("ci95_pp", [0.0, 0.0])
            lines.extend(
                [
                    "",
                    f"{shots}-shot paired fixed vs baseline: {paired['fixed_wins']} wins, {paired['fixed_losses']} losses, {paired['ties']} ties. "
                    f"Delta={delta_pp:+.2f} pp, 95% CI=[{ci[0]:+.2f}, {ci[1]:+.2f}], McNemar p={p_str}, identical text={ident:.1f}%.",
                    "",
                ]
            )
    report_path = destination / f"REPORT_{suite.upper()}.md"
    content = "\n".join(lines) + "\n"
    report_path.write_text(content, encoding="utf-8")
    (destination / "REPORT.md").write_text(content, encoding="utf-8")
    return report_path
