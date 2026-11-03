"""Paired GSM8K evaluation for baseline and fixed-route ARS."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence

import torch
from tqdm.auto import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

from src.configs.config import decode_only_for_shot
from src.data.prompts import build_prompt_unified
from src.evaluation.answers import score_tiers, should_stop_on_repeat


class AnswerStop(StoppingCriteria):
    """Stop greedy decoding after an answer marker or repeated continuation."""

    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0, self.prompt_length :], skip_special_tokens=True)
        return bool(re.search(r"(?:Final answer|####)\s*[:$-]?\s*[-+]?\d", text) or should_stop_on_repeat(text))


def _gold(value) -> float:
    return float(str(value).replace(",", "").replace("$", "").strip())


@torch.no_grad()
def _generate(
    model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    tokens = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        stopping_criteria=StoppingCriteriaList([AnswerStop(tokenizer, inputs["input_ids"].shape[1])]),
    )
    prompt_tokens = inputs["input_ids"].shape[1]
    return tokenizer.decode(tokens[0, prompt_tokens:], skip_special_tokens=True)


def _score(text: str, gold: float) -> Dict[str, bool]:
    return score_tiers(text, gold)


def _summary(results: List[Dict[str, object]]) -> Dict[str, float]:
    total = len(results)
    strict = sum(item["strict"] for item in results)
    marker = sum(item["marker"] for item in results)
    lenient = sum(item["lenient"] for item in results)
    guard_trips = sum(item["guard_trip"] for item in results)
    nonfinite = sum(item["nonfinite"] for item in results)
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
) -> Dict[str, int]:
    wins = losses = ties = 0
    for control, treatment in zip(baseline, fixed):
        if treatment["lenient"] and not control["lenient"]:
            wins += 1
        elif control["lenient"] and not treatment["lenient"]:
            losses += 1
        else:
            ties += 1
    return {"fixed_wins": wins, "fixed_losses": losses, "ties": ties}


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
) -> Dict[str, object]:
    """Run greedy paired baseline/fixed evaluation and return a serializable record."""
    rows = list(rows)
    record: Dict[str, object] = {
        "model": cfg.model_name,
        "route": list(cfg.fixed_layers),
        "shots": {},
    }
    for n_shot in shots:
        prompt_rows = [
            (row, build_prompt_unified(row["question"], n_shot=n_shot)) for row in rows
        ]
        arm_results: Dict[str, List[Dict[str, object]]] = {}
        for arm in arms:
            if arm not in {"baseline", "fixed"}:
                raise ValueError(f"unsupported benchmark arm: {arm}")
            model.set_current_n_shot(n_shot)
            model.set_decode_only_steering(decode_only_for_shot(n_shot, cfg))
            model.set_routing_override(
                "none" if arm == "baseline" else "only", cfg.fixed_layers
            )
            results = []
            progress = tqdm(prompt_rows, desc=f"{n_shot}-shot {arm}", leave=False)
            for row, prompt in progress:
                model.set_prompt_shield_mask(None)
                text = _generate(
                    model, tokenizer, prompt, device, cfg.generation_max_new_tokens
                )
                guard = dict(model.guard_state)
                cap = (
                    None
                    if contract is None
                    else contract["envelope"]["relative_norm_cap"]
                )
                guard_trip = cap is not None and float(
                    guard.get("rel_max", 0.0)
                ) > float(cap)
                if (
                    arm == "fixed"
                    and cfg.guard_revert_to_k0
                    and (guard_trip or guard.get("nonfinite", False))
                ):
                    model.set_routing_override("none", cfg.fixed_layers)
                    text = _generate(
                        model, tokenizer, prompt, device, cfg.generation_max_new_tokens
                    )
                    model.set_routing_override("only", cfg.fixed_layers)
                scored = _score(text, _gold(row["answer"]))
                results.append(
                    {
                        **scored,
                        "text": text,
                        "guard_trip": bool(guard_trip),
                        "nonfinite": bool(guard.get("nonfinite", False)),
                    }
                )
                progress.set_postfix(correct=sum(item["lenient"] for item in results))
            arm_results[arm] = results
        shot_record: Dict[str, object] = {
            arm: _summary(values) for arm, values in arm_results.items()
        }
        if "baseline" in arm_results and "fixed" in arm_results:
            shot_record["paired"] = _paired_counts(
                arm_results["baseline"], arm_results["fixed"]
            )
            shot_record["paired"]["mcnemar_p_value"] = mcnemar_exact(
                shot_record["paired"]["fixed_wins"], shot_record["paired"]["fixed_losses"]
            )
            identical = sum(
                baseline["text"] == fixed["text"]
                for baseline, fixed in zip(arm_results["baseline"], arm_results["fixed"])
            )
            shot_record["paired"]["identical_text_rate"] = identical / max(len(rows), 1)
            shot_record["paired"]["null_intervention_warning"] = (
                shot_record["paired"]["identical_text_rate"] > 0.95
            )
            generation_dir = Path(cfg.report_dir) / "generations"
            generation_dir.mkdir(parents=True, exist_ok=True)
            paired_rows = [
                {"question": row["question"], "gold": row["answer"], "baseline": baseline, "fixed": fixed}
                for row, baseline, fixed in zip(rows, arm_results["baseline"], arm_results["fixed"])
            ]
            output_path = generation_dir / f"gsm8k_{n_shot}shot.jsonl"
            output_path.write_text(
                "".join(json.dumps(item) + "\n" for item in paired_rows), encoding="utf-8"
            )
        record["shots"][str(n_shot)] = shot_record
    return record


def write_record(record: Dict[str, object], path: str) -> None:
    Path(path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def write_report(record: Dict[str, object], report_dir: str) -> Path:
    destination = Path(report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ARS Phase 1 Benchmark",
        "",
        "| Shots | Arm | Canonical | Lenient | Guard trips | Non-finite |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for shots, result in record["shots"].items():
        for arm in ("baseline", "fixed"):
            if arm not in result:
                continue
            summary = result[arm]
            lines.append(
                f"| {shots} | {arm} | {summary['canonical_accuracy']:.2f}% | "
                f"{summary['lenient_accuracy']:.2f}% | {summary['guard_trip_rate']:.3f} | {summary['nonfinite_count']} |"
            )
        if "paired" in result:
            paired = result["paired"]
            lines.extend(
                [
                    "",
                    f"{shots}-shot paired fixed vs baseline: {paired['fixed_wins']} wins, {paired['fixed_losses']} losses, {paired['ties']} ties.",
                    "",
                ]
            )
    report_path = destination / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
