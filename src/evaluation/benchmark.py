"""Paired GSM8K evaluation for baseline and fixed-route ARS."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence

import torch

from src.configs.config import decode_only_for_shot
from src.data.prompts import build_prompt_unified
from src.evaluation.answers import extract_numeric_answer


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
    )
    prompt_tokens = inputs["input_ids"].shape[1]
    return tokenizer.decode(tokens[0, prompt_tokens:], skip_special_tokens=True)


def _score(text: str, gold: float) -> Dict[str, bool]:
    prediction = extract_numeric_answer(text, lenient=True)
    correct = prediction is not None and abs(prediction - gold) <= 1e-6
    return {
        "canonical": bool(correct and re.search(r"####\s*[-+]?\d", text)),
        "lenient": bool(correct),
    }


def _summary(results: List[Dict[str, object]]) -> Dict[str, float]:
    total = len(results)
    canonical = sum(item["canonical"] for item in results)
    lenient = sum(item["lenient"] for item in results)
    guard_trips = sum(item["guard_trip"] for item in results)
    nonfinite = sum(item["nonfinite"] for item in results)
    return {
        "n": total,
        "canonical_accuracy": 100.0 * canonical / max(total, 1),
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
            for row, prompt in prompt_rows:
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
                        "guard_trip": bool(guard_trip),
                        "nonfinite": bool(guard.get("nonfinite", False)),
                    }
                )
            arm_results[arm] = results
        shot_record: Dict[str, object] = {
            arm: _summary(values) for arm, values in arm_results.items()
        }
        if "baseline" in arm_results and "fixed" in arm_results:
            shot_record["paired"] = _paired_counts(
                arm_results["baseline"], arm_results["fixed"]
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
