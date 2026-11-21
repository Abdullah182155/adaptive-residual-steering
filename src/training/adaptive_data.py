"""Question-level train/validation split for saved positive RFT rollouts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.answers import score_tiers
from src.training.rft_collect import reasoning_key, tokenize_rft_example


# Shared objective for static-action and adaptive-router selection.
# The pilot evaluates 0-shot and 2-shot; 4-shot remains weighted for later runs.
TASK_SELECTION_SHOT_WEIGHTS = {0: 0.7, 2: 0.2, 4: 0.1}


def weighted_shot_average(values: Mapping[int, float], metric_name: str) -> tuple[float, str]:
    """Compute a normalized, auditable weighted average over evaluated shot counts."""
    if not values:
        raise ValueError("at least one shot metric is required")
    active = {int(shot): float(value) for shot, value in values.items()}
    unsupported = sorted(set(active) - set(TASK_SELECTION_SHOT_WEIGHTS))
    if unsupported:
        raise ValueError(f"no task-selection weight is configured for shot counts: {unsupported}")
    weights = {shot: TASK_SELECTION_SHOT_WEIGHTS[shot] for shot in active}
    if any(weight < 0.0 for weight in weights.values()) or sum(weights.values()) <= 0.0:
        raise ValueError("task-selection shot weights must be nonnegative with a positive active sum")
    weight_total = sum(weights.values())
    score = sum(weights[shot] * active[shot] for shot in active) / weight_total
    expression = " + ".join(
        f"{weights[shot] / weight_total:g}*{metric_name}_{shot}shot"
        for shot in sorted(active)
    )
    return score, expression


def spread_items(items: Sequence[Any], limit: int) -> list[Any]:
    """Select a deterministic, evenly spaced subset without favoring early rows."""
    if limit < 1:
        return []
    values = list(items)
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[len(values) // 2]]
    indices = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    return [values[index] for index in indices]


def build_adaptive_rft_splits(
    rollout_files: Sequence[str | Path],
    tokenizer,
    cfg,
    *,
    max_questions: int | None = None,
    max_rows_per_question: int = 2,
    validation_fraction: float = 0.1,
    seed: int = 42,
    diversify_prompt_templates: bool = False,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Reuse strict-positive rollouts and split by question to prevent leakage."""
    if not rollout_files or max_rows_per_question < 1:
        raise ValueError("rollout files and a positive rows-per-question limit are required")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")

    records_by_question: dict[int, dict] = {}
    for path in rollout_files:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                question_id = int(record["question_index"])
                existing = records_by_question.get(question_id)
                if existing is None:
                    records_by_question[question_id] = record
                elif existing.get("question") != record.get("question"):
                    raise ValueError(f"question_index collision in rollout files: {question_id}")

    positive_records = []
    for record in records_by_question.values():
        unique: dict[str, tuple[int, str]] = {}
        for completion_index, sample in enumerate(record.get("samples", [])):
            if sample.get("strict_ok") and sample.get("text"):
                unique.setdefault(
                    reasoning_key(sample["text"]), (completion_index, sample["text"])
                )
        if unique:
            positive_records.append((record, list(unique.values())))
    random.Random(seed).shuffle(positive_records)
    if max_questions is not None:
        if max_questions < 2:
            raise ValueError("max_questions must be at least two for a train/validation split")
        positive_records = positive_records[:max_questions]
    if len(positive_records) < 2:
        raise ValueError("at least two questions with verified positive rollouts are required")

    n_validation = max(1, round(len(positive_records) * validation_fraction))
    validation_records = positive_records[:n_validation]
    training_records = positive_records[n_validation:]
    train_question_indices = [int(record["question_index"]) for record, _ in training_records]
    validation_question_indices = [int(record["question_index"]) for record, _ in validation_records]
    rng = random.Random(seed + 1)

    def tokenize(records: list[tuple[dict, list[tuple[int, str]]]]) -> list[dict]:
        rows: list[dict] = []
        for record, completions in records:
            rng.shuffle(completions)
            question_index = int(record["question_index"])
            shot = rng.choices([0, 2, 4], weights=[0.60, 0.25, 0.15])[0]
            template_id = (question_index + shot) % 3 if diversify_prompt_templates else 0
            for completion_index, completion in completions[:max_rows_per_question]:
                encoded = tokenize_rft_example(
                    tokenizer,
                    record["question"],
                    completion,
                    cfg,
                    question_index=question_index,
                    completion_index=completion_index,
                    n_shot=shot,
                    template_id=template_id,
                    max_length=cfg.max_tok_len,
                )
                if encoded is not None:
                    encoded["question"] = str(record["question"])
                    encoded["gold_answer"] = str(record.get("final_num", record.get("answer", "")))
                    rows.append(encoded)
        return rows

    train_rows = tokenize(training_records)
    validation_rows = tokenize(validation_records)
    if not train_rows or not validation_rows:
        raise ValueError("rollouts did not yield tokenizable train and validation examples")
    stats = {
        "questions_with_positives": len(positive_records),
        "train_questions": len(training_records),
        "validation_questions": len(validation_records),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "prompt_templates_diversified": bool(diversify_prompt_templates),
        "requested_prompt_settings_shared_per_question": True,
        "effective_shot_counts": {
            str(shot): sum(int(row["effective_shots"]) == shot for row in train_rows + validation_rows)
            for shot in (0, 2, 4)
        },
        "prompt_group_shot_counts": {
            str(shot): sum(
                int(group[0]["effective_shots"]) == shot
                for group in _group_rows_by_prompt(train_rows + validation_rows).values()
            )
            for shot in (0, 2, 4)
        },
        "prompt_groups": len(_group_rows_by_prompt(train_rows + validation_rows)),
        "train_question_indices": train_question_indices,
        "validation_question_indices": validation_question_indices,
    }
    print(f"[adaptive-data] {stats}", flush=True)
    return train_rows, validation_rows, stats


def build_adaptive_preference_pairs(
    rollout_files: Sequence[str | Path],
    tokenizer,
    cfg,
    question_indices: Sequence[int],
    *,
    max_pairs_per_question: int = 1,
    max_length: int = 768,
    seed: int = 42,
    diversify_prompt_templates: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair verified positives with well-formatted, numerically wrong rollouts.

    Both completions use the same question, shot count, prompt template, and
    tokenized prompt. Malformed outputs and correct-but-audit-rejected outputs
    are excluded so the preference signal focuses on answer correctness.
    """
    if max_pairs_per_question < 1:
        raise ValueError("max_pairs_per_question must be positive")
    allowed = {int(value) for value in question_indices}
    records_by_question: dict[int, dict[str, Any]] = {}
    for path in rollout_files:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                question_id = int(record["question_index"])
                if question_id not in allowed:
                    continue
                existing = records_by_question.get(question_id)
                if existing is not None and existing.get("question") != record.get("question"):
                    raise ValueError(f"question_index collision in rollout files: {question_id}")
                records_by_question[question_id] = record

    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []
    positive_count = negative_count = malformed_count = 0
    common_shot_retries = skipped_oversize_pairs = skipped_prompt_mismatch_pairs = 0
    for question_id in sorted(records_by_question):
        record = records_by_question[question_id]
        gold = record.get("final_num", record.get("answer"))
        gold_value = float(str(gold).replace(",", "").replace("$", "").strip())
        positives: dict[tuple[str, ...], tuple[int, str]] = {}
        negatives: dict[tuple[str, ...], tuple[int, str]] = {}
        for completion_index, sample in enumerate(record.get("samples", [])):
            text = str(sample.get("text", "")).strip()
            if not text:
                continue
            score = score_tiers(text, gold_value)
            key = reasoning_key(text)
            if sample.get("strict_ok"):
                positives.setdefault(key, (completion_index, text))
            elif score["format_ok"] and not score["lenient"]:
                negatives.setdefault(key, (completion_index, text))
            elif not score["format_ok"]:
                malformed_count += 1
        if not positives or not negatives:
            continue
        positive_items = list(positives.values())
        negative_items = list(negatives.values())
        rng.shuffle(positive_items)
        rng.shuffle(negative_items)
        positive_count += len(positive_items)
        negative_count += len(negative_items)
        for pair_index, ((positive_index, positive_text), (negative_index, negative_text)) in enumerate(
            zip(positive_items, negative_items)
        ):
            if pair_index >= max_pairs_per_question:
                break
            shots = rng.choices([0, 2, 4], weights=[0.60, 0.25, 0.15])[0]
            template_id = (question_id + pair_index) % 3 if diversify_prompt_templates else 0
            common = {
                "question_index": question_id,
                "template_id": template_id,
                "max_length": max_length,
            }
            requested_shots = shots
            positive_row = negative_row = None
            while requested_shots >= 0:
                positive_row = tokenize_rft_example(
                    tokenizer, record["question"], positive_text, cfg,
                    completion_index=positive_index, n_shot=requested_shots, **common,
                )
                negative_row = tokenize_rft_example(
                    tokenizer, record["question"], negative_text, cfg,
                    completion_index=negative_index, n_shot=requested_shots, **common,
                )
                if positive_row is None or negative_row is None:
                    skipped_oversize_pairs += 1
                    positive_row = negative_row = None
                    break
                positive_shots = int(positive_row["effective_shots"])
                negative_shots = int(negative_row["effective_shots"])
                if positive_shots == negative_shots:
                    break
                common_shot_retries += 1
                next_shots = min(positive_shots, negative_shots)
                if next_shots >= requested_shots:
                    positive_row = negative_row = None
                    skipped_prompt_mismatch_pairs += 1
                    break
                requested_shots = next_shots
            if positive_row is None or negative_row is None:
                continue
            if positive_row["prompt_signature"] != negative_row["prompt_signature"]:
                skipped_prompt_mismatch_pairs += 1
                continue
            for row in (positive_row, negative_row):
                row["question"] = str(record["question"])
                row["gold_answer"] = str(gold)
            pairs.append({"positive": positive_row, "negative": negative_row})

    stats = {
        "questions_considered": len(records_by_question),
        "questions_with_pairs": len({pair["positive"]["question_index"] for pair in pairs}),
        "pairs": len(pairs),
        "positive_rollouts_seen": positive_count,
        "well_formatted_wrong_rollouts_seen": negative_count,
        "malformed_rollouts_excluded": malformed_count,
        "pairs_retokenized_to_common_shots": common_shot_retries,
        "pairs_skipped_oversize": skipped_oversize_pairs,
        "pairs_skipped_prompt_mismatch": skipped_prompt_mismatch_pairs,
        "question_disjoint_split": True,
        "same_prompt_per_pair": True,
    }
    return pairs, stats


def _group_rows_by_prompt(rows: Sequence[dict]) -> dict[tuple[int, tuple[int, ...]], list[dict]]:
    grouped: dict[tuple[int, tuple[int, ...]], list[dict]] = {}
    for index, row in enumerate(rows):
        question_index = int(row.get("question_index", index))
        prompt_signature = tuple(row.get("prompt_signature", ()))
        key = (question_index, prompt_signature)
        grouped.setdefault(key, []).append(row)
    return grouped
