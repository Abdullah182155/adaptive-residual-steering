"""Question-level train/validation split for saved positive RFT rollouts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

from src.training.rft_collect import reasoning_key, tokenize_rft_example


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
    }
    print(f"[adaptive-data] {stats}", flush=True)
    return train_rows, validation_rows, stats


def _group_rows_by_prompt(rows: Sequence[dict]) -> dict[tuple[int, tuple[int, ...]], list[dict]]:
    grouped: dict[tuple[int, tuple[int, ...]], list[dict]] = {}
    for index, row in enumerate(rows):
        question_index = int(row.get("question_index", index))
        prompt_signature = tuple(row.get("prompt_signature", ()))
        key = (question_index, prompt_signature)
        grouped.setdefault(key, []).append(row)
    return grouped
