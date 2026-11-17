"""Question-level train/validation split for saved positive RFT rollouts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

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
) -> tuple[list[dict], list[dict], dict[str, int]]:
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
        unique = {
            reasoning_key(sample["text"]): sample["text"]
            for sample in record.get("samples", [])
            if sample.get("strict_ok") and sample.get("text")
        }
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

    def tokenize(records: list[tuple[dict, list[str]]]) -> list[dict]:
        rows: list[dict] = []
        for record, completions in records:
            rng.shuffle(completions)
            question_index = int(record["question_index"])
            for completion in completions[:max_rows_per_question]:
                shot = rng.choices([0, 2, 4], weights=[0.60, 0.25, 0.15])[0]
                encoded = tokenize_rft_example(
                    tokenizer,
                    record["question"],
                    completion,
                    cfg,
                    question_index=question_index,
                    n_shot=shot,
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
    }
    print(f"[adaptive-data] {stats}", flush=True)
    return train_rows, validation_rows, stats
