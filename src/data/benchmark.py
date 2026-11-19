"""GSM8K test-set sampling and dataset materialization helpers."""

from __future__ import annotations

import random
from typing import Any, Callable

from datasets import load_dataset


def load_gsm8k_test_rows(
    limit: int | None,
    seed: int = 42,
    *,
    dataset_loader: Callable[..., Any] | None = None,
) -> list[dict[str, object]]:
    """Load a deterministic shuffled GSM8K test subset with source indices."""
    if dataset_loader is None:
        dataset_loader = load_dataset
    dataset = dataset_loader("openai/gsm8k", "main", split="test")
    total = len(dataset)
    sample_size = total if limit is None else min(limit, total)
    if sample_size < 1:
        raise ValueError("benchmark sample size must be positive")
    source_indices = list(range(total))
    if sample_size < total:
        source_indices = random.Random(seed).sample(source_indices, sample_size)
    selected = dataset.select(source_indices)
    rows = []
    for source_index, example in zip(source_indices, selected):
        answer = example["answer"].split("####")[-1].strip()
        rows.append(
            {
                "question": example["question"],
                "answer": answer,
                "source_index": source_index,
            }
        )
    return rows


def materialize_dataset_split(
    dataset: Any, limit: int | None = None
) -> list[dict[str, object]]:
    """Convert a dataset-like split into row dictionaries, preserving order."""
    total = len(dataset)
    count = min(limit, total) if limit is not None else total
    if hasattr(dataset, "select"):
        selected = dataset.select(range(count))
        return [selected[index] for index in range(count)]
    return list(dataset[:count])
