"""Pluggable arithmetic out-of-distribution dataset loader with schema verification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from datasets import load_dataset


SPECS: Dict[str, Tuple[str, int]] = {
    "svamp": ("ChilleD/SVAMP", 1000),
    "multiarith": ("ChilleD/MultiArith", 600),
    "gsm_plus": ("qintongli/GSM-Plus", 1000),
}


def _extract_fields(name: str, ex: Dict[str, Any]) -> Tuple[str, Any, Optional[str]]:
    """Extract (question, answer, group) with loud schema validation."""
    try:
        if name == "svamp":
            if "question_concat" in ex:
                q = ex["question_concat"]
            elif "Body" in ex and "Question" in ex:
                q = f"{ex['Body'].strip()} {ex['Question'].strip()}"
            elif "question" in ex:
                q = ex["question"]
            else:
                raise KeyError(f"No question or Body/Question field in SVAMP example: {sorted(ex.keys())}")
            ans = ex.get("Answer", ex.get("answer"))
            if ans is None:
                raise KeyError(f"No Answer/answer field in SVAMP example: {sorted(ex.keys())}")
            return str(q), ans, ex.get("Body")

        if name == "multiarith":
            q = ex.get("question", ex.get("Question"))
            ans = ex.get("final_ans", ex.get("answer", ex.get("Answer")))
            if q is None:
                raise KeyError(f"No question/Question field in MultiArith example: {sorted(ex.keys())}")
            if ans is None:
                raise KeyError(f"No final_ans/answer field in MultiArith example: {sorted(ex.keys())}")
            return str(q), ans, None

        if name == "gsm_plus":
            q = ex.get("question", ex.get("problem", ex.get("Question")))
            ans = ex.get("answer", ex.get("final_answer", ex.get("Answer")))
            if q is None or ans is None:
                raise KeyError(f"Missing required fields in GSM-Plus example: {sorted(ex.keys())}")
            return str(q), ans, None

        # Generic fallback
        question_keys = ("question", "Question", "problem", "Body")
        answer_keys = ("answer", "Answer", "final_ans", "final_answer")
        q = next((ex[k] for k in question_keys if k in ex), None)
        ans = next((ex[k] for k in answer_keys if k in ex), None)
        if q is None or ans is None:
            raise KeyError(f"Unexpected schema for {name}; available columns: {sorted(ex.keys())}")
        return str(q), ans, None

    except KeyError as err:
        raise KeyError(f"{name}: unexpected schema; columns are {sorted(ex.keys())}") from err


def load_ood_rows(name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load a normalized question/answer view of one supported OOD benchmark with loud validation."""
    if name not in SPECS:
        raise ValueError(f"unsupported OOD dataset: {name}. Supported: {sorted(SPECS.keys())}")
    dataset_name, expected_count = SPECS[name]
    try:
        dataset = load_dataset(dataset_name, split="test")
    except Exception:
        dataset = load_dataset(dataset_name)

    rows: List[Dict[str, Any]] = []
    # If dataset has multiple splits (e.g. dict of splits), iterate across all or test
    if isinstance(dataset, dict):
        splits_to_read = ["test"] if "test" in dataset else list(dataset.keys())
        for split_name in splits_to_read:
            for ex in dataset[split_name]:
                q, a, group = _extract_fields(name, ex)
                clean_ans = str(a).split("####")[-1].strip()
                rows.append({"idx": len(rows), "question": q, "answer": clean_ans, "group": group})
    else:
        for ex in dataset:
            q, a, group = _extract_fields(name, ex)
            clean_ans = str(a).split("####")[-1].strip()
            rows.append({"idx": len(rows), "question": q, "answer": clean_ans, "group": group})

    assert len(rows) > 0, f"OOD loader for '{name}' returned 0 rows! Expected ~{expected_count}."

    # Validate that every loaded row has a numeric-parsable answer
    for r in rows:
        try:
            float(str(r["answer"]).replace(",", "").replace("$", ""))
        except ValueError as err:
            raise ValueError(f"Invalid non-numeric answer '{r['answer']}' in {name} row {r['idx']}") from err

    print(f"[ood] {name}: successfully loaded n={len(rows)} rows (expected ~{expected_count})")
    return rows[:limit] if limit is not None else rows
