"""Pluggable out-of-distribution dataset loader with schema verification.

Supports both numeric (open-ended math) and MCQ (multiple-choice) datasets.
Each row has: idx, question, answer, group, answer_type ("numeric" | "mcq").
MCQ rows also have "choices" (list of choice strings).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset


# (hf_name, expected_count, answer_type)
SPECS: Dict[str, Tuple[str, int, str]] = {
    "svamp": ("ChilleD/SVAMP", 1000, "numeric"),
    "multiarith": ("ChilleD/MultiArith", 600, "numeric"),
    "gsm_plus": ("qintongli/GSM-Plus", 1000, "numeric"),
    "math500": ("HuggingFaceH4/MATH-500", 500, "numeric"),
    "arc": ("allenai/ai2_arc", 1172, "mcq"),
    "aqua_rat": ("deepmind/aqua_rat", 254, "mcq"),
}


def _extract_fields(name: str, ex: Dict[str, Any]) -> Dict[str, Any]:
    """Extract standardized fields per dataset. Returns dict with question, answer, group, choices."""
    if name == "svamp":
        if "question_concat" in ex:
            q = ex["question_concat"]
        elif "Body" in ex and "Question" in ex:
            q = f"{ex['Body'].strip()} {ex['Question'].strip()}"
        elif "question" in ex:
            q = ex["question"]
        else:
            raise KeyError(f"No question field in SVAMP: {sorted(ex.keys())}")
        ans = ex.get("Answer", ex.get("answer"))
        if ans is None:
            raise KeyError(f"No answer field in SVAMP: {sorted(ex.keys())}")
        return {"question": str(q), "answer": str(ans), "group": ex.get("Body")}

    if name == "multiarith":
        q = ex.get("question", ex.get("Question"))
        ans = ex.get("final_ans", ex.get("answer", ex.get("Answer")))
        if q is None or ans is None:
            raise KeyError(f"Missing fields in MultiArith: {sorted(ex.keys())}")
        return {"question": str(q), "answer": str(ans), "group": None}

    if name == "gsm_plus":
        q = ex.get("question", ex.get("problem", ex.get("Question")))
        ans = ex.get("answer", ex.get("final_answer", ex.get("Answer")))
        if q is None or ans is None:
            raise KeyError(f"Missing fields in GSM-Plus: {sorted(ex.keys())}")
        return {
            "question": str(q),
            "answer": str(ans),
            "group": ex.get("perturbation_type"),
        }

    if name == "math500":
        q = ex.get("problem", ex.get("question"))
        ans = ex.get("answer", ex.get("solution"))
        if q is None or ans is None:
            raise KeyError(f"Missing fields in MATH-500: {sorted(ex.keys())}")
        return {"question": str(q), "answer": str(ans), "group": ex.get("subject")}

    if name == "arc":
        q = ex.get("question", "")
        choices_obj = ex.get("choices", {})
        labels = choices_obj.get("label", [])
        texts = choices_obj.get("text", [])
        choice_strs = [f"({lab}) {txt}" for lab, txt in zip(labels, texts)]
        ans = ex.get("answerKey", "")
        return {
            "question": q,
            "answer": str(ans),
            "group": None,
            "choices": choice_strs,
        }

    if name == "aqua_rat":
        q = ex.get("question", "")
        options = ex.get("options", [])
        ans = ex.get("correct", "")
        return {
            "question": str(q),
            "answer": str(ans),
            "group": None,
            "choices": options,
        }

    raise ValueError(f"Unknown dataset: {name}")


def _is_numeric(s: str) -> bool:
    try:
        float(s.replace(",", "").replace("$", "").replace("%", "").strip())
        return True
    except (ValueError, AttributeError):
        return False


def load_ood_rows(
    name: str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load normalized rows from a supported OOD benchmark.

    Returns list of dicts with keys: idx, question, answer, group,
    answer_type, and optionally choices (for MCQ).
    """
    if name not in SPECS:
        raise ValueError(
            f"Unsupported OOD dataset: {name}. Supported: {sorted(SPECS.keys())}"
        )
    dataset_name, expected_count, answer_type = SPECS[name]

    if name == "arc":
        dataset = load_dataset(dataset_name, "ARC-Challenge", split="test")
    elif name == "aqua_rat":
        dataset = load_dataset(dataset_name, "raw", split="test")
    elif name == "math500":
        dataset = load_dataset(dataset_name, split="test")
    else:
        try:
            dataset = load_dataset(dataset_name, split="test")
        except Exception:
            dataset = load_dataset(dataset_name)

    raw_items: list = []
    if isinstance(dataset, dict):
        splits = ["test"] if "test" in dataset else list(dataset.keys())
        for sp in splits:
            raw_items.extend(dataset[sp])
    else:
        raw_items = list(dataset)

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for ex in raw_items:
        fields = _extract_fields(name, ex)
        clean_ans = str(fields["answer"]).split("####")[-1].strip()

        if answer_type == "numeric" and (
            not _is_numeric(clean_ans) or clean_ans.lower() == "none"
        ):
            skipped += 1
            continue

        row: Dict[str, Any] = {
            "idx": len(rows),
            "question": fields["question"],
            "answer": clean_ans,
            "group": fields.get("group"),
            "answer_type": answer_type,
        }
        if "choices" in fields:
            row["choices"] = fields["choices"]
        rows.append(row)

    if skipped > 0:
        print(f"[ood] {name}: skipped {skipped} non-numeric answers")

    assert len(rows) > 0, (
        f"OOD loader for '{name}' returned 0 rows! Expected ~{expected_count}."
    )

    print(
        f"[ood] {name}: loaded n={len(rows)} rows "
        f"(expected ~{expected_count}, type={answer_type})"
    )
    return rows[:limit] if limit is not None else rows


def format_mcq_question(question: str, choices: List[str]) -> str:
    """Format an MCQ question with choices for prompting."""
    choices_text = "\n".join(choices)
    return f"{question}\n{choices_text}"


def extract_mcq_answer(text: str) -> Optional[str]:
    """Extract MCQ answer letter (A-E) from generated text."""
    text = text.strip()
    m = re.search(
        r"(?:answer|choice|option)\s*(?:is|:)\s*\(?([A-Ea-e])\)?", text, re.IGNORECASE
    )
    if m:
        return m.group(1).upper()
    m = re.search(r"\(?([A-Ea-e])\)\s*$", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"^\s*\(?([A-Ea-e])\)?[\s.]", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\(?([A-Ea-e])\)", text)
    if m:
        return m.group(1).upper()
    return None
