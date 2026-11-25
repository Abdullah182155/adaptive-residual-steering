"""Answer parsing and scoring for the symbolic MATH-500 benchmark."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Iterable


def _unwrap_literal_answer(value: str) -> str:
    """Remove common answer wrappers before literal MATH-500 comparison."""
    text = str(value).strip()
    for pattern in (r"\\boxed\s*\{(.*)\}", r"\\text\s*\{(.*)\}", r"\\mathrm\s*\{(.*)\}"):
        match = re.fullmatch(pattern, text, flags=re.DOTALL)
        if match:
            text = match.group(1).strip()
    return text.strip(" \t\r\n$`'\".,;:")


def _is_literal_math_answer(gold: str) -> bool:
    """MATH-500 includes a small number of text-valued answers (e.g. a name)."""
    text = _unwrap_literal_answer(gold)
    return len(text) > 1 and re.fullmatch(r"[A-Za-z][A-Za-z '\-]*", text) is not None


@lru_cache(maxsize=1024)
def _parse_math_reference(gold: str) -> tuple:
    """Cache parsed references because each benchmark question is scored repeatedly."""
    from math_verify import parse

    return tuple(parse(gold))


def _extract_literal_math_answer(text: str) -> str | None:
    """Read a literal answer from a boxed result, explicit answer line, or final line."""
    text = str(text)
    boxed = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    boxed.append(text[start:index])
                    break
    if boxed:
        return _unwrap_literal_answer(boxed[-1])

    answer_lines = re.findall(
        r"(?im)(?:final\s+answer|the\s+answer|answer)\s*(?:is\s+|[:\-]\s*)([^\n]+)",
        text,
    )
    if answer_lines:
        return _unwrap_literal_answer(answer_lines[-1])
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return _unwrap_literal_answer(lines[-1]) if lines else None


def validate_math_gold_answers(rows: Iterable[dict]) -> list[dict[str, Any]]:
    """Preflight MATH-500 gold answers before an expensive inference run."""
    invalid = []
    for row in rows:
        answer = str(row.get("answer", "")).strip()
        if not answer:
            invalid.append({"source_index": row.get("source_index"), "reason": "empty"})
        elif not _is_literal_math_answer(answer):
            try:
                parsed = _parse_math_reference(answer)
            except Exception:
                parsed = []
            if not parsed:
                invalid.append({"source_index": row.get("source_index"), "reason": "unparseable_gold"})
    return invalid


def score_math_answer(text: str, gold: str) -> dict[str, Any]:
    """Score symbolic answers by equivalence and text answers by literal match."""
    if _is_literal_math_answer(gold):
        prediction = _extract_literal_math_answer(text)
        normalized_gold = " ".join(_unwrap_literal_answer(gold).casefold().split())
        normalized_prediction = " ".join((prediction or "").casefold().split())
        correct = bool(normalized_prediction and normalized_prediction == normalized_gold)
        return {
            "format_ok": prediction is not None,
            "pred_strict": prediction,
            "pred_marker": prediction,
            "pred_lenient": prediction,
            "strict": correct,
            "marker": correct,
            "lenient": correct,
            "scoring_method": "math_verify_exact_or_literal_v1",
        }

    from math_verify import parse, verify

    try:
        gold_parsed = _parse_math_reference(gold)
    except Exception:
        gold_parsed = ()
    try:
        prediction_parsed = parse(text)
        correct = bool(gold_parsed and prediction_parsed and verify(gold_parsed, prediction_parsed))
    except Exception as error:
        prediction_parsed = []
        correct = False
        grading_error = type(error).__name__
    else:
        grading_error = None
    result = {
        "format_ok": bool(prediction_parsed),
        "pred_strict": None,
        "pred_marker": None,
        "pred_lenient": None,
        "strict": correct,
        "marker": correct,
        "lenient": correct,
        "scoring_method": "math_verify_exact_or_literal_v1",
    }
    if grading_error:
        result["grading_error"] = grading_error
    return result
