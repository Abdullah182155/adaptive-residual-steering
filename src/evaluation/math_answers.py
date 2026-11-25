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
    """Cache parsed references, including bare LaTeX used by MATH-500 answers."""
    from math_verify import parse

    text = str(gold).strip()
    candidates = [text]
    # MATH-500's `answer` column can contain raw LaTeX without `$...$` or another
    # math environment. Math-Verify's LaTeX extractor expects an environment, so
    # retry that exact answer wrapped as inline math after the plain parse fails.
    if text and not any(marker in text for marker in ("$", r"\(", r"\[")):
        candidates.append(f"${text}$")

    errors = []
    for candidate in candidates:
        try:
            parsed = parse(candidate, fallback_mode="no_fallback")
        except Exception as exc:
            errors.append(exc)
            continue
        if parsed:
            return tuple(parsed)

    if errors and len(errors) == len(candidates):
        raise errors[-1]
    return ()


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
        source_index = row.get("source_index", row.get("idx"))
        if not answer:
            invalid.append({"source_index": source_index, "reason": "empty", "answer_preview": answer})
        elif not _is_literal_math_answer(answer):
            try:
                parsed = _parse_math_reference(answer)
            except Exception as exc:
                parsed = []
                invalid.append({
                    "source_index": source_index,
                    "reason": f"parser_error:{type(exc).__name__}",
                    "answer_preview": answer[:200],
                })
                continue
            if not parsed:
                invalid.append({
                    "source_index": source_index,
                    "reason": "unparseable_gold",
                    "answer_preview": answer[:200],
                })
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
