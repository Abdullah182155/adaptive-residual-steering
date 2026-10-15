"""Unified numeric answer extractor for ARS / GSM8K evaluation.

A-1 (run-analysis fix): adds cut_continuation() and should_stop_on_repeat()
to prevent hallucinated follow-up questions from polluting extraction.
Also fixes extraction priority: 'Final answer: <words>' line → last number
is now tried before the last '=' scan.

ISSUE-13 fix: replaces three divergent extractors with one canonical module.
The old `re.finditer(r'(?:=|is)\\s*...')` pattern matched 'is' inside
'this'/'his'. This module uses '=' only for the fallback scan.
"""

import re
from typing import Optional

_NUM = r"-?\d[\d,]*(?:\.\d+)?"

# Markers that indicate hallucinated follow-up questions / discussion — cut here
_STOP_MARKERS = (
    r"\n\s*Question\s*:",
    r"\n\s*Q\s*:",
    r"\n\s*Problem\s*:",
    r"<\|endoftext\|>",
    r"\n\s*Topic\s*:",
    r"\n\s*Follow-?up[^\n]*:",
    r"\n\s*Complete detailed[^\n]*",
    r"\n\s*(?:Discussion|Exercise|Ph\.?D\.?-level[^\n]*)\s*:?",
)

# Priority-ordered explicit answer markers (must capture group 1 = the number)
_MARKERS = (
    r"####\s*\$?(" + _NUM + ")",
    r"\\+boxed\{\s*\$?(" + _NUM + r")\s*\}",
    r"[Ff]inal\s+[Aa]nswer(?:\s+is)?\s*[:\-]?\s*\$?(" + _NUM + ")",
    r"[Tt]he\s+answer\s+is\s*[:\-]?\s*\$?(" + _NUM + ")",
)


def _to_float(s: str) -> Optional[float]:
    """Parse a numeric string, tolerating commas, $, % and trailing dots."""
    try:
        return float(s.replace(",", "").replace("$", "").replace("%", "").strip().rstrip("."))
    except (ValueError, AttributeError):
        return None


def cut_continuation(text: str) -> str:
    """Drop everything from the first hallucinated follow-up question onward.

    Phi-2 and similar models sometimes continue generating past the answer
    into invented follow-up questions (e.g. 'Question: There are 10 apples...').
    Extraction on this tail produces wrong answers. Cut it away symmetrically
    for all arms so baseline and ARS are evaluated on equivalent text.
    """
    positions = []
    for p in _STOP_MARKERS:
        m = re.search(p, text)
        if m:
            positions.append(m.start())
    if positions:
        return text[: min(positions)].rstrip()
    return text


def should_stop_on_repeat(text: str, max_repeats: int = 3, min_len: int = 25) -> bool:
    """Return True if any normalised line appears >= max_repeats times.

    Used inside AnswerStoppingCriteria to abort runaway repetition loops
    that infinite-loop on the same sentence (observed in 2-shot Phi-2 runs).
    Numbers are normalised to '#' so that slight numeric variations don't
    prevent detection of structural loops.
    """
    counts: dict = {}
    for line in text.split("\n"):
        line = line.strip()
        if len(line) < min_len:
            continue
        key = re.sub(r"\d+", "#", line)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= max_repeats:
            return True
    return False


def has_answer_marker(text: str) -> bool:
    """Return True if any explicit answer marker is present (after cutting continuation)."""
    text = cut_continuation(text)
    return any(re.search(p, text) for p in _MARKERS)


def truncate_at_answer(text: str) -> str:
    """Cut text after the earliest explicit answer marker; leave unchanged if none found."""
    text = cut_continuation(text)
    ends = [m.end() for p in _MARKERS if (m := re.search(p, text))]
    if ends:
        return text[: min(ends)].strip()
    # Fallback: cut at end of 'Final answer: <words>' line even without a number
    m = re.search(r"[Ff]inal\s+[Aa]nswer\s*(?:is)?\s*[:\-]?[^\n]*", text)
    return text[: m.end()].strip() if m else text


def extract_numeric_answer(text: str) -> Optional[float]:
    """Return the numeric answer from text, or None if not found.

    Priority:
      1. Explicit marker (####, \\boxed{}, 'Final answer: N', 'The answer is N')
      2. 'Final answer: <words>' line — last number anywhere in that line
         (handles 'Final answer: Indras has 6 letters and her sister has 13 letters' → 13)
      3. Last '= N' in text ('=' only — never 'is' to avoid 'this'/'his' false matches)
      4. 'therefore ...' heuristic — last resort only
    """
    text = cut_continuation(text)

    # 1. Explicit markers with numeric capture — highest confidence
    for pat in _MARKERS:
        m = re.search(pat, text)
        if m:
            return _to_float(m.group(1))

    # 2. 'Final answer: <words>' line — last number in the line
    m = re.search(r"[Ff]inal\s+[Aa]nswer\s*(?:is)?\s*[:\-]?([^\n]*)", text)
    if m:
        nums = re.findall(_NUM, m.group(1))
        if nums:
            return _to_float(nums[-1])

    # 3. Last explicit equality — '=' only, never 'is'
    last = None
    for m in re.finditer(r"=\s*\$?(" + _NUM + ")", text):
        last = m.group(1)
    if last is not None:
        return _to_float(last)

    # 4. 'therefore' heuristic — least reliable, last resort
    m = re.search(r"[Tt]herefore[^.\n]*", text)
    if m:
        nums = re.findall(_NUM, m.group(0))
        if nums:
            return _to_float(nums[-1])

    return None
