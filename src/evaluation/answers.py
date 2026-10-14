"""Unified numeric answer extractor for ARS / GSM8K evaluation.

ISSUE-13 fix: replaces three divergent extractors with one canonical module:
  - verifier.py:  extract_final_number, extract_final_answer
  - multi_dataset.py: extract_numeric_answer  (had 'is' substring bug)
  - losses.py: _extract_final_number

The old `re.finditer(r"(?:=|is)\\s*\\$?(-?[\\d,]+\\.?\\d*)", text)` pattern matched
'is' anywhere, including inside 'this' and 'his', producing wrong answers on
sentences like "she gave this 5 away" → 5 or "his 12 books" → 12.
This module uses '=' only for the fallback scan; explicit answer markers take priority.
"""

import re
from typing import Optional

_NUM = r"-?\d[\d,]*(?:\.\d+)?"

# Priority-ordered explicit answer markers
_MARKERS = (
    r"####\s*\$?(" + _NUM + ")",
    r"\\boxed\{\s*\$?(" + _NUM + r")\s*\}",
    r"[Ff]inal\s+[Aa]nswer(?:\s+is)?\s*[:\-]?\s*\$?(" + _NUM + ")",
    r"[Tt]he\s+answer\s+is\s*[:\-]?\s*\$?(" + _NUM + ")",
)


def _to_float(s: str) -> Optional[float]:
    """Parse a numeric string, tolerating commas, $, % and trailing dots."""
    try:
        return float(s.replace(",", "").replace("$", "").replace("%", "").strip().rstrip("."))
    except (ValueError, AttributeError):
        return None


def has_answer_marker(text: str) -> bool:
    """Return True if any explicit answer marker is present."""
    return any(re.search(p, text) for p in _MARKERS)


def truncate_at_answer(text: str) -> str:
    """Cut text after the earliest explicit answer marker; leave unchanged if none found."""
    ends = [m.end() for p in _MARKERS if (m := re.search(p, text))]
    return text[: min(ends)].strip() if ends else text


def extract_numeric_answer(text: str) -> Optional[float]:
    """Return the numeric answer from text, or None if not found.

    Priority:
      1. Explicit marker (####, \\boxed{}, 'Final answer:', 'The answer is')
      2. 'therefore …' heuristic (last number after 'therefore')
      3. Last '= N' in text ('=' only — never the substring 'is')
    """
    # 1. Explicit markers — highest confidence
    for pat in _MARKERS:
        m = re.search(pat, text)
        if m:
            return _to_float(m.group(1))

    # 2. 'therefore' heuristic
    m = re.search(r"[Tt]herefore[^.\n]*", text)
    if m:
        nums = re.findall(_NUM, m.group(0))
        if nums:
            return _to_float(nums[-1])

    # 3. Last explicit equality — '=' only, never 'is' (avoids 'this'/'his' false matches)
    last = None
    for m in re.finditer(r"=\s*\$?(" + _NUM + ")", text):
        last = m.group(1)
    return _to_float(last) if last is not None else None
