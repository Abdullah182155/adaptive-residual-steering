"""
Canonical multi-tier answer extractor for ARS evaluation.

Phase 0.2: Implement 4-tier extractor (T1-T4) with strict, canonical, and lenient metrics.
Per §5.3 of the Steering-v5 blueprint.
"""

import re
from typing import Tuple, Optional, Dict, Any, List
import numpy as np


def normalize_text(text: str) -> str:
    """Apply T0 normalizations in order."""
    # Cut continuation and anything after first Question: or Problem:
    text = re.split(r'(?:Question:|Problem:)', text)[0]
    
    # Replace unicode minus
    text = text.replace('−', '-').replace('–', '-')
    
    # Strip markdown emphasis
    text = text.replace('**', '').replace('__', '').replace('``', '')
    
    # Replace \$ with $
    text = text.replace(r'\$', '$')
    
    # Unwrap LaTeX
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\boxed\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\(([^)]+)\)', r'\1', text)
    
    # Normalize non-breaking spaces
    text = text.replace('\u00a0', ' ')
    
    # Remove thousands separators only when pattern matches \d{1,3}(,\d{3})+
    text = re.sub(r'(\d{1,3}(?:,\d{3})+)', lambda m: m.group(1).replace(',', ''), text)
    
    return text


def parse_number(num_str: str) -> Optional[float]:
    """Parse NUM/FRAC/MIXED into float."""
    # Remove trailing .
    num_str = num_str.rstrip('.')
    
    # Fraction: a/b
    frac_match = re.match(r'^\s*([+-]?\$?\s*[\d,]+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*$', num_str)
    if frac_match:
        num = float(frac_match.group(1).replace(',', '').replace('$', '').strip())
        den = float(frac_match.group(2).strip())
        if den == 0:
            return None
        result = num / den
        # Handle percent
        if '%' in num_str:
            return result  # GSM8K golds are in asked unit, don't divide by 100
        return result
    
    # \frac{a}{b}
    frac_latex = re.match(r'\\frac\{([^}]+)\}\{([^}]+)\}', num_str)
    if frac_latex:
        a = parse_number(frac_latex.group(1))
        b = parse_number(frac_latex.group(2))
        if a is not None and b is not None and b != 0:
            return a / b
    
    # Mixed: 3 1/2
    mixed = re.match(r'^\s*(\d+)\s+(\d+)/(\d+)\s*$', num_str)
    if mixed:
        whole = int(mixed.group(1))
        num = int(mixed.group(2))
        den = int(mixed.group(3))
        if den == 0:
            return None
        return whole + num / den
    
    # Check for "X Y/Z" format in the original string before tokenization
    # This handles cases where "3 1/2" gets split into tokens
    if re.match(r'^\d+\s+\d+/\d+$', num_str):
        parts = num_str.split()
        if len(parts) == 2:
            mixed_val = parse_number(parts[0] + ' ' + parts[1])
            if mixed_val is not None:
                return mixed_val
    
    # Regular number with optional $, commas, percent
    num_match = re.match(r'^\s*([+-]?\$?\s*[\d,]+(?:\.\d+)?)%?\s*$', num_str)
    if num_match:
        cleaned = num_match.group(1).replace(',', '').replace('$', '').strip()
        result = float(cleaned)
        if '%' in num_str:
            return result  # GSM8K golds are in asked unit
        return result
    
    return None


def extract_tier1(text: str, question: str, exemplar_boundary: int) -> Tuple[Optional[float], str]:
    """
    Tier 1: Marker-based extraction (strict/canonical).
    
    Markers in order of earliest position: ####, Final answer, The final answer is, The answer is, \boxed
    """
    # Find earliest marker position
    markers = [
        ('####', r'####'),
        ('Final answer', r'Final answer'),
        ('The final answer is', r'The final answer is'),
        ('The answer is', r'The answer is'),
        ('\\boxed', r'\\\\boxed'),
    ]
    
    earliest_pos = len(text)
    matched_marker = None
    
    for name, pattern in markers:
        match = re.search(pattern, text)
        if match:
            pos = match.start()
            if pos < earliest_pos:
                earliest_pos = pos
                matched_marker = name
    
    if matched_marker is None:
        return None, "no_marker"
    
    # Extract first NUM/FRAC/MIXED after marker
    # Allow optional : - =, "is", optional $, optional "x =" or "answer =" prefix
    remaining = text[earliest_pos:]
    
    # Skip marker
    if matched_marker == '####':
        remaining = remaining[4:]
    elif matched_marker == 'Final answer':
        remaining = remaining[12:]
    elif matched_marker == 'The final answer is':
        remaining = remaining[20:]
    elif matched_marker == 'The answer is':
        remaining = remaining[14:]
    elif matched_marker == '\\boxed':
        # Handle \boxed{...} - need to extract content between braces
        brace_match = re.search(r'\{([^}]+)\}', remaining)
        if brace_match:
            remaining = brace_match.group(1)
        else:
            remaining = remaining[7:]
    
    # Skip optional prefixes
    remaining = re.sub(r'^\s*(?:x\s*=\s*|answer\s*=\s*|is\s*)', '', remaining)
    remaining = re.sub(r'^\s*[-:]\s*', '', remaining)
    remaining = re.sub(r'^\s*\$?\s*', '', remaining)
    
    # Find first number (handle mixed numbers by looking ahead)
    tokens = remaining.split()
    for i, token in enumerate(tokens):
        # Check if this could be part of a mixed number "X Y/Z"
        if i < len(tokens) - 1 and re.match(r'^\d+$', token) and re.match(r'^\d+/\d+$', tokens[i+1]):
            combined = f"{token} {tokens[i+1]}"
            num = parse_number(combined)
            if num is not None:
                if earliest_pos < exemplar_boundary:
                    return None, "exemplar_echo"
                return num, matched_marker
        
        num = parse_number(token)
        if num is not None:
            # Check if marker is inside echoed exemplar (before test item boundary)
            if earliest_pos < exemplar_boundary:
                return None, "exemplar_echo"
            return num, matched_marker
    
    return None, "no_number_after_marker"


def extract_tier2(text: str) -> Tuple[Optional[float], str]:
    """
    Tier 2: Conclusion sentence in last 2 non-empty lines with cue.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 2:
        return None, "not_enough_lines"
    
    # Check last 2 lines
    last_lines = lines[-2:]
    cues = ['so', 'therefore', 'thus', 'total', 'altogether', 'in all', 'remain', 'left', 'has', 'have']
    
    for line in last_lines:
        if any(cue in line.lower() for cue in cues):
            # Extract exactly one NUM
            for token in line.split():
                num = parse_number(token)
                if num is not None:
                    return num, "cue_sentence"
    
    return None, "no_cue_sentence"


def extract_tier3(text: str) -> Tuple[Optional[float], str]:
    """
    Tier 3: Last non-empty line contains exactly one NUM.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return None, "empty"
    
    last_line = lines[-1]
    numbers = []
    for token in last_line.split():
        num = parse_number(token)
        if num is not None:
            numbers.append(num)
    
    if len(numbers) == 1:
        return numbers[0], "single_num_last_line"
    
    return None, "not_single_num"


def extract_tier4(text: str) -> Tuple[Optional[float], str]:
    """
    Tier 4: Value after the last = (diagnostic only).
    """
    # Find last = with pattern " = number"
    eq_matches = list(re.finditer(r'=\s*([+-]?\$?\s*[\d,]+(?:\.\d+)?)%?\s*$', text, re.MULTILINE))
    if eq_matches:
        last_match = eq_matches[-1]
        return parse_number(last_match.group(1)), "last_equals"
    return None, "no_equals"


def extract_canonical(
    text: str,
    question: str,
    exemplar_boundary: int = 0,
    tolerance: float = 0.01
) -> Dict[str, Any]:
    """
    Canonical multi-tier extraction.
    
    Returns dict with:
    - strict: T1 result (with checks for echoed exemplars)
    - canonical: T1+T2 result (headline metric)
    - lenient: T1-T3 result
    - marker_rate: share of items with T1 hit
    - tier_histogram: dict of which tier fired
    - multi_answer: True if multiple different T1 values found
    """
    normalized = normalize_text(text)
    
    # Find exemplar boundary (position of "Answer: Let's think step by step." in test item)
    # Look for the LAST occurrence since that's the test item (exemplars come first)
    test_answer_indices = [m.start() for m in re.finditer(r"answer: let's think step by step", normalized.lower())]
    if test_answer_indices:
        test_answer_start = test_answer_indices[-1]
    else:
        test_answer_start = 0
    exemplar_boundary = test_answer_start
    
    # Extract from each tier (stop at first hit)
    t1_val, t1_reason = extract_tier1(normalized, question, exemplar_boundary)
    
    # Check for multiple T1 values
    t1_vals = []
    if t1_val is not None:
        marker_patterns = [r'####', r'Final answer', r'The final answer is', r'The answer is', r'\\\\boxed']
        for pattern in marker_patterns:
            for match in re.finditer(pattern, normalized):
                # Extract number after this marker
                remaining = normalized[match.end():]
                remaining = re.sub(r'^\s*(?:x\s*=\s*|answer\s*=\s*|is\s*)', '', remaining)
                remaining = re.sub(r'^\s*[-:]\s*', '', remaining)
                remaining = re.sub(r'^\s*\$?\s*', '', remaining)
                for token in remaining.split():
                    num = parse_number(token)
                    if num is not None:
                        t1_vals.append(num)
                        break
                break
    
    multi_answer = len(set(t1_vals)) > 1 if t1_vals else False
    
    if t1_val is not None and t1_reason == "exemplar_echo":
        # Reject if inside echoed exemplar
        t1_val = None
    
    t2_val, t2_reason = extract_tier2(normalized) if t1_val is None else (None, "")
    t3_val, t3_reason = extract_tier3(normalized) if t1_val is None and t2_val is None else (None, "")
    t4_val, t4_reason = extract_tier4(normalized)  # Diagnostic only
    
    # Determine which tier fired
    tier_histogram = {"T1": 0, "T2": 0, "T3": 0, "T4": 0, "none": 0}
    canonical_val = None
    
    if t1_val is not None:
        canonical_val = t1_val
        tier_histogram["T1"] = 1
    elif t2_val is not None:
        canonical_val = t2_val
        tier_histogram["T2"] = 1
    # T3 does NOT count in canonical (only lenient)
    else:
        tier_histogram["none"] = 1
    
    # Lenient is T1-T3
    lenient_val = t1_val if t1_val is not None else (t2_val if t2_val is not None else t3_val)
    
    # Update tier histogram for T3 if it fired
    if t3_val is not None and t1_val is None and t2_val is None:
        tier_histogram["T3"] = 1
    
    # Marker rate
    marker_rate = 1.0 if t1_val is not None else 0.0
    
    return {
        "strict": t1_val,
        "canonical": canonical_val,
        "lenient": lenient_val,
        "marker_rate": marker_rate,
        "tier_histogram": tier_histogram,
        "multi_answer": multi_answer,
        "t1_reason": t1_reason if t1_val is not None else "",
        "t2_reason": t2_reason if t2_val is not None else "",
        "t3_reason": t3_reason if t3_val is not None else "",
        "t4_val": t4_val,  # Diagnostic
        "t4_reason": t4_reason,
    }
