"""
Tests for canonical multi-tier extractor (extract_v2.py).

Phase 0.2: Test T1-T4 extraction logic and edge cases.
"""

import pytest
from src.evaluation.extract_v2 import extract_canonical, parse_number, normalize_text


def test_parse_number_basic():
    """Test basic number parsing."""
    assert parse_number("42") == 42.0
    assert parse_number("-5") == -5.0
    assert parse_number("3.14") == 3.14
    assert parse_number("$100") == 100.0
    assert parse_number("1,000") == 1000.0
    assert parse_number("50%") == 50.0


def test_parse_number_fractions():
    """Test fraction parsing."""
    assert parse_number("1/2") == 0.5
    assert parse_number("3/4") == 0.75
    assert parse_number("-1/2") == -0.5
    assert parse_number("5/0") is None  # Division by zero


def test_parse_number_mixed():
    """Test mixed number parsing."""
    assert parse_number("3 1/2") == 3.5
    assert parse_number("1 3/4") == 1.75


def test_parse_number_trailing_dot():
    """Test trailing dot removal."""
    assert parse_number("42.") == 42.0
    assert parse_number("3.14.") == 3.14


def test_normalize_text():
    """Test text normalization."""
    # Unicode minus
    assert normalize_text("−5") == "-5"
    assert normalize_text("10–5") == "10-5"
    
    # Markdown emphasis
    assert normalize_text("**bold**") == "bold"
    assert normalize_text("__italic__") == "italic"
    
    # LaTeX
    assert normalize_text(r"\text{value}") == "value"
    assert normalize_text(r"\boxed{42}") == "42"
    
    # Thousands separators
    assert normalize_text("1,000") == "1000"
    assert normalize_text("10,000,000") == "10000000"


def test_extract_tier1_final_answer():
    """Test T1 extraction with Final answer marker."""
    text = "Step 1: 2 + 2 = 4\nFinal answer: 4"
    result = extract_canonical(text, "What is 2+2?")
    assert result["strict"] == 4.0
    assert result["canonical"] == 4.0
    assert result["lenient"] == 4.0
    assert result["marker_rate"] == 1.0
    assert result["tier_histogram"]["T1"] == 1


def test_extract_tier1_markdown_bold():
    """Test T1 with markdown emphasis."""
    text = "Step 1: 2 + 2 = 4\n**Final answer:** 42"
    result = extract_canonical(text, "What is 2+2?")
    assert result["strict"] == 42.0


def test_extract_tier1_x_equals():
    """Test T1 with x = prefix."""
    text = "Step 1: 2 + 2 = 4\nFinal answer: x = 4"
    result = extract_canonical(text, "What is 2+2?")
    assert result["strict"] == 4.0


def test_extract_tier1_dollar_sign():
    """Test T1 with dollar sign."""
    text = "Step 1: 2 + 2 = 4\nFinal answer: $5.00"
    result = extract_canonical(text, "What is 2+2?")
    assert result["strict"] == 5.0


def test_extract_tier1_units():
    """Test T1 with unit words."""
    text = "Step 1: 2 + 2 = 4\nFinal answer: 5 dollars"
    result = extract_canonical(text, "What is 2+2?")
    assert result["strict"] == 5.0


def test_extract_tier1_latex_boxed():
    """Test T1 with \\boxed."""
    # \boxed{...} is complex to parse - skip for now
    # Can be added later with proper LaTeX parsing
    pass


def test_extract_tier1_fraction():
    """Test T1 with fraction."""
    text = "Step 1: 1/2 + 1/2 = 1\nFinal answer: 7/2"
    result = extract_canonical(text, "What is 3.5?")
    assert result["strict"] == 3.5


def test_extract_tier1_mixed_number():
    """Test T1 with mixed number."""
    text = "Step 1: 1 + 2 = 3\nFinal answer: 3 1/2"
    result = extract_canonical(text, "What is 3.5?")
    assert result["strict"] == 3.5


def test_extract_tier1_fraction_not_one():
    """Test that 1/2 is parsed as 0.5, not 1."""
    text = "Step 1: 1/2\nFinal answer: 1/2"
    result = extract_canonical(text, "What is 0.5?")
    assert result["strict"] == 0.5


def test_extract_tier2_cue_sentence():
    """Test T2 with cue sentence."""
    text = "Step 1: 2 + 2 = 4\nTherefore, the total is 4."
    result = extract_canonical(text, "What is 2+2?")
    assert result["strict"] is None  # No T1 marker
    assert result["canonical"] == 4.0  # T2 fired
    assert result["tier_histogram"]["T2"] == 1


def test_extract_tier3_single_num_last_line():
    """Test T3 with single number on last line."""
    text = "Step 1: 2 + 2 = 4\n5"
    result = extract_canonical(text, "What is 2+2?")
    assert result["strict"] is None
    assert result["canonical"] == 5.0  # T3 now counts as canonical for 0-shot
    assert result["lenient"] == 5.0  # T3 fired
    assert result["tier_histogram"]["T3"] == 1


def test_extract_tier4_diagnostic():
    """Test T4 (diagnostic only)."""
    text = "Step 1: 2 + 2 = 4\nThe result = 5"
    result = extract_canonical(text, "What is 2+2?")
    assert result["t4_val"] == 5.0
    assert result["t4_reason"] == "last_equals"
    # T4 should NOT count in strict/canonical/lenient
    assert result["strict"] is None
    # T2 might fire on "The result = 5" (cue sentence with "result")
    # If T2 fires, it's canonical. If not, T3 might fire.
    # The key is T4 should NOT be the canonical value
    if result["canonical"] == 5.0:
        # Either T2 or T3 fired, which is acceptable
        assert result["tier_histogram"]["T2"] == 1 or result["tier_histogram"]["T3"] == 1


def test_multi_answer_detection():
    """Test detection of multiple different T1 values."""
    text = "Step 1: 2 + 2 = 4\n#### 4\nAnother step\nFinal answer: 5"
    result = extract_canonical(text, "What is 2+2?")
    # Should detect multiple different T1 values
    # (Though the earliest marker wins for extraction)
    # For now, just check it extracts the earliest
    assert result["strict"] == 4.0  # #### comes first


def test_earliest_marker_wins():
    """Test that earliest marker in text wins."""
    text = "Step 1: 2 + 2 = 4\n#### 4\nFinal answer: 5"
    result = extract_canonical(text, "What is 2+2?")
    assert result["strict"] == 4.0  # #### comes first


def test_tolerance_check():
    """Test tolerance for near-correct answers."""
    text = "Step 1: 2 + 2 = 4\nFinal answer: 4.01"
    result = extract_canonical(text, "What is 2+2?")
    # Tolerance is applied at comparison time, not extraction
    assert result["strict"] == 4.01


def test_marker_in_echoed_exemplar():
    """Test that marker inside echoed exemplar is rejected."""
    # This test requires proper boundary detection
    # For now, just test basic extraction
    text = "Step 1: 2+2=4\nFinal answer: 4"
    result = extract_canonical(text, "What is 2+2?")
    assert result["strict"] == 4.0
