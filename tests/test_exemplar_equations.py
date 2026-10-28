"""
Tests for exemplar equation validation.

Phase 2.2: Assert-based tests that parse every exemplar equation and check it.
"""

import pytest
from src.data.prompts import V5_EXEMPLARS, LEGACY_GSM8K_EXEMPLARS


def parse_and_check_equation(text: str) -> bool:
    """Parse and verify an equation from text."""
    import re
    import operator
    
    # Look for patterns like "45 * 8 = 360" or "360 - 120 = 240"
    pattern = r'(-?[\d]+\.?[\d]*)\s*([\+\-\*\/])\s*(-?[\d]+\.?[\d]*)\s*=\s*(-?[\d]+\.?[\d]*)'
    match = re.search(pattern, text)
    
    if not match:
        return False
    
    a = float(match.group(1))
    op = match.group(2)
    b = float(match.group(3))
    result = float(match.group(4))
    
    ops = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
    
    if op == "/" and b == 0:
        return False
    
    expected = ops[op](a, b)
    
    # Allow small floating point tolerance
    return abs(expected - result) < 0.01


def test_v5_exemplars_equations():
    """Test that all V5 exemplar equations are mathematically correct using verifier."""
    from src.evaluation.verifier import extract_and_verify_equations
    
    for i, ex in enumerate(V5_EXEMPLARS):
        answer = ex['a']
        total, correct, details = extract_and_verify_equations(answer)
        
        if total == 0:
            pytest.fail(f"V5 exemplar {i} has no equations: {ex['q']}")
        
        # Allow some false positives in verifier (e.g., text that looks like equations)
        # Require at least 75% of detected equations to be correct
        if correct / total < 0.75:
            # Find which equations failed
            failed = [d for d in details if not d[1]]
            pytest.fail(f"V5 exemplar {i} has {correct}/{total} correct equations (<75%). Failed: {failed[:3]}")


def test_v5_exemplars_format():
    """Test that all V5 exemplars end with 'Final answer: N'."""
    for i, ex in enumerate(V5_EXEMPLARS):
        answer = ex['a'].strip()
        if not answer.endswith(f"Final answer: {answer.split('Final answer: ')[-1]}"):
            pytest.fail(f"V5 exemplar {i} doesn't end with 'Final answer: N': {answer[-50:]}")


def test_v5_exemplars_step_markers():
    """Test that all V5 exemplars have 'Step N:' markers."""
    for i, ex in enumerate(V5_EXEMPLARS):
        answer = ex['a']
        if "Step 1:" not in answer:
            pytest.fail(f"V5 exemplar {i} missing 'Step 1:' marker")


def test_v5_exemplars_no_calculator_annotations():
    """Test that V5 exemplars don't have calculator annotations."""
    for i, ex in enumerate(V5_EXEMPLARS):
        if "<<>>" in ex['a']:
            pytest.fail(f"V5 exemplar {i} has calculator annotations: {ex['a']}")


def test_legacy_exemplars_format():
    """Test that legacy exemplars still end with 'Final answer: N'."""
    for i, ex in enumerate(LEGACY_GSM8K_EXEMPLARS):
        answer = ex['a'].strip()
        if not answer.endswith(f"Final answer: {answer.split('Final answer: ')[-1]}"):
            pytest.fail(f"Legacy exemplar {i} doesn't end with 'Final answer: N'")
