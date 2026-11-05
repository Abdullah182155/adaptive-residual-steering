"""Tests for OOD dataset loading and MCQ answer extraction."""

from src.evaluation.ood import extract_mcq_answer, format_mcq_question, SPECS


def test_specs_registry():
    assert "svamp" in SPECS
    assert "math500" in SPECS
    assert "arc" in SPECS
    assert "aqua_rat" in SPECS
    assert "gsm_plus" in SPECS
    for name, (hf_id, count, atype) in SPECS.items():
        assert atype in ("numeric", "mcq"), f"{name} has invalid type {atype}"
        assert count > 0


def test_extract_mcq_answer():
    assert extract_mcq_answer("The answer is (B)") == "B"
    assert extract_mcq_answer("Answer: C") == "C"
    assert extract_mcq_answer("The correct choice is D.") == "D"
    assert extract_mcq_answer("A) 125 is correct") == "A"
    assert extract_mcq_answer("(E)") == "E"
    assert extract_mcq_answer("no letter here 123") is None


def test_format_mcq_question():
    q = "What is 2+2?"
    choices = ["(A) 3", "(B) 4", "(C) 5", "(D) 6"]
    result = format_mcq_question(q, choices)
    assert "What is 2+2?" in result
    assert "(A) 3" in result
    assert "(D) 6" in result
