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


def test_ood_generation_delegates_to_canonical_generator(monkeypatch):
    import torch
    import src.evaluation.ood_benchmark as benchmark

    observed = {}

    def fake_generate_batch(model, tokenizer, prompts, device, **kwargs):
        observed.update({"prompts": prompts, "device": device, **kwargs})
        return [type("Result", (), {"text": "42"})() for _ in prompts]

    monkeypatch.setattr(benchmark, "generate_batch", fake_generate_batch)
    outputs = benchmark._generate_batch(object(), object(), ["a", "long prompt"], torch.device("cpu"))
    assert outputs == ["42", "42"]
    assert observed["max_prompt_len"] == 1024
    assert observed["repetition_penalty"] == 1.0
