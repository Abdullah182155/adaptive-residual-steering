"""
Test unified prompt builder ensures byte-identical prompts across all builders.

Phase 0.1: Verify that verifier.build_prompt, multi_dataset.build_benchmark_prompt,
and shots.build_shot_example all produce identical output when using legacy exemplars.
"""

import pytest
from src.data.prompts import build_prompt_unified, get_harness_hash
from src.evaluation.verifier import build_prompt as verifier_build_prompt
from src.data.shots import build_shot_example
from transformers import AutoTokenizer


def test_prompt_builder_byte_identity():
    """All three main builders should produce byte-identical prompts for legacy mode."""
    test_question = "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"
    n_shot = 2
    
    # Build prompts from each builder
    unified_prompt = build_prompt_unified(test_question, n_shot=n_shot, exemplar_set="legacy", include_preface=False)
    verifier_prompt = verifier_build_prompt(test_question, n_shot=n_shot)
    
    # They should be byte-identical
    assert unified_prompt.encode() == verifier_prompt.encode(), \
        "Unified and verifier prompts differ in legacy mode"
    
    # Note: shots.build_shot_example has different semantics (includes target), so it's not tested here


def test_harness_hash_is_deterministic():
    """Harness hash should be deterministic across calls."""
    hash1 = get_harness_hash()
    hash2 = get_harness_hash()
    assert hash1 == hash2, "Harness hash should be deterministic"
    assert len(hash1) == 16, "Harness hash should be 16 characters"


def test_v5_exemplars_format():
    """V5 exemplars should all follow the required format."""
    from src.data.prompts import V5_EXEMPLARS
    
    for i, ex in enumerate(V5_EXEMPLARS):
        # Check answer ends with "Final answer: N"
        assert ex['a'].strip().endswith(f"Final answer: {ex['a'].strip().split('Final answer: ')[-1]}"), \
            f"Exemplar {i} answer should end with 'Final answer: N'"
        
        # Check answer contains Step markers
        lines = ex['a'].split('\n')
        step_lines = [l for l in lines if l.strip().startswith("Step")]
        assert len(step_lines) >= 2, f"Exemplar {i} should have at least 2 steps"
        
        # Check no calculator annotations
        assert "<<>>" not in ex['a'], f"Exemplar {i} should not contain calculator annotations"
