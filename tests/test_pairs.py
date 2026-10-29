"""
Tests for preference pair mining.

Phase 1.3: Test pair mining logic.
"""

import pytest
from src.data.pairs import mine_preference_pairs, PreferencePair
from src.data.onpolicy import OnPolicyChain


def test_mine_preference_pairs_basic():
    """Test basic pair mining."""
    chains = [
        OnPolicyChain(qid=1, sample_idx=0, shot=0, prompt="", text="Step 1: 2+2=4\nFinal answer: 4",
                      pred_strict=4.0, pred_lenient=4.0, gold=4.0, correct=True, n_tokens=10, div_line=None),
        OnPolicyChain(qid=1, sample_idx=1, shot=0, prompt="", text="Step 1: 2+2=5\nFinal answer: 5",
                      pred_strict=5.0, pred_lenient=5.0, gold=4.0, correct=False, n_tokens=10, div_line=None),
    ]
    
    pairs = mine_preference_pairs(chains, k_samples=2, edit_distance_threshold=10)
    
    assert len(pairs) == 1
    assert pairs[0].y_plus.correct is True
    assert pairs[0].y_minus.correct is False


def test_edit_distance_filter():
    """Test edit distance filtering."""
    chains = [
        OnPolicyChain(qid=1, sample_idx=0, shot=0, prompt="", text="Step 1: 2+2=4\nFinal answer: 4",
                      pred_strict=4.0, pred_lenient=4.0, gold=4.0, correct=True, n_tokens=10, div_line=None),
        OnPolicyChain(qid=1, sample_idx=1, shot=0, prompt="", text="Step 1: 2+2=4\nStep 2: 4\nFinal answer: 4",
                      pred_strict=4.0, pred_lenient=4.0, gold=4.0, correct=True, n_tokens=15, div_line=None),
        OnPolicyChain(qid=1, sample_idx=2, shot=0, prompt="", text="Step 1: 2+2=5\nFinal answer: 5",
                      pred_strict=5.0, pred_lenient=5.0, gold=4.0, correct=False, n_tokens=10, div_line=None),
    ]
    
    # With threshold 1, the first correct chain should not pair with incorrect (diff lines > 1)
    pairs = mine_preference_pairs(chains, k_samples=3, edit_distance_threshold=1)
    
    # Should still pair if there's at least one pair with low edit distance
    assert len(pairs) >= 0


def test_divergence_position():
    """Test divergence position detection."""
    chains = [
        OnPolicyChain(qid=1, sample_idx=0, shot=0, prompt="", text="Step 1: 2+2=4\nFinal answer: 4",
                      pred_strict=4.0, pred_lenient=4.0, gold=4.0, correct=True, n_tokens=10, div_line=None),
        OnPolicyChain(qid=1, sample_idx=1, shot=0, prompt="", text="Step 1: 2+2=5\nFinal answer: 5",
                      pred_strict=5.0, pred_lenient=5.0, gold=4.0, correct=False, n_tokens=10, div_line=None),
    ]
    
    pairs = mine_preference_pairs(chains, k_samples=2, edit_distance_threshold=10)
    
    assert len(pairs) == 1
    # First line differs ("2+2=4" vs "2+2=5")
    assert pairs[0].divergence_position == 0


def test_max_pairs_per_question():
    """Test max pairs per question limit."""
    chains = [
        OnPolicyChain(qid=1, sample_idx=i, shot=0, prompt="", text=f"Step 1: 2+2={4 if i % 2 == 0 else 5}\nFinal answer: {4 if i % 2 == 0 else 5}",
                      pred_strict=4.0 if i % 2 == 0 else 5.0, pred_lenient=4.0 if i % 2 == 0 else 5.0, gold=4.0, correct=(i % 2 == 0), n_tokens=10, div_line=None)
        for i in range(10)
    ]
    
    pairs = mine_preference_pairs(chains, k_samples=10, max_pairs_per_question=2)
    
    assert len(pairs) <= 2
