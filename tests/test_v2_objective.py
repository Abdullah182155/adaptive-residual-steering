"""Unit tests for ARS Phase 1 V2 typed reasoning objective, stopping criteria, and chain audit."""

from types import SimpleNamespace
import torch
import pytest

from src.data.dataset import annotate_char_types, token_types_from_offsets
from src.evaluation.benchmark import AnswerStop, STOP_RE
from src.evaluation.chain import (
    audit_chain,
    classify_outcome,
    gold_step_recall,
    gold_step_values,
    holm,
    is_numeric_prefix,
    mcnemar_one_sided,
    paired_bootstrap_ci,
)
from src.training.losses import (
    T_FINAL,
    T_FORMAT,
    T_NONE,
    T_OPERAND,
    T_OPERATOR,
    T_PROSE,
    T_RESULT,
    typed_objective,
    window_counts,
)


def test_step_index_is_format_not_arithmetic():
    """Verify that 'Step 1:' step index is tagged as T_FORMAT, NOT arithmetic."""
    text = "Q\nA\nStep 1: 48 / 2 = 24 clips.\nFinal answer: 24\n"
    start = text.index("Step")
    lab = annotate_char_types(text, start)
    # Step index '1:' must be tagged as T_FORMAT
    assert lab[text.index("1:")] == T_FORMAT
    # Result numeral '24' after '=' must be tagged as T_RESULT
    assert lab[text.index("24 clips")] == T_RESULT
    # Final answer '24' must be tagged as T_FINAL
    assert lab[text.rindex("24")] == T_FINAL
    # Operand '48' must be tagged as T_OPERAND
    assert lab[text.index("48")] == T_OPERAND
    # Operator '/' must be tagged as T_OPERATOR
    assert lab[text.index("/")] == T_OPERATOR


def test_loss_is_bounded_and_finite():
    """Verify that typed loss gradients are bounded and finite."""
    cfg = SimpleNamespace(
        focal_gamma=1.0,
        kl_weight=0.0,
        lambda_format=0.05,
        lambda_operator=0.10,
        lambda_operand=0.20,
        lambda_result=0.35,
        lambda_final=0.30,
    )
    V, T = 50, 6
    logits = torch.randn(1, T, V, requires_grad=True)
    labels = torch.randint(0, V, (1, T))
    types = torch.full((1, T), T_RESULT, dtype=torch.long)
    counts = window_counts([labels], [types])

    loss, parts = typed_objective(logits, None, labels, types, cfg, counts)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    # Upper bound on gradient norm is sqrt(2) * sum(lambdas) = sqrt(2) * 1.0 ~ 1.414
    assert logits.grad.norm().item() < (2**0.5) * 1.05


def test_answer_stop_does_not_truncate_complete_numbers():
    """Verify that AnswerStop does not prematurely fire on the first digit of multi-digit numbers."""
    class DummyTokenizer:
        def __init__(self):
            self.text = ""

        def decode(self, ids, skip_special_tokens=True):
            return self.text

    tok = DummyTokenizer()
    stop = AnswerStop(tok, prompt_length=5)
    dummy_ids = torch.zeros((1, 10), dtype=torch.long)

    # Incomplete single digit of multi-digit number without newline: MUST NOT stop
    tok.text = "Step 1: work\nFinal answer: 1"
    assert stop(dummy_ids, None) is False
    assert stop.reason is None

    # Multi-token comma-separated number without newline: MUST NOT stop
    tok.text = "Step 1: work\nFinal answer: 1,250"
    assert stop(dummy_ids, None) is False
    assert stop.reason is None

    # Complete comma-separated number terminated by newline: MUST stop
    tok.text = "Step 1: work\nFinal answer: 1,250\n"
    assert stop(dummy_ids, None) is True
    assert stop.reason == "answer_stop"

    # Complete currency number terminated by newline
    tok.text = "Step 1: work\nFinal answer: $1,250\n"
    assert stop(dummy_ids, None) is True
    assert stop.reason == "answer_stop"

    # Complete #### format terminated by newline
    tok.text = "Step 1: work\n#### 1,250\n"
    assert stop(dummy_ids, None) is True
    assert stop.reason == "answer_stop"


def test_chain_audit_arithmetic_verifier():
    """Verify that audit_chain accurately separates correct and incorrect equations."""
    chain = (
        "Step 1: 45 * 8 = 360 bottles.\n"
        "Step 2: 360 - 120 = 240 bottles.\n"
        "Step 3: 240 / 20 = 12 cartons.\n"
        "Final answer: 12\n"
    )
    res = audit_chain(chain, question="A machine fills 45 bottles...")
    assert res["total_equations"] == 3
    assert res["correct_equations"] == 3
    assert res["car"] == 1.0
    assert res["eaa"] == 1.0
    assert res["first_bad_equation_idx"] is None

    # Chain with an arithmetic drift
    bad_chain = (
        "Step 1: 45 * 8 = 360.\n"
        "Step 2: 360 - 120 = 200.\n"  # 360 - 120 is 240, not 200
        "Step 3: 200 / 20 = 10.\n"
    )
    bad_res = audit_chain(bad_chain)
    assert bad_res["total_equations"] == 3
    assert bad_res["correct_equations"] == 2
    assert bad_res["car"] == 0.0
    assert abs(bad_res["eaa"] - 2 / 3) < 1e-4
    assert bad_res["first_bad_equation_idx"] == 1


def test_classify_outcome_taxonomy():
    """Verify that classify_outcome assigns mutually exclusive failure classes in correct precedence."""
    # TRUNC_NUMBER
    trunc_res = {"text": "Final answer: 1\n", "pred_marker": 1.0, "stop_reason": "answer_stop"}
    assert classify_outcome(trunc_res, 1250.0) == "TRUNC_NUMBER"

    # LOOP_OR_RUNON
    loop_res = {"text": "Step 1: repeating...", "pred_marker": 42.0, "stop_reason": "max_new_tokens"}
    assert classify_outcome(loop_res, 100.0) == "LOOP_OR_RUNON"

    # NO_ANSWER
    no_ans_res = {"text": "I do not know.", "pred_marker": None, "stop_reason": "eos"}
    assert classify_outcome(no_ans_res, 10.0) == "NO_ANSWER"

    # EXTRACTION: intermediate equation reaches 50, but final is 20
    ext_res = {
        "text": "Step 1: 25 * 2 = 50.\nStep 2: 50 - 30 = 20.\nFinal answer: 20",
        "pred_marker": 20.0,
        "stop_reason": "answer_stop",
    }
    assert classify_outcome(ext_res, 50.0) == "EXTRACTION"

    # ARITHMETIC error
    arith_res = {
        "text": "Step 1: 10 + 10 = 25.\nFinal answer: 25",
        "pred_marker": 25.0,
        "stop_reason": "answer_stop",
    }
    assert classify_outcome(arith_res, 20.0) == "ARITHMETIC"


def test_paired_bootstrap_ci_and_mcnemar():
    """Verify paired bootstrap confidence intervals and McNemar test."""
    b_flags = [True] * 40 + [False] * 60
    s_flags = [True] * 60 + [False] * 40
    delta, lo, hi = paired_bootstrap_ci(b_flags, s_flags, b=1000, seed=42)
    assert abs(delta - 20.0) < 1e-4
    assert lo < 20.0 < hi

    # One-sided McNemar
    p = mcnemar_one_sided(wins=25, losses=5)
    assert p < 0.001
    assert mcnemar_one_sided(wins=5, losses=25) > 0.999
