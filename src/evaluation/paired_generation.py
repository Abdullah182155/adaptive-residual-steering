"""Batch greedy generation and paired evaluation utilities (M4+)."""
from __future__ import annotations
import re
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Sequence
import torch
from torch import Tensor
from transformers import StoppingCriteriaList

from src.data.onpolicy import BatchAnswerStopping
from src.evaluation.answers import cut_continuation, extract_numeric_answer  # noqa: F401
from src.data.prompts import build_prompt_unified


def generate_batch(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: torch.device,
    *,
    max_new_tokens: int = 512,
    question_span_mask: Optional[Tensor] = None,
) -> List[str]:
    """Greedy batch generation with same settings as generate_response:
    repetition_penalty=1.1, no_repeat_ngram_size=6, use_cache=True.
    Left padding. BatchAnswerStopping. Post-trims identical to generate_response regex truncation.

    question_span_mask (optional): (B, T_padded) bool mask of router-context tokens. It is
    forwarded to model.set_router_context_mask only if the model supports it and the shape
    matches the left-padded batch; otherwise it is ignored.
    """
    prompts = list(prompts)
    if not prompts:
        return []

    model.eval()
    old_side = getattr(tokenizer, "padding_side", "left")
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(device)
    finally:
        tokenizer.padding_side = old_side

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    prompt_len = int(input_ids.shape[1])

    if (
        question_span_mask is not None
        and hasattr(model, "set_router_context_mask")
        and tuple(question_span_mask.shape) == tuple(input_ids.shape)
    ):
        model.set_router_context_mask(question_span_mask.to(device=device, dtype=torch.bool))

    stopping = StoppingCriteriaList([BatchAnswerStopping(tokenizer, prompt_len)])
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
            do_sample=False,
            use_cache=True,
        )

    outputs: List[str] = []
    for row in output_ids:
        new_ids = row[prompt_len:]
        raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        # Phase 0.3: Use cut_continuation instead of _TRIM_PATTERNS
        outputs.append(cut_continuation(raw_text))
    return outputs


def _restore_override(model, old) -> None:
    if old is None:
        model.set_routing_override(None)
    else:
        model.set_routing_override(old[0], old[1])


def paired_greedy_eval(
    model,
    tokenizer,
    questions: Sequence[str],
    golds: Sequence[float],
    configure_a: Callable,
    configure_b: Callable,
    device: torch.device,
    *,
    batch_size: int = 8,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Evaluate two model configurations (A vs B) on the same questions.

    configure_a(model) / configure_b(model) are callables that set routing overrides
    (e.g. lambda m: m.set_routing_override("none", None)).

    Returns:
        acc_a, acc_b: float (accuracy 0..1)
        b: int, rows where B correct and A wrong
        c: int, rows where A correct and B wrong
        delta: float, acc_b - acc_a
        per_item: List[dict] with keys: question, gold, pred_a, pred_b, correct_a, correct_b
    """
    questions = list(questions)
    golds = list(golds)
    n = len(questions)
    prompts = [build_prompt_unified(q, n_shot=0, exemplar_set="legacy", include_preface=False) for q in questions]

    old_override = getattr(model, "_routing_override", None)
    ctx = model.counterfactual_mode() if hasattr(model, "counterfactual_mode") else nullcontext()

    def _run(configure) -> List[Optional[float]]:
        configure(model)
        preds: List[Optional[float]] = []
        for start in range(0, n, batch_size):
            texts = generate_batch(
                model, tokenizer, prompts[start : start + batch_size], device,
                max_new_tokens=max_new_tokens,
            )
            preds.extend(extract_numeric_answer(t) for t in texts)
        return preds

    try:
        with ctx:
            preds_a = _run(configure_a)
            preds_b = _run(configure_b)
    finally:
        _restore_override(model, old_override)

    def _ok(pred, gold) -> bool:
        return pred is not None and gold is not None and abs(float(pred) - float(gold)) <= 0.01

    per_item: List[Dict[str, Any]] = []
    n_a = n_b = b = c = 0
    for q, g, pa, pb in zip(questions, golds, preds_a, preds_b):
        ca, cb = _ok(pa, g), _ok(pb, g)
        n_a += int(ca)
        n_b += int(cb)
        b += int(cb and not ca)
        c += int(ca and not cb)
        per_item.append(
            {"question": q, "gold": g, "pred_a": pa, "pred_b": pb, "correct_a": ca, "correct_b": cb}
        )

    acc_a = n_a / max(1, n)
    acc_b = n_b / max(1, n)
    return {
        "acc_a": acc_a,
        "acc_b": acc_b,
        "b": b,
        "c": c,
        "delta": acc_b - acc_a,
        "per_item": per_item,
    }
