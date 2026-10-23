"""Contract checks for Phi2WithRSC (Task M6a).

Three checks corresponding to plan items III-1, III-2, III-3.
All raise FatalContractError on failure.
III-4 lives in M9.
"""
from __future__ import annotations
from typing import Optional, Sequence
import torch
from src.utils.failure_policy import FatalContractError


def check_baseline_identity(model, tokenizer, prompts: Sequence[str], tol: float) -> None:
    """III-1: override='none' logits must equal hooks_disabled() logits."""
    old_override = model._routing_override
    model.eval()
    tokenizer.padding_side = "right"
    inputs = tokenizer(
        list(prompts), return_tensors="pt", padding=True,
        truncation=True, max_length=256,
    )
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    try:
        with torch.no_grad():
            model.set_routing_override("none")
            out_steered = model(**inputs).logits
            with model.hooks_disabled():
                out_base = model(**inputs).logits
        if not torch.allclose(out_steered, out_base, atol=tol):
            max_err = (out_steered - out_base).abs().max().item()
            raise FatalContractError(
                "E_BASELINE_IDENTITY",
                f"override='none' logits differ from hooks_disabled() by up to {max_err:.2e} (tol={tol})",
            )
    finally:
        model._routing_override = old_override


def check_hook_noop(model, tokenizer, prompts: Sequence[str]) -> None:
    """III-2: all-False matrix => per-layer outputs equal hooks_disabled outputs."""
    old_override = model._routing_override
    model.eval()
    n = len(prompts)
    nc = len(model.target_layers)
    all_false = torch.zeros(n, nc, dtype=torch.bool)
    tokenizer.padding_side = "right"
    inputs = tokenizer(
        list(prompts), return_tensors="pt", padding=True,
        truncation=True, max_length=256,
    )
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    try:
        with torch.no_grad():
            model.set_routing_override("matrix", all_false.to(inputs["input_ids"].device))
            out_matrix = model(**inputs).logits
            with model.hooks_disabled():
                out_base = model(**inputs).logits
        if not torch.allclose(out_matrix, out_base, atol=1e-6):
            max_err = (out_matrix - out_base).abs().max().item()
            raise FatalContractError(
                "E_HOOK_CONTRACT",
                f"all-False matrix row logits differ from hooks_disabled() by up to {max_err:.2e}",
            )
    finally:
        model._routing_override = old_override


def check_route_plumbing(
    model, tokenizer, prompts: Sequence[str], forced: Sequence[Sequence[int]]
) -> None:
    """III-3: forced matrix, batch>1, left-padded, generate(max_new_tokens=4);
    every routing_step_log entry equals forced tuples for prefill AND >=1 decode step."""
    old_override = model._routing_override
    old_padding = tokenizer.padding_side
    forced_mat = model.subset_matrix(list(forced))
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        list(prompts), return_tensors="pt", padding=True,
        truncation=True, max_length=256,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.routing_step_log.clear()
    model.set_routing_trace(True)
    model.set_routing_override("matrix", forced_mat.to(device))
    try:
        model.eval()
        with torch.no_grad():
            # Run one explicit prefill forward to populate routing_step_log.
            # Then run generate (which also calls forward on real models,
            # adding decode-step entries).
            model(**inputs)
            model.generate(**inputs, max_new_tokens=4, do_sample=False)
        if not model.routing_step_log:
            raise FatalContractError(
                "E_ROUTE_PLUMBING", "routing_step_log is empty after forward+generate"
            )
        expected = [
            tuple(model.target_layers[i] for i, v in enumerate(row) if v)
            for row in forced_mat.tolist()
        ]
        for entry in model.routing_step_log:
            for row_idx, actual_route in enumerate(entry.get("routes", [])):
                if tuple(actual_route) != expected[row_idx]:
                    raise FatalContractError(
                        "E_ROUTE_PLUMBING",
                        f"Row {row_idx}: expected {expected[row_idx]}, got {tuple(actual_route)}",
                    )
    finally:
        model.set_routing_trace(False)
        model._routing_override = old_override
        tokenizer.padding_side = old_padding


def run_contract_checks(
    model, tokenizer, device, n_prompts: int = 16, tol: Optional[float] = None
) -> None:
    """Run all M6a contract checks (III-1, III-2, III-3). All raise FatalContractError on failure."""
    if tol is None:
        param = next(model.parameters())
        tol = 1e-3 if param.dtype == torch.float16 else 1e-5

    prompts = [f"What is {i} + {i}? Answer:" for i in range(n_prompts)]
    forced_subsets = [model.target_layers[:1]] * n_prompts  # single-layer forced route

    check_baseline_identity(model, tokenizer, prompts[:4], tol)
    check_hook_noop(model, tokenizer, prompts[:4])
    check_route_plumbing(model, tokenizer, prompts[:2], forced_subsets[:2])
