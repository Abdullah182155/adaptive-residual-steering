"""Audit an ARS checkpoint before using it for benchmark claims.

The audit has two layers:

1. It verifies every ARS tensor in the checkpoint has an exact destination
   (name and shape) in a freshly constructed wrapper.  Unlike
   ``load_state_dict(..., strict=False)``, an ARS mismatch is a hard failure.
2. It runs a deterministic synthetic causal model through the real wrapper
   hooks and proves that the loaded checkpoint changes logits when steering is
   explicitly enabled.  This validates checkpoint application and hook wiring
   without requiring a local Phi-2 download.

Optionally pass ``--model-name microsoft/phi-2`` on a machine where the model
is available to repeat the forward-difference proof on the actual backbone.
The synthetic proof establishes integration correctness, not benchmark quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import torch
import torch.nn as nn

from src.configs.config import RSCConfig
from src.models.wrapper import Phi2WithRSC


ARS_PREFIXES = ("gates.", "steer_nets.", "router.")


def is_ars_key(key: str) -> bool:
    """Return whether a state-dict key belongs to an ARS trainable component."""
    return key.startswith(ARS_PREFIXES)


def load_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    """Load a checkpoint payload and normalize legacy ``steer_state`` wrapping."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "steer_state" in payload:
        payload = payload["steer_state"]
    if not isinstance(payload, dict) or not all(torch.is_tensor(v) for v in payload.values()):
        raise TypeError("Checkpoint must be a tensor state dictionary or contain a tensor 'steer_state' dictionary.")
    return payload


def tensor_inventory(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Calculate a finite, per-tensor norm inventory for all saved ARS tensors."""
    entries = []
    total_numel = 0
    for name, value in sorted(state.items()):
        if not is_ars_key(name):
            continue
        value32 = value.detach().float()
        total_numel += value.numel()
        entries.append(
            {
                "name": name,
                "shape": list(value.shape),
                "numel": value.numel(),
                "l2_norm": float(value32.norm().item()),
                "abs_max": float(value32.abs().max().item()) if value.numel() else 0.0,
                "finite": bool(torch.isfinite(value32).all().item()),
            }
        )
    if not entries:
        raise ValueError("Checkpoint contains no ARS tensors (expected gates.*, steer_nets.*, or router.*).")
    return {"tensor_count": len(entries), "total_numel": total_numel, "tensors": entries}


class _IdentityBlock(nn.Module):
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return hidden_states


class SyntheticCausalLM(nn.Module):
    """Minimal causal-LM-shaped backbone that exercises the production hooks."""

    def __init__(self, hidden_size: int = 2560, num_hidden_layers: int = 32, vocab_size: int = 97):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_IdentityBlock() for _ in range(num_hidden_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        hidden_states = self.embed(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return SimpleNamespace(logits=self.lm_head(hidden_states))


def verify_ars_load(model: Phi2WithRSC, state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Load checkpoint and fail if any saved ARS tensor cannot be applied exactly."""
    expected = {name: tensor for name, tensor in model.state_dict().items() if is_ars_key(name)}
    saved = {name: tensor for name, tensor in state.items() if is_ars_key(name)}
    missing = sorted(set(expected) - set(saved))
    unexpected = sorted(set(saved) - set(expected))
    shape_mismatches = {
        name: {"checkpoint": list(saved[name].shape), "model": list(expected[name].shape)}
        for name in sorted(set(expected) & set(saved))
        if tuple(saved[name].shape) != tuple(expected[name].shape)
    }
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "ARS checkpoint incompatibility: "
            + json.dumps(
                {"missing_ars_keys": missing, "unexpected_ars_keys": unexpected, "shape_mismatches": shape_mismatches},
                indent=2,
            )
        )
    incompatible = model.load_state_dict(saved, strict=False)
    missing_after_load = sorted(key for key in incompatible.missing_keys if is_ars_key(key))
    unexpected_after_load = sorted(key for key in incompatible.unexpected_keys if is_ars_key(key))
    if missing_after_load or unexpected_after_load:
        raise RuntimeError(
            f"PyTorch did not apply all ARS tensors: missing={missing_after_load}, unexpected={unexpected_after_load}"
        )
    return {
        "saved_ars_tensors": len(saved),
        "model_ars_tensors": len(expected),
        "non_ars_missing_keys": len(incompatible.missing_keys),
    }


@torch.no_grad()
def forward_difference_proof(model: Phi2WithRSC, input_ids: torch.Tensor) -> Dict[str, Any]:
    """Prove that the loaded SteerNet path changes logits under a fixed intervention."""
    model.eval()
    attention_mask = torch.ones_like(input_ids)
    model.set_routing_override("none")
    model.set_gate_freeze(False)
    baseline = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()

    model.set_routing_override("only", model.target_layers)
    model.set_gate_freeze(True, 1.0)
    steered = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    difference = steered - baseline

    model.set_routing_override(None)
    model.set_gate_freeze(False)
    return {
        "baseline_l2": float(baseline.norm().item()),
        "steered_l2": float(steered.norm().item()),
        "difference_l2": float(difference.norm().item()),
        "difference_abs_max": float(difference.abs().max().item()),
        "logits_changed": bool(not torch.equal(baseline, steered)),
    }


def synthetic_audit(state: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, Any]:
    """Run exact-load and hook-difference checks with Phi-2-compatible dimensions."""
    torch.manual_seed(20261012)
    backbone = SyntheticCausalLM().to(device)
    wrapper = Phi2WithRSC(backbone, RSCConfig(), device=device).to(device)
    load_result = verify_ars_load(wrapper, state)
    probe = torch.tensor([[1, 7, 13, 29, 31, 43, 47, 53]], device=device)
    proof = forward_difference_proof(wrapper, probe)
    if not proof["logits_changed"] or proof["difference_l2"] <= 0.0:
        raise RuntimeError("Loaded ARS hooks produced exactly the baseline logits; the intervention path is inactive.")
    return {"load": load_result, "forward_proof": proof}


def actual_model_audit(state: Dict[str, torch.Tensor], model_name: str, device: torch.device) -> Dict[str, Any]:
    """Repeat exact-load and fixed-intervention proof on a locally available HF model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    wrapper = Phi2WithRSC(backbone, RSCConfig(), device=device).to(device)
    load_result = verify_ars_load(wrapper, state)
    probe = tokenizer("ARS checkpoint hook audit.", return_tensors="pt").input_ids.to(device)
    proof = forward_difference_proof(wrapper, probe)
    if not proof["logits_changed"] or proof["difference_l2"] <= 0.0:
        raise RuntimeError("Actual-backbone proof failed: the enabled intervention left logits unchanged.")
    return {"load": load_result, "forward_proof": proof}


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict ARS checkpoint and hook audit")
    parser.add_argument("checkpoint", type=Path, help="Path to final_ars_weights.pt or an equivalent ARS state dict")
    parser.add_argument("--model-name", help="Optional Hugging Face model id/path for an actual-backbone forward proof")
    parser.add_argument("--device", default="cpu", help="Torch device for audit (default: cpu)")
    parser.add_argument("--report", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device(args.device)
    state = load_checkpoint(args.checkpoint)
    report: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "inventory": tensor_inventory(state),
    }
    exit_code = 0
    try:
        report["synthetic_phi2_shape_audit"] = synthetic_audit(state, device)
    except RuntimeError as exc:
        # Emit the complete inventory and incompatibility report before failing.
        # A benchmark notebook must never turn this condition into random-module
        # evaluation by continuing after a permissive checkpoint load.
        report["synthetic_phi2_shape_audit"] = {"passed": False, "error": str(exc)}
        exit_code = 2
    if args.model_name and exit_code == 0:
        report["actual_backbone_audit"] = actual_model_audit(state, args.model_name, device)

    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
