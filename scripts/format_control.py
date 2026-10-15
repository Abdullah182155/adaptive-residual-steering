"""A-0 / P1 Format Control Experiment.

Tests whether the SteerNet adds genuine mathematical reasoning or merely
acts as a prompt format cue (e.g. forcing "Step 1:").

Compares 4 arms paired by item across >=300 GSM8K test rows:
  1. baseline_plain:       no prefix, steering disabled
  2. baseline_step_prefix: "Step 1:" prefix appended to prompt, steering disabled
  3. steernet_plain:       no prefix, single-layer SteerNet enabled (gate=1.0)
  4. steernet_step_prefix: "Step 1:" prefix, single-layer SteerNet enabled

If baseline_step_prefix ≈ steernet_plain:
  => SteerNet is acting as a format adapter. RFT targets (C-4) are necessary.
If steernet_step_prefix - baseline_step_prefix has lower CI > 0:
  => SteerNet adds genuine reasoning beyond the format cue.
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.configs.config import RSCConfig
from src.data.dataset import gsm8k_to_cot
from src.evaluation.deployment import paired_delta_ci
from src.evaluation.multi_dataset import build_benchmark_prompt, evaluate_sample_answer
from src.evaluation.verifier import generate_response
from src.models.wrapper import RSCPhi2Model


def _arm(model, tokenizer, rows, device, override, prefix: str = "", max_new_tokens: int = 384) -> np.ndarray:
    model.eval()
    mode, layers = override
    model.set_routing_override(mode, layers)
    model.set_gate_freeze(True, 1.0)
    flags = []
    for row in rows:
        prompt = build_benchmark_prompt("gsm8k", row["question"], n_shot=0)
        if prefix:
            prompt = prompt + "\n" + prefix if not prompt.endswith("\n") else prompt + prefix
        resp = generate_response(model, tokenizer, prompt, device, max_new_tokens=max_new_tokens)
        full_text = (prefix + " " + resp) if prefix else resp
        eval_result = evaluate_sample_answer("gsm8k", full_text, row["final_num"])
        flags.append(int(eval_result["correct"]))
    return np.asarray(flags)


def run_format_control(model, tokenizer, rows, device, layer: int = 17, max_new_tokens: int = 384) -> dict:
    print(f"\n========================================================")
    print(f"Format Control Experiment (P1) on n={len(rows)} samples (layer={layer})")
    print(f"========================================================")

    arms = {
        "baseline_plain": _arm(model, tokenizer, rows, device, ("none", None), max_new_tokens=max_new_tokens),
        "baseline_step_prefix": _arm(model, tokenizer, rows, device, ("none", None), prefix="Step 1:", max_new_tokens=max_new_tokens),
        "steernet_plain": _arm(model, tokenizer, rows, device, ("only", [layer]), max_new_tokens=max_new_tokens),
        "steernet_step_prefix": _arm(model, tokenizer, rows, device, ("only", [layer]), prefix="Step 1:", max_new_tokens=max_new_tokens),
    }

    print("\n--- Arm Accuracies ---")
    for name, flags in arms.items():
        print(f"  {name:22s} acc={flags.mean() * 100:5.1f}%  n={len(flags)}")

    print("\n--- Paired Bootstrap Comparisons (95% CI) ---")
    comparisons = [
        ("baseline_plain", "baseline_step_prefix", "Format effect on baseline"),
        ("baseline_step_prefix", "steernet_plain", "SteerNet vs simple format cue"),
        ("baseline_step_prefix", "steernet_step_prefix", "SteerNet gain above format cue (reasoning)"),
    ]
    for a, b, desc in comparisons:
        d, lo, hi = paired_delta_ci(arms[a], arms[b])
        sig = "*" if lo > 0 else ""
        print(f"  {a} -> {b} ({desc}):")
        print(f"    Δ={d * 100:+.1f}pp  CI95=[{lo * 100:+.1f}, {hi * 100:+.1f}]{sig}")

    model.set_routing_override(None)
    model.set_gate_freeze(False)
    return arms


def main():
    parser = argparse.ArgumentParser(description="Run ARS Format Control Experiment (P1)")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/steer_phase1_adaptive_v11.pt", help="SteerNet checkpoint path")
    parser.add_argument("--layer", type=int, default=17, help="Target layer for SteerNet evaluation")
    parser.add_argument("--n_samples", type=int, default=300, help="Number of test samples (recommended >= 300)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = RSCConfig()
    print(f"Loading tokenizer & base model {cfg.model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.clean_up_tokenization_spaces = False

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype).to(device)
    model = RSCPhi2Model(base_model, cfg)

    # Load SteerNet checkpoint
    from src.training.trainer import load_steer_weights
    load_steer_weights(model, args.checkpoint, device=device)

    # Load evaluation dataset
    from src.evaluation.multi_dataset import load_benchmark_dataset
    test_rows = load_benchmark_dataset("gsm8k", split="test", n_samples=args.n_samples)

    run_format_control(model, tokenizer, test_rows, device, layer=args.layer)


if __name__ == "__main__":
    main()
