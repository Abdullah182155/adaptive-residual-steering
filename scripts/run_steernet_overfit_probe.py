"""Run the acceptance probe that Phase 1 must pass before ARS training.

This is intentionally a small, reproducible memorisation experiment, not a
benchmark.  A frozen Phi-2 plus SteerNet must materially improve both labelled
completion loss and answer accuracy on a fixed training subset before a router
or gate is allowed to claim it can improve multi-step reasoning.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Support both ``python scripts/...`` and ``python -m`` invocation on Kaggle.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.configs.config import RSCConfig
from src.data.dataset import gsm8k_to_cot
from src.evaluation.verifier import epoch_end_accuracy
from src.models.wrapper import Phi2WithRSC
from src.training.trainer import evaluate_loss, train_phase1_steernet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def score(model, tokenizer, data, cfg, device, route: str, k: int | None = None) -> dict:
    if route == "baseline":
        model.set_routing_override("none")
        model.set_gate_freeze(False)
    elif route == "fixed":
        model.cfg.fixed_steer_eval_k = int(k)
        model.set_routing_override("balanced")
        model.set_gate_freeze(True, 1.0)
    else:
        raise ValueError(route)
    model.eval()
    return {
        "completion_nll": evaluate_loss(model, data, cfg, device),
        "answer_accuracy": epoch_end_accuracy(
            model, tokenizer, data, device, n_samples=len(data), seed=cfg.seed
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-1 SteerNet overfit acceptance probe")
    parser.add_argument("--model-name", default="microsoft/phi-2")
    parser.add_argument("--output-dir", type=Path, default=Path("./probe_runs/phase1_overfit"))
    parser.add_argument("--train-examples", type=int, default=96)
    parser.add_argument("--validation-examples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--min-relative-nll-reduction", type=float, default=0.05)
    parser.add_argument("--min-accuracy-gain", type=float, default=0.05)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, torch_dtype=torch_dtype
    ).to(device)

    total = args.train_examples + args.validation_examples
    cfg = RSCConfig(
        n_samples=total,
        seed=args.seed,
        phase1_epochs=args.epochs,
        phase1_lr=args.lr,
        phase1_patience=args.epochs + 1,
        accuracy_eval_n=args.validation_examples,
        max_tok_len=384,
        # The probe isolates steering capacity.  Few-shot robustness is tested
        # only after this acceptance gate passes.
        fewshot_train_prob=0.0,
        lora_save_path=str(args.output_dir / "phase1_steernet.pt"),
        checkpoint_dir=str(args.output_dir / "checkpoints"),
    )
    train_data, val_data = gsm8k_to_cot(
        tokenizer, cfg=cfg, n_samples=total, val_split=args.validation_examples / total, seed=args.seed
    )
    wrapper = Phi2WithRSC(backbone, cfg, device=device).to(device)

    before = {"train": score(wrapper, tokenizer, train_data, cfg, device, "baseline"),
              "validation": score(wrapper, tokenizer, val_data, cfg, device, "baseline")}
    _, best_val, phase_logs = train_phase1_steernet(wrapper, train_data, val_data, cfg, device)
    after = {
        "train": {str(k): score(wrapper, tokenizer, train_data, cfg, device, "fixed", k) for k in range(1, 5)},
        "validation": {str(k): score(wrapper, tokenizer, val_data, cfg, device, "fixed", k) for k in range(1, 5)},
    }

    best_train_nll = min(value["completion_nll"] for value in after["train"].values())
    best_train_acc = max(value["answer_accuracy"] for value in after["train"].values())
    nll_reduction = (before["train"]["completion_nll"] - best_train_nll) / max(
        before["train"]["completion_nll"], 1e-8
    )
    accuracy_gain = best_train_acc - before["train"]["answer_accuracy"]
    acceptance = {
        "relative_nll_reduction": nll_reduction,
        "accuracy_gain": accuracy_gain,
        "min_relative_nll_reduction": args.min_relative_nll_reduction,
        "min_accuracy_gain": args.min_accuracy_gain,
        "passed": nll_reduction >= args.min_relative_nll_reduction
        and accuracy_gain >= args.min_accuracy_gain,
    }
    report = {
        "purpose": "Phase-1 steering capacity acceptance probe",
        "config": vars(cfg),
        "baseline": before,
        "phase1_best_validation_nll": best_val,
        "phase1_logs": phase_logs,
        "fixed_route_scores": after,
        "acceptance": acceptance,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"Saved probe report: {report_path}")
    return 0 if acceptance["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
