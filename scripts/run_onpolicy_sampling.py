"""CLI script to run on-policy reasoning trace sampling (M3).

Usage:
    python scripts/run_onpolicy_sampling.py [--ckpt CKPT] [--out_path PATH]
                                            [--n_questions N] [--batch_size B] [--seed S]
"""
import argparse
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.configs.config import RSCConfig
from src.data.dataset import gsm8k_to_cot
from src.data.onpolicy import sample_chains
from src.models.wrapper import Phi2WithRSC
from src.utils.failure_policy import FatalContractError, get_manifest

try:
    from src.training.trainer import load_checkpoint
except ImportError:  # fall back to the weight loader present in trainer.py
    from src.training.trainer import load_steer_weights as load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="On-policy reasoning trace sampler (M3)")
    p.add_argument("--ckpt", type=str, default=None, help="Optional steering checkpoint to load")
    p.add_argument("--out_path", type=str, default="./checkpoints/onpolicy_store.jsonl")
    p.add_argument("--n_questions", type=int, default=None, help="Default: cfg.onpolicy_n_questions")
    p.add_argument("--batch_size", type=int, default=None, help="Default: cfg.onpolicy_batch_size")
    p.add_argument("--seed", type=int, default=None, help="Default: cfg.seed")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RSCConfig()
    if args.n_questions is not None:
        cfg.onpolicy_n_questions = args.n_questions
    if args.batch_size is not None:
        cfg.onpolicy_batch_size = args.batch_size
    seed = cfg.seed if args.seed is None else args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = get_manifest()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model = Phi2WithRSC(base, cfg, device=device)
    if args.ckpt and os.path.exists(args.ckpt):
        load_checkpoint(model, args.ckpt, device=device)

    train_data, _ = gsm8k_to_cot(tokenizer, cfg)
    # train_data is curriculum-sorted by difficulty, so draw a random subset, not the head.
    n = min(cfg.onpolicy_n_questions, len(train_data))
    idxs = random.Random(seed).sample(range(len(train_data)), n)
    questions = [
        {"qid": i, "question": train_data[i]["question"], "final_num": train_data[i]["final_num"]}
        for i in idxs
    ]

    store = sample_chains(model, tokenizer, questions, cfg, device, seed=seed, out_path=args.out_path)
    stats = store.stats()
    manifest.advisory("onpolicy_stats", stats, "Summary of sampled on-policy chains")
    print(f"[onpolicy] wrote {args.out_path}: {stats}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FatalContractError as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(2)
