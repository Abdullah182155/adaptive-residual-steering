import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.configs.config import RSCConfig
from src.data.dataset import gsm8k_to_cot
from src.data.onpolicy import OnPolicyStore
from src.models.wrapper import Phi2WithRSC
from src.routing.catalog import Catalog
from src.training.trainer import (
    load_steer_weights,
    run_k_scaling_ablation,
    train_s1_steernet,
)


def parse_args():
    parser = argparse.ArgumentParser(description="ARS M4: S1' SteerNet subset trainer (on-policy + layer-dropout)")
    parser.add_argument("--ckpt", type=str, default=None, help="Phase-1 checkpoint to initialise SteerNets from")
    parser.add_argument("--onpolicy_store", type=str, required=True, help="On-policy store JSONL path (from M3)")
    parser.add_argument("--out_dir", type=str, default="./checkpoints/s1", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation", action="store_true", help="Also run the k-scaling ablation (before main training)")
    return parser.parse_args()


def dataset_to_rows(ds):
    """Convert the tokenized GSM8K dataset into unpadded row dicts."""
    rows = []
    for i in range(len(ds)):
        ex = ds[i]
        keep = ex["attention_mask"].bool()
        rows.append(
            {
                "input_ids": ex["input_ids"][keep].long(),
                "labels": ex["labels"][keep].long(),
                "is_onpolicy": False,
                "shot": 0,
                "qid": i,
            }
        )
    return rows


def load_store(path):
    for name in ("load", "from_jsonl", "load_jsonl", "read_jsonl"):
        fn = getattr(OnPolicyStore, name, None)
        if fn is not None:
            return fn(path)
    raise RuntimeError("OnPolicyStore has no load/from_jsonl/load_jsonl/read_jsonl classmethod")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[*] S1' trainer | Device: {device}")

    cfg = RSCConfig()
    cfg.seed = args.seed
    cfg.checkpoint_dir = args.out_dir
    cfg.s1_save_path = os.path.join(args.out_dir, "s1_steernet.pt")

    print("[*] Loading tokenizer & base model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
        trust_remote_code=True,
    ).to(device)
    model = Phi2WithRSC(base_model, cfg, device=device)

    if args.ckpt and os.path.exists(args.ckpt):
        load_steer_weights(model, args.ckpt, device=device)

    layers = getattr(cfg, "catalog_layers", None) or model.target_layers
    catalog = Catalog.build(tuple(layers), getattr(cfg, "catalog_max_k", 3))
    print(f"[*] Catalog: {len(catalog)} subsets over layers {catalog.layers}")

    print("[*] Loading gold GSM8K rows and on-policy store...")
    train_data, _ = gsm8k_to_cot(tokenizer, cfg)
    gold_rows_all = dataset_to_rows(train_data)
    store = load_store(args.onpolicy_store)
    train_rows, val_rows = store.build_training_rows(
        tokenizer, cfg, gold_rows_all, val_fraction=0.1, seed=args.seed
    )

    gold_rows = [r for r in train_rows if not r.get("is_onpolicy")]
    onpolicy_rows = [r for r in train_rows if r.get("is_onpolicy")]
    val_onp = [r for r in val_rows if r.get("is_onpolicy")][: cfg.s1_val_onpolicy_n]
    val_gold = [r for r in val_rows if not r.get("is_onpolicy")][: cfg.s1_val_onpolicy_n]
    sel_rows = val_onp + val_gold
    print(f"[*] gold_train={len(gold_rows)}  onpolicy_train={len(onpolicy_rows)}  sel_rows={len(sel_rows)}")

    if args.ablation:
        print("[*] Running k-scaling ablation...")
        abl = run_k_scaling_ablation(model, tokenizer, train_rows, sel_rows, cfg, device)
        with open(os.path.join(args.out_dir, "s1_k_ablation.json"), "w") as f:
            json.dump(abl, f, indent=2)
        print(f"[*] Ablation winners: k_scaling={abl['winner_k_scaling']}  k_cond={abl['winner_k_cond']}")

    logs = train_s1_steernet(model, tokenizer, gold_rows, onpolicy_rows, sel_rows, cfg, device)
    logs_out = dict(logs)
    if logs_out.get("best_subset") is not None:
        logs_out["best_subset"] = list(logs_out["best_subset"])
    with open(os.path.join(args.out_dir, "s1_logs.json"), "w") as f:
        json.dump(logs_out, f, indent=2)
    print(f"[*] Done. Best subset: {logs_out.get('best_subset')}  weights: {cfg.s1_save_path}")


if __name__ == "__main__":
    main()
