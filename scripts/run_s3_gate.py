"""CLI: Train token gates using dominance hinge and counterfactual token BCE (M7 / S3)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args():
    p = argparse.ArgumentParser(description="Train S3 token gates (dominance hinge + counterfactual BCE).")
    p.add_argument("--table_path", type=str, default=None, help="Path to the CFTable .npz")
    p.add_argument("--steernet_path", type=str, default="./checkpoints/s1_steernet.pt", help="Path to S1' SteerNet weights")
    p.add_argument("--router_path", type=str, default="./checkpoints/router_catalog_v5.pt", help="Path to catalog router weights")
    p.add_argument("--out_path", type=str, default=None, help="Output path for gate checkpoint (default: cfg.gate_v5_save_path)")
    p.add_argument("--epochs", type=int, default=None, help="Override cfg.gate_epochs")
    p.add_argument("--lr", type=float, default=None, help="Override cfg.gate_lr")
    p.add_argument("--sel_n", type=int, default=None, help="Override cfg.gate_sel_n")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.configs.config import RSCConfig
    from src.models.wrapper import Phi2WithRSC
    from src.training.trainer import train_s3_gate
    from src.analysis.cf_table import CFTable

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = RSCConfig()
    cfg.seed = args.seed
    if args.epochs is not None:
        cfg.gate_epochs = args.epochs
    if args.lr is not None:
        cfg.gate_lr = args.lr
    if args.out_path is not None:
        cfg.gate_v5_save_path = args.out_path
    if args.sel_n is not None:
        cfg.gate_sel_n = args.sel_n

    print(f"[*] S3 Gate Training | device={device}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
        trust_remote_code=True,
    ).to(device)

    model = Phi2WithRSC(base_model, cfg, device=device).to(device)

    # Load frozen SteerNet weights
    if os.path.exists(args.steernet_path):
        print(f"[*] Loading SteerNet weights from {args.steernet_path}")
        sn_state = torch.load(args.steernet_path, map_location=device)
        model.steer_nets.load_state_dict(sn_state.get("steer_nets", sn_state), strict=False)

    # Load frozen Router weights
    if os.path.exists(args.router_path):
        print(f"[*] Loading Router weights from {args.router_path}")
        r_state = torch.load(args.router_path, map_location=device)
        model.router.load_state_dict(r_state.get("router", r_state), strict=False)

    table = None
    if args.table_path and os.path.exists(args.table_path):
        print(f"[*] Loading CFTable from {args.table_path}")
        table = CFTable.load(args.table_path)

    # Build dummy or training rows from dataset
    from src.data.dataset import load_gsm8k
    dataset = load_gsm8k()
    train_rows = dataset.get("train", [])
    val_rows = dataset.get("test", [])

    results = train_s3_gate(
        model, tokenizer, train_rows, val_rows, cfg, device, table=table
    )
    print(f"[✓] S3 Gate training complete. Adopted: {results['gate_adopted_nll']}")


if __name__ == "__main__":
    main()
