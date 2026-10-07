import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.configs.config import RSCConfig
from src.models.wrapper import Phi2WithRSC
from src.data.dataset import gsm8k_to_cot
from src.training.trainer import train_router_bootstrap

def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive Residual Steering (ARS) Training CLI")
    parser.add_argument("--phase", type=str, default="bootstrap", choices=["bootstrap", "full"],
                        help="Training phase to run ('bootstrap' for Router only, 'full' for complete staged run)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch_size", type=int, default=None, help="Override micro-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] ARS Experiment Runner | Device: {device}")

    cfg = RSCConfig()
    if args.epochs:
        cfg.router_bootstrap_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.router_bootstrap_lr = args.lr

    print("[*] Loading base tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
        trust_remote_code=True,
    ).to(device)

    print("[*] Wrapping base model with ARS modules...")
    model = Phi2WithRSC(base_model, cfg, device=device)

    print("[*] Loading GSM8K CoT dataset...")
    train_data, eval_data = gsm8k_to_cot(tokenizer, cfg)

    if args.phase == "bootstrap":
        print("[*] Running Phase 1.5: Router Bootstrap...")
        best_state, best_loss = train_router_bootstrap(model, train_data, eval_data, cfg, device)
        print(f"[✓] Router bootstrap completed. Best loss: {best_loss:.4f}")
    else:
        print("[*] Full staged pipeline execution mode.")

if __name__ == "__main__":
    main()
