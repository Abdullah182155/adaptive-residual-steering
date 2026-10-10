import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.configs.config import RSCConfig
from src.models.wrapper import Phi2WithRSC
from src.data.dataset import gsm8k_to_cot
from src.training.trainer import (
    train_rsc,
    train_phase1_steernet,
    train_router_bootstrap,
    load_steer_weights,
)
from src.evaluation.benchmark import eval_gsm8k_benchmark

def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive Residual Steering (ARS) Training CLI")
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "phase1", "bootstrap", "eval"],
        help="Training phase: 'phase1' (SteerNet only), 'bootstrap' (Router only), 'all' (complete multi-stage pipeline), 'eval' (benchmark evaluation)",
    )
    parser.add_argument("--eval_stage", type=str, default="baseline", choices=["baseline", "steernet", "ars"])
    parser.add_argument("--eval_samples", type=int, default=50, help="Number of test problems for evaluation")
    parser.add_argument("--load_weights", type=str, default=None, help="Path to checkpoint to load before eval/training")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch_size", type=int, default=None, help="Override micro-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--steer_magnitude_mode", type=str, default="direct", choices=["direct", "decoupled"], help="SteerNet magnitude mode")
    parser.add_argument("--fixed_layers", type=str, default=None, help="Comma-separated fixed layers (e.g. '11,15,20') to bypass router")
    parser.add_argument("--gating_mode", type=str, default="learned", choices=["learned", "constant", "step"], help="Token gating mode")
    parser.add_argument("--curriculum_mode", type=str, default="full", choices=["full", "streamlined"], help="Curriculum mode ('full' 6-phase or 'streamlined' 2-stage)")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] ARS Experiment Runner | Device: {device}")

    cfg = RSCConfig()
    if args.epochs:
        cfg.phase1_epochs = args.epochs
        cfg.router_bootstrap_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.phase1_lr = args.lr
        cfg.router_bootstrap_lr = args.lr
    if args.steer_magnitude_mode:
        cfg.steer_magnitude_mode = args.steer_magnitude_mode
    if args.fixed_layers:
        cfg.fixed_layers = [int(x.strip()) for x in args.fixed_layers.split(",") if x.strip()]
        cfg.use_router = False
    if args.gating_mode:
        cfg.gating_mode = args.gating_mode
    if args.curriculum_mode:
        cfg.curriculum_mode = args.curriculum_mode

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

    if args.load_weights and os.path.exists(args.load_weights):
        load_steer_weights(model, args.load_weights, device=device)

    if args.phase == "eval":
        if args.eval_stage == "baseline":
            model.set_routing_override("none")
            stage_name = "Stage A - Frozen Baseline"
        elif args.eval_stage == "steernet":
            model.set_routing_override("all")
            model.set_gate_freeze(True, 1.0)
            stage_name = "Stage B - SteerNet Only"
        else:
            model.set_routing_override(None)
            model.set_gate_freeze(False)
            stage_name = "Stage C - Full ARS"

        eval_gsm8k_benchmark(
            model=model,
            tokenizer=tokenizer,
            device=device,
            stage_name=stage_name,
            n_test=args.eval_samples,
            n_shot=0,
            display_examples=3,
        )
        return

    print("[*] Loading GSM8K CoT dataset...")
    train_data, eval_data = gsm8k_to_cot(tokenizer, cfg)

    if args.phase == "phase1":
        print("[*] Running Phase 1: Pure SteerNet Training...")
        train_phase1_steernet(model, train_data, eval_data, cfg, device)
    elif args.phase == "bootstrap":
        print("[*] Running Phase 1.5: Router Bootstrap...")
        train_router_bootstrap(model, train_data, eval_data, cfg, device)
    elif args.phase == "all":
        print("[*] Running Full Staged Pipeline (Phase 1 -> 1.5 -> 2A -> 2B -> 2C -> 2D)...")
        train_rsc(model, tokenizer, train_data, eval_data, cfg, device)

if __name__ == "__main__":
    main()
