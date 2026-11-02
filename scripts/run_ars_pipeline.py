"""Run the only supported ARS path: calibration, S1, S5, then GSM8K."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.configs.config import RSCConfig, shot_aware_scale_map
from src.data.dataset import gsm8k_to_cot
from src.deploy.calibrate import measure_envelope, write_contract
from src.models.wrapper import Phi2WithRSC
from src.training.trainer import train_phase1_steernet

def main():
    parser = argparse.ArgumentParser(description="Phase 1 fixed-route ARS pipeline")
    parser.add_argument("--stage", choices=("all", "train", "calibrate"), default="all")
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args(); cfg = RSCConfig()
    if args.samples is not None: cfg.n_samples = args.samples
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(cfg.model_name, trust_remote_code=True).to(device)
    model = Phi2WithRSC(backbone, cfg, device); model.set_shot_aware_scale(shot_aware_scale_map(cfg))
    train, validation = gsm8k_to_cot(tokenizer, cfg); result = {}
    if args.stage in ("all", "train"): result["training"] = train_phase1_steernet(model, train, validation, cfg, device)
    if args.stage in ("all", "calibrate"):
        result["contract"] = write_contract(cfg.contract_path, cfg=cfg, envelope=measure_envelope(model, validation, cfg, device))
    Path("experiments").mkdir(exist_ok=True)
    Path("experiments/history.json").write_text(json.dumps([result], indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__": main()
