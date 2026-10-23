import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.configs.config import RSCConfig
from src.data.dataset import gsm8k_to_cot
from src.deploy.calibrate import build_model, run_deployment_calibration


def parse_args():
    p = argparse.ArgumentParser(description="ARS S5 deployment calibration -> deployment_contract.json")
    p.add_argument("--contract_path", default=None)
    p.add_argument("--steernet_ckpt", default=None)
    p.add_argument("--router_ckpt",   default=None)
    p.add_argument("--gate_ckpt",     default=None)
    p.add_argument("--manifest",      default="")
    p.add_argument("--sel_n", type=int, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = RSCConfig()
    if args.sel_n:
        cfg.calib_sel_n = args.sel_n
    if args.contract_path:
        cfg.contract_path = args.contract_path
    print(f"[*] S5 calibration | device={device}")
    model, tok    = build_model(cfg, device)
    _, eval_data  = gsm8k_to_cot(tok, cfg)
    contract = run_deployment_calibration(
        model, tok, eval_data, cfg, device,
        steernet_ckpt=args.steernet_ckpt,
        router_ckpt=args.router_ckpt,
        gate_ckpt=args.gate_ckpt,
        manifest_path=args.manifest,
        out_path=cfg.contract_path,
    )
    print(f"s*_fixed   : {contract['s_fixed_star']}")
    print(f"theta*_abs : {contract['theta_abs']}")
    print(f"gate       : {contract['gate']['adopted']}")
    print(f"env cap    : {contract['envelope']['rel_norm_cap']:.5f}")


if __name__ == "__main__":
    main()
