import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.configs.config import RSCConfig
from src.deploy.calibrate import build_model, load_contract, load_contract_weights
from src.evaluation.benchmark import ARM_SPEC, run_full_benchmark, write_record


def parse_args():
    p = argparse.ArgumentParser(description="ARS 5-arm production benchmark (GSM8K test)")
    p.add_argument("--contract",  default=None, help="deployment_contract.json path")
    p.add_argument("--out",       default="experiment_record.json")
    p.add_argument("--shots",     type=int, nargs="*", default=None,
                   help="default: cfg.bench_shots (0 8)")
    p.add_argument("--n_test",    type=int, default=None, help="default: full test set (1319)")
    p.add_argument("--secondary", action="store_true",
                   help="also run cfg.bench_secondary_shots on a subset")
    p.add_argument("--arms",      nargs="*", default=None, choices=[k for k, _, _ in ARM_SPEC])
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--no_verify_sha", action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg      = RSCConfig()
    contract = load_contract(args.contract or cfg.contract_path)
    print(f"[*] benchmark | device={device} | contract git={contract['git_commit']}")
    model, tok = build_model(cfg, device)
    load_contract_weights(model, contract, device=device, verify=not args.no_verify_sha)
    record = run_full_benchmark(
        model, tok, contract, cfg, device,
        shots=args.shots, secondary=args.secondary,
        n_test=args.n_test, cache_dir=args.cache_dir, arms=args.arms,
    )
    write_record(record, args.out)
    print(f"[*] wrote {args.out}")
    for k, blk in record["shots"].items():
        print(f"\n{k}-shot (n={blk['n']}):")
        for key, a in blk["arms"].items():
            print(
                f"  {a['label']:<14} "
                f"strict={a['accuracy_strict']:.2f}%  "
                f"lenient={a['accuracy_lenient']:.2f}%  "
                f"abstain={a['summary']['abstain_rate']:.2f}  "
                f"trips={a['summary']['guard_trip_rate']:.3f}"
            )


if __name__ == "__main__":
    main()
