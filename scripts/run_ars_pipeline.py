"""Single CLI for the ARS pipeline: --stage {S0, S1, S0b, S2, S3, S5, bench, all}.

S5 and bench are implemented here.
S0/S1/S0b/S2/S3 are dispatched through STAGE_MAP to the M1-M7 entrypoints;
override any of them with --stage_map map.json ({"S1": "pkg.mod:fn"}).

Each stage function is called as
  fn(model=, tokenizer=, train_data=, eval_data=, cfg=, device=, state=) -> dict
and may return {"halt": True} (failure policy) to stop the run.

Decisions and checkpoints are persisted in <checkpoint_dir>/pipeline_state.json
so later stages (and reruns) pick them up.
"""

import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.configs.config import RSCConfig

STAGE_ORDER = ["S0", "S1", "S0b", "S2", "S3", "S5", "bench"]
STAGE_MAP = {
    "S0":  "src.audit.signal_audit:run_s0",         # M1/M2 counterfactual table + signal audit
    "S1":  "src.training.subset_trainer:run_s1",    # M4 S1' SteerNet subset trainer
    "S0b": "src.sampling.trace_sampler:run_s0b",    # M3 on-policy reasoning traces
    "S2":  "src.routing.distill:run_s2",            # M5 catalog router distillation
    "S3":  "src.training.gate_trainer:run_s3",      # M7 direct task-loss gate + dominance hinge
}


def parse_args():
    p = argparse.ArgumentParser(description="ARS end-to-end pipeline")
    p.add_argument("--stage",     default="all", choices=STAGE_ORDER + ["all"])
    p.add_argument("--stage_map", default=None, help="JSON overriding STAGE_MAP entrypoints")
    p.add_argument("--contract",  default=None)
    p.add_argument("--record",    default="experiment_record.json")
    p.add_argument("--shots",     type=int, nargs="*", default=None)
    p.add_argument("--n_test",    type=int, default=None)
    p.add_argument("--secondary", action="store_true")
    p.add_argument("--report",    action="store_true", help="run make_report after bench")
    p.add_argument("--report_dir", default="report")
    p.add_argument("--manifest",  default="")
    return p.parse_args()


def _state_path(cfg):
    return os.path.join(cfg.checkpoint_dir, "pipeline_state.json")


def load_state(cfg):
    p = _state_path(cfg)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_state(cfg, state):
    with open(_state_path(cfg), "w") as f:
        json.dump(state, f, indent=2, default=str)


def resolve(spec: str):
    mod, fn = spec.split(":")
    try:
        return getattr(importlib.import_module(mod), fn)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"cannot resolve stage entrypoint '{spec}': {e}. "
            f"Point it at your implementation via --stage_map."
        ) from e


def stage_s5(model, tok, eval_data, cfg, device, state, args):
    from src.deploy.calibrate import run_deployment_calibration
    ck = {k: (state.get(s) or {}).get("ckpt") for k, s in (
        ("steernet_ckpt", "S1"),
        ("router_ckpt",   "S2"),
        ("gate_ckpt",     "S3"),
    )}
    contract = run_deployment_calibration(
        model, tok, eval_data, cfg, device,
        manifest_path=args.manifest,
        out_path=args.contract or cfg.contract_path,
        **ck,
    )
    return {
        "decision": {
            "s_fixed_star":  contract["s_fixed_star"],
            "theta_abs":     contract["theta_abs"],
            "gate_adopted":  contract["gate"]["adopted"],
            "rel_norm_cap":  contract["envelope"]["rel_norm_cap"],
        },
        "contract": args.contract or cfg.contract_path,
    }


def stage_bench(model, tok, cfg, device, args):
    from src.deploy.calibrate import load_contract, load_contract_weights
    from src.evaluation.benchmark import run_full_benchmark, write_record
    contract = load_contract(args.contract or cfg.contract_path)
    load_contract_weights(model, contract, device=device)
    record = run_full_benchmark(
        model, tok, contract, cfg, device,
        shots=args.shots, secondary=args.secondary, n_test=args.n_test,
    )
    write_record(record, args.record)
    out = {
        "record": args.record,
        "accuracy": {
            k: {a: round(v["accuracy_strict"], 2) for a, v in blk["arms"].items()}
            for k, blk in record["shots"].items()
        },
    }
    if args.report:
        sys.path.insert(0, os.path.dirname(__file__))
        import make_report
        out["report"] = make_report.write_report(record, args.report_dir)
    return out


def main():
    args      = parse_args()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg       = RSCConfig()
    stage_map = dict(STAGE_MAP)
    if args.stage_map:
        with open(args.stage_map) as f:
            stage_map.update(json.load(f))
    stages = STAGE_ORDER if args.stage == "all" else [args.stage]
    state  = load_state(cfg)

    from src.deploy.calibrate import build_model
    model, tok = build_model(cfg, device)

    train_data = eval_data = None
    if any(s in stages for s in ("S0", "S1", "S0b", "S2", "S3", "S5")):
        from src.data.dataset import gsm8k_to_cot
        train_data, eval_data = gsm8k_to_cot(tok, cfg)

    for s in stages:
        print("\n" + "=" * 80 + f"\n[pipeline] stage {s}\n" + "=" * 80)
        if s == "S5":
            result = stage_s5(model, tok, eval_data, cfg, device, state, args)
        elif s == "bench":
            result = stage_bench(model, tok, cfg, device, args)
        else:
            fn     = resolve(stage_map[s])
            result = fn(
                model=model, tokenizer=tok,
                train_data=train_data, eval_data=eval_data,
                cfg=cfg, device=device, state=state,
            ) or {}
        state[s] = result
        save_state(cfg, state)
        print(f"[pipeline] {s} result:\n{json.dumps(result, indent=2, default=str)}")
        if isinstance(result, dict) and result.get("halt"):
            print(f"[pipeline] stage {s} requested HALT (failure policy). Stopping.")
            sys.exit(2)


if __name__ == "__main__":
    main()
