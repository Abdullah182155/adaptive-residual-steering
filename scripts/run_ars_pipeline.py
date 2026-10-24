"""Single CLI for the ARS pipeline: --stage {S0, S1, S0b, S2, S3, S5, bench, all}.

Stages are natively wired to the core ARS engine. Override any entrypoint with --stage_map map.json.
Each stage records its outputs into <checkpoint_dir>/pipeline_state.json.
"""

import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

from src.configs.config import RSCConfig

STAGE_ORDER = ["S0", "S1", "S0b", "S2", "S3", "S5", "bench"]
STAGE_MAP = {}


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
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
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


def stage_s0(model, tok, eval_data, cfg, device, state, args):
    from src.analysis.cf_table import build_cf_table, sample_audit_examples
    from src.analysis.signal_audit import compute_signal_audit
    from src.routing.catalog import Catalog
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    table_path = getattr(cfg, "cf_table_path", os.path.join(cfg.checkpoint_dir, "cf_table_s0.npz"))
    layers = getattr(cfg, "catalog_layers", None) or model.target_layers
    catalog = Catalog.build(tuple(layers), getattr(cfg, "catalog_max_k", 3))

    if os.path.exists(table_path):
        print(f"[S0] Loading existing CF table from {table_path}")
        from src.analysis.cf_table import CFTable
        table = CFTable.load(table_path)
    else:
        print(f"[S0] Building CF table over {len(catalog)} subsets...")
        examples = sample_audit_examples(eval_data, cfg)
        table = build_cf_table(model, tok, examples, catalog, cfg, device)
        table.save(table_path)
        print(f"[S0] Saved CF table to {table_path}")

    audit = compute_signal_audit(table, cfg, seed=cfg.seed)
    decision = audit.decision
    print(f"[S0] Decision 1: {decision.recommendation}")
    return {
        "table_path": table_path,
        "recommendation": decision.recommendation,
        "decision": decision.as_dict() if hasattr(decision, "as_dict") else str(decision),
        "halt": decision.recommendation == "HALT",
    }


def stage_s1(model, tok, train_data, eval_data, cfg, device, state, args):
    from src.training.trainer import train_s1_steernet
    from src.data.onpolicy import OnPolicyStore
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.checkpoint_dir, "s1_steernet.pt")

    store_path = (state.get("S0b") or {}).get("store_path", "./checkpoints/onpolicy_store.jsonl")
    if os.path.exists(store_path):
        store = OnPolicyStore.load(store_path)
        train_rows, val_rows = store.build_training_rows(train_data, val_frac=0.1, seed=cfg.seed)
    else:
        train_rows = train_data
        val_rows = eval_data[:min(len(eval_data), 100)]

    gold_rows = [r for r in train_rows if not (isinstance(r, dict) and r.get("is_onpolicy"))]
    onpolicy_rows = [r for r in train_rows if (isinstance(r, dict) and r.get("is_onpolicy"))]
    sel_rows = eval_data[:min(len(eval_data), getattr(cfg, "calib_sel_n", 120))]

    logs = train_s1_steernet(model, tok, gold_rows, onpolicy_rows, sel_rows, cfg, device)
    return {"ckpt": ckpt_path, "logs": logs}


def stage_s0b(model, tok, train_data, cfg, device, state, args):
    from src.data.onpolicy import sample_chains
    from src.training.trainer import load_steer_weights

    s1_ckpt = (state.get("S1") or {}).get("ckpt") or os.path.join(cfg.checkpoint_dir, "s1_steernet.pt")
    if os.path.exists(s1_ckpt):
        load_steer_weights(model, s1_ckpt, device=device)

    out_path = os.path.join(cfg.checkpoint_dir, "onpolicy_store.jsonl")
    n_q = getattr(cfg, "onpolicy_n_questions", 100)
    questions = [ex["question"] if isinstance(ex, dict) else ex for ex in train_data[:n_q]]

    store = sample_chains(model, tok, questions, cfg, device, seed=cfg.seed)
    store.save(out_path)
    return {"store_path": out_path, "n_samples": len(store)}


def stage_s2(model, tok, cfg, device, state, args):
    from src.routing.distill import distill_router
    from src.analysis.cf_table import CFTable
    from src.routing.router import JointLayerRouter
    from src.utils.failure_policy import get_manifest

    table_path = (state.get("S0") or {}).get("table_path") or getattr(cfg, "cf_table_path", os.path.join(cfg.checkpoint_dir, "cf_table_s0.npz"))
    table = CFTable.load(table_path)

    layers = getattr(cfg, "catalog_layers", None) or model.target_layers
    n_catalog = len(table.catalog.subsets) if hasattr(table.catalog, "subsets") else len(table.catalog)
    router = JointLayerRouter(layers, int(table.pooled.shape[1]), 32, cfg.max_active_layers, n_catalog=n_catalog).to(device)

    manifest = get_manifest()
    report = distill_router(router, table, cfg, device, manifest, seed=cfg.seed, verbose=True)
    ckpt = getattr(cfg, "router_catalog_save_path", os.path.join(cfg.checkpoint_dir, "router_catalog_v5.pt"))
    torch.save(router.state_dict(), ckpt)
    return {"ckpt": ckpt, "val_loss": report.val_loss, "top1_agreement": report.top1_agreement}


def stage_s3(model, tok, train_data, eval_data, cfg, device, state, args):
    from src.training.trainer import train_s3_gate, load_steer_weights
    from src.analysis.cf_table import CFTable

    s1_ckpt = (state.get("S1") or {}).get("ckpt") or os.path.join(cfg.checkpoint_dir, "s1_steernet.pt")
    if os.path.exists(s1_ckpt):
        load_steer_weights(model, s1_ckpt, device=device)

    table_path = (state.get("S0") or {}).get("table_path") or getattr(cfg, "cf_table_path", os.path.join(cfg.checkpoint_dir, "cf_table_s0.npz"))
    table = CFTable.load(table_path) if os.path.exists(table_path) else None

    val_rows = eval_data[:min(len(eval_data), 100)]
    results = train_s3_gate(model, tok, train_data, val_rows, cfg, device, table=table)
    ckpt = getattr(cfg, "gate_v5_save_path", os.path.join(cfg.checkpoint_dir, "gate_v5.pt"))
    return {"ckpt": ckpt, "results": results}


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
        if s == "S0":
            result = stage_s0(model, tok, eval_data, cfg, device, state, args)
        elif s == "S1":
            result = stage_s1(model, tok, train_data, eval_data, cfg, device, state, args)
        elif s == "S0b":
            result = stage_s0b(model, tok, train_data, cfg, device, state, args)
        elif s == "S2":
            result = stage_s2(model, tok, cfg, device, state, args)
        elif s == "S3":
            result = stage_s3(model, tok, train_data, eval_data, cfg, device, state, args)
        elif s == "S5":
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
