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

STAGE_ORDER = ["S1", "S0b", "S0", "S2", "S3", "S5", "bench"]
STAGE_MAP = {}


def parse_args():
    p = argparse.ArgumentParser(description="ARS end-to-end pipeline")
    p.add_argument("--stage",     default="all", choices=STAGE_ORDER + ["all"])
    p.add_argument("--stage_map", default=None, help="JSON overriding STAGE_MAP entrypoints")
    p.add_argument("--ckpt",      default=None, help="Initial SteerNet checkpoint to load")
    p.add_argument("--contract",  default=None)
    p.add_argument("--record",    default="experiment_record.json")
    p.add_argument("--shots",     type=int, nargs="*", default=None)
    p.add_argument("--n_test",    type=int, default=None)
    p.add_argument("--secondary", action="store_true")
    p.add_argument("--report",    action="store_true", help="run make_report after bench")
    p.add_argument("--report_dir", default="report")
    p.add_argument("--manifest",         default="")
    p.add_argument("--profile",          choices=["quick", "medium", "full"], default=None,
                   help="Scale preset: quick (~5m), medium (~20m), full (production)")
    p.add_argument("--cf_n_audit",       type=int, default=None, help="Total train examples for S0 audit table")
    p.add_argument("--cf_heldout_n",     type=int, default=None, help="Heldout examples for S0 audit table")
    p.add_argument("--cf_shard_size",    type=int, default=None, help="Shard size for CF table")
    p.add_argument("--n_questions",      type=int, default=None, help="Number of questions for S0b sampling")
    p.add_argument("--onpolicy_samples", type=int, nargs=2, default=None, help="Onpolicy samples (round1 round2)")
    p.add_argument("--s1_epochs",        type=int, default=None, help="Epochs for S1 SteerNet training")
    p.add_argument("--s1_train_n",       type=int, default=None, help="Training rows for S1 SteerNet training")
    p.add_argument("--s1_val_n",         type=int, default=None, help="Validation rows for S1")
    p.add_argument("--gate_epochs",      type=int, default=None, help="Epochs for S3 Gate training")
    p.add_argument("--gate_train_n",     type=int, default=None, help="Training rows for S3 Gate training")
    p.add_argument("--gate_sel_n",       type=int, default=None, help="Selection rows for S3 Gate")
    p.add_argument("--calib_sel_n",      type=int, default=None, help="Selection rows for S5 calibration")
    p.add_argument("--batch_size",              type=int, default=None, help="Batch size override")
    p.add_argument("--steer_scale",             type=float, default=None, help="Steering magnitude multiplier")
    p.add_argument("--s1_lr",                   type=float, default=None, help="Learning rate for S1 SteerNet")
    p.add_argument("--gate_lr",                 type=float, default=None, help="Learning rate for S3 Gate")
    p.add_argument("--onpolicy_temperature",    type=float, default=None, help="Temperature for S0b sampling")
    p.add_argument("--onpolicy_max_new_tokens", type=int, default=None, help="Max tokens for S0b sampling")
    p.add_argument("--router_tau",              type=float, default=None, help="Softmax temperature for router target distribution")
    p.add_argument("--router_k_free",           type=int, default=None, help="Number of layers exempt from size penalty (K<=k_free free)")
    p.add_argument("--router_c_size",           type=float, default=None, help="Penalty per excess layer beyond k_free in router distillation")
    p.add_argument("--set",                     action="append", default=[], help="Arbitrary config override KEY=VAL")
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



def _slice_dataset(ds, n: int):
    """Safely slice a dataset (HF Dataset, list, or dict of columns) into a list of row dicts."""
    if ds is None:
        return []
    total = len(ds)
    limit = min(n, total)
    if limit <= 0:
        return []
    if hasattr(ds, "select"):
        return [ds[i] for i in range(limit)]
    elif isinstance(ds, list):
        return ds[:limit]
    elif isinstance(ds, dict):
        keys = list(ds.keys())
        first_col_len = len(ds[keys[0]]) if keys else 0
        limit = min(limit, first_col_len)
        return [{k: ds[k][i] for k in keys} for i in range(limit)]
    else:
        try:
            return [ds[i] for i in range(limit)]
        except Exception:
            return list(ds)[:limit]


def _dataset_to_rows(ds, n: int = None):
    """Convert dataset or slice into unpadded 1D row dicts for SteerNet training."""
    if ds is None:
        return []
    rows = []
    limit = len(ds) if n is None else min(n, len(ds))
    for i in range(limit):
        ex = ds[i]
        if isinstance(ex, dict) and "attention_mask" in ex:
            keep = ex["attention_mask"].bool() if hasattr(ex["attention_mask"], "bool") else [bool(x) for x in ex["attention_mask"]]
            inp = ex["input_ids"]
            lbl = ex["labels"]
            if hasattr(inp, "__getitem__") and hasattr(inp, "long"):
                inp = inp[keep].long()
            elif isinstance(inp, list):
                inp = torch.tensor([inp[j] for j, m in enumerate(keep) if m], dtype=torch.long)
            if hasattr(lbl, "__getitem__") and hasattr(lbl, "long"):
                lbl = lbl[keep].long()
            elif isinstance(lbl, list):
                lbl = torch.tensor([lbl[j] for j, m in enumerate(keep) if m], dtype=torch.long)
            rows.append({
                "input_ids": inp,
                "labels": lbl,
                "is_onpolicy": False,
                "shot": 0,
                "n_shot": 0,
                "split": "train",
                "qid": f"train:{i}",
                "question": ex.get("question", ""),
                "final_num": ex.get("final_num", None),
            })
        else:
            rows.append(ex)
    return rows

def stage_s0(model, tok, train_data, eval_data, cfg, device, state, args):
    from src.analysis.cf_table import build_cf_table, sample_audit_examples
    from src.analysis.signal_audit import compute_signal_audit, write_audit
    from src.routing.catalog import Catalog
    from src.utils.failure_policy import get_manifest
    from src.training.trainer import load_steer_weights
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    table_path = getattr(cfg, "cf_table_path", os.path.join(cfg.checkpoint_dir, "cf_table_s0.npz"))
    layers = getattr(cfg, "catalog_layers", None) or model.target_layers
    catalog = Catalog.build(tuple(layers), getattr(cfg, "catalog_max_k", 3))
    if hasattr(model, "set_catalog"):
        model.set_catalog(catalog)

    # P0-5: Load initial SteerNet weights if available
    init_ckpt = getattr(args, "ckpt", None) or os.path.join(cfg.checkpoint_dir, "s1_steernet.pt")
    has_steer = False
    if init_ckpt and os.path.exists(init_ckpt):
        print(f"[S0] Loading SteerNet weights from {init_ckpt} for CF table")
        load_steer_weights(model, init_ckpt, device=device)
        has_steer = True
    else:
        print(f"[S0] Notice: No SteerNet checkpoint found at {init_ckpt}; evaluating unsteered baseline.")

    rebuild = False
    if os.path.exists(table_path) and has_steer and init_ckpt and os.path.exists(init_ckpt):
        if os.path.getmtime(init_ckpt) > os.path.getmtime(table_path):
            print(f"[S0] SteerNet checkpoint is newer than existing CF table. Rebuilding CF table.")
            rebuild = True

    if os.path.exists(table_path) and not rebuild:
        print(f"[S0] Loading existing CF table from {table_path}")
        from src.analysis.cf_table import CFTable
        table = CFTable.load(table_path)
    else:
        print(f"[S0] Building CF table over {len(catalog)} subsets...")
        examples = sample_audit_examples(train_data, eval_data, tok, cfg, seed=cfg.seed)
        steer_scale = getattr(cfg, "steer_scale", 1.0)
        table = build_cf_table(
            model, tok, examples, catalog, cfg, device,
            steer_scale=steer_scale, out_dir=cfg.checkpoint_dir, seed=cfg.seed
        )
        table.save(table_path)
        print(f"[S0] Saved CF table to {table_path}")

    manifest = get_manifest()
    audit = compute_signal_audit(table, cfg, seed=cfg.seed)
    audit_json = os.path.join(cfg.checkpoint_dir, "audit_s0.json")
    audit_md = os.path.join(cfg.checkpoint_dir, "AUDIT_REPORT_S0.md")
    write_audit(audit, manifest, audit_json, audit_md)
    print(f"[S0] Decision 1: {audit.decision_1}")
    return {
        "table_path": table_path,
        "audit_path": audit_json,
        "decision_1": audit.decision_1,
        "flags": audit.flags,
        "decision_2_inputs": audit.decision_2_inputs,
    }


def stage_s1(model, tok, train_data, eval_data, cfg, device, state, args):
    from src.training.trainer import train_s1_steernet
    from src.data.onpolicy import OnPolicyStore
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.checkpoint_dir, "s1_steernet.pt")
    cfg.s1_save_path = ckpt_path
    if getattr(args, "s1_epochs", None):
        cfg.s1_epochs = args.s1_epochs
    if getattr(args, "batch_size", None):
        cfg.batch_size = args.batch_size

    s1_train_n = getattr(args, "s1_train_n", None) or getattr(cfg, "s1_train_n", None)
    gold_rows_all = _dataset_to_rows(train_data, n=s1_train_n)
    print(f"[S1] SteerNet training on n={len(gold_rows_all)} gold rows")
    store_path = (state.get("S0b") or {}).get("store_path", os.path.join(cfg.checkpoint_dir, "onpolicy_store.jsonl"))
    if os.path.exists(store_path):
        store = OnPolicyStore.load(store_path)
        train_rows, val_rows = store.build_training_rows(tok, cfg, gold_rows_all, val_fraction=0.1, seed=cfg.seed)
    else:
        train_rows = gold_rows_all
        val_rows = _dataset_to_rows(eval_data, n=min(len(eval_data), 100))

    gold_rows = [r for r in train_rows if not (isinstance(r, dict) and r.get("is_onpolicy"))]
    onpolicy_rows = [r for r in train_rows if (isinstance(r, dict) and r.get("is_onpolicy"))]
    val_onp = [r for r in val_rows if (isinstance(r, dict) and r.get("is_onpolicy"))][: getattr(cfg, "s1_val_onpolicy_n", 60)]
    val_gold = [r for r in val_rows if not (isinstance(r, dict) and r.get("is_onpolicy"))][: getattr(cfg, "s1_val_onpolicy_n", 60)]
    sel_rows = (val_onp + val_gold) if (val_onp or val_gold) else val_rows[:min(len(val_rows), 120)]

    logs = train_s1_steernet(model, tok, gold_rows, onpolicy_rows, sel_rows, cfg, device)
    return {"ckpt": ckpt_path, "logs": logs}


def stage_s0b(model, tok, train_data, cfg, device, state, args):
    import random
    from src.data.onpolicy import sample_chains
    from src.training.trainer import load_steer_weights

    s1_ckpt = (state.get("S1") or {}).get("ckpt") or os.path.join(cfg.checkpoint_dir, "s1_steernet.pt")
    if os.path.exists(s1_ckpt):
        load_steer_weights(model, s1_ckpt, device=device)

    out_path = os.path.join(cfg.checkpoint_dir, "onpolicy_store.jsonl")
    n_q = getattr(args, "n_questions", None) or getattr(cfg, "onpolicy_n_questions", 100)
    cfg.onpolicy_n_questions = n_q
    if getattr(args, "batch_size", None):
        cfg.onpolicy_batch_size = args.batch_size
    n = min(n_q, len(train_data))
    idxs = random.Random(cfg.seed).sample(range(len(train_data)), n)
    questions = [
        {"qid": i, "question": train_data[i]["question"], "final_num": train_data[i].get("final_num", None)}
        for i in idxs
    ]

    print(f"[S0b] Sampling on-policy chains for {len(questions)} questions (batch_size={cfg.onpolicy_batch_size})...")
    store = sample_chains(model, tok, questions, cfg, device, seed=cfg.seed, out_path=out_path)
    return {"store_path": out_path, "n_samples": len(store.chains) if hasattr(store, "chains") else len(store)}


def stage_s2(model, tok, cfg, device, state, args):
    from src.routing.distill import distill_router
    from src.analysis.cf_table import CFTable
    from src.utils.failure_policy import get_manifest
    from src.training.trainer import save_steer_weights

    table_path = (state.get("S0") or {}).get("table_path") or getattr(cfg, "cf_table_path", os.path.join(cfg.checkpoint_dir, "cf_table_s0.npz"))
    table = CFTable.load(table_path)

    # P0-3: Ensure catalog is attached to model.router before distillation
    model.set_catalog(table.catalog)

    manifest = get_manifest()
    report = distill_router(model.router, table, cfg, device, manifest, seed=cfg.seed, verbose=True)
    ckpt = getattr(cfg, "router_catalog_save_path", os.path.join(cfg.checkpoint_dir, "router_catalog_v5.pt"))
    save_steer_weights(model, ckpt, cfg=cfg, phase="s2_router", prefixes=("router.",))
    return {"ckpt": ckpt, "val_loss": report.val_loss, "top1_agreement": report.top1_agreement}


def stage_s3(model, tok, train_data, eval_data, cfg, device, state, args):
    from src.training.trainer import train_s3_gate, load_steer_weights, save_steer_weights
    from src.analysis.cf_table import CFTable
    from src.routing.catalog import Catalog

    # P0-3 & P0-6: Load frozen SteerNet AND distilled router
    table_path = (state.get("S0") or {}).get("table_path") or getattr(cfg, "cf_table_path", os.path.join(cfg.checkpoint_dir, "cf_table_s0.npz"))
    table = CFTable.load(table_path) if os.path.exists(table_path) else None

    if table is not None:
        model.set_catalog(table.catalog)
    else:
        layers = getattr(cfg, "catalog_layers", None) or model.target_layers
        catalog = Catalog.build(tuple(layers), getattr(cfg, "catalog_max_k", 3))
        model.set_catalog(catalog)

    s1_ckpt = (state.get("S1") or {}).get("ckpt") or os.path.join(cfg.checkpoint_dir, "s1_steernet.pt")
    if os.path.exists(s1_ckpt):
        load_steer_weights(model, s1_ckpt, device=device)

    router_ckpt = (state.get("S2") or {}).get("ckpt") or getattr(cfg, "router_catalog_save_path", os.path.join(cfg.checkpoint_dir, "router_catalog_v5.pt"))
    if os.path.exists(router_ckpt):
        print(f"[S3] Loading distilled router from {router_ckpt}")
        load_steer_weights(model, router_ckpt, device=device)

    ckpt = getattr(cfg, "gate_v5_save_path", os.path.join(cfg.checkpoint_dir, "gate_v5.pt"))
    cfg.gate_v5_save_path = ckpt
    if getattr(args, "gate_epochs", None):
        cfg.gate_epochs = args.gate_epochs
    if getattr(args, "batch_size", None):
        cfg.batch_size = args.batch_size
    val_n = getattr(args, "gate_sel_n", None) or getattr(cfg, "gate_sel_n", None) or 100
    val_rows = _slice_dataset(eval_data, min(len(eval_data), val_n))
    train_n = getattr(args, "gate_train_n", None) or getattr(cfg, "gate_train_n", None) or len(train_data)
    train_rows = _slice_dataset(train_data, train_n)
    print(f"[S3] Gate training on n={len(train_rows)} rows (epochs={cfg.gate_epochs}, bs={cfg.batch_size}), val on n={len(val_rows)} rows")
    results = train_s3_gate(model, tok, train_rows, val_rows, cfg, device, table=table)
    save_steer_weights(model, ckpt, cfg=cfg, phase="s3_gate", prefixes=("gates.",))
    return {"ckpt": ckpt, "results": results}


def stage_s5(model, tok, eval_data, cfg, device, state, args):
    from src.deploy.calibrate import run_deployment_calibration
    from src.routing.catalog import Catalog
    layers = getattr(cfg, "catalog_layers", None) or model.target_layers
    catalog = Catalog.build(tuple(layers), getattr(cfg, "catalog_max_k", 3))
    if hasattr(model, "set_catalog"):
        model.set_catalog(catalog)

    ck = {k: (state.get(s) or {}).get("ckpt") for k, s in (
        ("steernet_ckpt", "S1"),
        ("router_ckpt",   "S2"),
        ("gate_ckpt",     "S3"),
    )}
    contract = run_deployment_calibration(
        model, tok, eval_data, cfg, device,
        catalog=catalog,
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
    contract_path = args.contract or cfg.contract_path
    if not os.path.exists(contract_path):
        raise FileNotFoundError(f"Deployment contract not found: {contract_path}. Please run Stage S5 first.")
    contract = load_contract(contract_path)
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



def apply_cfg_overrides(cfg, args):
    """Apply profile presets, dedicated CLI flags, and arbitrary --set KEY=VAL overrides."""
    profile = getattr(args, "profile", None)
    if profile == "quick":
        cfg.cf_audit_plan = ((0, 20), (4, 10), (8, 10))
        cfg.cf_heldout_n = 10
        cfg.cf_shard_size = 10
        cfg.onpolicy_n_questions = 30
        cfg.onpolicy_samples = (1, 1)
        cfg.s1_epochs = 2
        cfg.s1_train_n = 50
        cfg.s1_val_onpolicy_n = 20
        cfg.gate_epochs = 2
        cfg.gate_train_n = 50
        cfg.gate_sel_n = 30
        cfg.calib_sel_n = 30
        print("[profile: quick] Applied fast-execution scale preset")
    elif profile == "medium":
        cfg.cf_audit_plan = ((0, 100), (4, 50), (8, 50))
        cfg.cf_heldout_n = 40
        cfg.cf_shard_size = 20
        cfg.onpolicy_n_questions = 100
        cfg.onpolicy_samples = (2, 2)
        cfg.s1_epochs = 3
        cfg.s1_train_n = 200
        cfg.s1_val_onpolicy_n = 60
        cfg.gate_epochs = 3
        cfg.gate_train_n = 200
        cfg.gate_sel_n = 60
        cfg.calib_sel_n = 60
        print("[profile: medium] Applied balanced-scale preset")

    if getattr(args, "cf_shard_size", None) is not None:
        cfg.cf_shard_size = int(args.cf_shard_size)
    if getattr(args, "cf_heldout_n", None) is not None:
        cfg.cf_heldout_n = int(args.cf_heldout_n)
    if getattr(args, "cf_n_audit", None) is not None:
        total = int(args.cf_n_audit)
        n0 = max(1, total // 2)
        n4 = max(1, total // 4)
        n8 = max(1, total - n0 - n4)
        cfg.cf_audit_plan = ((0, n0), (4, n4), (8, n8))
    if getattr(args, "n_questions", None) is not None:
        cfg.onpolicy_n_questions = int(args.n_questions)
    if getattr(args, "onpolicy_samples", None) is not None:
        cfg.onpolicy_samples = tuple(int(x) for x in args.onpolicy_samples)
    if getattr(args, "s1_epochs", None) is not None:
        cfg.s1_epochs = int(args.s1_epochs)
    if getattr(args, "s1_train_n", None) is not None:
        cfg.s1_train_n = int(args.s1_train_n)
    if getattr(args, "s1_val_n", None) is not None:
        cfg.s1_val_onpolicy_n = int(args.s1_val_n)
    if getattr(args, "gate_epochs", None) is not None:
        cfg.gate_epochs = int(args.gate_epochs)
    if getattr(args, "gate_train_n", None) is not None:
        cfg.gate_train_n = int(args.gate_train_n)
    if getattr(args, "gate_sel_n", None) is not None:
        cfg.gate_sel_n = int(args.gate_sel_n)
    if getattr(args, "calib_sel_n", None) is not None:
        cfg.calib_sel_n = int(args.calib_sel_n)
    if getattr(args, "batch_size", None) is not None:
        cfg.batch_size = int(args.batch_size)
        cfg.onpolicy_batch_size = int(args.batch_size)
    if getattr(args, "steer_scale", None) is not None:
        cfg.steer_scale = float(args.steer_scale)
    if getattr(args, "s1_lr", None) is not None:
        cfg.s1_lr = float(args.s1_lr)
    if getattr(args, "gate_lr", None) is not None:
        cfg.gate_lr = float(args.gate_lr)
    if getattr(args, "onpolicy_temperature", None) is not None:
        cfg.onpolicy_temperature = float(args.onpolicy_temperature)
    if getattr(args, "onpolicy_max_new_tokens", None) is not None:
        cfg.onpolicy_max_new_tokens = int(args.onpolicy_max_new_tokens)
    if getattr(args, "router_tau", None) is not None:
        cfg.router_tau = float(args.router_tau)
    if getattr(args, "router_k_free", None) is not None:
        cfg.router_k_free = int(args.router_k_free)
    if getattr(args, "router_c_size", None) is not None:
        cfg.router_c_size = float(args.router_c_size)

    if getattr(args, "set", None):
        for item in args.set:
            if "=" in item:
                k, v = item.split("=", 1)
                k = k.strip()
                v = v.strip()
                if v.lower() == "true":
                    val = True
                elif v.lower() == "false":
                    val = False
                else:
                    try:
                        val = int(v)
                    except ValueError:
                        try:
                            val = float(v)
                        except ValueError:
                            val = v
                setattr(cfg, k, val)
                print(f"[config override] {k} = {val}")


def main():
    args      = parse_args()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg       = RSCConfig()
    apply_cfg_overrides(cfg, args)
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
            result = stage_s0(model, tok, train_data, eval_data, cfg, device, state, args)
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
