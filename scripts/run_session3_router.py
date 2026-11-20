"""Train and compare the adaptive subset router without changing Session 2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.utils.runtime import configure_pipeline_runtime
configure_pipeline_runtime()

import torch

from src.configs.config import RSCConfig
from src.data.benchmark import load_gsm8k_test_rows as load_benchmark_rows
from src.evaluation.benchmark import run_benchmark, write_record, write_report
from src.models.loading import load_frozen_phi2 as build_model
from src.steering.steernet import load_layer_norms
from src.training.adaptive_data import build_adaptive_rft_splits
from src.training.adaptive_router_training import (
    load_adaptive_router,
    select_static_bank_action,
    train_adaptive_router,
)
from src.training.adaptive_subset_training import train_adaptive_steering_bank


V2_ROUTE = (12, 15, 18, 21)
DEFAULT_CANDIDATES = (8, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24, 26)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train_bank", "train_router", "benchmark"), required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("report/session3"))
    parser.add_argument("--candidate_layers", type=int, nargs="+", default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--norm_calibration", type=Path, required=True)
    parser.add_argument("--v2_norm_calibration", type=Path, required=True)
    parser.add_argument("--v2_checkpoint", type=Path, required=True)
    parser.add_argument("--bank_checkpoint", type=Path, default=Path("checkpoints/session3_steering_bank.pt"))
    parser.add_argument("--router_checkpoint", type=Path, default=Path("checkpoints/session3_router.pt"))
    parser.add_argument("--rollout_files", type=Path, nargs="*")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_questions", type=int, default=2000)
    parser.add_argument("--max_rows_per_question", type=int, default=2)
    parser.add_argument("--max_router_rows", type=int, default=128)
    parser.add_argument("--route_cost", type=float, default=0.01)
    parser.add_argument("--static_screen_questions", type=int, default=16)
    parser.add_argument("--diversify_prompts", action="store_true")
    parser.add_argument("--route_examples_per_layer", type=int, default=256)
    parser.add_argument("--screen_validation_rows", type=int, default=24)
    parser.add_argument("--max_pair_routes", type=int, default=12)
    parser.add_argument("--max_triple_routes", type=int, default=8)
    parser.add_argument("--max_quad_routes", type=int, default=6)
    parser.add_argument("--bench_n", type=int, default=800)
    parser.add_argument("--bench_seed", type=int, default=42)
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _configure(args):
    cfg = RSCConfig(steernet_arch="v2")
    cfg.validate()
    candidates = tuple(sorted(set(int(value) for value in args.candidate_layers)))
    if not set(V2_ROUTE).issubset(candidates):
        raise ValueError(f"candidate layers must include the fixed V2 route {list(V2_ROUTE)}")
    if len(candidates) < 4 or any(layer < 0 or layer >= 32 for layer in candidates):
        raise ValueError("candidate layer set must contain 4–32 valid Phi-2 layer indices")
    cfg.candidate_layers = list(candidates)
    # The wrapper uses fixed_layers as its hook registry; routing still selects
    # a per-example subset from this complete registry.
    cfg.fixed_layers = list(candidates)
    cfg.objective = "nll"
    cfg.use_gate = False
    cfg.diversify_prompts = bool(args.diversify_prompts)
    cfg.steernet_use_mag_head = False
    cfg.steernet_use_conv = False
    cfg.steer_scale = 1.0
    cfg.phase1_lr = 1.5e-4
    cfg.max_tok_len = 768
    cfg.generation_repetition_penalty = 1.0
    cfg.generation_no_repeat_ngram = 0
    cfg.layer_norms_calibration = str(args.norm_calibration)
    cfg.report_dir = str(args.output_dir)
    return cfg, candidates


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(cfg, device):
    model, tokenizer = build_model(cfg, device)
    if any(parameter.requires_grad for parameter in model.base_model.parameters()):
        raise RuntimeError("Phi-2 backbone must remain frozen")
    return model, tokenizer


def _apply_norm_refs(model, norm_path: Path) -> dict[int, float]:
    norms = load_layer_norms(str(norm_path), statistic="p99_norm")
    missing = [layer for layer in model.target_layers if norms.get(layer, 0.0) <= 0]
    if missing:
        raise ValueError(f"p99 norm calibration is missing candidate layers: {missing}")
    for layer in model.target_layers:
        model.steer_nets[str(layer)].layer_norm_ref = float(norms[layer])
    return norms


def _load_v2_weights(model, path: Path, norm_path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"V2 fixed-route checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    if metadata.get("steernet_arch") != "v2" or tuple(metadata.get("route", ())) != V2_ROUTE:
        raise ValueError("checkpoint is not the verified Session 2 V2 route [12, 15, 18, 21]")
    if int(metadata.get("rank", -1)) != 16:
        raise ValueError("V2 checkpoint rank must be 16")
    source = checkpoint["steer_nets"]
    for layer in V2_ROUTE:
        prefix = f"{layer}."
        layer_state = {key[len(prefix):]: value for key, value in source.items() if key.startswith(prefix)}
        if not layer_state:
            raise ValueError(f"V2 checkpoint has no adapter weights for layer {layer}")
        model.steer_nets[str(layer)].load_state_dict(layer_state, strict=True)
    original_norms = load_layer_norms(str(norm_path), statistic="p99_norm")
    missing = [layer for layer in V2_ROUTE if original_norms.get(layer, 0.0) <= 0]
    if missing:
        raise ValueError(f"V2 p99 calibration is missing its fixed route layers: {missing}")
    for layer in V2_ROUTE:
        model.steer_nets[str(layer)].layer_norm_ref = float(original_norms[layer])
    return {"metadata": metadata, "norm_refs": original_norms}


def _load_bank(model, path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "ars_adaptive_steering_v1":
        raise ValueError("unsupported adaptive steering bank checkpoint")
    if tuple(checkpoint["candidate_layers"]) != tuple(model.target_layers):
        raise ValueError("steering-bank candidate layers do not match the active model")
    model.steer_nets.load_state_dict(checkpoint["steer_nets"], strict=True)
    for layer, value in checkpoint["norm_refs"].items():
        model.steer_nets[str(layer)].layer_norm_ref = float(value)
    return checkpoint


def _load_rollouts(args) -> list[Path]:
    paths = list(args.rollout_files or [])
    if not paths:
        paths = [args.output_dir / "rollouts" / "shard0.jsonl", args.output_dir / "rollouts" / "shard1.jsonl"]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"verified rollout files are missing: {missing}")
    return paths


def run(args) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    has_bank_artifacts = args.bank_checkpoint.is_file() and args.norm_calibration.is_file()
    has_router_artifacts = has_bank_artifacts and args.router_checkpoint.is_file()
    if args.stage == "benchmark" and not has_bank_artifacts:
        # Preserve a useful baseline/V2 paired report when adaptive training did
        # not finish; the notebook can still complete and report the missing arm.
        args.candidate_layers = list(V2_ROUTE)
        args.norm_calibration = args.v2_norm_calibration
    cfg, candidates = _configure(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[session3] stage={args.stage} candidates={list(candidates)} device={device}", flush=True)
    model, tokenizer = _load_model(cfg, device)

    if args.stage == "train_bank":
        rollouts = _load_rollouts(args)
        _apply_norm_refs(model, args.norm_calibration)
        _load_v2_weights(model, args.v2_checkpoint, args.v2_norm_calibration)
        train_rows, validation_rows, stats = build_adaptive_rft_splits(
            rollouts,
            tokenizer,
            cfg,
            max_questions=args.max_questions,
            max_rows_per_question=args.max_rows_per_question,
            seed=args.seed,
            diversify_prompt_templates=args.diversify_prompts,
        )
        result = train_adaptive_steering_bank(
            model,
            train_rows,
            validation_rows,
            cfg,
            device,
            epochs=args.epochs,
            save_path=args.bank_checkpoint,
            seed=args.seed,
            route_examples_per_layer=args.route_examples_per_layer,
            screen_validation_rows=args.screen_validation_rows,
            max_pair_routes=args.max_pair_routes,
            max_triple_routes=args.max_triple_routes,
            max_quad_routes=args.max_quad_routes,
            anchor_routes=[V2_ROUTE],
        )
        print(json.dumps({"data": stats, "training": result}, default=str), flush=True)
        return 0

    if args.stage == "train_router":
        rollouts = _load_rollouts(args)
        _apply_norm_refs(model, args.norm_calibration)
        bank_state = _load_bank(model, args.bank_checkpoint)
        train_rows, validation_rows, stats = build_adaptive_rft_splits(
            rollouts,
            tokenizer,
            cfg,
            max_questions=args.max_questions,
            max_rows_per_question=args.max_rows_per_question,
            seed=args.seed,
            diversify_prompt_templates=args.diversify_prompts,
        )
        static_action = select_static_bank_action(
            model,
            validation_rows,
            candidates,
            bank_state.get("curriculum", {}).get("catalog", []),
            device,
            route_cost=args.route_cost,
            max_questions=args.static_screen_questions,
        )
        static_action.update({
            "format": "ars_static_bank_action_v1",
            "bank_checkpoint": str(args.bank_checkpoint),
            "bank_sha256": _sha256(args.bank_checkpoint),
            "candidate_layers": list(candidates),
            "seed": args.seed,
        })
        static_action_path = args.output_dir / "session3_static_bank_action.json"
        temporary = static_action_path.with_suffix(static_action_path.suffix + ".tmp")
        temporary.write_text(json.dumps(static_action, indent=2) + "\n", encoding="utf-8")
        temporary.replace(static_action_path)
        result = train_adaptive_router(
            model,
            train_rows,
            validation_rows,
            candidates,
            device,
            epochs=args.epochs,
            save_path=args.router_checkpoint,
            probe_layer=min(candidates) - 1,
            seed=args.seed,
            max_train_rows=args.max_router_rows,
            route_cost=args.route_cost,
            # Hold gain at the bank's training value to isolate route selection.
            alpha_values=(1.0,),
            route_catalog=bank_state.get("curriculum", {}).get("catalog"),
        )
        print(json.dumps({"data": stats, "training": {k: v for k, v in result.items() if k != "router"}}, default=str), flush=True)
        return 0

    if args.stage != "benchmark":
        raise ValueError(f"unknown stage {args.stage}")
    static_action = None
    if has_bank_artifacts:
        _apply_norm_refs(model, args.norm_calibration)
        _load_bank(model, args.bank_checkpoint)
        static_action_path = args.output_dir / "session3_static_bank_action.json"
        if static_action_path.is_file():
            candidate_action = json.loads(static_action_path.read_text(encoding="utf-8"))
            if candidate_action.get("bank_sha256") == _sha256(args.bank_checkpoint):
                static_action = candidate_action
            else:
                print("[session3] Ignoring static bank action selected for a different bank checkpoint", flush=True)
        if has_router_artifacts:
            router, router_prefill, probe = load_adaptive_router(
                model, args.router_checkpoint, device
            )
        else:
            router_prefill, probe = None, None
        active_arms = ["baseline", "steer_only"]
        if static_action is not None:
            active_arms.append("static_bank")
        if has_router_artifacts:
            active_arms.append("router")
    else:
        router_prefill = None
        probe = None
        active_arms = ["baseline", "steer_only"]
    _load_v2_weights(model, args.v2_checkpoint, args.v2_norm_calibration)

    def arm_setup(arm: str) -> None:
        if arm == "steer_only":
            cfg.fixed_layers = list(V2_ROUTE)
            _load_v2_weights(model, args.v2_checkpoint, args.v2_norm_calibration)
        elif arm == "static_bank" and static_action is not None:
            cfg.static_bank_route = list(static_action["route"])
            cfg.static_bank_alpha = float(static_action["alpha"])
            cfg.fixed_layers = list(candidates)
            _load_bank(model, args.bank_checkpoint)
        elif arm == "router":
            cfg.fixed_layers = list(candidates)
            _load_bank(model, args.bank_checkpoint)

    rows = load_benchmark_rows(args.bench_n, seed=args.bench_seed)
    record = run_benchmark(
        model,
        tokenizer,
        rows,
        cfg,
        device,
        args.shots,
        active_arms,
        contract=None,
        router_prefill_fn=router_prefill,
        arm_setup=arm_setup,
        test_seed=args.bench_seed,
    )
    record["fixed_v2_route"] = list(V2_ROUTE)
    record["router_candidate_layers"] = list(candidates)
    record["router_checkpoint"] = str(args.router_checkpoint)
    record["static_bank_action"] = static_action
    record["static_bank_checkpoint"] = str(args.bank_checkpoint) if static_action else None
    record["fixed_v2_checkpoint"] = str(args.v2_checkpoint)
    record["router_status"] = "included" if "router" in active_arms else "not_available"
    record_path = args.output_dir / f"session3_gsm8k_{args.bench_n}_seed{args.bench_seed}.json"
    write_record(record, str(record_path))
    write_report(record, str(args.output_dir))
    if probe is not None:
        probe.close()
    print(f"[session3] wrote {record_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except Exception as error:
        print(f"[session3] phase failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
