"""Train and compare the adaptive subset router without changing Session 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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
from src.evaluation.ood import SPECS as OOD_SPECS, load_ood_rows
from src.evaluation.benchmark import run_benchmark, write_record, write_report
from src.models.loading import load_frozen_phi2 as build_model
from src.steering.steernet import load_layer_norms
from src.training.adaptive_data import (
    build_adaptive_preference_pairs,
    build_adaptive_rft_splits,
)
from src.training.adaptive_router_training import (
    load_adaptive_router,
    select_static_bank_action,
    train_adaptive_router,
)
from src.training.adaptive_subset_training import train_adaptive_steering_bank


V2_ROUTE = (12, 15, 18, 21)
DEFAULT_CANDIDATES = (8, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24, 26)
DEFAULT_ROUTER_ALPHAS = (1.0, 1.5, 2.0)


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
    parser.add_argument("--router_alphas", type=float, nargs="+", default=list(DEFAULT_ROUTER_ALPHAS))
    parser.add_argument("--static_screen_questions", type=int, default=16)
    parser.add_argument("--preference_weight", type=float, default=0.25)
    parser.add_argument("--require_all_arms", action="store_true")
    parser.add_argument("--ood_datasets", nargs="*", default=[])
    parser.add_argument("--ood_n", type=int, default=200)
    parser.add_argument("--task_validation_rows", type=int, default=32)
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
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
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


def _split_validation_rows(rows: Sequence[dict], seed: int) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Create disjoint bank-screen, router-validation, and action-selection sets."""
    grouped: dict[int, list[dict]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(int(row.get("question_index", index)), []).append(row)
    question_ids = list(grouped)
    random.Random(seed).shuffle(question_ids)
    if len(question_ids) < 3:
        # Tiny smoke runs cannot support independent selection sets. Keep them
        # executable, but mark their validation as diagnostic rather than final.
        return list(rows), list(rows), list(rows), {
            "independent": False,
            "reason": "fewer than three validation questions",
            "question_counts": [len(question_ids)] * 3,
        }
    chunks = [question_ids[i::3] for i in range(3)]
    partitions = [[row for qid in ids for row in grouped[qid]] for ids in chunks]
    return partitions[0], partitions[1], partitions[2], {
        "independent": True,
        "question_counts": [len(ids) for ids in chunks],
        "question_ids": [list(map(int, ids)) for ids in chunks],
    }


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
        train_rows, validation_rows, stats = build_adaptive_rft_splits(
            rollouts,
            tokenizer,
            cfg,
            max_questions=args.max_questions,
            max_rows_per_question=args.max_rows_per_question,
            seed=args.seed,
            diversify_prompt_templates=args.diversify_prompts,
        )
        bank_validation, _, _, validation_split = _split_validation_rows(validation_rows, args.seed + 101)
        train_pairs, train_pair_stats = build_adaptive_preference_pairs(
            rollouts, tokenizer, cfg, stats["train_question_indices"],
            max_length=cfg.max_tok_len, seed=args.seed + 211,
            diversify_prompt_templates=args.diversify_prompts,
        )
        bank_validation_ids = sorted({int(row["question_index"]) for row in bank_validation})
        validation_pairs, validation_pair_stats = build_adaptive_preference_pairs(
            rollouts, tokenizer, cfg, bank_validation_ids,
            max_length=cfg.max_tok_len, seed=args.seed + 307,
            diversify_prompt_templates=args.diversify_prompts,
        )
        stats["train_question_count"] = len(stats.pop("train_question_indices"))
        stats["validation_question_count"] = len(stats.pop("validation_question_indices"))
        result = train_adaptive_steering_bank(
            model,
            train_rows,
            bank_validation,
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
            preference_pairs=train_pairs,
            validation_preference_pairs=validation_pairs,
            preference_weight=args.preference_weight,
        )
        print(json.dumps({
            "data": stats,
            "preference_data": {"train": train_pair_stats, "validation": validation_pair_stats},
            "validation_split": validation_split,
            "bank_initialization": "fresh_adapter_initialization; fixed V2 checkpoint is not loaded into bank",
            "training": result,
        }, default=str), flush=True)
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
        _, router_validation, static_validation, validation_split = _split_validation_rows(
            validation_rows, args.seed + 101
        )
        train_pairs, train_pair_stats = build_adaptive_preference_pairs(
            rollouts, tokenizer, cfg, stats["train_question_indices"],
            max_length=cfg.max_tok_len, seed=args.seed + 211,
            diversify_prompt_templates=args.diversify_prompts,
        )
        router_validation_ids = sorted({int(row["question_index"]) for row in router_validation})
        router_validation_pairs, validation_pair_stats = build_adaptive_preference_pairs(
            rollouts, tokenizer, cfg, router_validation_ids,
            max_length=cfg.max_tok_len, seed=args.seed + 307,
            diversify_prompt_templates=args.diversify_prompts,
        )
        static_validation_ids = sorted({int(row["question_index"]) for row in static_validation})
        static_validation_pairs, static_pair_stats = build_adaptive_preference_pairs(
            rollouts, tokenizer, cfg, static_validation_ids,
            max_length=cfg.max_tok_len, seed=args.seed + 401,
            diversify_prompt_templates=args.diversify_prompts,
        )
        stats["train_question_count"] = len(stats.pop("train_question_indices"))
        stats["validation_question_count"] = len(stats.pop("validation_question_indices"))
        static_action = select_static_bank_action(
            model,
            static_validation,
            candidates,
            bank_state.get("curriculum", {}).get("catalog", []),
            device,
            route_cost=args.route_cost,
            alpha_values=(1.0, 1.5, 2.0),
            max_questions=args.static_screen_questions,
            tokenizer=tokenizer,
            cfg=cfg,
            preference_pairs=static_validation_pairs,
        )
        static_action.update({
            "format": "ars_static_bank_action_v1",
            "bank_checkpoint": str(args.bank_checkpoint),
            "bank_sha256": _sha256(args.bank_checkpoint),
            "candidate_layers": list(candidates),
            "seed": args.seed,
            "validation_split": validation_split,
            "preference_validation": static_pair_stats,
        })
        static_action_path = args.output_dir / "session3_static_bank_action.json"
        temporary = static_action_path.with_suffix(static_action_path.suffix + ".tmp")
        temporary.write_text(json.dumps(static_action, indent=2) + "\n", encoding="utf-8")
        temporary.replace(static_action_path)
        result = train_adaptive_router(
            model,
            train_rows,
            router_validation,
            candidates,
            device,
            epochs=args.epochs,
            save_path=args.router_checkpoint,
            probe_layer=min(candidates) - 1,
            seed=args.seed,
            max_train_rows=args.max_router_rows,
            route_cost=args.route_cost,
            # Learn question-conditioned route and gain jointly from the same oracle scores.
            alpha_values=args.router_alphas,
            route_catalog=bank_state.get("curriculum", {}).get("catalog"),
            preference_pairs=train_pairs,
            validation_preference_pairs=router_validation_pairs,
            bank_sha256=_sha256(args.bank_checkpoint),
            tokenizer=tokenizer,
            cfg=cfg,
            task_validation_limit=args.task_validation_rows,
            static_action=static_action,
        )
        result["preference_data"] = {"train": train_pair_stats, "validation": validation_pair_stats}
        result["validation_split"] = validation_split
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
                model, args.router_checkpoint, device,
                expected_bank_sha256=_sha256(args.bank_checkpoint),
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
    expected_arms = ["baseline", "steer_only", "static_bank", "router"]
    if args.require_all_arms and active_arms != expected_arms:
        raise RuntimeError(
            f"four-arm benchmark prerequisites missing: evaluated={active_arms}, expected={expected_arms}"
        )
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

    suites = ["gsm8k", *args.ood_datasets]
    unknown_ood = sorted(set(args.ood_datasets) - set(OOD_SPECS))
    if unknown_ood:
        raise ValueError(f"unsupported OOD benchmark suites: {unknown_ood}")
    if any(OOD_SPECS[name][2] != "numeric" for name in args.ood_datasets):
        raise ValueError("Session 3 four-arm OOD benchmark currently supports numeric-answer datasets only")
    records = []
    suite_status = {}
    for suite in suites:
        try:
            n_test = args.bench_n if suite == "gsm8k" else args.ood_n
            if suite == "gsm8k":
                rows = load_benchmark_rows(n_test, seed=args.bench_seed)
            else:
                source_rows = load_ood_rows(suite)
                chooser = random.Random(args.bench_seed)
                if n_test < len(source_rows):
                    selected_indices = sorted(chooser.sample(range(len(source_rows)), n_test))
                    source_rows = [source_rows[index] for index in selected_indices]
                rows = [
                    {"source_index": int(row["idx"]), "question": row["question"], "answer": row["answer"]}
                    for row in source_rows[:n_test]
                ]
            record = run_benchmark(
                model,
                tokenizer,
                rows,
                cfg,
                device,
                args.shots,
                active_arms,
                contract=None,
                suite=suite,
                router_prefill_fn=router_prefill,
                arm_setup=arm_setup,
                test_seed=args.bench_seed,
            )
            record["evaluated_arms"] = list(active_arms)
            record["expected_arms"] = expected_arms
            record["four_arm_complete"] = active_arms == expected_arms
            record["fixed_v2_route"] = list(V2_ROUTE)
            record["router_candidate_layers"] = list(candidates)
            record["router_checkpoint"] = str(args.router_checkpoint)
            record["static_bank_action"] = static_action
            record["static_bank_checkpoint"] = str(args.bank_checkpoint) if static_action else None
            record["fixed_v2_checkpoint"] = str(args.v2_checkpoint)
            record["router_status"] = "included" if "router" in active_arms else "not_available"
            if args.require_all_arms and not record["four_arm_complete"]:
                raise RuntimeError(f"incomplete four-arm record for suite {suite}: {active_arms}")
            record_path = args.output_dir / f"session3_{suite}_{n_test}_seed{args.bench_seed}.json"
            write_record(record, str(record_path))
            write_report(record, str(args.output_dir))
            print(f"[session3] wrote {record_path}", flush=True)
            records.append(str(record_path))
            suite_status[suite] = {"status": "passed_four_arms", "record": str(record_path), "n": len(rows)}
        except Exception as error:
            suite_status[suite] = {"status": f"failed: {type(error).__name__}: {error}"}
            print(f"[session3] {suite} benchmark failed; continuing to remaining suites: {error}", file=sys.stderr, flush=True)
    status_path = args.output_dir / "session3_benchmark_status.json"
    status_path.write_text(json.dumps({"suites": suite_status, "records": records}, indent=2) + "\n", encoding="utf-8")
    if suite_status.get("gsm8k", {}).get("status") != "passed_four_arms":
        raise RuntimeError(f"required GSM8K four-arm benchmark did not complete: {suite_status.get('gsm8k')}")
    if probe is not None:
        probe.close()
    print(json.dumps({"benchmark_records": records, "evaluated_arms": active_arms,
                      "suite_status": suite_status}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except Exception as error:
        print(f"[session3] phase failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
