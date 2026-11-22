"""Collect GSM8K rollouts, train fixed steering, and calibrate its strength."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.runtime import configure_pipeline_runtime
configure_pipeline_runtime()

import torch

from src.configs.config import RSCConfig
from src.data.benchmark import materialize_dataset_split as prepare_dataset_split
from src.data.dataset import gsm8k_to_cot
from src.evaluation.generation import equivalence_test
from src.models.loading import load_frozen_phi2 as build_model
from src.training.rft_collect import build_rft_rows, collect_rft_data
from src.training.rft_trainer import (
    evaluate_generation,
    fit_prompt_strength_controller,
    train_phase1_rft,
)
from src.training.trainer import load_steer_weights


LEGACY_FEATURE_ROUTE = [17, 20, 23, 26]
FIXED_LOW_RANK_ROUTE = [12, 15, 18, 21]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "equivalence",
            "rollout_quality_probe",
            "gate0",
            "collect",
            "train",
            "strength_teacher",
            "alpha_sweep",
            "all",
        ),
        default="equivalence",
    )
    parser.add_argument(
        "--n_questions",
        type=int,
        default=None,
        help=(
            "question count for the selected stage (defaults: quality probe=60, "
            "equivalence=100, collection=2000)"
        ),
    )
    parser.add_argument(
        "--quality_probe_nonblocking",
        "--gate0_nonblocking",
        dest="quality_probe_nonblocking",
        action="store_true",
        help="record rollout quality without aborting when its diagnostic threshold is missed",
    )
    parser.add_argument(
        "--strict_equivalence",
        action="store_true",
        help="block generation stages when batched equivalence thresholds fail",
    )
    parser.add_argument(
        "--allow_low_yield",
        action="store_true",
        help="allow low positive-sample yield for smoke collection runs",
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--retry_k",
        type=int,
        default=0,
        help="targeted retries for questions with no strict-positive initial sample",
    )
    parser.add_argument("--retry_temperature", type=float, default=0.9)
    parser.add_argument("--target_positive_samples", type=int, default=2)
    parser.add_argument(
        "--collection_shot_weights",
        type=float,
        nargs=3,
        default=(0.60, 0.25, 0.15),
        metavar=("ZERO_SHOT", "TWO_SHOT", "FOUR_SHOT"),
    )
    parser.add_argument("--diversify_collection_templates", action="store_true")
    parser.add_argument("--max_rft_rows_per_question", type=int, default=3)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--eval_questions",
        type=int,
        default=100,
        help=(
            "number of held-out questions used for baseline and per-epoch generation "
            "evaluation during training (default: 100)"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=None, help="RFT micro-batch size")
    parser.add_argument("--grad_accum", type=int, default=None, help="RFT gradient accumulation steps")
    parser.add_argument(
        "--train_shot_weights",
        type=float,
        nargs=3,
        default=(0.60, 0.25, 0.15),
        metavar=("ZERO_SHOT", "TWO_SHOT", "FOUR_SHOT"),
        help="sampling weights for 0/2/4-shot RFT rows",
    )
    parser.add_argument(
        "--diversify_prompt_templates",
        action="store_true",
        help="cycle through meaning-equivalent prompt instructions during RFT row building",
    )
    parser.add_argument(
        "--steering_architecture",
        "--steernet_arch",
        dest="steernet_arch",
        choices=(
            "legacy_feature_residual",
            "v5",
            "fixed_route_low_rank_residual",
            "v2",
            "prompt_conditioned_residual",
        ),
        default="v5",
        help=(
            "adapter architecture; fixed_route_low_rank_residual is the rank-16 "
            "calibrated fixed-route adapter; prompt_conditioned_residual is the "
            "rank-32 prompt-conditioned adapter; legacy labels remain accepted"
        ),
    )
    parser.add_argument(
        "--skip_collection",
        action="store_true",
        help="train only from existing rollout shards; never start rollout generation",
    )
    parser.add_argument(
        "--rollout_dir",
        help="directory containing existing shard0.jsonl and shard1.jsonl",
    )
    parser.add_argument(
        "--allow_partial_rollouts",
        action="store_true",
        help="when resuming, train from any non-empty existing shard if another shard failed",
    )
    parser.add_argument("--norm_calibration")
    parser.add_argument("--output_dir", default="report/rft")
    parser.add_argument("--checkpoint", default="checkpoints/steernet.pt")
    parser.add_argument(
        "--strength_teacher_questions",
        type=int,
        default=256,
        help="training questions used to fit prompt-conditioned steering strength",
    )
    parser.add_argument(
        "--strength_candidates",
        type=float,
        nargs="+",
        default=(0.0, 0.5, 0.75, 1.0, 1.25),
        help="bounded strengths compared with greedy exact-answer scoring on training questions",
    )
    parser.add_argument("--strength_epochs", type=int, default=40)
    parser.add_argument("--strength_batch_size", type=int, default=32)
    parser.add_argument(
        "--alpha_values", type=float, nargs="+", default=[0.0, 0.5, 0.75, 1.0, 1.25],
        help="post-cap steering intensities for held-out 0/2-shot calibration",
    )
    args = parser.parse_args(argv)
    architecture_aliases = {
        "legacy_feature_residual": "v5",
        "fixed_route_low_rank_residual": "v2",
    }
    requested_architecture = args.steernet_arch
    args.architecture_id = {
        "legacy_feature_residual": "ars.legacy_feature_residual",
        "v5": "ars.legacy_feature_residual",
        "fixed_route_low_rank_residual": "ars.fixed_route_low_rank_residual",
        "v2": "ars.fixed_route_low_rank_residual",
        "prompt_conditioned_residual": "ars.prompt_conditioned_residual",
    }[requested_architecture]
    args.steernet_arch = architecture_aliases.get(requested_architecture, requested_architecture)
    # Preserve the former argparse attribute for legacy callers.
    args.gate0_nonblocking = args.quality_probe_nonblocking
    if args.stage == "gate0":
        args.stage = "rollout_quality_probe"
    return args


def _require_rollout_quality_probe(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"rollout quality probe result not found at {path}; "
            "run --stage rollout_quality_probe before collection"
        )
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("passed", False):
        raise RuntimeError("rollout quality probe did not pass; refusing to start collection")
    return result


def _require_gate0(path: Path) -> dict:
    """Compatibility wrapper for checkpoints and callers using the former name."""
    return _require_rollout_quality_probe(path)


def _require_equivalence(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"generation equivalence result not found at {path}; run --stage equivalence first"
        )
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("quality_gate_passed", result.get("passed", False)):
        raise RuntimeError("generation equivalence failed; refusing collection or training")
    return result


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _enforce_collection_yield(stats: dict, allow_low_yield: bool = False) -> None:
    """Record and enforce the positive-completion yield gate for an RFT shard."""
    yield_passed = stats["positive_sample_rate"] >= 0.15
    stats["minimum_yield_passed"] = yield_passed
    stats["low_yield_allowed"] = bool(allow_low_yield)
    if yield_passed:
        return
    if allow_low_yield:
        print(
            "RFT positive-sample yield is below 15%; continuing because "
            "--allow_low_yield is enabled for this smoke run.",
            flush=True,
        )
        return
    raise RuntimeError(
            "RFT collection yield is below 15%; inspect the rollout quality probe "
            "and strict acceptance before training"
    )


def _run_rollout_quality_probe(
    model,
    tokenizer,
    train_rows,
    cfg,
    device,
    output_dir,
    k,
    temperature,
    n_questions=60,
    nonblocking=False,
):
    indices = random.Random(cfg.seed).sample(
        range(len(train_rows)), min(n_questions, len(train_rows))
    )
    rollout_path = output_dir / "rollout_quality_probe.jsonl"
    stats = collect_rft_data(
        model,
        tokenizer,
        train_rows,
        device,
        indices=indices,
        output_path=rollout_path,
        k=k,
        temperature=temperature,
        max_new_tokens=200,
        max_prompt_len=cfg.max_tok_len,
        shield_policy=cfg.shield_policy,
        compare_greedy=True,
        seed=cfg.seed,
        progress_desc=f"Rollout quality probe pass@{k}",
    )
    quality_gate_passed = (
        stats["n"] >= 1
        and stats["pass_at_k_lenient"] >= 0.70
        and stats["pass_at_k_delta_vs_greedy_pp"] > 0
    )
    stats["quality_gate_passed"] = quality_gate_passed
    stats["nonblocking"] = bool(nonblocking)
    stats["passed"] = quality_gate_passed or nonblocking
    _write_json(output_dir / "rollout_quality_probe.json", stats)
    _write_json(output_dir / "gate0_stats.json", stats)
    if not stats["passed"]:
        raise RuntimeError(
            "rollout quality probe failed: sampled pass@k must reach 70% "
            "and beat paired greedy accuracy"
        )
    if nonblocking and not quality_gate_passed:
        print(
            "Rollout quality threshold was not met; continuing because "
            "--quality_probe_nonblocking is enabled.",
            flush=True,
        )
    return stats


def _gate0(*args, **kwargs):
    """Compatibility alias for older notebooks calling the former stage name."""
    return _run_rollout_quality_probe(*args, **kwargs)


def run(args) -> dict | None:
    if (
        (args.n_questions is not None and args.n_questions < 1)
        or args.eval_questions < 1
        or args.k < 1
        or args.retry_k < 0
        or args.target_positive_samples < 1
        or args.max_rft_rows_per_question < 1
        or args.n_shards < 1
    ):
        raise ValueError(
            "n_questions, eval_questions, k, target_positive_samples, "
            "max_rft_rows_per_question, and n_shards must be positive; retry_k cannot be negative"
        )
    if (
        len(args.collection_shot_weights) != 3
        or any(not math.isfinite(weight) for weight in args.collection_shot_weights)
        or any(weight < 0 for weight in args.collection_shot_weights)
        or sum(args.collection_shot_weights) <= 0
    ):
        raise ValueError("collection_shot_weights must be three nonnegative values with positive sum")
    if (
        args.stage in {"rollout_quality_probe", "collect", "all"}
        and (not math.isfinite(args.temperature) or args.temperature <= 0)
    ) or (
        args.retry_k > 0
        and (not math.isfinite(args.retry_temperature) or args.retry_temperature <= 0)
    ):
        raise ValueError("collection and retry temperatures must be positive")
    if not 0 <= args.shard < args.n_shards:
        raise ValueError("shard must satisfy 0 <= shard < n_shards")
    if args.stage == "all" and args.n_shards != 1:
        raise ValueError("--stage all runs a single collection shard; use separate collect/train stages for sharding")
    if args.skip_collection and args.stage != "train":
        raise ValueError("--skip_collection is only valid with --stage train")
    if args.stage == "strength_teacher":
        if args.steernet_arch != "prompt_conditioned_residual":
            raise ValueError(
                "--stage strength_teacher requires "
                "--steering_architecture prompt_conditioned_residual"
            )
        if args.strength_teacher_questions < 10:
            raise ValueError("strength_teacher_questions must be at least 10")
        if args.strength_epochs < 1 or args.strength_batch_size < 1:
            raise ValueError("strength_epochs and strength_batch_size must be positive")
        if len(args.strength_candidates) < 2 or len(set(args.strength_candidates)) != len(args.strength_candidates):
            raise ValueError("strength_candidates must contain at least two distinct values")
        if any(
            not math.isfinite(value) or value < 0 or value > 1.25
            for value in args.strength_candidates
        ):
            raise ValueError("strength_candidates must be finite and within [0, 1.25]")

    equivalence_questions = args.n_questions or 100
    quality_probe_questions = args.n_questions or 60
    collection_questions = args.n_questions or 2000

    cfg = RSCConfig(steernet_arch=args.steernet_arch)
    cfg.diversify_prompts = bool(args.diversify_prompt_templates)
    route = (
        FIXED_LOW_RANK_ROUTE
        if args.steernet_arch in {"v2", "prompt_conditioned_residual"}
        else LEGACY_FEATURE_ROUTE
    )
    cfg.fixed_layers = list(route)
    cfg.candidate_layers = sorted(set(cfg.candidate_layers) | set(route))
    if args.steernet_arch in {"v2", "prompt_conditioned_residual"}:
        cfg.lora_rank = 16 if args.steernet_arch == "v2" else 32
        cfg.rho_max = 0.10
        cfg.steernet_norm_stat = "p99_norm"
        cfg.steernet_use_conv = False
        if args.norm_calibration is None:
            raise ValueError(
                "fixed low-rank steering requires --norm_calibration with p99 references"
            )
        cfg.layer_norms_calibration = args.norm_calibration
        calibration_path = Path(args.norm_calibration)
        if not calibration_path.is_file():
            raise FileNotFoundError(
                f"p99 norm calibration file not found: {calibration_path}"
            )
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibrated_layers = calibration.get("layers", {})
        missing_p99 = [
            str(layer)
            for layer in FIXED_LOW_RANK_ROUTE
            if "p99_norm" not in calibrated_layers.get(str(layer), {})
        ]
        if missing_p99:
            raise ValueError(
                f"p99 calibration is missing required steering layers: {missing_p99}"
            )
    cfg.steernet_use_mag_head = False
    cfg.mag_l1_weight = 0.0
    cfg.s1_answer_weight = 1.0
    cfg.on_policy = False
    cfg.use_gate = False
    cfg.max_tok_len = 768
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("batch_size must be positive")
        cfg.batch_size = args.batch_size
    if args.grad_accum is not None:
        if args.grad_accum < 1:
            raise ValueError("grad_accum must be positive")
        cfg.grad_accum = args.grad_accum
    cfg.validate()
    print(
        f"[steering-training] architecture={args.architecture_id} "
        f"route={cfg.fixed_layers} steering_rank={cfg.lora_rank} "
        f"strength_head={cfg.steernet_arch == 'prompt_conditioned_residual'} "
        f"rho_max={cfg.rho_max} norm_reference={cfg.steernet_norm_stat}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Phase 1 RFT expects a CUDA runtime such as Kaggle T4")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_dir = str(output_dir)
    cfg.checkpoint_dir = str(output_dir / "checkpoints")
    rollout_source_dir = Path(args.rollout_dir) if args.rollout_dir else output_dir
    if args.skip_collection:
        required_shards = [
            rollout_source_dir / "shard0.jsonl",
            rollout_source_dir / "shard1.jsonl",
        ]
        missing = [str(path) for path in required_shards if not path.is_file()]
        available = [path for path in required_shards if path.is_file() and path.stat().st_size > 0]
        if missing and not (args.allow_partial_rollouts and available):
            raise FileNotFoundError(
                "fixed-steering training is configured to reuse the supplied rollout "
                f"shards; missing files: {missing}"
            )
        if not available:
            raise ValueError("at least one non-empty RFT rollout shard is required")
        if any(path.exists() and path.stat().st_size == 0 for path in required_shards):
            raise ValueError("existing RFT rollout shards must not be empty")
    equivalence_path = output_dir / "generation_equivalence.json"
    if args.strict_equivalence and args.stage in {
        "rollout_quality_probe", "collect", "train"
    }:
        _require_equivalence(equivalence_path)
    if args.stage == "collect" or (args.stage == "train" and not args.skip_collection):
        quality_probe_path = output_dir / "rollout_quality_probe.json"
        if not quality_probe_path.is_file():
            quality_probe_path = output_dir / "gate0_stats.json"
        _require_rollout_quality_probe(quality_probe_path)

    model, tokenizer = build_model(cfg, device)
    raw_train, raw_dev = gsm8k_to_cot(tokenizer, cfg)
    train_rows = prepare_dataset_split(raw_train.with_format(None))
    if args.stage in {"collect", "all"} and collection_questions > len(train_rows):
        raise ValueError(
            f"requested {collection_questions} rollout questions, but only "
            f"{len(train_rows)} training questions are available"
        )
    if args.stage in {"equivalence", "all"}:
        dev_limit = max(100, equivalence_questions)
    elif args.stage == "alpha_sweep":
        dev_limit = max(300, args.n_questions or 300)
    elif args.stage == "train":
        dev_limit = args.eval_questions
    else:
        dev_limit = 100
    dev_rows = prepare_dataset_split(raw_dev.with_format(None), limit=dev_limit)

    if args.stage == "strength_teacher":
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"trained steering checkpoint not found: {checkpoint_path}"
            )
        load_steer_weights(model, checkpoint_path, cfg, device)
        report = fit_prompt_strength_controller(
            model,
            tokenizer,
            train_rows,
            cfg,
            device,
            n_questions=args.strength_teacher_questions,
            candidate_strengths=args.strength_candidates,
            shot_weights=args.train_shot_weights,
            epochs=args.strength_epochs,
            batch_size=args.strength_batch_size,
            seed=cfg.seed,
            save_path=checkpoint_path,
        )
        report_path = output_dir / "prompt_strength_training.json"
        _write_json(report_path, report)
        print(json.dumps(report, indent=2), flush=True)
        print(f"Saved prompt-strength report: {report_path}", flush=True)
        return report

    if args.stage == "alpha_sweep":
        if not Path(args.checkpoint).is_file():
            raise FileNotFoundError(f"trained checkpoint not found: {args.checkpoint}")
        if not args.alpha_values or any(not torch.isfinite(torch.tensor(value)) or value < 0 for value in args.alpha_values):
            raise ValueError("alpha_values must be finite and nonnegative")
        load_steer_weights(model, args.checkpoint, cfg, device)
        # RFT checkpoint selection uses the first 100 validation rows; keep
        # intensity calibration on a disjoint segment of the same held-out split.
        count = min(args.n_questions or 200, max(0, len(dev_rows) - 100))
        calibration_rows = dev_rows[100 : 100 + count]
        if not calibration_rows:
            raise ValueError("alpha calibration requires held-out rows beyond the RFT selection slice")
        result = {"n": len(calibration_rows), "alpha_values": args.alpha_values, "shots": {}}
        for n_shot in (0, 2):
            baseline_flags, baseline_metrics = evaluate_generation(
                model, tokenizer, calibration_rows, cfg, device,
                n_shot=n_shot, batch_size=1, steered=False,
            )
            entries = []
            for alpha in args.alpha_values:
                flags, metrics = evaluate_generation(
                    model, tokenizer, calibration_rows, cfg, device,
                    n_shot=n_shot, batch_size=1, steered=True,
                    intervention_gain=alpha,
                )
                delta = 100 * sum(
                    int(current) - int(base)
                    for current, base in zip(flags, baseline_flags)
                ) / len(baseline_flags)
                entries.append({"alpha": alpha, **metrics, "paired_delta_lenient_pp": delta})
            best = max(entries, key=lambda item: (item["paired_delta_lenient_pp"], -item["alpha"]))
            result["shots"][str(n_shot)] = {
                "baseline": baseline_metrics,
                "candidates": entries,
                "selected_alpha": best["alpha"],
            }
        _write_json(Path(args.output_dir) / "alpha_calibration.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return result

    if args.stage in {"equivalence", "all"}:
        model.set_routing_override("none", cfg.fixed_layers)
        result = equivalence_test(
            model,
            tokenizer,
            dev_rows,
            device,
            cfg,
            n=min(equivalence_questions, len(dev_rows)),
            batch_size=2,
        )
        quality_gate_passed = bool(result["passed"])
        result["quality_gate_passed"] = quality_gate_passed
        result["used_for_session1_gate"] = bool(args.strict_equivalence)
        _write_json(equivalence_path, result)
        if not quality_gate_passed:
            print(
                "Batched equivalence thresholds were not met. Session 1 uses "
                "single-question generation; this result is diagnostic unless "
                "--strict_equivalence is enabled.",
                flush=True,
            )
        if not quality_gate_passed and args.strict_equivalence:
            raise RuntimeError(
                "batched generation failed Phase 1 equivalence thresholds; "
                f"n={result['n']} delta={result['delta_lenient_pp']:+.2f}pp "
                f"identity={result['string_identity']:.1%}; inspect "
                f"{equivalence_path} mismatch_examples"
            )
        if args.stage == "equivalence":
            return result

    if args.stage in {"rollout_quality_probe", "all"}:
        quality_probe_stats = _run_rollout_quality_probe(
            model,
            tokenizer,
            train_rows,
            cfg,
            device,
            output_dir,
            args.k,
            args.temperature,
            n_questions=quality_probe_questions,
            nonblocking=args.quality_probe_nonblocking,
        )
        if args.stage == "rollout_quality_probe":
            return quality_probe_stats

    if args.stage in {"collect", "all"}:
        count = collection_questions
        sampled_order = random.Random(cfg.seed).sample(
            range(len(train_rows)), min(count, len(train_rows))
        )
        shard_indices = sampled_order[args.shard :: args.n_shards]
        collect_stats = collect_rft_data(
            model,
            tokenizer,
            train_rows,
            device,
            indices=shard_indices,
            output_path=output_dir / f"shard{args.shard}.jsonl",
            k=args.k,
            temperature=args.temperature,
            max_new_tokens=200,
            max_prompt_len=cfg.max_tok_len,
            shield_policy=cfg.shield_policy,
            retry_k=args.retry_k,
            retry_temperature=args.retry_temperature,
            target_positive_samples=args.target_positive_samples,
            shot_weights=args.collection_shot_weights,
            diversify_templates=args.diversify_collection_templates,
            seed=cfg.seed + args.shard,
            progress_desc=f"RFT shard {args.shard}/{args.n_shards}",
            progress_position=args.shard,
        )
        _enforce_collection_yield(collect_stats, allow_low_yield=args.allow_low_yield)
        _write_json(output_dir / f"shard{args.shard}_stats.json", collect_stats)
        if args.stage == "collect":
            return collect_stats

    if args.stage in {"train", "all"}:
        source_dir = rollout_source_dir
        shard_files = sorted(source_dir.glob("shard*.jsonl"))
        if args.allow_partial_rollouts:
            shard_files = [
                path for path in shard_files
                if path.stat().st_size > 0
                and (source_dir / f"{path.stem}_stats.json").is_file()
            ]
        if not shard_files:
            raise FileNotFoundError(f"no RFT rollout shards found in {source_dir}")
        if args.skip_collection:
            required_shards = {"shard0.jsonl", "shard1.jsonl"}
            found_shards = {path.name for path in shard_files}
            missing_shards = sorted(required_shards - found_shards)
            if missing_shards and not args.allow_partial_rollouts:
                raise FileNotFoundError(
                    f"--skip_collection requires the two complete rollout shards; "
                    f"missing {missing_shards} in {source_dir}"
                )
        rft_rows, row_stats = build_rft_rows(
            shard_files,
            tokenizer,
            cfg,
            max_length=cfg.max_tok_len,
            seed=cfg.seed,
            shot_weights=args.train_shot_weights,
            diversify_prompt_templates=args.diversify_prompt_templates,
            max_rows_per_question=args.max_rft_rows_per_question,
        )
        rows_path = Path(cfg.checkpoint_dir) / "rft_rows.pt"
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"rows": rft_rows, "stats": row_stats}, rows_path)
        result = train_phase1_rft(
            model,
            rft_rows,
            dev_rows,
            cfg,
            device,
            tokenizer,
            epochs=args.epochs,
            dev_limit=args.eval_questions,
            save_path=args.checkpoint,
        )
        result["rft_rows"] = row_stats
        result["training_data"] = {
            "rollout_files": [str(path) for path in shard_files],
            "shot_weights": list(args.train_shot_weights),
            "max_rows_per_question": args.max_rft_rows_per_question,
            "prompt_templates_diversified": args.diversify_prompt_templates,
        }
        _write_json(output_dir / "rft_training_result.json", result)
        return result
    return None


if __name__ == "__main__":
    run(parse_args())
