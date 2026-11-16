"""Run RFT collection or train V5/V2 adapters from existing GSM8K rollouts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_ars_pipeline import build_model, prepare_dataset_split
from src.configs.config import RSCConfig
from src.data.dataset import gsm8k_to_cot
from src.evaluation.generation import equivalence_test
from src.training.rft_collect import build_rft_rows, collect_rft_data
from src.training.rft_trainer import train_phase1_rft


STAGE1_ROUTE = [17, 20, 23, 26]
STAGE2_ROUTE = [12, 15, 18, 21]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("equivalence", "gate0", "collect", "train", "all"),
        default="equivalence",
    )
    parser.add_argument(
        "--n_questions",
        type=int,
        default=None,
        help=(
            "question count for the selected stage (defaults: Gate 0=60, "
            "equivalence=100, collection=2000)"
        ),
    )
    parser.add_argument(
        "--gate0_nonblocking",
        action="store_true",
        help="record Gate 0 quality without aborting when its threshold is missed",
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
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steernet_arch", choices=("v5", "v2"), default="v5")
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
    return parser.parse_args(argv)


def _require_gate0(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Gate 0 result not found at {path}; run --stage gate0 before RFT training"
        )
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("passed", False):
        raise RuntimeError("Gate 0 did not pass; refusing to start RFT training")
    return result


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
        "RFT collection yield is below 15%; inspect Gate 0 and strict acceptance before training"
    )


def _gate0(
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
    rollout_path = output_dir / "gate0.jsonl"
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
        progress_desc=f"Gate 0 pass@{k}",
    )
    quality_gate_passed = (
        stats["n"] >= 1
        and stats["pass_at_k_lenient"] >= 0.70
        and stats["pass_at_k_delta_vs_greedy_pp"] > 0
    )
    stats["quality_gate_passed"] = quality_gate_passed
    stats["nonblocking"] = bool(nonblocking)
    stats["passed"] = quality_gate_passed or nonblocking
    _write_json(output_dir / "gate0_stats.json", stats)
    if not stats["passed"]:
        raise RuntimeError(
            "Gate 0 failed: sampled pass@k must reach 70% and beat paired greedy accuracy"
        )
    if nonblocking and not quality_gate_passed:
        print(
            "Gate 0 quality threshold was not met; continuing because "
            "--gate0_nonblocking is enabled.",
            flush=True,
        )
    return stats


def run(args) -> dict | None:
    if (
        (args.n_questions is not None and args.n_questions < 1)
        or args.k < 1
        or args.n_shards < 1
    ):
        raise ValueError("n_questions, k, and n_shards must be positive")
    if not 0 <= args.shard < args.n_shards:
        raise ValueError("shard must satisfy 0 <= shard < n_shards")
    if args.stage == "all" and args.n_shards != 1:
        raise ValueError("--stage all runs a single collection shard; use separate collect/train stages for sharding")
    if args.skip_collection and args.stage != "train":
        raise ValueError("--skip_collection is only valid with --stage train")

    equivalence_questions = args.n_questions or 100
    gate0_questions = args.n_questions or 60
    collection_questions = args.n_questions or 2000

    cfg = RSCConfig(steernet_arch=args.steernet_arch)
    route = STAGE2_ROUTE if args.steernet_arch == "v2" else STAGE1_ROUTE
    cfg.fixed_layers = list(route)
    cfg.candidate_layers = sorted(set(cfg.candidate_layers) | set(route))
    if args.steernet_arch == "v2":
        cfg.lora_rank = 16
        cfg.rho_max = 0.10
        cfg.steernet_norm_stat = "p99_norm"
        cfg.steernet_use_conv = False
        if args.norm_calibration is None:
            raise ValueError("Session 2 V2 training requires --norm_calibration from p99 calibration")
        cfg.layer_norms_calibration = args.norm_calibration
    cfg.steernet_use_mag_head = False
    cfg.mag_l1_weight = 0.0
    cfg.s1_answer_weight = 1.0
    cfg.on_policy = False
    cfg.use_gate = False
    cfg.max_tok_len = 768
    cfg.validate()
    print(
        f"[rft] architecture={cfg.steernet_arch} route={cfg.fixed_layers} "
        f"rank={cfg.lora_rank} rho_max={cfg.rho_max} norm_stat={cfg.steernet_norm_stat}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Phase 1 RFT expects a CUDA runtime such as Kaggle T4")
    output_dir = Path(args.output_dir)
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
                "Session 2 is configured to reuse Session 1 data and will not collect "
                f"new rollouts; missing files: {missing}"
            )
        if not available:
            raise ValueError("at least one non-empty RFT rollout shard is required")
        if any(path.exists() and path.stat().st_size == 0 for path in required_shards):
            raise ValueError("existing RFT rollout shards must not be empty")
        if args.steernet_arch == "v2":
            calibration_path = Path(args.norm_calibration)
            if not calibration_path.is_file():
                raise FileNotFoundError(
                    f"Session 2 p99 norm calibration file not found: {calibration_path}"
                )
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            calibrated_layers = calibration.get("layers", {})
            missing_p99 = [
                str(layer)
                for layer in STAGE2_ROUTE
                if "p99_norm" not in calibrated_layers.get(str(layer), {})
            ]
            if missing_p99:
                raise ValueError(
                    f"p99 calibration is missing Session 2 layers: {missing_p99}"
                )
    equivalence_path = output_dir / "generation_equivalence.json"
    if args.strict_equivalence and args.stage in {"gate0", "collect", "train"}:
        _require_equivalence(equivalence_path)
    if args.stage == "collect" or (args.stage == "train" and not args.skip_collection):
        _require_gate0(output_dir / "gate0_stats.json")

    model, tokenizer = build_model(cfg, device)
    raw_train, raw_dev = gsm8k_to_cot(tokenizer, cfg)
    train_rows = prepare_dataset_split(raw_train.with_format(None))
    dev_limit = (
        max(100, equivalence_questions)
        if args.stage in {"equivalence", "all"}
        else 100
    )
    dev_rows = prepare_dataset_split(raw_dev.with_format(None), limit=dev_limit)

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

    if args.stage in {"gate0", "all"}:
        gate0_stats = _gate0(
            model,
            tokenizer,
            train_rows,
            cfg,
            device,
            output_dir,
            args.k,
            args.temperature,
            n_questions=gate0_questions,
            nonblocking=args.gate0_nonblocking,
        )
        if args.stage == "gate0":
            return gate0_stats

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
                    f"--skip_collection requires the existing two Session 1 shards; "
                    f"missing {missing_shards} in {source_dir}"
                )
        rft_rows, row_stats = build_rft_rows(
            shard_files,
            tokenizer,
            cfg,
            max_length=cfg.max_tok_len,
            seed=cfg.seed,
        )
        rows_path = Path(cfg.checkpoint_dir) / "rft_rows.pt"
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"rows": rft_rows, "stats": row_stats}, rows_path)
        return train_phase1_rft(
            model,
            rft_rows,
            dev_rows,
            cfg,
            device,
            tokenizer,
            epochs=args.epochs,
            save_path=args.checkpoint,
        )
    return None


if __name__ == "__main__":
    run(parse_args())
