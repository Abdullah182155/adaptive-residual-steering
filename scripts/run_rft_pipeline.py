"""Run Gate 0, split RFT collection, and Phase 1 RFT training on GSM8K."""

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
        "--equivalence_nonblocking",
        action="store_true",
        help="record small-sample equivalence quality without aborting a smoke run",
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
    if not result.get("passed", False):
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

    equivalence_questions = args.n_questions or 100
    gate0_questions = args.n_questions or 60
    collection_questions = args.n_questions or 2000

    cfg = RSCConfig()
    cfg.fixed_layers = list(STAGE1_ROUTE)
    cfg.candidate_layers = sorted(set(cfg.candidate_layers) | set(STAGE1_ROUTE))
    cfg.steernet_use_mag_head = False
    cfg.mag_l1_weight = 0.0
    cfg.s1_answer_weight = 1.0
    cfg.on_policy = False
    cfg.use_gate = False
    cfg.max_tok_len = 768
    cfg.validate()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Phase 1 RFT expects a CUDA runtime such as Kaggle T4")
    output_dir = Path(args.output_dir)
    equivalence_path = output_dir / "generation_equivalence.json"
    if args.stage in {"gate0", "collect", "train"}:
        _require_equivalence(equivalence_path)
    if args.stage in {"collect", "train"}:
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
        result["nonblocking"] = bool(args.equivalence_nonblocking)
        result["passed"] = quality_gate_passed or args.equivalence_nonblocking
        _write_json(equivalence_path, result)
        if not result["passed"]:
            raise RuntimeError(
                "batched generation failed Phase 1 equivalence thresholds; "
                f"n={result['n']} delta={result['delta_lenient_pp']:+.2f}pp "
                f"identity={result['string_identity']:.1%}; inspect "
                f"{equivalence_path} mismatch_examples"
            )
        if args.equivalence_nonblocking and not quality_gate_passed:
            print(
                "Generation equivalence thresholds were not met; continuing "
                "because --equivalence_nonblocking is enabled for this smoke run.",
                flush=True,
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
        shard_files = sorted(output_dir.glob("shard*.jsonl"))
        if not shard_files:
            raise FileNotFoundError(f"no RFT rollout shards found in {output_dir}")
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
