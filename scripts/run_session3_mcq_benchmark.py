"""Run paired four-arm evaluation on ARC-Challenge, AQuA-RAT, or MMLU-Pro."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.runtime import configure_pipeline_runtime

configure_pipeline_runtime()

import torch
from datasets import load_dataset
from tqdm import tqdm

from scripts.run_session3_router import (
    FIXED_ROUTE,
    _apply_norm_refs,
    _configure,
    _load_fixed_adapter_weights,
    _load_bank,
    _load_model,
    _sha256,
)
from src.evaluation.benchmark import _configure_arm, _generation_metrics, mcnemar_exact
from src.evaluation.generation import build_question_feature_masks, generate_batch
from src.training.adaptive_router_training import load_adaptive_router


LABELS = "ABCDEFGHIJ"
ARMS = ("baseline", "steer_only", "static_bank", "router")
DATASET_REVISIONS = {
    "arc": ("allenai/ai2_arc", "ARC-Challenge"),
    "aqua_rat": ("deepmind/aqua_rat", "raw"),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None),
}


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_REVISIONS), required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--candidate_layers", type=int, nargs="+", default=[8, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24, 26])
    parser.add_argument("--norm_calibration", type=Path, required=True)
    parser.add_argument(
        "--fixed_adapter_norm_calibration", "--v2_norm_calibration",
        dest="fixed_adapter_norm_calibration", type=Path, required=True,
    )
    parser.add_argument(
        "--fixed_adapter_checkpoint", "--v2_checkpoint",
        dest="fixed_adapter_checkpoint", type=Path, required=True,
    )
    parser.add_argument("--bank_checkpoint", type=Path, required=True)
    parser.add_argument("--router_checkpoint", type=Path, required=True)
    parser.add_argument("--static_action", type=Path, required=True)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--max_prompt_tokens", type=int, default=1800)
    return parser.parse_args(argv)


def _normalise_mcq(
    dataset_name: str, raw: dict[str, Any], source_index: int, split_name: str
) -> dict[str, Any]:
    if dataset_name == "arc":
        choice_data = raw.get("choices") or {}
        original_labels = [str(value).strip().upper() for value in choice_data.get("label", [])]
        choice_texts = [str(value) for value in choice_data.get("text", [])]
        if not original_labels or len(original_labels) != len(choice_texts):
            raise ValueError(f"ARC row {source_index} has malformed choices")
        try:
            answer_index = original_labels.index(str(raw.get("answerKey", "")).strip().upper())
        except ValueError as error:
            raise ValueError(f"ARC row {source_index} has an unknown answerKey") from error
        source_id = f"{split_name}:{raw.get('id', source_index)}"
        category = None
        question = str(raw.get("question", ""))
    elif dataset_name == "aqua_rat":
        option_labels: list[str] = []
        choice_texts = []
        for option in raw.get("options", []):
            match = re.fullmatch(r"\s*\(?([A-E])\s*[).:]\s*(.*?)\s*", str(option), re.IGNORECASE)
            if match is None:
                raise ValueError(f"AQuA-RAT row {source_index} has a malformed option: {option!r}")
            option_labels.append(match.group(1).upper())
            choice_texts.append(match.group(2))
        if not choice_texts or len(choice_texts) > len(LABELS) or len(set(option_labels)) != len(option_labels):
            raise ValueError(f"AQuA-RAT row {source_index} has malformed choices")
        answer_match = re.fullmatch(r"\s*\(?([A-E])\)?\s*", str(raw.get("correct", "")), re.IGNORECASE)
        if answer_match is None:
            raise ValueError(f"AQuA-RAT row {source_index} has an unknown correct option")
        try:
            answer_index = option_labels.index(answer_match.group(1).upper())
        except ValueError as error:
            raise ValueError(f"AQuA-RAT row {source_index} has a correct option absent from its choices") from error
        source_id = f"{split_name}:{raw.get('id', source_index)}"
        category = None
        question = str(raw.get("question", ""))
    else:
        choice_texts = [str(value) for value in raw.get("options", [])]
        if not choice_texts or len(choice_texts) > len(LABELS):
            raise ValueError(f"MMLU-Pro row {source_index} has malformed options")
        if raw.get("answer_index") is not None:
            answer_index = int(raw["answer_index"])
        else:
            answer = str(raw.get("answer", "")).strip().upper()
            if answer not in LABELS[:len(choice_texts)]:
                raise ValueError(f"MMLU-Pro row {source_index} has an unknown answer")
            answer_index = LABELS.index(answer)
        source_id = f"{split_name}:{raw.get('question_id', source_index)}"
        category = raw.get("category")
        question = str(raw.get("question", ""))
    if not question.strip() or not 0 <= answer_index < len(choice_texts):
        raise ValueError(f"Malformed question or answer at row {source_index}")
    choices = [f"{LABELS[i]}. {text}" for i, text in enumerate(choice_texts)]
    return {
        "source_index": int(source_index),
        "source_id": source_id,
        "question": question,
        "choices": choices,
        "gold": LABELS[answer_index],
        "category": category,
    }


def _load_splits(dataset_name: str):
    dataset_id, config = DATASET_REVISIONS[dataset_name]
    if dataset_name == "arc":
        dataset = load_dataset(dataset_id, config)
        train_split = dataset["train"]
        test_split = dataset["test"]
        train = [_normalise_mcq(dataset_name, row, i, "train") for i, row in enumerate(train_split)]
    elif dataset_name == "aqua_rat":
        dataset = load_dataset(dataset_id, config)
        train_split = dataset["train"]
        test_split = dataset["test"]
        train = [_normalise_mcq(dataset_name, row, i, "train") for i, row in enumerate(train_split)]
    else:
        dataset = load_dataset(dataset_id)
        # MMLU-Pro has no public train split in the standard release. Use the
        # small validation split for demonstrations and keep test untouched.
        train = [_normalise_mcq(dataset_name, row, i, "validation") for i, row in enumerate(dataset["validation"])]
        test_split = dataset["test"]
    test = [_normalise_mcq(dataset_name, row, i, "test") for i, row in enumerate(test_split)]
    fingerprints = {
        "demonstration_split_fingerprint": getattr(
            train_split if dataset_name in {"arc", "aqua_rat"} else dataset["validation"],
            "_fingerprint",
            None,
        ),
        "test_split_fingerprint": getattr(test_split, "_fingerprint", None),
    }
    return train, test, fingerprints


def _format_item(row: dict[str, Any]) -> str:
    return f"Question: {row['question']}\nChoices:\n" + "\n".join(row["choices"])


def _build_prompt(row: dict[str, Any], demonstrations: Sequence[dict[str, Any]]) -> str:
    parts = [
        "Answer the multiple-choice question. You may reason briefly, then write "
        "Final answer: followed by one option letter.\n"
    ]
    for example in demonstrations:
        parts.append(f"{_format_item(example)}\nFinal answer: {example['gold']}\n\n")
    parts.append(f"{_format_item(row)}\nFinal answer:")
    return "".join(parts)


def _extract_choice(text: str, n_choices: int) -> str | None:
    allowed = LABELS[:n_choices]
    patterns = (
        r"(?:final\s+answer|answer|option|choice)\s*(?:is|:)?\s*\(?([A-J])\)?\b",
        r"^\s*\(?([A-J])\)?[.)]?\s*$",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        for value in reversed(matches):
            choice = value.upper()
            if choice in allowed:
                return choice
    return None


def _paired_summary(baseline: Sequence[bool], treatment: Sequence[bool], seed: int) -> dict[str, Any]:
    if len(baseline) != len(treatment) or not baseline:
        raise ValueError("paired outcomes must have equal nonzero lengths")
    wins = sum(t and not b for b, t in zip(baseline, treatment))
    losses = sum(b and not t for b, t in zip(baseline, treatment))
    ties = len(baseline) - wins - losses
    deltas = [int(t) - int(b) for b, t in zip(baseline, treatment)]
    delta = 100.0 * sum(deltas) / len(deltas)
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(5000):
        bootstrap.append(100.0 * sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
    bootstrap.sort()
    lo = bootstrap[int(0.025 * len(bootstrap))]
    hi = bootstrap[min(int(0.975 * len(bootstrap)), len(bootstrap) - 1)]
    return {
        "n": len(baseline),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "delta_pp": delta,
        "bootstrap_ci95_pp": [lo, hi],
        "mcnemar_p_value": mcnemar_exact(wins, losses),
    }


def _load_existing(path: Path) -> dict[int, dict[str, Any]]:
    values: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return values
    with path.open(encoding="utf-8") as stream:
        lines = stream.readlines()
        for line_no, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                if line_no == len(lines) and not line.endswith(("\n", "\r")):
                    break
                raise ValueError(f"invalid saved prediction at {path}:{line_no}: {error}") from error
            index = int(item["source_index"])
            if index in values:
                raise ValueError(f"duplicate question index in {path}:{line_no}: {index}")
            values[index] = item
    return values


def _route_and_alpha(model, arm: str, cfg, static_action: dict[str, Any]) -> tuple[list[int], float]:
    if arm == "steer_only":
        return list(FIXED_ROUTE), 1.0
    if arm == "static_bank":
        return list(static_action["route"]), float(static_action["alpha"])
    if arm == "router":
        routes = model.current_routes()
        route = list(routes[0]) if routes else []
        gains = getattr(model, "_route_gains", None)
        if gains is None:
            alpha = 1.0
        else:
            values = torch.as_tensor(gains).detach().cpu().flatten()
            alpha = float(values[0]) if values.numel() else 1.0
        return route, alpha
    return [], 0.0


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("MCQ evaluation expects a CUDA GPU runtime such as Kaggle T4")
    if args.n_test < 1 or any(shot not in (0, 2) for shot in args.shots):
        raise ValueError("n_test must be positive; supported shot counts are 0 and 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.fixed_adapter_checkpoint, args.fixed_adapter_norm_calibration, args.bank_checkpoint, args.router_checkpoint, args.norm_calibration, args.static_action):
        if not path.is_file():
            raise FileNotFoundError(path)

    static_action = json.loads(args.static_action.read_text(encoding="utf-8"))
    bank_digest = _sha256(args.bank_checkpoint)
    if static_action.get("bank_sha256") != bank_digest:
        raise ValueError("static-bank action was selected for a different bank checkpoint")

    train_rows, full_test_rows, dataset_fingerprints = _load_splits(args.dataset)
    rng = random.Random(args.seed)
    if args.n_test < len(full_test_rows):
        selected = sorted(rng.sample(range(len(full_test_rows)), args.n_test))
        test_rows = [full_test_rows[index] for index in selected]
    else:
        test_rows = full_test_rows
    if len(test_rows) < 2:
        raise ValueError("at least two test questions are required for paired statistics")
    demo_rng = random.Random(args.seed + 10_007)
    demo_count = min(2, len(train_rows))
    demos = demo_rng.sample(train_rows, demo_count)
    if any(demo["source_id"] in {row["source_id"] for row in test_rows} for demo in demos):
        raise ValueError("few-shot demonstrations overlap with the evaluation questions")

    candidate_layers = sorted(set(args.candidate_layers))
    cfg, candidates = _configure(SimpleNamespace(
        candidate_layers=candidate_layers,
        diversify_prompts=False,
        norm_calibration=args.norm_calibration,
        output_dir=args.output_dir,
    ))
    device = torch.device("cuda:0")
    model, tokenizer = _load_model(cfg, device)
    model.eval()
    _apply_norm_refs(model, args.norm_calibration)
    _load_bank(model, args.bank_checkpoint)
    _load_fixed_adapter_weights(model, args.fixed_adapter_checkpoint, args.fixed_adapter_norm_calibration)
    _, router_prefill, probe = load_adaptive_router(
        model, args.router_checkpoint, device, expected_bank_sha256=bank_digest
    )
    static_action_route = [int(layer) for layer in static_action["route"]]
    if not set(static_action_route).issubset(candidates):
        raise ValueError("static route contains a layer outside the bank candidate set")
    cfg.static_bank_route = static_action_route
    cfg.static_bank_alpha = float(static_action["alpha"])
    cfg.fixed_layers = list(candidates)

    tokenizer.padding_side = "left"
    all_arm_results: dict[str, dict[int, list[dict[str, Any]]]] = {}
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prompts_by_shot: dict[int, dict[int, str]] = {}
    skipped_long_prompts: list[dict[str, Any]] = []
    for shot in args.shots:
        selected_demos = demos if shot else []
        prompts_by_shot[shot] = {}
        for row in test_rows:
            prompt = _build_prompt(row, selected_demos)
            length = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
            if length > args.max_prompt_tokens:
                skipped_long_prompts.append({"source_index": row["source_index"], "shot": shot, "tokens": length})
                continue
            prompts_by_shot[shot][row["source_index"]] = prompt
    skipped_ids = {int(item["source_index"]) for item in skipped_long_prompts}
    if skipped_ids:
        test_rows = [row for row in test_rows if row["source_index"] not in skipped_ids]
        for shot in args.shots:
            prompts_by_shot[shot] = {
                source_index: prompt
                for source_index, prompt in prompts_by_shot[shot].items()
                if source_index not in skipped_ids
            }
    if len(test_rows) < 2:
        raise ValueError("fewer than two paired questions fit the selected prompt-token limit")
    print({"requested_test_n": args.n_test, "paired_test_n": len(test_rows), "skipped_long_questions": len(skipped_ids)}, flush=True)

    config = {
        "schema": "ars_mcq_run_v1",
        "dataset": args.dataset,
        "dataset_id": DATASET_REVISIONS[args.dataset][0],
        "dataset_fingerprints": dataset_fingerprints,
        "test_n": len(test_rows),
        "requested_test_n": args.n_test,
        "skipped_long_prompt_ids": sorted(skipped_ids),
        "seed": args.seed,
        "shots": list(args.shots),
        "arms": list(ARMS),
        "demonstration_ids": [item["source_id"] for item in demos],
        "test_source_ids": [row["source_id"] for row in test_rows],
        "artifact_sha256": {
            "fixed_adapter_checkpoint": _sha256(args.fixed_adapter_checkpoint),
            "fixed_adapter_norms": _sha256(args.fixed_adapter_norm_calibration),
            "fixed_adapter_architecture_id": "ars.fixed_route_low_rank_residual",
            "fixed_adapter_architecture_version": "1.0.0",
            "bank_checkpoint": bank_digest,
            "router_checkpoint": _sha256(args.router_checkpoint),
            "bank_norms": _sha256(args.norm_calibration),
            "static_bank_action": _sha256(args.static_action),
        },
        "prompt_contract": "compact_mcq_final_answer_letter_v1",
        "runner_sha256": _sha256(Path(__file__).resolve()),
    }
    config_path = args.output_dir / "mcq_run_config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("resume configuration differs from the existing run; use a new output directory")
    else:
        _atomic_json(config_path, config)

    try:
        for shot in args.shots:
            all_arm_results[shot] = {}
            for arm in ARMS:
                arm_path = prediction_dir / f"{args.dataset}_{shot}shot_{arm}.jsonl"
                existing = _load_existing(arm_path)
                expected_indices = {row["source_index"] for row in test_rows}
                if not set(existing).issubset(expected_indices):
                    raise ValueError(f"saved predictions do not match this test subset: {arm_path}")
                if arm == "steer_only":
                    cfg.fixed_layers = list(FIXED_ROUTE)
                    _load_fixed_adapter_weights(model, args.fixed_adapter_checkpoint, args.fixed_adapter_norm_calibration)
                elif arm in {"static_bank", "router"}:
                    cfg.fixed_layers = list(candidates)
                    _load_bank(model, args.bank_checkpoint)
                else:
                    cfg.fixed_layers = list(candidates)
                _configure_arm(model, arm, cfg)
                route_prefill = router_prefill if arm == "router" else None
                missing_rows = [row for row in test_rows if row["source_index"] not in existing]
                if missing_rows:
                    warmup_row = missing_rows[0]
                    warmup_prompt = prompts_by_shot[shot][warmup_row["source_index"]]
                    warmup_masks = build_question_feature_masks(
                        tokenizer, [warmup_prompt], [warmup_row["question"]]
                    )
                    warmup_features = torch.tensor(
                        [[float(shot) / 2.0, min(len(tokenizer(warmup_row["question"])["input_ids"]), 256) / 256.0]],
                        dtype=torch.float32,
                    )
                    generate_batch(
                        model,
                        tokenizer,
                        [warmup_prompt],
                        device,
                        prompt_shields=[None],
                        max_new_tokens=args.max_new_tokens,
                        repetition_penalty=1.0,
                        no_repeat_ngram_size=0,
                        max_prompt_len=args.max_prompt_tokens,
                        route_prefill_fn=route_prefill,
                        route_feature_masks=warmup_masks if route_prefill is not None else None,
                        route_context_features=warmup_features if route_prefill is not None else None,
                    )
                    _configure_arm(model, arm, cfg)
                if missing_rows:
                    mode = "a" if arm_path.exists() else "w"
                    with arm_path.open(mode, encoding="utf-8") as stream:
                        for row in tqdm(missing_rows, desc=f"{args.dataset} {shot}-shot {arm}", unit="question"):
                            prompt = prompts_by_shot[shot][row["source_index"]]
                            feature_masks = build_question_feature_masks(tokenizer, [prompt], [row["question"]])
                            context_features = torch.tensor(
                                [[float(shot) / 2.0, min(len(tokenizer(row["question"])["input_ids"]), 256) / 256.0]],
                                dtype=torch.float32,
                            )
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            generation_started = time.perf_counter()
                            generated = generate_batch(
                                model,
                                tokenizer,
                                [prompt],
                                device,
                                prompt_shields=[None],
                                max_new_tokens=args.max_new_tokens,
                                repetition_penalty=1.0,
                                no_repeat_ngram_size=0,
                                max_prompt_len=args.max_prompt_tokens,
                                route_prefill_fn=route_prefill,
                                route_feature_masks=feature_masks if route_prefill is not None else None,
                                route_context_features=context_features if route_prefill is not None else None,
                            )[0]
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            generation_latency_seconds = time.perf_counter() - generation_started
                            prediction = _extract_choice(generated.text, len(row["choices"]))
                            selected_route, selected_alpha = _route_and_alpha(model, arm, cfg, static_action)
                            item = {
                                "source_index": row["source_index"],
                                "source_id": row["source_id"],
                                "question": row["question"],
                                "choices": row["choices"],
                                "gold": row["gold"],
                                "prediction": prediction,
                                "correct": prediction == row["gold"],
                                "extracted": prediction is not None,
                                "text": generated.text,
                                "n_new_tokens": generated.n_new_tokens,
                                "prompt_tokens": generated.prompt_width,
                                "generation_latency_seconds": generation_latency_seconds,
                                "stop_reason": generated.stop_reason,
                                "category": row["category"],
                                "selected_layers": selected_route,
                                "selected_k": len(selected_route),
                                "selected_alpha": selected_alpha,
                                "shot": shot,
                                "arm": arm,
                            }
                            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                            stream.flush()
                            existing[row["source_index"]] = item
                all_arm_results[shot][arm] = [existing[row["source_index"]] for row in test_rows]

        record: dict[str, Any] = {
            **config,
            "four_arm_complete": True,
            "shots": {},
            "demonstration_metadata": {
                "source_split": "validation" if args.dataset == "mmlu_pro" else "train",
                "ids": [item["source_id"] for item in demos],
                "policy": "fixed two examples shared by every arm and test question",
            },
            "performance_measurement": {
                "latency_scope": "per-example generate_batch call, including tokenization, model generation, stopping, and decoding",
                "cuda_synchronization": True,
                "warmup_generations_per_arm_shot": 1,
                "questions_per_second": "timed examples divided by summed generation latency",
                "output_tokens_per_second": "output tokens from timed generations divided by summed generation latency",
            },
            "runtime_metadata": {
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "generation_batch_size": 1,
            },
        }
        for shot, arm_results in all_arm_results.items():
            by_arm = {
                arm: [bool(item["correct"]) for item in values]
                for arm, values in arm_results.items()
            }
            usage = Counter("+".join(map(str, item["selected_layers"])) or "k0" for item in arm_results["router"])
            k_counts = Counter(str(item["selected_k"]) for item in arm_results["router"])
            layer_counts = Counter(str(layer) for item in arm_results["router"] for layer in item["selected_layers"])
            alpha_counts = Counter(str(item["selected_alpha"]) for item in arm_results["router"])
            shot_data: dict[str, Any] = {}
            for arm, values in arm_results.items():
                shot_data[arm] = {
                    "n": len(values),
                    "accuracy_pct": 100.0 * sum(by_arm[arm]) / max(len(values), 1),
                    "extraction_rate_pct": 100.0 * sum(bool(item["extracted"]) for item in values) / max(len(values), 1),
                }
                shot_data[arm].update(_generation_metrics(values))
            shot_data["router_route_usage"] = {
                "route_counts": dict(usage),
                "k_counts": dict(k_counts),
                "layer_counts": dict(layer_counts),
                "alpha_counts": dict(alpha_counts),
                "unique_routes": len(usage),
                "most_used_route_share": max(usage.values(), default=0) / max(len(test_rows), 1),
            }
            shot_data["paired_vs_baseline"] = {
                arm: _paired_summary(by_arm["baseline"], by_arm[arm], args.seed + shot * 31 + i)
                for i, arm in enumerate(("steer_only", "static_bank", "router"))
            }
            shot_data["router_vs_steer_only"] = _paired_summary(
                by_arm["steer_only"], by_arm["router"], args.seed + shot * 41
            )
            shot_data["static_bank_vs_steer_only"] = _paired_summary(
                by_arm["steer_only"], by_arm["static_bank"], args.seed + shot * 47
            )
            shot_data["router_vs_static_bank"] = _paired_summary(
                by_arm["static_bank"], by_arm["router"], args.seed + shot * 43
            )
            record["shots"][str(shot)] = shot_data
        record_path = args.output_dir / f"mcq_{args.dataset}_{len(test_rows)}_seed{args.seed}.json"
        _atomic_json(record_path, record)
        print(json.dumps({"record": str(record_path), "four_arm_complete": True, "shots": record["shots"]}, indent=2), flush=True)
        return 0
    finally:
        if probe is not None:
            probe.close()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
