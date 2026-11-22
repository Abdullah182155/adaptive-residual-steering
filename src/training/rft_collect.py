"""Rejection-sampled positive rollout collection and RFT tokenization."""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

import torch
from tqdm import tqdm

from src.data.prompts import build_prompt_with_shield
from src.evaluation.answers import score_tiers
from src.evaluation.chain import audit_chain
from src.evaluation.generation import build_prompts, generate_texts, truncate_completion


_FINAL_LINE = re.compile(
    r"^[ \t]*Final answer:[ \t]*\$?-?[\d,]+(?:\.\d+)?[ \t]*\.?[ \t]*$",
    re.MULTILINE,
)
_STEP_LINE = re.compile(r"^Step (\d+): .+")


def reasoning_key(text: str) -> tuple[str, ...]:
    """Deduplicate completions by their numerical reasoning trace."""
    return tuple(re.findall(r"-?\d[\d,]*(?:\.\d+)?", text))


def strict_accept(
    completion: str,
    question: str,
    gold: float | str,
    *,
    max_steps: int = 12,
) -> Optional[str]:
    """Return a canonical positive completion only when format, answer, and math pass."""
    text = truncate_completion(completion).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or len(lines) > max_steps + 1:
        return None
    steps = [_STEP_LINE.fullmatch(line) for line in lines[:-1]]
    if not all(steps):
        return None
    if [int(match.group(1)) for match in steps] != list(range(1, len(lines))):
        return None
    if not _FINAL_LINE.fullmatch(lines[-1]) or text.count("Final answer:") != 1:
        return None
    try:
        gold_value = float(str(gold).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None
    if not score_tiers(text, gold_value)["strict"]:
        return None
    audit = audit_chain(text, question)
    if audit["total_equations"] < 1 or audit["correct_equations"] != audit["total_equations"]:
        return None
    return text


def _prompt_for_rollout(
    tokenizer,
    question: str,
    index: int,
    requested_shots: int,
    policy: str,
    max_prompt_len: int,
    template_id: int = 0,
):
    """Reduce only the shot count when a rollout prompt exceeds the context budget."""
    for n_shot in range(requested_shots, -1, -2):
        if n_shot == 1:
            n_shot = 0
        prompts, shields = build_prompts(
            tokenizer,
            [question],
            [index],
            n_shot,
            policy=policy,
            template_ids=[template_id],
        )
        encoded = tokenizer(prompts[0], return_tensors="pt", truncation=False)
        if encoded["input_ids"].shape[1] <= max_prompt_len:
            return n_shot, prompts, shields
    return None, None, None


@torch.no_grad()
def collect_rft_data(
    model,
    tokenizer,
    rows: Sequence[dict],
    device: torch.device | str,
    *,
    indices: Sequence[int],
    output_path: str | Path,
    k: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 200,
    max_prompt_len: int = 768,
    shield_policy: str = "prompt",
    compare_greedy: bool = False,
    retry_k: int = 0,
    retry_temperature: float = 0.9,
    target_positive_samples: int = 2,
    shot_weights: Sequence[float] = (0.60, 0.25, 0.15),
    diversify_templates: bool = False,
    seed: int = 0,
    progress_desc: str = "RFT rollout collection",
    progress_position: int = 0,
) -> dict[str, Any]:
    """Collect initial rollouts and selectively retry questions with no strict positive.

    Collection runs with steering hooks disabled. When ``compare_greedy`` is enabled,
    a paired greedy sample is also recorded for the Gate 0 pass@k check.
    """
    if k < 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("RFT collection requires k >= 1 and sampling temperature > 0")
    if retry_k < 0 or not math.isfinite(retry_temperature) or (retry_k > 0 and retry_temperature <= 0):
        raise ValueError("retry_k must be nonnegative and retry_temperature must be positive")
    if target_positive_samples < 1:
        raise ValueError("target_positive_samples must be positive")
    if (
        len(shot_weights) != 3
        or any(not math.isfinite(weight) for weight in shot_weights)
        or any(weight < 0 for weight in shot_weights)
        or sum(shot_weights) <= 0
    ):
        raise ValueError("shot_weights must contain three nonnegative 0/2/4-shot weights")
    if any(index < 0 or index >= len(rows) for index in indices):
        raise IndexError("RFT collection index is outside the provided dataset")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    sampled_correct = greedy_correct = strict_correct = strict_question_correct = 0
    initial_lenient_questions = initial_strict_questions = 0
    questions_retried = retry_samples = retry_prompt_skips_total = 0
    unique_strict_total = two_positive_questions = 0
    sampled_total = 0
    n_questions = n_skipped = 0
    shot_counts = {"0": 0, "2": 0, "4": 0}
    template_counts = {str(index): 0 for index in range(3)}
    shot_stats = {
        str(shots): {"questions": 0, "attempts": 0, "strict_positive_samples": 0}
        for shots in (0, 2, 4)
    }
    model.eval()
    model.set_routing_override("none")
    hooks_context = model.hooks_disabled() if hasattr(model, "hooks_disabled") else torch.no_grad()
    progress = tqdm(
        indices,
        total=len(indices),
        desc=progress_desc,
        unit="question",
        position=progress_position,
        file=sys.stdout,
        disable=False,
        mininterval=0.5,
    )
    with output_file.open("w", encoding="utf-8") as stream, hooks_context:
        for index in progress:
            row = rows[index]
            question = row["question"]
            gold_raw = row.get("final_num", row.get("answer"))
            gold = float(str(gold_raw).replace(",", "").replace("$", "").strip())
            question_rng = random.Random(int(seed) + int(index) * 104729)
            requested_shots = (
                0
                if compare_greedy
                else question_rng.choices((0, 2, 4), weights=shot_weights, k=1)[0]
            )
            initial_template = question_rng.randrange(3) if diversify_templates else 0
            n_shot, prompts, shields = _prompt_for_rollout(
                tokenizer,
                question,
                index,
                requested_shots,
                shield_policy,
                max_prompt_len,
                template_id=initial_template,
            )
            if prompts is None:
                n_skipped += 1
                continue

            completions = generate_texts(
                model,
                tokenizer,
                prompts,
                device,
                prompt_shields=shields,
                max_new_tokens=max_new_tokens,
                num_return_sequences=k,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.0,
                max_prompt_len=max_prompt_len,
            )
            samples = []

            def append_scored_sample(
                completion: str,
                *,
                attempt_index: int,
                phase: str,
                effective_shots: int,
                template_id: int,
                sample_temperature: float,
            ) -> None:
                nonlocal strict_correct, sampled_total
                text = truncate_completion(completion)
                strict_text = strict_accept(text, question, gold)
                score = score_tiers(text, gold)
                sample = {
                    "text": strict_text if strict_text is not None else text,
                    "lenient": bool(score["lenient"]),
                    "strict_ok": strict_text is not None,
                    "attempt_index": int(attempt_index),
                    "phase": phase,
                    "temperature": float(sample_temperature),
                    "n_shot": int(effective_shots),
                    "template_id": int(template_id),
                }
                if strict_text is not None:
                    strict_correct += 1
                    shot_stats[str(effective_shots)]["strict_positive_samples"] += 1
                samples.append(sample)
                sampled_total += 1
                shot_counts[str(effective_shots)] += 1
                template_counts[str(template_id)] += 1
                shot_stats[str(effective_shots)]["attempts"] += 1

            for attempt_index, completion in enumerate(completions):
                append_scored_sample(
                    completion,
                    attempt_index=attempt_index,
                    phase="initial",
                    effective_shots=n_shot,
                    template_id=initial_template,
                    sample_temperature=temperature,
                )

            initial_samples = samples[:]
            initial_lenient = any(sample["lenient"] for sample in initial_samples)
            initial_strict = any(sample["strict_ok"] for sample in initial_samples)
            initial_lenient_questions += int(initial_lenient)
            initial_strict_questions += int(initial_strict)
            shot_stats[str(n_shot)]["questions"] += 1

            def positive_signatures() -> set[tuple[str, ...]]:
                return {
                    reasoning_key(sample["text"])
                    for sample in samples
                    if sample["strict_ok"]
                }

            # Only questions with no verified positive receive extra generation.
            retry_prompt_skips = 0
            if retry_k and not initial_strict:
                retry_specs = []
                for retry_index in range(retry_k):
                    template_id = (initial_template + retry_index + 1) % 3 if diversify_templates else 0
                    retry_shots, retry_prompts, retry_shields = _prompt_for_rollout(
                        tokenizer,
                        question,
                        index,
                        requested_shots,
                        shield_policy,
                        max_prompt_len,
                        template_id=template_id,
                    )
                    if retry_prompts is not None:
                        retry_specs.append(
                            (retry_index, retry_shots, template_id, retry_prompts[0], retry_shields[0])
                        )
                    else:
                        retry_prompt_skips += 1
                if retry_specs:
                    questions_retried += 1
                    retry_outputs = generate_texts(
                        model,
                        tokenizer,
                        [spec[3] for spec in retry_specs],
                        device,
                        prompt_shields=[spec[4] for spec in retry_specs],
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=1,
                        temperature=retry_temperature,
                        top_p=top_p,
                        repetition_penalty=1.0,
                        max_prompt_len=max_prompt_len,
                    )
                    for spec, completion in zip(retry_specs, retry_outputs):
                        retry_index, retry_shots, template_id, _, _ = spec
                        retry_samples += 1
                        append_scored_sample(
                            completion,
                            attempt_index=k + retry_index,
                            phase="targeted_retry",
                            effective_shots=retry_shots,
                            template_id=template_id,
                            sample_temperature=retry_temperature,
                        )
            retry_prompt_skips_total += retry_prompt_skips

            signatures = positive_signatures()
            unique_strict_total += len(signatures)
            two_positive_questions += int(len(signatures) >= target_positive_samples)
            sampled_correct += int(any(sample["lenient"] for sample in samples))
            strict_question_correct += int(any(sample["strict_ok"] for sample in samples))

            record: dict[str, Any] = {
                "question_index": index,
                "question": question,
                "final_num": gold_raw,
                "n_shot": n_shot,
                "initial_template_id": initial_template,
                "initial_k": k,
                "retry_k_requested": retry_k if not initial_strict else 0,
                "retry_prompt_skips": retry_prompt_skips,
                "retry_trigger": "no_initial_strict_positive" if not initial_strict else None,
                "samples": samples,
            }
            if compare_greedy:
                greedy = generate_texts(
                    model,
                    tokenizer,
                    prompts,
                    device,
                    prompt_shields=shields,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=1.0,
                    max_prompt_len=max_prompt_len,
                )[0]
                greedy_ok = bool(score_tiers(greedy, gold)["lenient"])
                greedy_correct += int(greedy_ok)
                record["greedy"] = {"text": greedy, "lenient": greedy_ok}

            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            n_questions += 1
            progress.set_postfix(
                pass_at_k=f"{sampled_correct / n_questions:.1%}",
                positive_rate=f"{strict_correct / max(sampled_total, 1):.1%}",
                retried=questions_retried,
            )
            sys.stdout.flush()
    progress.close()

    stats = {
        "n": n_questions,
        "skipped_long_prompts": n_skipped,
        "k": k,
        "retry_k": retry_k,
        "retry_temperature": retry_temperature,
        "target_positive_samples": target_positive_samples,
        "shot_weights": list(map(float, shot_weights)),
        "diversify_templates": bool(diversify_templates),
        "seed": int(seed),
        "attempted_samples": sampled_total,
        "initial_lenient_questions": initial_lenient_questions,
        "initial_strict_questions": initial_strict_questions,
        "questions_retried": questions_retried,
        "retry_samples": retry_samples,
        "retry_prompt_skips": retry_prompt_skips_total,
        "unique_strict_positive_samples": unique_strict_total,
        "questions_with_target_positives": two_positive_questions,
        "shot_counts": shot_counts,
        "template_counts": template_counts,
        "shot_stats": shot_stats,
        "pass_at_k_lenient": sampled_correct / max(n_questions, 1),
        "pass_at_k_strict": strict_question_correct / max(n_questions, 1),
        "positive_sample_rate": strict_correct / max(sampled_total, 1),
        "output_path": str(output_file),
    }
    if compare_greedy:
        stats["greedy_lenient_accuracy"] = greedy_correct / max(n_questions, 1)
        stats["pass_at_k_delta_vs_greedy_pp"] = 100 * (
            stats["pass_at_k_lenient"] - stats["greedy_lenient_accuracy"]
        )
    print(f"[rft-collect] {stats}", flush=True)
    return stats


def tokenize_rft_example(
    tokenizer,
    question: str,
    completion: str,
    cfg,
    *,
    question_index: int = 0,
    completion_index: int = 0,
    n_shot: int = 0,
    template_id: int = 0,
    max_length: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """Tokenize completion-only targets, shrinking demonstrations without truncation."""
    max_length = int(max_length or cfg.max_tok_len)
    policy = getattr(cfg, "shield_policy", "prompt")
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        shot_candidates = list(range(n_shot, -1, -2))
        if not shot_candidates or shot_candidates[-1] != 0:
            shot_candidates.append(0)
        for active_shots in shot_candidates:
            prompt, _, _ = build_prompt_with_shield(
                tokenizer,
                question,
                n_shot=active_shots,
                policy=policy,
                question_idx=question_index,
                template_id=template_id,
            )
            text = prompt + completion.strip()
            encoded = tokenizer(
                text,
                truncation=False,
                padding=False,
                return_offsets_mapping=True,
            )
            ids = list(encoded["input_ids"])
            offsets = list(encoded["offset_mapping"])
            if len(ids) > max_length:
                continue

            prompt_encoded = tokenizer(
                prompt,
                truncation=False,
                padding=False,
                return_offsets_mapping=True,
            )
            prompt_offsets = [
                (int(start), int(end))
                for start, end in prompt_encoded["offset_mapping"]
                if int(end) > int(start)
            ]
            question_start = prompt.rfind(question)
            question_end = question_start + len(question) if question_start >= 0 else -1
            router_feature_mask = tuple(
                int(question_start >= 0 and int(end) > question_start and int(start) < question_end)
                for start, end in prompt_encoded["offset_mapping"]
            )
            if not any(router_feature_mask):
                router_feature_mask = tuple(
                    int(int(end) > int(start)) for start, end in prompt_encoded["offset_mapping"]
                )
            shield_char_end = prompt_offsets[-1][0] if prompt_offsets and policy == "prompt" else 0
            labels = [
                int(token_id) if end > start and start >= len(prompt) else -100
                for token_id, (start, end) in zip(ids, offsets)
            ]
            if not any(label != -100 for label in labels):
                return None
            shield = [
                int(end > start and end <= shield_char_end)
                for start, end in offsets
            ]
            attention = [1] * len(ids)
            padding = max_length - len(ids)
            input_ids = ids + [int(tokenizer.pad_token_id or 0)] * padding
            attention += [0] * padding
            labels += [-100] * padding
            shield += [0] * padding
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "prompt_shield_mask": torch.tensor(shield, dtype=torch.bool),
                "question_index": int(question_index),
                "completion_index": int(completion_index),
                "effective_shots": int(active_shots),
                "template_id": int(template_id),
                "prompt_signature": tuple(int(token_id) for token_id in prompt_encoded["input_ids"]),
                "router_feature_mask": router_feature_mask,
            }
        return None
    finally:
        tokenizer.padding_side = original_padding_side


def build_rft_rows(
    shard_files: Sequence[str | Path],
    tokenizer,
    cfg,
    *,
    max_rows_per_question: int = 3,
    max_length: int = 768,
    seed: int = 0,
    shot_weights: Sequence[float] = (0.60, 0.25, 0.15),
    diversify_prompt_templates: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build verified RFT rows with configurable shot mix and safe template diversity."""
    if max_rows_per_question < 1:
        raise ValueError("max_rows_per_question must be positive")
    if len(shot_weights) != 3 or any(weight < 0 for weight in shot_weights) or sum(shot_weights) <= 0:
        raise ValueError("shot_weights must contain three nonnegative 0/2/4-shot weights")
    rng = random.Random(seed)
    rows: list[dict[str, torch.Tensor]] = []
    stats = {
        "questions": 0,
        "questions_with_positive": 0,
        "rows": 0,
        "shot_counts": {"0": 0, "2": 0, "4": 0},
        "template_counts": {"0": 0, "1": 0, "2": 0},
        "shot_weights": list(map(float, shot_weights)),
        "prompt_templates_diversified": bool(diversify_prompt_templates),
    }
    for shard_path in shard_files:
        with Path(shard_path).open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                stats["questions"] += 1
                unique = {
                    reasoning_key(sample["text"]): sample["text"]
                    for sample in record["samples"]
                    if sample.get("strict_ok")
                }
                if not unique:
                    continue
                stats["questions_with_positive"] += 1
                candidates = list(unique.values())
                rng.shuffle(candidates)
                for completion_index, completion in enumerate(candidates[:max_rows_per_question]):
                    n_shot = rng.choices([0, 2, 4], weights=shot_weights)[0]
                    template_id = (
                        (int(record["question_index"]) + len(rows)) % 3
                        if diversify_prompt_templates
                        else 0
                    )
                    row = tokenize_rft_example(
                        tokenizer,
                        record["question"],
                        completion,
                        cfg,
                        question_index=int(record["question_index"]),
                        completion_index=completion_index,
                        n_shot=n_shot,
                        template_id=template_id,
                        max_length=max_length,
                    )
                    if row is not None:
                        rows.append(row)
                        stats["rows"] += 1
                        stats["shot_counts"][str(n_shot)] += 1
                        stats["template_counts"][str(template_id)] += 1
    if not rows:
        raise ValueError("no verified positive RFT rows were produced")
    print(f"[rft-rows] {stats}")
    return rows, stats
