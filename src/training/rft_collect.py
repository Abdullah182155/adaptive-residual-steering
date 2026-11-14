"""Rejection-sampled positive rollout collection and RFT tokenization."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

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


def _prompt_for_rollout(tokenizer, question: str, index: int, requested_shots: int, policy: str, max_prompt_len: int):
    """Reduce only the shot count when a rollout prompt exceeds the context budget."""
    for n_shot in range(requested_shots, -1, -2):
        if n_shot == 1:
            n_shot = 0
        prompts, shields = build_prompts(tokenizer, [question], [index], n_shot, policy=policy)
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
) -> dict[str, Any]:
    """Sample 2/4-shot completions and persist every question's scored rollouts.

    Collection runs with steering hooks disabled. When ``compare_greedy`` is enabled,
    a paired greedy sample is also recorded for the Gate 0 pass@k check.
    """
    if k < 1 or (k > 1 and temperature <= 0):
        raise ValueError("RFT collection requires k >= 1 and sampling temperature > 0")
    if any(index < 0 or index >= len(rows) for index in indices):
        raise IndexError("RFT collection index is outside the provided dataset")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sampled_correct = greedy_correct = strict_correct = strict_question_correct = 0
    sampled_total = 0
    n_questions = n_skipped = 0
    model.eval()
    model.set_routing_override("none")
    hooks_context = model.hooks_disabled() if hasattr(model, "hooks_disabled") else torch.no_grad()
    with output_file.open("w", encoding="utf-8") as stream, hooks_context:
        for index in indices:
            row = rows[index]
            question = row["question"]
            gold_raw = row.get("final_num", row.get("answer"))
            gold = float(str(gold_raw).replace(",", "").replace("$", "").strip())
            requested_shots = 0 if compare_greedy else (2 if index % 2 == 0 else 4)
            n_shot, prompts, shields = _prompt_for_rollout(
                tokenizer, question, index, requested_shots, shield_policy, max_prompt_len
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
            for completion in completions:
                text = truncate_completion(completion)
                strict_text = strict_accept(text, question, gold)
                score = score_tiers(text, gold)
                samples.append(
                    {
                        "text": text,
                        "lenient": bool(score["lenient"]),
                        "strict_ok": strict_text is not None,
                    }
                )
                strict_correct += int(strict_text is not None)
            sampled_correct += int(any(sample["lenient"] for sample in samples))
            strict_question_correct += int(any(sample["strict_ok"] for sample in samples))
            sampled_total += len(samples)

            record: dict[str, Any] = {
                "question_index": index,
                "question": question,
                "final_num": gold_raw,
                "n_shot": n_shot,
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
            if n_questions % 25 == 0:
                print(
                    f"[rft-collect] {n_questions}/{len(indices)} questions; "
                    f"pass@{k}={sampled_correct / n_questions:.1%}",
                    flush=True,
                )

    stats = {
        "n": n_questions,
        "skipped_long_prompts": n_skipped,
        "k": k,
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
    print(f"[rft-collect] {stats}")
    return stats


def tokenize_rft_example(
    tokenizer,
    question: str,
    completion: str,
    cfg,
    *,
    question_index: int = 0,
    n_shot: int = 0,
    max_length: Optional[int] = None,
) -> Optional[dict[str, torch.Tensor]]:
    """Tokenize completion-only targets, shrinking demonstrations without truncation."""
    max_length = int(max_length or cfg.max_tok_len)
    policy = getattr(cfg, "shield_policy", "prompt")
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for active_shots in range(n_shot, -1, -1):
            prompt, _, _ = build_prompt_with_shield(
                tokenizer,
                question,
                n_shot=active_shots,
                policy=policy,
                question_idx=question_index,
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
) -> tuple[list[dict[str, torch.Tensor]], dict[str, int]]:
    """Build mixed 0/2/4-shot training rows from verified positive rollouts only."""
    if max_rows_per_question < 1:
        raise ValueError("max_rows_per_question must be positive")
    rng = random.Random(seed)
    rows: list[dict[str, torch.Tensor]] = []
    stats = {"questions": 0, "questions_with_positive": 0, "rows": 0}
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
                for completion in candidates[:max_rows_per_question]:
                    n_shot = rng.choices([0, 2, 4], weights=[0.60, 0.25, 0.15])[0]
                    row = tokenize_rft_example(
                        tokenizer,
                        record["question"],
                        completion,
                        cfg,
                        question_index=int(record["question_index"]),
                        n_shot=n_shot,
                        max_length=max_length,
                    )
                    if row is not None:
                        rows.append(row)
                        stats["rows"] += 1
    if not rows:
        raise ValueError("no verified positive RFT rows were produced")
    print(f"[rft-rows] {stats}")
    return rows, stats
