"""On-policy completion generation, verification, and tokenization helpers."""

from __future__ import annotations

import random
import re
from typing import Any, Sequence

import torch

from src.configs.config import RSCConfig
from src.data.prompts import build_prompt_unified, build_prompt_with_shield
from src.evaluation.answers import score_tiers
from src.evaluation.chain import safe_eval_expr
from src.evaluation.generation import generate_texts
from src.training.losses import (
    T_FINAL,
    T_NONE,
    T_OPERAND,
    T_OPERATOR,
    T_RESULT,
    annotate_char_types,
    token_types_from_offsets,
)


@torch.no_grad()
def generate_onpolicy_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device,
    max_new_tokens: int = 256,
    num_return_sequences: int = 1,
    temperature: float = 0.0,
    max_prompt_length: int = 1024,
) -> list[str]:
    """Generate completions through the shared left-padded decode engine."""
    return generate_texts(
        model,
        tokenizer,
        list(prompts),
        device,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        top_p=0.95 if temperature > 0 else 1.0,
        max_prompt_len=max_prompt_length,
        repetition_penalty=1.0,
    )


def verify_arithmetic_steps(completion: str, tol: float = 0.02) -> bool:
    """Reject explicit equations with a provably incorrect right-hand side.

    Expressions the bounded AST evaluator cannot parse are skipped.
    """
    equation_pattern = re.compile(r"([\d\s\+\-\*\/\.\(\)×÷]+)\s*=\s*(-?\d[\d,]*(?:\.\d+)?)")
    for match in equation_pattern.finditer(completion):
        lhs_raw = match.group(1).replace("×", "*").replace("÷", "/").replace(",", "").strip()
        rhs_raw = match.group(2).replace(",", "")
        if re.fullmatch(r"-?\d+(\.\d+)?", lhs_raw):
            continue
        lhs_value = safe_eval_expr(lhs_raw)
        if lhs_value is None:
            continue
        rhs_value = float(rhs_raw)
        denominator = max(1.0, abs(rhs_value))
        if abs(lhs_value - rhs_value) > tol * denominator:
            return False
    return True


def tokenize_onpolicy(
    tokenizer: Any,
    question: str,
    completion: str,
    cfg: RSCConfig,
    n_shot: int = 0,
) -> dict[str, torch.Tensor]:
    """Tokenize a completion while masking prompt and padding from its loss."""
    prompt_text, _prompt_encoding, _shield_mask = build_prompt_with_shield(
        tokenizer,
        question,
        n_shot=n_shot,
        policy=getattr(cfg, "shield_policy", "context"),
    )
    full_text = prompt_text + completion
    target_start = len(prompt_text)
    encoding = tokenizer(
        full_text,
        truncation=True,
        max_length=cfg.max_tok_len,
        padding="max_length",
        return_offsets_mapping=True,
    )

    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    attention_mask = encoding["attention_mask"]
    labels = [
        token_id if attended and end > start and start >= target_start else -100
        for token_id, (start, end), attended in zip(input_ids, offsets, attention_mask)
    ]
    shield_char_end = target_start if n_shot > 0 else 0
    shield = [
        1 if attended and end > 0 and end <= shield_char_end else 0
        for (_start, end), attended in zip(offsets, attention_mask)
    ]
    char_labels = annotate_char_types(full_text, target_start)
    token_types = token_types_from_offsets(full_text, char_labels, offsets)
    token_types = [
        token_type if label != -100 else T_NONE
        for token_type, label in zip(token_types, labels)
    ]
    calc_mask = [
        1 if token_type in (T_OPERAND, T_OPERATOR, T_RESULT, T_FINAL) else 0
        for token_type in token_types
    ]

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
        "prompt_shield_mask": torch.tensor(shield),
        "token_types": torch.tensor(token_types),
        "calc_mask": torch.tensor(calc_mask),
    }


@torch.no_grad()
def collect_onpolicy_data(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    device: torch.device,
    cfg: RSCConfig,
    epoch: int = 0,
) -> list[dict[str, Any]]:
    """Collect verified correct completions and tokenize them as training rows."""
    orig_indices = [index for index, row in enumerate(rows) if "question" in row]
    n_generate = max(1, int(len(orig_indices) * cfg.on_policy_frac))
    permutation = torch.randperm(
        len(orig_indices), generator=torch.Generator().manual_seed(cfg.seed + epoch)
    )[:n_generate].tolist()
    indices = [orig_indices[index] for index in permutation]

    rng = random.Random(cfg.seed + epoch + 1000)
    questions = [rows[index]["question"] for index in indices]
    gold_answers = [rows[index]["final_num"] for index in indices]
    gold_completions = [rows[index].get("answer", "") for index in indices]
    shot_probs = getattr(cfg, "on_policy_shot_probs", None) or cfg.train_shot_probs
    shot_choices = list(range(len(shot_probs)))
    shot_counts = [rng.choices(shot_choices, weights=shot_probs)[0] for _ in questions]
    prompts = [
        build_prompt_unified(question, n_shot=shots)
        for question, shots in zip(questions, shot_counts)
    ]

    checkpointing_was_enabled = getattr(model.base_model, "is_gradient_checkpointing", False)
    if checkpointing_was_enabled and hasattr(model.base_model, "gradient_checkpointing_disable"):
        model.base_model.gradient_checkpointing_disable()

    model.eval()
    model.set_routing_override("only", cfg.fixed_layers)
    samples_per_question = getattr(cfg, "on_policy_k", 1)
    temperature = getattr(cfg, "on_policy_temperature", 0.0)
    generated: list[str] = []
    generation_batch_size = cfg.on_policy_gen_batch
    batch_count = (len(prompts) + generation_batch_size - 1) // generation_batch_size
    for batch_index, start in enumerate(range(0, len(prompts), generation_batch_size)):
        batch_prompts = prompts[start : start + generation_batch_size]
        generated.extend(
            generate_onpolicy_batch(
                model,
                tokenizer,
                batch_prompts,
                device,
                num_return_sequences=samples_per_question,
                temperature=temperature,
                max_prompt_length=cfg.max_tok_len,
            )
        )
        if (batch_index + 1) % 25 == 0 or (batch_index + 1) == batch_count:
            print(f"  on-policy gen: {batch_index + 1}/{batch_count} batches", flush=True)

    if checkpointing_was_enabled and hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    new_rows: list[dict[str, Any]] = []
    tried = 0
    correct = 0
    gold_fallbacks = 0
    step_rejected = 0
    for question_index, (question, gold, gold_completion, shots) in enumerate(
        zip(questions, gold_answers, gold_completions, shot_counts)
    ):
        try:
            gold_number = float(str(gold).replace(",", "").replace("$", "").strip())
        except (ValueError, AttributeError):
            continue
        base = question_index * samples_per_question
        found = False
        for sample_index in range(samples_per_question):
            tried += 1
            completion = generated[base + sample_index]
            if score_tiers(completion, gold_number)["lenient"]:
                if getattr(cfg, "on_policy_verify_steps", True) and not verify_arithmetic_steps(completion):
                    step_rejected += 1
                    continue
                correct += 1
                row = tokenize_onpolicy(tokenizer, question, completion, cfg, n_shot=shots)
                row["question"] = question
                row["final_num"] = gold
                new_rows.append(row)
                found = True
                break
        if not found and gold_completion and getattr(cfg, "on_policy_gold_fallback", True):
            row = tokenize_onpolicy(tokenizer, question, gold_completion, cfg, n_shot=shots)
            row["question"] = question
            row["final_num"] = gold
            new_rows.append(row)
            gold_fallbacks += 1

    print(
        f"[on-policy] {n_generate} questions × {samples_per_question} samples = {tried} tried "
        f"→ {correct} correct ({100 * correct / max(n_generate, 1):.1f}%)"
        f"{f', {step_rejected} step-rejected' if step_rejected else ''}"
        f" + {gold_fallbacks} gold fallback → {len(new_rows)} new rows",
        flush=True,
    )

    model.train()
    model.base_model.eval()
    return new_rows
