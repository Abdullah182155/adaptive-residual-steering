"""GSM8K dataset processing with multi-context mixture and typed reasoning tokens."""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from src.configs.config import RSCConfig
from src.data.prompts import PREFACE, V5_EXEMPLARS, build_prompt_unified
from src.training.losses import (
    T_FINAL,
    T_FORMAT,
    T_NONE,
    T_OPERAND,
    T_OPERATOR,
    T_PROSE,
    T_RESULT,
    annotate_char_types,
    token_types_from_offsets,
)


class PromptTemplateBank:
    """Randomises wording/format of a CoT prompt while holding math and final answer fixed."""

    HEADERS = [
        ("Question: {q}\n", "Answer: Let's think step by step.\n"),
        ("Problem: {q}\n", "Solution: I'll work through this step by step.\n"),
        ("Q: {q}\n", "A: Let's break this down.\n"),
        ("### Problem\n{q}\n", "### Reasoning\n"),
        ("Here is a math problem: {q}\n", "Let's solve it methodically.\n"),
    ]
    STEP_PREFIXES = ["Step {i}: ", "{i}. ", "({i}) ", "Step {i} - ", "> "]
    FINAL_PREFIXES = [
        "Final answer: {ans}\n",
        "The answer is {ans}.\n",
        "So the result is {ans}.\n",
        "#### {ans}\n",
    ]

    @classmethod
    def render(
        cls,
        question: str,
        steps: List[str],
        final_answer: str,
        rng: random.Random,
        fixed: bool = True,
        return_target_start: bool = False,
    ):
        if fixed:
            head_tpl, ans_tpl = cls.HEADERS[0]
            step_tpl = cls.STEP_PREFIXES[0]
            final_tpl = cls.FINAL_PREFIXES[0]
        else:
            head_tpl, ans_tpl = rng.choice(cls.HEADERS)
            step_tpl = rng.choice(cls.STEP_PREFIXES)
            final_tpl = rng.choice(cls.FINAL_PREFIXES)

        body_lines = []
        for i, s in enumerate(steps):
            s = s if s.endswith(".") else s + "."
            body_lines.append(step_tpl.format(i=i + 1) + s)
        body = "\n".join(body_lines)

        prefix = head_tpl.format(q=question) + ans_tpl
        text = prefix + body + "\n" + final_tpl.format(ans=final_answer)
        return (text, len(prefix)) if return_target_start else text


def _stratified_shots(n: int, probs: List[float], seed: int) -> List[int]:
    """Sample stratified shot counts matching the target distribution exactly."""
    counts = [int(round(p * n)) for p in probs]
    diff = n - sum(counts)
    counts[0] += diff
    shots = [k for k, c in enumerate(counts) for _ in range(c)]
    random.Random(seed).shuffle(shots)
    return shots


def split_answer(answer: str) -> Tuple[List[str], str]:
    """Parse GSM8K gold solution into reasoning sentences and final number."""
    parts = answer.split("####")
    reasoning = re.sub(r"<<[^>]*>>", "", parts[0]).strip()
    final_num = parts[1].strip() if len(parts) > 1 else ""
    steps = [s.strip() for s in re.split(r"(?<!\d)\.(?!\d)\s+", reasoning) if s.strip()]
    return steps, final_num


def gold_to_canonical_completion(steps: List[str], final_num: str) -> str:
    """Render canonical CoT completion with Step N: and Final answer:."""
    body_lines = [f"Step {i+1}: {s if s.endswith('.') else s + '.'}" for i, s in enumerate(steps)]
    return "\n".join(body_lines) + f"\nFinal answer: {final_num}\n"


def gsm8k_to_cot(
    tokenizer,
    cfg: Optional[RSCConfig] = None,
    n_samples: Optional[int] = None,
    max_length: Optional[int] = None,
    val_split: Optional[float] = None,
    seed: Optional[int] = None,
):
    """Loads GSM8K, builds multi-context stratified mixture, and tokenizes with typed targets."""
    if cfg is None:
        cfg = RSCConfig()
    if n_samples is not None:
        cfg.n_samples = n_samples
    if max_length is not None:
        cfg.max_tok_len = max_length
    if seed is not None:
        cfg.seed = seed
    split_ratio = val_split if val_split is not None else getattr(cfg, "val_split", 0.08)

    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
    except Exception:
        try:
            ds = load_dataset("openai/gsm8k", "main", split="train", verification_mode="no_checks")
        except Exception:
            ds = load_dataset("gsm8k", "main", split="train")

    print(f"[data] GSM8K official train loaded: {len(ds)} examples (n_samples cap={cfg.n_samples})")
    if cfg.n_samples < len(ds):
        ds = ds.shuffle(seed=cfg.seed).select(range(cfg.n_samples))
        print(f"[data] Capped to n_samples={len(ds)} from the official train split")

    raw_questions = list(ds["question"])
    raw_answers = list(ds["answer"])

    shot_probs = getattr(cfg, "train_shot_probs", [0.40, 0.15, 0.20, 0.10, 0.15])
    shots = _stratified_shots(len(ds), shot_probs, cfg.seed)
    use_preface = getattr(cfg, "train_use_preface", True)
    v5_demo_frac = getattr(cfg, "train_v5_demo_frac", 0.50)

    def make_train_demo(source_idx: int) -> Dict[str, str]:
        steps, final_num = split_answer(raw_answers[source_idx])
        sol = gold_to_canonical_completion(steps, final_num).strip()
        return {"q": raw_questions[source_idx], "a": sol}

    def format_example(example, idx):
        question = example["question"]
        ans_text = example["answer"]
        sentences, final_num = split_answer(ans_text)
        completion = gold_to_canonical_completion(sentences, final_num)

        rng = random.Random(cfg.seed + idx)
        k = shots[idx]

        # Draw k demonstrations
        if k > 0:
            if rng.random() < v5_demo_frac:
                exemplars = list(V5_EXEMPLARS[:k])
            else:
                candidates = [j for j in range(len(raw_questions)) if j != idx]
                donors = rng.sample(candidates, k=min(k, len(candidates)))
                exemplars = [make_train_demo(j) for j in donors]
        else:
            exemplars = None

        # Build prompt and shrink demonstrations if length exceeds max_tok_len
        # Never truncate the target completion!
        while True:
            cur_exemplars = exemplars[:k] if (exemplars and k > 0) else None
            prompt = build_prompt_unified(
                question,
                n_shot=k,
                exemplars=cur_exemplars,
                include_preface=use_preface,
            )
            full_text = prompt + completion
            token_count = len(tokenizer.encode(full_text))
            if k == 0 or token_count <= cfg.max_tok_len:
                break
            k -= 1

        target_start = len(prompt)
        tail = f"Question: {question}\nAnswer: Let's think step by step.\n"
        shield_char_end = (len(prompt) - len(tail)) if k > 0 else 0

        return {
            "text": full_text,
            "target_start": target_start,
            "shield_char_end": shield_char_end,
            "n_shot": k,
            "steps_list": sentences,
            "final_num": final_num,
        }

    ds = ds.map(format_example, with_indices=True)

    def tokenize_fn(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.max_tok_len,
            padding="max_length",
            return_offsets_mapping=True,
        )
        labels = []
        shield_masks = []
        token_types = []
        calc_masks = []

        for text, target_start, shield_char_end, ids, offsets, am in zip(
            examples["text"],
            examples["target_start"],
            examples["shield_char_end"],
            out["input_ids"],
            out["offset_mapping"],
            out["attention_mask"],
        ):
            # Supervised labels: only tokens starting at or after target_start
            lbl = []
            for tok_id, (s, e), mask in zip(ids, offsets, am):
                if mask and e > s and s >= target_start:
                    lbl.append(tok_id)
                else:
                    lbl.append(-100)

            # Prompt shield mask: covers preface + demos when k > 0
            shield = [
                1 if (mask and e > s and e <= shield_char_end) else 0
                for (s, e), mask in zip(offsets, am)
            ]

            # Categorize token types
            char_labels = annotate_char_types(text, target_start)
            tt = token_types_from_offsets(text, char_labels, offsets)
            tt = [t if l != -100 else T_NONE for t, l in zip(tt, lbl)]

            # Legacy calc mask: math-bearing tokens
            calc = [
                1 if t in (T_OPERAND, T_OPERATOR, T_RESULT, T_FINAL) else 0 for t in tt
            ]

            labels.append(lbl)
            shield_masks.append(shield)
            token_types.append(tt)
            calc_masks.append(calc)

        out.pop("offset_mapping")
        out["labels"] = labels
        out["prompt_shield_mask"] = shield_masks
        out["token_types"] = token_types
        out["calc_mask"] = calc_masks
        return out

    keep_raw_cols = ["question", "steps_list", "final_num"]
    remove_cols = [c for c in ds.column_names if c not in keep_raw_cols]
    ds = ds.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    ds.set_format(
        "torch",
        columns=[
            "input_ids",
            "attention_mask",
            "labels",
            "prompt_shield_mask",
            "token_types",
            "calc_mask",
        ],
        output_all_columns=True,
    )

    splits = ds.train_test_split(test_size=split_ratio, seed=cfg.seed)
    print(
        f"[data] Multi-context split: train={len(splits['train'])} val={len(splits['test'])} "
        f"(val_split={split_ratio}, max_tok_len={cfg.max_tok_len}). "
        f"Shots mixture: {shot_probs}."
    )
    return splits["train"], splits["test"]
