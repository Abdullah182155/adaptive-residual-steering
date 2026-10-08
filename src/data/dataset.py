import re
import random
from typing import Tuple, List, Dict, Any, Optional
from datasets import load_dataset
from src.configs.config import RSCConfig

class PromptTemplateBank:
    """Randomises wording/format of a CoT prompt while holding math and final answer fixed."""

    HEADERS = [
        ("Question: {q}\n", "Answer: Let's think step by step.\n"),  # original
        ("Problem: {q}\n", "Solution: I'll work through this step by step.\n"),
        ("Q: {q}\n", "A: Let's break this down.\n"),
        ("### Problem\n{q}\n", "### Reasoning\n"),
        ("Here is a math problem: {q}\n", "Let's solve it methodically.\n"),
    ]
    STEP_PREFIXES = ["Step {i}: ", "{i}. ", "({i}) ", "Step {i} - ", "> "]
    FINAL_PREFIXES = [
        "#### {ans}\n",
        "Final answer: {ans}\n",
        "The answer is {ans}.\n",
        "So the result is {ans}.\n",
    ]

    @classmethod
    def render(
        cls, question: str, steps: List[str], final_answer: str, rng: random.Random, fixed: bool = False,
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

    @classmethod
    def two_views(cls, question: str, steps: List[str], final_answer: str, seed: int):
        r1 = random.Random(seed)
        r2 = random.Random(seed + 999983)
        return (
            cls.render(question, steps, final_answer, r1),
            cls.render(question, steps, final_answer, r2),
        )

def gsm8k_to_cot(
    tokenizer,
    cfg: Optional[RSCConfig] = None,
    n_samples: Optional[int] = None,
    max_length: Optional[int] = None,
    val_split: Optional[float] = None,
    seed: Optional[int] = None,
):
    """Loads GSM8K, applies formatting, curriculum sorting, and tokenization."""
    if cfg is None:
        cfg = RSCConfig()
    if n_samples is not None:
        cfg.n_samples = n_samples
    if max_length is not None:
        cfg.max_tok_len = max_length
    if seed is not None:
        cfg.seed = seed
    split_ratio = val_split if val_split is not None else getattr(cfg, "val_split", 0.08)

    ds = load_dataset("openai/gsm8k", "main", split="train")
    if cfg.n_samples < len(ds):
        ds = ds.shuffle(seed=cfg.seed).select(range(cfg.n_samples))

    def format_example(example, idx):
        question = example["question"]
        ans_text = example["answer"]

        parts = ans_text.split("####")
        reasoning = parts[0].strip()
        final_num = parts[1].strip() if len(parts) > 1 else ""

        sentences = [s.strip() for s in re.split(r"(?<!\d)\.(?!\d)\s+", reasoning) if s.strip()]

        rng = random.Random(cfg.seed + idx)
        text, target_start = PromptTemplateBank.render(
            question, sentences, final_num, rng, fixed=not cfg.diversify_prompts,
            return_target_start=True,
        )
        return {
            "text": text, "target_start": target_start,
            "steps_list": sentences, "final_num": final_num,
        }

    ds = ds.map(format_example, with_indices=True)

    def get_difficulty(ex):
        text = ex.get("text", "")
        ops_count = len(re.findall(r"[\+\-\*\/]", text))
        nums = re.findall(r"-?\d+\.?\d*", text)
        distinct_nums = len(set(nums))
        answer_start = text.find("Answer: Let's think")
        chain_len = len(text[answer_start:]) if answer_start >= 0 else 0
        score = ops_count * 2 + distinct_nums + chain_len / 100.0
        return {"difficulty_score": score}

    ds = ds.map(get_difficulty)
    ds = ds.sort("difficulty_score", reverse=False)
    ds = ds.remove_columns(["difficulty_score"])

    def tokenize_fn(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.max_tok_len,
            padding="max_length",
        )
        labels = []
        for text, target_start, ids in zip(
            examples["text"], examples["target_start"], out["input_ids"]
        ):
            lbl = list(ids)
            prefix_len = len(tokenizer.encode(text[:target_start], add_special_tokens=False))
            for j in range(min(prefix_len, len(lbl))):
                lbl[j] = -100
            for j in range(len(lbl)):
                if lbl[j] == tokenizer.pad_token_id:
                    lbl[j] = -100
            labels.append(lbl)
        out["labels"] = labels
        return out

    keep_raw_cols = ["question", "steps_list", "final_num"]
    remove_cols = [c for c in ds.column_names if c not in keep_raw_cols]
    ds = ds.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    ds.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True
    )

    splits = ds.train_test_split(test_size=split_ratio, seed=cfg.seed)
    return splits["train"], splits["test"]
