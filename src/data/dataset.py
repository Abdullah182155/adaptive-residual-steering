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

_CALC_TOKEN_CACHE: Dict[Tuple[str, int, int], int] = {}


def _calc_token_cache_key(tokenizer, token_id: int) -> Tuple[str, int, int]:
    """Identify token IDs within a tokenizer namespace, not globally."""
    vocab_size = getattr(tokenizer, "_ars_cached_vocab_size", None)
    if vocab_size is None:
        try:
            vocab_size = getattr(tokenizer, "vocab_size", None)
            if vocab_size is None:
                vocab_size = len(tokenizer)
            vocab_size = int(vocab_size)
        except Exception:
            vocab_size = 0
        try:
            tokenizer._ars_cached_vocab_size = vocab_size
        except Exception:
            pass
    return (tokenizer.__class__.__name__, vocab_size, int(token_id))

def _is_calc_token_id(tokenizer, token_id: int) -> int:
    """Classify token as math/calculation/outcome token vs English prose."""
    cache_key = _calc_token_cache_key(tokenizer, token_id)
    if cache_key in _CALC_TOKEN_CACHE:
        return _CALC_TOKEN_CACHE[cache_key]
    try:
        s = tokenizer.decode([token_id])
    except Exception:
        s = ""
    if "Step" in s:
        res = 0
    elif re.search(r"[\d\+\-\*\/=\$\%#]", s):
        res = 1
    else:
        res = 0
    _CALC_TOKEN_CACHE[cache_key] = res
    return res


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

    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
    except Exception:
        try:
            ds = load_dataset("openai/gsm8k", "main", split="train", verification_mode="no_checks")
        except Exception:
            ds = load_dataset("gsm8k", "main", split="train")
    if cfg.n_samples < len(ds):
        ds = ds.shuffle(seed=cfg.seed).select(range(cfg.n_samples))

    # Keep a deterministic training-only demonstration pool.  Evaluation uses
    # few-shot prompts, so a zero-shot-only curriculum is a distribution shift
    # rather than a meaningful robustness test.
    raw_questions = list(ds["question"])
    raw_answers = list(ds["answer"])

    def split_answer(answer: str):
        parts = answer.split("####")
        reasoning = parts[0].strip()
        final_num = parts[1].strip() if len(parts) > 1 else ""
        steps = [s.strip() for s in re.split(r"(?<!\d)\.(?!\d)\s+", reasoning) if s.strip()]
        return steps, final_num

    def make_demo(source_idx: int, rng: random.Random) -> str:
        steps, final_num = split_answer(raw_answers[source_idx])
        # Use exactly the benchmark's Question/Answer convention so training
        # and multi-shot evaluation share their structural grammar.
        rendered = PromptTemplateBank.render(raw_questions[source_idx], steps, final_num, rng, fixed=True)
        return rendered.rstrip() + "\n\n"

    def format_example(example, idx):
        question = example["question"]
        ans_text = example["answer"]

        sentences, final_num = split_answer(ans_text)

        rng = random.Random(cfg.seed + idx)
        current_text, current_target_start = PromptTemplateBank.render(
            question, sentences, final_num, rng, fixed=not cfg.diversify_prompts,
            return_target_start=True,
        )
        demo_prefix = ""
        max_demos = max(0, int(getattr(cfg, "fewshot_train_max_demos", 0)))
        if max_demos and rng.random() < float(getattr(cfg, "fewshot_train_prob", 0.0)):
            n_demos = rng.randint(1, max_demos)
            candidates = [j for j in range(len(raw_questions)) if j != idx]
            # Sampling is replacement-free and deterministic per example.
            for source_idx in rng.sample(candidates, k=min(n_demos, len(candidates))):
                demo_prefix += make_demo(source_idx, rng)
        text = demo_prefix + current_text
        target_start = len(demo_prefix) + current_target_start
        return {
            "text": text, "target_start": target_start,
            "steps_list": sentences, "final_num": final_num,
            "shield_char_end": len(demo_prefix),
            "fallback_text": current_text,
            "fallback_target_start": current_target_start,
        }

    ds = ds.map(format_example, with_indices=True)

    def get_difficulty(ex):
        text = ex.get("text", "")
        target_start = ex.get("target_start", 0)  # BUG-04 fix: use pre-computed offset instead of hardcoded string
        ops_count = len(re.findall(r"[\+\-\*\/]", text))
        nums = re.findall(r"-?\d+\.?\d*", text)
        distinct_nums = len(set(nums))
        # BUG-04 fix: was `text.find("Answer: Let's think")` which only matched template-0.
        # Now correctly uses target_start for all template variants.
        chain_len = len(text[target_start:]) if target_start > 0 else 0
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
        shield_masks = []
        calc_masks = []
        for row, (text, target_start, shield_char_end, fallback_text, fallback_target_start, ids) in enumerate(zip(
            examples["text"], examples["target_start"], examples["shield_char_end"],
            examples["fallback_text"], examples["fallback_target_start"], out["input_ids"]
        )):
            lbl = list(ids)
            prefix_len = len(tokenizer.encode(text[:target_start], add_special_tokens=False))
            for j in range(min(prefix_len, len(lbl))):
                lbl[j] = -100
            for j in range(len(lbl)):
                if lbl[j] == tokenizer.pad_token_id:
                    lbl[j] = -100
            # A long demonstration must never silently truncate every target
            # token.  Fall back to the current 0-shot item so every optimizer
            # example has a real supervised completion.
            if sum(token != -100 for token in lbl) < 4:
                fallback = tokenizer(
                    fallback_text, truncation=True, max_length=cfg.max_tok_len, padding="max_length"
                )
                ids = fallback["input_ids"]
                out["input_ids"][row] = ids
                out["attention_mask"][row] = fallback["attention_mask"]
                lbl = list(ids)
                prefix_len = len(tokenizer.encode(fallback_text[:fallback_target_start], add_special_tokens=False))
                for j in range(min(prefix_len, len(lbl))):
                    lbl[j] = -100
                for j in range(len(lbl)):
                    if lbl[j] == tokenizer.pad_token_id:
                        lbl[j] = -100
                shield_char_end = 0
            labels.append(lbl)
            # Do not assume independently tokenizing a character prefix always
            # matches the full prompt at its boundary.  Shorten until its IDs
            # are an exact prefix of the full token sequence.
            shield_ids = tokenizer.encode(text[:shield_char_end], add_special_tokens=False) if shield_char_end else []
            shield_len = min(len(shield_ids), len(ids))
            while shield_len and list(ids[:shield_len]) != shield_ids[:shield_len]:
                shield_len -= 1
            shield_masks.append([1 if j < shield_len else 0 for j in range(len(ids))])

            # Engine v2: Compute calculation/outcome token mask
            calc_mask = [0] * len(lbl)
            for j in range(len(lbl)):
                if lbl[j] != -100:
                    calc_mask[j] = _is_calc_token_id(tokenizer, int(ids[j]))
            if sum(calc_mask) == 0:
                valid_indices = [j for j, l in enumerate(lbl) if l != -100]
                for idx in valid_indices[-8:]:
                    calc_mask[idx] = 1
            calc_masks.append(calc_mask)

        out["labels"] = labels
        out["prompt_shield_mask"] = shield_masks
        out["calc_mask"] = calc_masks
        return out

    keep_raw_cols = ["question", "steps_list", "final_num"]
    remove_cols = [c for c in ds.column_names if c not in keep_raw_cols]
    ds = ds.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    ds.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels", "prompt_shield_mask", "calc_mask"], output_all_columns=True
    )

    splits = ds.train_test_split(test_size=split_ratio, seed=cfg.seed)
    return splits["train"], splits["test"]
