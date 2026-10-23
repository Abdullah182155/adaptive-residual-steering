from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence
import torch
from src.evaluation.verifier import build_prompt


def from_dataset_row(row: Mapping) -> dict:
    """Strips padding; question_span_mask = labels==-100 & attention_mask."""
    inp = row["input_ids"]
    mask = row["attention_mask"]
    labels = row["labels"]
    if torch.is_tensor(inp):
        valid = mask == 1
        q_mask = (labels == -100) & valid
        return {
            "input_ids": inp[valid],
            "attention_mask": mask[valid],
            "labels": labels[valid],
            "question_span_mask": q_mask[valid],
            "n_shot": row.get("n_shot", 0),
        }
    return dict(row)


def build_shot_example(tokenizer, question: str, steps: List[str], final_num: str, n_shot: int,
                       max_len: int = 1536, final_style: str = "auto") -> dict:
    """Construct prompt + target CoT. Target tokens have offset >= target start.
    Question span tokens have char start in [last 'Question:' char index, target_char_start).
    Returns dict(input_ids, attention_mask, labels, question_span_mask, n_shot) as 1D tensors, unpadded.

    final_style: "auto" -> "#### x" for 0-shot, "Final answer: x" otherwise; or force "hash" / "final_answer".
    max_len is advisory (callers filter on length); sequences are never truncated here because
    truncating would silently cut the target."""
    prompt_prefix = build_prompt(question, n_shot=n_shot)

    # Target rendering
    cot_body = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(steps))
    if final_style == "auto":
        use_hash = n_shot == 0
    elif final_style == "hash":
        use_hash = True
    elif final_style == "final_answer":
        use_hash = False
    else:
        raise ValueError(f"final_style must be auto|hash|final_answer, got {final_style!r}")
    ans_line = f"#### {final_num}" if use_hash else f"Final answer: {final_num}"

    full_target = cot_body + "\n" + ans_line if cot_body else ans_line
    full_text = prompt_prefix + full_target
    target_char_start = len(prompt_prefix)

    # Locate question in prompt (last "Question:" => excludes few-shot exemplars)
    q_pos = prompt_prefix.rfind("Question:")
    if q_pos == -1:
        q_pos = 0

    enc = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    input_ids = torch.tensor(ids, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    q_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for i, (start, end) in enumerate(offsets):
        if start < target_char_start:
            labels[i] = -100
        if q_pos <= start < target_char_start:
            q_mask[i] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "question_span_mask": q_mask,
        "n_shot": n_shot,
    }


def question_span_mask_from_prompts(tokenizer, prompts: List[str], padding_side: str = "left", max_len: int = 2048) -> torch.BoolTensor:
    """Generation-time left-padded batch question span mask."""
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    finally:
        tokenizer.padding_side = prev_side
    input_ids = enc["input_ids"]
    B, T = input_ids.shape
    q_mask = torch.zeros(B, T, dtype=torch.bool)
    for b, p in enumerate(prompts):
        q_start = p.rfind("Question:")
        if q_start == -1:
            q_start = 0
        single_enc = tokenizer(p, return_offsets_mapping=True, add_special_tokens=False)
        # Only the tokens that survived batch truncation are present in the row.
        n_real = int(enc["attention_mask"][b].sum().item())
        single_offsets = single_enc["offset_mapping"][:n_real]
        offset_in_row = T - n_real if padding_side == "left" else 0
        for i, (s, e) in enumerate(single_offsets):
            if s >= q_start:
                q_mask[b, offset_in_row + i] = True
    return q_mask
