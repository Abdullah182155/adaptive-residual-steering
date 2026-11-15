"""Shared left-padded generation for evaluation and RFT collection."""

from __future__ import annotations

from dataclasses import dataclass
import re
import sys
from typing import Any, Callable, Optional, Sequence

import torch
from tqdm import tqdm
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

from src.data.prompts import build_prompt_with_shield
from src.evaluation.answers import (
    cut_continuation,
    score_tiers,
    should_stop_on_repeat,
    truncate_at_answer,
)


_FINAL_LINE = re.compile(
    r"^[ \t]*Final answer:[ \t]*\$?-?[\d,]+(?:\.\d+)?[ \t]*\.?[ \t]*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class GenerationResult:
    """Text and token-level stopping details for one generated sequence."""

    text: str
    stop_reason: str
    n_new_tokens: int
    first_token_id: Optional[int] = None
    prompt_width: int = 0
    token_ids: tuple[int, ...] = ()


class _StopTracker:
    """Track token, final-answer, and repetition stops independently per sequence."""

    def __init__(
        self,
        prompt_width: int,
        stop_token_ids: Sequence[int],
        batch_size: int,
        tokenizer,
        device: torch.device,
    ):
        self.prompt_width = prompt_width
        self.stop_token_ids = torch.tensor(list(stop_token_ids), dtype=torch.long)
        self.finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.tokenizer = tokenizer
        self.reasons: list[Optional[str]] = [None] * batch_size

    def update(self, input_ids: torch.LongTensor) -> torch.BoolTensor:
        if input_ids.shape[0] != self.finished.numel():
            raise ValueError("generation batch size changed while tracking stop tokens")
        generated = input_ids[:, self.prompt_width :]
        if generated.numel():
            stop_ids = self.stop_token_ids.to(input_ids.device)
            matches = torch.isin(generated, stop_ids).any(dim=1)
            newly_stopped = matches & ~self.finished
            for row in newly_stopped.nonzero(as_tuple=False).flatten().tolist():
                self.reasons[row] = "eos"
            self.finished |= matches

            active = (~self.finished).nonzero(as_tuple=False).flatten().tolist()
            if active:
                active_tokens = generated[active]
                decoded = self.tokenizer.batch_decode(
                    active_tokens[:, -64:], skip_special_tokens=True
                )
                for row, text in zip(active, decoded):
                    if re.search(
                        r"(?:^|\n)[ \t]*(?:Final answer|####)\s*[:$-]?\s*"
                        r"\$?-?[\d,]+(?:\.\d+)?[^\n]*\n",
                        text,
                    ):
                        self.finished[row] = True
                        self.reasons[row] = "answer_stop"
                    elif generated.shape[1] % 8 == 0:
                        full_text = self.tokenizer.decode(
                            generated[row], skip_special_tokens=True
                        )
                        if should_stop_on_repeat(full_text):
                            self.finished[row] = True
                            self.reasons[row] = "repeat_stop"
        return self.finished


class _PerSequenceEosProcessor(LogitsProcessor):
    """Force already-finished sequences to EOS while others continue decoding."""

    def __init__(self, tracker: _StopTracker, terminal_token_id: int):
        self.tracker = tracker
        self.terminal_token_id = terminal_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        finished = self.tracker.update(input_ids)
        if not bool(finished.any()):
            return scores
        scores = scores.clone()
        scores[finished] = -torch.inf
        scores[finished, self.terminal_token_id] = 0
        return scores


class _AllSequencesStopped(StoppingCriteria):
    """Stop the batch only after every individual sequence has reached a stop token."""

    def __init__(self, tracker: _StopTracker):
        self.tracker = tracker

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return bool(self.tracker.update(input_ids).all())


def truncate_completion(text: str) -> str:
    """Remove hallucinated continuation and keep the first explicit final answer."""
    cleaned = cut_continuation(text)
    if _FINAL_LINE.search(cleaned):
        return truncate_at_answer(cleaned).rstrip()
    return cleaned.rstrip()


def build_prompts(
    tokenizer,
    questions: Sequence[str],
    question_indices: Sequence[int],
    n_shot: int,
    policy: str = "prompt",
) -> tuple[list[str], list[Optional[torch.Tensor]]]:
    """Build prompts and matching prefill shield masks for a batch of questions."""
    if len(questions) != len(question_indices):
        raise ValueError("questions and question_indices must have matching lengths")
    prompts: list[str] = []
    shields: list[Optional[torch.Tensor]] = []
    for question, index in zip(questions, question_indices):
        prompt, _, shield = build_prompt_with_shield(
            tokenizer,
            question,
            n_shot=n_shot,
            policy=policy,
            question_idx=index,
        )
        prompts.append(prompt)
        shields.append(shield if bool(shield.any()) else None)
    return prompts, shields


def _normalise_stop_ids(tokenizer, stop_token_ids: Optional[Sequence[int]]) -> list[int]:
    raw = stop_token_ids if stop_token_ids is not None else tokenizer.eos_token_id
    if raw is None:
        raise ValueError("tokenizer must define eos_token_id or explicit stop_token_ids")
    ids = [int(raw)] if isinstance(raw, int) else [int(token_id) for token_id in raw]
    eos = tokenizer.eos_token_id
    if eos is not None:
        ids.extend([int(eos)] if isinstance(eos, int) else [int(token_id) for token_id in eos])
    if not ids:
        raise ValueError("stop_token_ids cannot be empty")
    return list(dict.fromkeys(ids))


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: torch.device | str,
    *,
    prompt_shields: Optional[Sequence[Optional[torch.Tensor]]] = None,
    max_new_tokens: int = 256,
    num_return_sequences: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    max_prompt_len: int = 1024,
    stop_token_ids: Optional[Sequence[int]] = None,
    route_prefill_fn=None,
) -> list[GenerationResult]:
    """Generate completions with left padding and exact padded-width slicing.

    Results are grouped by input prompt, then by return sequence. Oversized prompts
    raise instead of silently truncating context or answer-relevant text.
    """
    if not prompts:
        return []
    if num_return_sequences < 1:
        raise ValueError("num_return_sequences must be positive")
    if num_return_sequences > 1 and temperature <= 0:
        raise ValueError("multiple return sequences require temperature > 0")
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be nonnegative and top_p must be in (0, 1]")
    if prompt_shields is None:
        prompt_shields = [None] * len(prompts)
    if len(prompt_shields) != len(prompts):
        raise ValueError("prompt_shields must have one entry per prompt")

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer must provide a padding token or EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    prompt_width = int(encoded["input_ids"].shape[1])
    if prompt_width > max_prompt_len:
        raise ValueError(
            f"padded prompt width {prompt_width} exceeds max_prompt_len={max_prompt_len}; "
            "reduce the shot count"
        )
    target_device = torch.device(device)
    inputs = {key: value.to(target_device) for key, value in encoded.items()}
    stop_ids = _normalise_stop_ids(tokenizer, stop_token_ids)
    if tokenizer.eos_token_id is None:
        terminal_token_id = stop_ids[0]
    else:
        eos_ids = _normalise_stop_ids(tokenizer, None)
        terminal_token_id = eos_ids[0]

    if any(shield is not None for shield in prompt_shields):
        attention_mask = inputs["attention_mask"].bool()
        shield_batch = torch.zeros_like(attention_mask)
        for row, shield in enumerate(prompt_shields):
            if shield is None:
                continue
            mask = torch.as_tensor(shield, dtype=torch.bool).flatten()
            live_length = int(attention_mask[row].sum())
            if mask.numel() != live_length:
                raise ValueError(
                    f"shield length {mask.numel()} does not match prompt length {live_length}"
                )
            shield_batch[row, prompt_width - live_length :] = mask.to(shield_batch.device)
        if not hasattr(model, "set_prompt_shield_mask"):
            raise TypeError("model does not support prompt shielding")
        model.set_prompt_shield_mask(shield_batch.repeat_interleave(num_return_sequences, 0))
    elif hasattr(model, "set_prompt_shield_mask"):
        model.set_prompt_shield_mask(None)

    expanded_size = len(prompts) * num_return_sequences
    tracker = _StopTracker(prompt_width, stop_ids, expanded_size, tokenizer, target_device)
    processors = LogitsProcessorList(
        [_PerSequenceEosProcessor(tracker, terminal_token_id)]
    )
    stopping = StoppingCriteriaList([_AllSequencesStopped(tracker)])
    generation_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "num_return_sequences": num_return_sequences,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": stop_ids,
        "logits_processor": processors,
        "stopping_criteria": stopping,
    }
    if temperature > 0:
        generation_kwargs.update(temperature=temperature, top_p=top_p)
    if repetition_penalty != 1.0:
        generation_kwargs["repetition_penalty"] = repetition_penalty
    if no_repeat_ngram_size > 0:
        generation_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    try:
        if route_prefill_fn is not None:
            route_prefill_fn(inputs["input_ids"], inputs["attention_mask"])
        output = model.generate(**generation_kwargs)
    finally:
        if hasattr(model, "set_prompt_shield_mask"):
            model.set_prompt_shield_mask(None)

    if output.shape[0] != expanded_size:
        raise RuntimeError(
            f"model returned {output.shape[0]} sequences; expected {expanded_size}"
        )
    results: list[GenerationResult] = []
    stop_set = set(stop_ids)
    for index, sequence in enumerate(output):
        generated = sequence[prompt_width:]
        stop_positions = [
            index for index, token in enumerate(generated.tolist()) if token in stop_set
        ]
        if stop_positions:
            end = stop_positions[0] + 1
            reason = tracker.reasons[index] or "eos"
        else:
            end = int(generated.numel())
            reason = tracker.reasons[index] or (
                "max_new_tokens" if end >= max_new_tokens else "length"
            )
        text = tokenizer.decode(generated[:end], skip_special_tokens=True)
        results.append(
            GenerationResult(
                text=truncate_completion(text),
                stop_reason=reason,
                n_new_tokens=end,
                first_token_id=int(generated[0]) if generated.numel() else None,
                prompt_width=prompt_width,
                token_ids=tuple(int(token_id) for token_id in generated[:end].tolist()),
            )
        )
    return results


@torch.no_grad()
def generate_texts(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: torch.device | str,
    *,
    prompt_shields: Optional[Sequence[Optional[torch.Tensor]]] = None,
    max_new_tokens: int = 256,
    num_return_sequences: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    max_prompt_len: int = 1024,
    stop_token_ids: Optional[Sequence[int]] = None,
) -> list[str]:
    """Text-only convenience wrapper around :func:`generate_batch`."""
    results = generate_batch(
        model,
        tokenizer,
        prompts,
        device,
        prompt_shields=prompt_shields,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        max_prompt_len=max_prompt_len,
        stop_token_ids=stop_token_ids,
    )
    return [result.text for result in results]


@torch.no_grad()
def equivalence_test(
    model,
    tokenizer,
    rows: Sequence[dict],
    device: torch.device | str,
    cfg=None,
    *,
    n: int = 100,
    batch_size: int = 2,
) -> dict[str, Any]:
    """Compare greedy batch-1 and batched text, paired accuracy, and identity.

    The harness passes only when lenient accuracy differs by at most 1.5 percentage
    points and at least 90% of generated strings are identical.
    """
    selected = list(rows[:n])
    if not selected:
        raise ValueError("equivalence_test requires at least one row")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    policy = getattr(cfg, "shield_policy", "prompt")
    max_new_tokens = getattr(cfg, "generation_max_new_tokens", 256)
    max_prompt_len = getattr(cfg, "max_tok_len", 768)
    one_at_a_time: list[GenerationResult] = []
    with tqdm(
        total=len(selected),
        desc="Equivalence batch-1",
        unit="question",
        file=sys.stdout,
        disable=False,
        mininterval=0.5,
    ) as progress:
        for index, row in enumerate(selected):
            prompt, shields = build_prompts(
                tokenizer, [row["question"]], [index], 0, policy=policy
            )
            one_at_a_time.extend(
                generate_batch(
                    model,
                    tokenizer,
                    prompt,
                    device,
                    prompt_shields=shields,
                    max_new_tokens=max_new_tokens,
                    max_prompt_len=max_prompt_len,
                    repetition_penalty=1.0,
                )
            )
            progress.update(1)
            sys.stdout.flush()

    batched: list[GenerationResult] = []
    n_batches = (len(selected) + batch_size - 1) // batch_size
    with tqdm(
        total=n_batches,
        desc=f"Equivalence batched (B={batch_size})",
        unit="batch",
        file=sys.stdout,
        disable=False,
        mininterval=0.5,
    ) as progress:
        for start in range(0, len(selected), batch_size):
            batch_rows = selected[start : start + batch_size]
            indices = list(range(start, start + len(batch_rows)))
            prompts, shields = build_prompts(
                tokenizer,
                [row["question"] for row in batch_rows],
                indices,
                0,
                policy=policy,
            )
            batched.extend(
                generate_batch(
                    model,
                    tokenizer,
                    prompts,
                    device,
                    prompt_shields=shields,
                    max_new_tokens=max_new_tokens,
                    max_prompt_len=max_prompt_len,
                    repetition_penalty=1.0,
                )
            )
            progress.update(1)
            sys.stdout.flush()

    golds = [
        float(str(row.get("final_num", row.get("answer"))).replace(",", "").replace("$", "").strip())
        for row in selected
    ]
    one_texts = [result.text for result in one_at_a_time]
    batch_texts = [result.text for result in batched]
    one_scores = [score_tiers(text, gold) for text, gold in zip(one_texts, golds)]
    batch_scores = [score_tiers(text, gold) for text, gold in zip(batch_texts, golds)]
    acc_one = sum(score["lenient"] for score in one_scores) / len(selected)
    acc_batch = sum(score["lenient"] for score in batch_scores) / len(selected)
    mismatch_indices = [
        index
        for index, (one_text, batch_text) in enumerate(zip(one_texts, batch_texts))
        if one_text != batch_text
    ]

    def first_divergence(
        one_result: GenerationResult, batch_result: GenerationResult
    ) -> tuple[Optional[int], Optional[int], Optional[int]]:
        for token_index, (one_token, batch_token) in enumerate(
            zip(one_result.token_ids, batch_result.token_ids)
        ):
            if one_token != batch_token:
                return token_index, one_token, batch_token
        if len(one_result.token_ids) != len(batch_result.token_ids):
            token_index = min(len(one_result.token_ids), len(batch_result.token_ids))
            one_token = (
                one_result.token_ids[token_index]
                if token_index < len(one_result.token_ids)
                else None
            )
            batch_token = (
                batch_result.token_ids[token_index]
                if token_index < len(batch_result.token_ids)
                else None
            )
            return token_index, one_token, batch_token
        return None, None, None

    identity = 1 - len(mismatch_indices) / len(selected)
    delta_pp = 100 * (acc_batch - acc_one)
    result = {
        "n": len(selected),
        "batch1_lenient_accuracy": 100 * acc_one,
        "batched_lenient_accuracy": 100 * acc_batch,
        "delta_lenient_pp": delta_pp,
        "string_identity": identity,
        "n_string_mismatches": len(mismatch_indices),
        "mismatch_examples": [
            _mismatch_example(
                index,
                selected,
                one_texts,
                batch_texts,
                one_at_a_time,
                batched,
                one_scores,
                batch_scores,
                first_divergence,
            )
            for index in mismatch_indices[:5]
        ],
        "passed": abs(delta_pp) <= 1.5 and identity >= 0.90,
    }
    print(
        f"[generation-equivalence] n={result['n']} "
        f"delta={delta_pp:+.2f}pp identity={identity:.1%} passed={result['passed']}"
    )
    return result


def _mismatch_example(
    index: int,
    rows: Sequence[dict],
    one_texts: Sequence[str],
    batch_texts: Sequence[str],
    one_results: Sequence[GenerationResult],
    batch_results: Sequence[GenerationResult],
    one_scores: Sequence[dict],
    batch_scores: Sequence[dict],
    first_divergence: Callable[
        [GenerationResult, GenerationResult],
        tuple[Optional[int], Optional[int], Optional[int]],
    ],
) -> dict[str, Any]:
    """Build a compact per-question diagnostic for generation differences."""
    one_result = one_results[index]
    batch_result = batch_results[index]
    divergence_index, one_token, batch_token = first_divergence(one_result, batch_result)
    return {
        "row_index": index,
        "question": str(rows[index]["question"])[:300],
        "batch1_text": one_texts[index][:500],
        "batched_text": batch_texts[index][:500],
        "batch1_first_token_id": one_result.first_token_id,
        "batched_first_token_id": batch_result.first_token_id,
        "first_divergent_token_index": divergence_index,
        "batch1_divergent_token_id": one_token,
        "batched_divergent_token_id": batch_token,
        "batch1_prompt_width": one_result.prompt_width,
        "batched_prompt_width": batch_result.prompt_width,
        "batch1_stop_reason": one_result.stop_reason,
        "batched_stop_reason": batch_result.stop_reason,
        "batch1_new_tokens": one_result.n_new_tokens,
        "batched_new_tokens": batch_result.n_new_tokens,
        "batch1_lenient": bool(one_scores[index]["lenient"]),
        "batched_lenient": bool(batch_scores[index]["lenient"]),
    }
