import re
import operator
import random
import time
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import torch
import numpy as np
from transformers import StoppingCriteria, StoppingCriteriaList
from tqdm.auto import tqdm
from src.data.prompts import build_prompt_unified

# Marker for BatchAnswerStopping
_ANSWER_START_RE = re.compile(r"####\s*\$?-?\d|[Ff]inal\s+answer\s*[:\-]?\s*\$?-?\d")
_RUNAWAY = "\nQuestion:"


class BatchAnswerStopping(StoppingCriteria):
    """Per-row batch stopping criteria for HuggingFace generate.

    Tracks each row independently: a row is finished on EOS, on an answer marker
    ('#### <num>', 'Final answer: <num>') once `grace_tokens` more tokens have been
    produced (so the number can complete), or immediately on runaway continuation
    ('\\nQuestion:'). Returns True only when every row in the batch is finished.
    """

    def __init__(self, tokenizer, prompt_len: int, grace_tokens: int = 12, window: int = 32):
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.grace_tokens = int(grace_tokens)
        self.window = int(window)
        self.eos_id = getattr(tokenizer, "eos_token_id", None)
        self.row_done: List[bool] = []
        self._marker_at: List[Optional[int]] = []

    def _init_rows(self, n: int) -> None:
        if len(self.row_done) != n:
            self.row_done = [False] * n
            self._marker_at = [None] * n

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        n = input_ids.shape[0]
        self._init_rows(n)
        cur_len = input_ids.shape[1] - self.prompt_len
        if cur_len <= 0:
            return False
        new_ids = input_ids[:, self.prompt_len:]
        for b in range(n):
            if self.row_done[b]:
                continue
            row = new_ids[b]
            if self.eos_id is not None and bool((row == self.eos_id).any()):
                self.row_done[b] = True
                continue
            if self._marker_at[b] is None:
                text = self.tokenizer.decode(row[-self.window:], skip_special_tokens=True)
                if _RUNAWAY in text:
                    self.row_done[b] = True
                    continue
                if _ANSWER_START_RE.search(text):
                    self._marker_at[b] = cur_len
            if self._marker_at[b] is not None and cur_len >= self._marker_at[b] + self.grace_tokens:
                self.row_done[b] = True
        return all(self.row_done)

GSM8K_8SHOT_EXEMPLARS = [
    {
        "q": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "a": "Step 1: There are 15 trees originally.\nStep 2: Then there were 21 trees after some more were planted.\nStep 3: So there must have been 21 - 15 = 6 trees planted.\nFinal answer: 6",
    },
    {
        "q": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "a": "Step 1: There are originally 3 cars.\nStep 2: 2 more cars arrive.\nStep 3: 3 + 2 = 5 cars.\nFinal answer: 5",
    },
    {
        "q": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "a": "Step 1: Originally, Leah had 32 chocolates.\nStep 2: Her sister had 42.\nStep 3: So in total they had 32 + 42 = 74.\nStep 4: After eating 35, they had 74 - 35 = 39.\nFinal answer: 39",
    },
    {
        "q": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "a": "Step 1: Jason started with 20 lollipops.\nStep 2: Then he had 12 after giving some to Denny.\nStep 3: So he gave Denny 20 - 12 = 8 lollipops.\nFinal answer: 8",
    },
    {
        "q": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "a": "Step 1: Shawn started with 5 toys.\nStep 2: He got 2 toys from his mom.\nStep 3: He got 2 toys from his dad.\nStep 4: 5 + 2 + 2 = 9 toys.\nFinal answer: 9",
    },
    {
        "q": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "a": "Step 1: There were originally 9 computers.\nStep 2: 5 more computers were added each day.\nStep 3: From Monday to Thursday is 4 days.\nStep 4: So 5 * 4 = 20 computers were added.\nStep 5: 9 + 20 = 29 computers.\nFinal answer: 29",
    },
    {
        "q": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "a": "Step 1: Michael started with 58 golf balls.\nStep 2: After losing 23 on Tuesday, he had 58 - 23 = 35.\nStep 3: After losing 2 more on Wednesday, he had 35 - 2 = 33.\nFinal answer: 33",
    },
    {
        "q": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "a": "Step 1: Olivia started with $23.\nStep 2: She bought 5 bagels for $3 each.\nStep 3: 5 bagels cost 5 * 3 = 15 dollars.\nStep 4: 23 - 15 = 8 dollars.\nFinal answer: 8",
    },
]

def build_prompt(question: str, n_shot: int = 0) -> str:
    """Legacy wrapper for unified prompt builder. Use build_prompt_unified directly for new code."""
    return build_prompt_unified(question, n_shot=n_shot, exemplar_set="legacy", include_preface=False)

def extract_and_verify_equations(text: str) -> Tuple[int, int, List[Tuple[str, bool]]]:
    """Enhanced equation verifier: handles standard and reversed equations."""
    pattern = (
        r"(-?[\d]+\.?[\d]*)\s*([\+\-\*\/])\s*(-?[\d]+\.?[\d]*)\s*=\s*(-?[\d]+\.?[\d]*)"
        r"|(-?[\d]+\.?[\d]*)\s*=\s*(-?[\d]+\.?[\d]*)\s*([\+\-\*\/])\s*(-?[\d]+\.?[\d]*)"
    )
    ops = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
    total, correct = 0, 0
    details = []

    for m in re.finditer(pattern, text):
        try:
            if m.group(1) is not None:
                a, op_str, b, result = float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
            else:
                result, a, op_str, b = float(m.group(5)), float(m.group(6)), m.group(7), float(m.group(8))
            if op_str == "/" and b == 0:
                continue
            expected = ops[op_str](a, b)
            is_correct = abs(expected - result) <= 0.01
            total += 1
            correct += int(is_correct)
            details.append((f"{a}{op_str}{b}={result}", is_correct))
        except (ValueError, ZeroDivisionError, TypeError):
            continue

    return total, correct, details

def extract_final_number(text: str) -> Optional[float]:
    """Pull the numeric answer from text. Alias of extract_numeric_answer (ISSUE-13 unified extractor)."""
    from src.evaluation.answers import extract_numeric_answer
    return extract_numeric_answer(text)

def extract_final_answer(text: str) -> Optional[float]:
    """Extract numeric answer via strict priority levels. Alias of extract_numeric_answer (ISSUE-13 unified extractor)."""
    from src.evaluation.answers import extract_numeric_answer
    return extract_numeric_answer(text)


def evaluate_response(response: str, expected: Optional[float], min_steps: int = 1) -> Dict[str, Any]:
    extracted = extract_final_answer(response)
    tolerance = 0.01
    correct = (extracted is not None and expected is not None and abs(extracted - expected) <= tolerance)
    step_count = max(
        len(re.findall(r"Step \d+", response, re.IGNORECASE)),
        len(re.findall(r"\n\d+[\.\)]\s", response)),
        len(re.findall(r"=\s*\$?-?\d", response)),
    )
    has_answer = extracted is not None
    valid = has_answer and step_count >= min_steps
    quality = "valid" if valid else ("incomplete" if step_count > 0 else "invalid")
    return {
        "correct": correct,
        "extracted": extracted,
        "steps": step_count,
        "has_answer": has_answer,
        "quality": quality,
    }

def _safe_decode(tokenizer, token_ids, skip_special_tokens: bool = True) -> str:
    """Decode tokens tolerating mock/custom tokenizers without clean_up_tokenization_spaces."""
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


class AnswerStoppingCriteria(StoppingCriteria):
    """Stops generation with a grace period once an answer marker is emitted."""

    def __init__(self, tokenizer, prompt_length: int, max_new: int = 512, grace_tokens: int = 48):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.max_new = max_new
        self.grace_tokens = grace_tokens
        if hasattr(tokenizer, "encode"):
            hash_ids = tokenizer.encode("####", add_special_tokens=False)
            final_ids = tokenizer.encode("\nFinal answer:", add_special_tokens=False)
            final_ids_cap = tokenizer.encode("\nFinal Answer:", add_special_tokens=False)
            self.stop_sequences = [torch.tensor(hash_ids), torch.tensor(final_ids), torch.tensor(final_ids_cap)]
        else:
            hash_ids, final_ids, final_ids_cap = [1001], [1002], [1003]
            self.stop_sequences = [torch.tensor(hash_ids), torch.tensor(final_ids), torch.tensor(final_ids_cap)]
        self._marker_seen_at = None

    def __call__(self, input_ids: torch.Tensor, scores, **kwargs) -> bool:
        cur_len = input_ids.shape[1] - self.prompt_length
        if cur_len > self.max_new:
            return True

        # Text-level early stops: cut at hallucinated follow-up questions and
        # repetition loops. These apply symmetrically to all arms (baseline and
        # ARS) so they do not introduce measurement bias. Check every 8 tokens
        # to avoid per-token decode overhead.
        if cur_len >= 8 and cur_len % 8 == 0:
            from src.evaluation.answers import _STOP_MARKERS, should_stop_on_repeat
            # Decode last ~160 tokens for follow-up question detection
            tail_ids = input_ids[0, self.prompt_length:]
            tail_start = max(0, tail_ids.shape[0] - 160)
            tail_text = _safe_decode(self.tokenizer, tail_ids[tail_start:])
            if any(re.search(p, tail_text) for p in _STOP_MARKERS):
                return True
            if cur_len % 16 == 0:
                full_tail = _safe_decode(self.tokenizer, tail_ids)
                if should_stop_on_repeat(full_tail):
                    return True

        if self._marker_seen_at is not None:
            return cur_len >= self._marker_seen_at + self.grace_tokens

        for batch_idx in range(input_ids.shape[0]):
            new_ids = input_ids[batch_idx, self.prompt_length :]
            if len(new_ids) < 3:
                continue
            window = new_ids[-20:]
            for stop_seq in self.stop_sequences:
                stop_seq = stop_seq.to(input_ids.device)
                L = len(stop_seq)
                if len(window) >= L:
                    for i in range(len(window) - L + 1):
                        if torch.all(window[i : i + L] == stop_seq):
                            self._marker_seen_at = cur_len
                            return False
        return False

def epoch_end_accuracy(
    model, tokenizer, eval_data, device, n_samples=40, max_new_tokens=384,
    seed=0, progress_desc: Optional[str] = None,
) -> float:
    """Lightweight final-answer accuracy check for checkpoint selection (R6 fix).

    Routes generation through AnswerStoppingCriteria, cut_continuation, and
    truncate_at_answer to maintain symmetry with benchmark evaluation harness.
    """
    rng = random.Random(seed)
    buckets = {"easy": [], "medium": [], "hard": []}
    for idx, ex in enumerate(eval_data):
        n_steps = len(ex.get("steps_list", [])) if isinstance(ex, dict) else 0
        bucket = "easy" if n_steps <= 1 else "medium" if n_steps <= 3 else "hard"
        buckets[bucket].append(idx)
    # BUG-09 fix: old code used min() across ALL buckets → tiny bucket shrank total eval to 6-9 samples.
    # New approach: sample proportionally per bucket (at least 1 each), then fill remainder from all eval_data.
    target_n = min(n_samples, len(eval_data))
    non_empty = {k: v for k, v in buckets.items() if v}
    per_bucket = max(1, target_n // max(1, len(non_empty)))
    idxs = []
    for values in non_empty.values():
        idxs.extend(rng.sample(values, min(per_bucket, len(values))))
    # Fill remainder from any remaining eval examples to reach target_n
    if len(idxs) < target_n:
        remaining = sorted(set(range(len(eval_data))) - set(idxs))
        rng.shuffle(remaining)
        idxs.extend(remaining[: target_n - len(idxs)])

    was_training = model.training
    model.eval()
    correct, total = 0, 0

    # Keep audit bars in notebook output.  A cleared bar looks like a stalled
    # cell in Kaggle even though Phi-2 generation is still progressing.
    iterator = tqdm(idxs, desc=progress_desc, unit="sample", leave=True) if progress_desc else idxs
    with torch.no_grad():
        for i in iterator:
            ids = eval_data[i]["input_ids"]
            lbls = eval_data[i]["labels"]
            pad_id = tokenizer.pad_token_id
            q_mask = (lbls == -100) & (ids != pad_id)
            if q_mask.sum() == 0:
                continue
            q_ids = ids[q_mask].unsqueeze(0).to(device)
            gold_mask = lbls != -100
            if gold_mask.sum() == 0:
                continue
            gold_text = _safe_decode(tokenizer, ids[gold_mask]).strip()
            from src.evaluation.answers import cut_continuation, truncate_at_answer, extract_numeric_answer
            gold_num = extract_numeric_answer(gold_text)
            if gold_num is None:
                gold_num = extract_final_answer(gold_text)
            if gold_num is None:
                continue

            q_attn = torch.ones_like(q_ids)
            stopping = StoppingCriteriaList([AnswerStoppingCriteria(tokenizer, q_ids.shape[1], max_new_tokens)])
            gen = model.generate(
                input_ids=q_ids, attention_mask=q_attn, max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True, pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id, stopping_criteria=stopping,
            )
            raw_gen = _safe_decode(tokenizer, gen[0][q_ids.size(1) :]).strip()
            gen_text = truncate_at_answer(cut_continuation(raw_gen))
            pred_num = extract_numeric_answer(gen_text)
            if pred_num is None:
                pred_num = extract_final_answer(gen_text)
            total += 1
            if pred_num is not None and abs(pred_num - gold_num) < 0.01:
                correct += 1
            if progress_desc:
                iterator.set_postfix(acc=f"{correct / total:.1%}", valid=total)

    if was_training:
        model.train()
    return correct / max(1, total)

def generate_response(
    model, tokenizer, prompt: str, device, max_new_tokens=512,
    collect_trace: bool = False,
    collect_routing_trace: bool = False,
) -> str:
    """Generate without silently truncating a benchmark's final question.

    A few-shot result is invalid if the prompt plus requested completion does
    not fit the backbone context.  The previous right-truncation could drop
    the test question while still returning a numerical-looking score.
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
    prompt_length = inputs["input_ids"].shape[1]
    backbone = getattr(model, "base_model", model)
    model_cfg = getattr(backbone, "config", None)
    context_limit = getattr(model_cfg, "max_position_embeddings", None)
    if context_limit is None:
        context_limit = getattr(model_cfg, "n_positions", None)
    if context_limit is not None and prompt_length + max_new_tokens > int(context_limit):
        raise ValueError(
            "Benchmark prompt exceeds model context budget: "
            f"prompt={prompt_length}, max_new_tokens={max_new_tokens}, context={context_limit}. "
            "Reduce shots or completion budget; do not silently truncate the test question."
        )
    stop_criterion = AnswerStoppingCriteria(tokenizer, prompt_length, max_new_tokens)
    stopping = StoppingCriteriaList([stop_criterion])

    # Multi-shot prompt shielding: protect few-shot demonstrations from steering representation drift
    if hasattr(model, "set_prompt_shield_mask"):
        q_toks = tokenizer.encode("Question:", add_special_tokens=False)
        inp_list = inputs["input_ids"][0].tolist()
        last_match = -1
        for i in range(len(inp_list) - len(q_toks) + 1):
            if inp_list[i : i + len(q_toks)] == q_toks:
                last_match = i
        if last_match > 0:
            shield_mask = torch.zeros((inputs["input_ids"].shape[0], prompt_length), dtype=torch.bool, device=device)
            shield_mask[:, :last_match] = True
            model.set_prompt_shield_mask(shield_mask)

    if collect_trace and hasattr(model, "set_gate_trace"):
        model.set_gate_trace(True)
    if (collect_trace or collect_routing_trace) and hasattr(model, "set_routing_trace"):
        model.set_routing_trace(True)
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping,
                do_sample=False,
                # ISSUE-12 fix: removed repetition_penalty=1.1 and no_repeat_ngram_size=6.
                # Those flags penalised recurring arithmetic tokens and banned 6-token prompt
                # spans (including few-shot template endings like "dollars.\nFinal answer:"),
                # producing asymmetric confounds between baseline and ARS arms.
                use_cache=True,
            )
    finally:
        if collect_trace or collect_routing_trace:
            # Copy before disabling collection, whose public setter clears
            # transient state.  The resulting object is JSON-safe and can be
            # attached to the qualitative result without retaining tensors.
            gate_trace = {
                int(layer): list(values)
                for layer, values in (getattr(model, "last_gate_trace", None) or {}).items()
            } if collect_trace else {}
            routing_trace = {
                int(layer): dict(values)
                for layer, values in (getattr(model, "last_routing_trace", None) or {}).items()
            }
            model.last_generation_trace = {
                "gate": gate_trace,
                "routing": routing_trace,
                "generated_token_ids": output_ids[0][prompt_length:].detach().cpu().tolist() if "output_ids" in locals() else [],
            }
            if hasattr(model, "set_gate_trace"):
                model.set_gate_trace(False)
            if hasattr(model, "set_routing_trace"):
                model.set_routing_trace(False)
        if hasattr(model, "set_prompt_shield_mask"):
            model.set_prompt_shield_mask(None)

    new_ids = output_ids[0][prompt_length:]
    raw_text = _safe_decode(tokenizer, new_ids).strip()

    from src.evaluation.answers import cut_continuation, truncate_at_answer
    raw_text = cut_continuation(raw_text)
    raw_text = truncate_at_answer(raw_text)
    return raw_text

