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
    prompt = ""
    for i in range(min(n_shot, len(GSM8K_8SHOT_EXEMPLARS))):
        ex = GSM8K_8SHOT_EXEMPLARS[i]
        prompt += f"Question: {ex['q']}\nAnswer: Let's think step by step.\n{ex['a']}\n\n"
    prompt += f"Question: {question}\nAnswer: Let's think step by step.\n"
    return prompt

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
    """Pull the number after '#### ' if present, else fallback to the last number in text."""
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            return None
    return None

def extract_final_answer(text: str) -> Optional[float]:
    """Extract numeric answer via strict priority levels."""
    def safe_float(s: str) -> Optional[float]:
        try:
            s = s.replace(",", "").replace("$", "").strip()
            if not s or not any(c.isdigit() for c in s):
                return None
            return float(s)
        except (ValueError, AttributeError):
            return None

    m = re.search(r"####\s*\$?(-?\d[\d,]*\.?\d*)", text)
    if m:
        return safe_float(m.group(1))

    m = re.search(r"[Ff]inal\s+answer(?:\s+is)?\s*[:\-]?\s*\$?(-?[\d,]+\.?\d*)", text)
    if m:
        return safe_float(m.group(1))

    m = re.search(r"[Tt]he\s+answer\s+is\s*[:\-]?\s*\$?(-?[\d,]+\.?\d*)", text)
    if m:
        return safe_float(m.group(1))

    m = re.search(r"[Tt]herefore[^.]*?(\d[\d,]*\.?\d*)[^.]*\.", text)
    if m:
        sentence_match = re.search(r"[Tt]herefore[^.]*\.", text)
        if sentence_match:
            nums = re.findall(r"-?[\d,]+\.?\d*", sentence_match.group())
            if nums:
                return safe_float(nums[-1])

    matches = list(re.finditer(r"(?:=|is)\s*\$?(-?[\d,]+\.?\d*)", text))
    if matches:
        return safe_float(matches[-1].group(1))

    return None

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

class AnswerStoppingCriteria(StoppingCriteria):
    """Stops generation with a grace period once an answer marker is emitted."""

    def __init__(self, tokenizer, prompt_length: int, max_new: int = 512, grace_tokens: int = 12):
        self.prompt_length = prompt_length
        self.max_new = max_new
        self.grace_tokens = grace_tokens
        hash_ids = tokenizer.encode("####", add_special_tokens=False)
        final_ids = tokenizer.encode("\nFinal answer:", add_special_tokens=False)
        self.stop_sequences = [torch.tensor(hash_ids), torch.tensor(final_ids)]
        self._marker_seen_at = None

    def __call__(self, input_ids: torch.Tensor, scores, **kwargs) -> bool:
        if input_ids.shape[1] - self.prompt_length > self.max_new:
            return True
        cur_len = input_ids.shape[1] - self.prompt_length
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

def epoch_end_accuracy(model, tokenizer, eval_data, device, n_samples=40, max_new_tokens=200, seed=0) -> float:
    """Lightweight final-answer accuracy check for checkpoint selection."""
    rng = random.Random(seed)
    buckets = {"easy": [], "medium": [], "hard": []}
    for idx, ex in enumerate(eval_data):
        n_steps = len(ex.get("steps_list", [])) if isinstance(ex, dict) else 0
        bucket = "easy" if n_steps <= 1 else "medium" if n_steps <= 3 else "hard"
        buckets[bucket].append(idx)
    per_bucket = min(n_samples // 3, *(len(v) for v in buckets.values()))
    idxs = []
    for values in buckets.values():
        idxs.extend(rng.sample(values, per_bucket))
    remainder = min(n_samples, len(eval_data)) - len(idxs)
    if remainder > 0:
        remaining = sorted(set(range(len(eval_data))) - set(idxs))
        idxs.extend(rng.sample(remaining, min(remainder, len(remaining))))

    was_training = model.training
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for i in idxs:
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
            gold_text = tokenizer.decode(ids[gold_mask], skip_special_tokens=True)
            gold_num = extract_final_number(gold_text)
            if gold_num is None:
                continue

            q_attn = torch.ones_like(q_ids)
            gen = model.generate(
                input_ids=q_ids, attention_mask=q_attn, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
            )
            gen_text = tokenizer.decode(gen[0][q_ids.size(1) :], skip_special_tokens=True)
            pred_num = extract_final_number(gen_text)
            total += 1
            if pred_num is not None and abs(pred_num - gold_num) < 0.01:
                correct += 1

    if was_training:
        model.train()
    return correct / max(1, total)

def generate_response(model, tokenizer, prompt: str, device, max_new_tokens=512) -> str:
    """Generate response with repetition penalty and answer stopping criteria."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    prompt_length = inputs["input_ids"].shape[1]
    stop_criterion = AnswerStoppingCriteria(tokenizer, prompt_length, max_new_tokens)
    stopping = StoppingCriteriaList([stop_criterion])

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=6,
            use_cache=True,
        )

    new_ids = output_ids[0][prompt_length:]
    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    for pat in [
        r"####\s*-?\d[\d,]*\.?\d*",
        r"[Ff]inal\s+answer\s*[:\-]?\s*-?\d[\d,]*\.?\d*",
        r"[Tt]herefore[^.\n]*\.",
        r"[Tt]he answer is[^.\n]*\.",
    ]:
        m = re.search(pat, raw_text)
        if m:
            raw_text = raw_text[: m.end()].strip()
            break

    return raw_text
