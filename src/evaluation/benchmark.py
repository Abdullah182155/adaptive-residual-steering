"""GSM8K evaluator for the baseline and fixed SteerNet route."""
from typing import Dict, Iterable
import torch
from src.data.prompts import build_prompt_unified
from src.evaluation.answers import extract_numeric_answer

@torch.no_grad()
def evaluate_gsm8k(model, tokenizer, rows: Iterable[dict], device, n_shot: int, max_new_tokens: int = 256) -> Dict[str, float]:
    model.set_current_n_shot(n_shot); total = correct = 0
    for row in rows:
        inputs = tokenizer(build_prompt_unified(row["question"], n_shot=n_shot), return_tensors="pt").to(device)
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        correct += extract_numeric_answer(text, lenient=True) == row["answer"]; total += 1
    return {"n": total, "accuracy": 100.0 * correct / max(total, 1), "shot": n_shot}
