import json
import time
import torch
import numpy as np
from datasets import load_dataset
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.configs.config import RSCConfig
from src.models.wrapper import Phi2WithRSC
from src.evaluation.verifier import (
    build_prompt,
    generate_response,
    extract_final_answer,
    evaluate_response,
)

def eval_gsm8k_benchmark(
    model,
    tokenizer,
    device: torch.device,
    stage_name: str,
    n_test: int = 100,
    n_shot: int = 0,
    display_examples: int = 3,
) -> Dict[str, Any]:
    """Runs reproducible GSM8K test-set evaluation matching Stage A / B / C protocol."""
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=42)
    ds = ds.select(range(min(n_test, len(ds))))

    correct = 0
    results = []
    total_time = 0.0
    total_tokens = 0

    print(f"\n{'=' * 80}")
    print(f"  GSM8K Benchmark: {stage_name} (n={len(ds)}, {n_shot}-shot)")
    print(f"{'=' * 80}")

    for idx, ex in enumerate(ds):
        q = ex["question"]
        gold_parts = ex["answer"].split("####")
        gold = float(gold_parts[1].replace(",", "").strip()) if len(gold_parts) > 1 else None

        prompt = build_prompt(q, n_shot=n_shot)
        t0 = time.time()
        resp = generate_response(model, tokenizer, prompt, device, max_new_tokens=512)
        elapsed = time.time() - t0
        total_time += elapsed

        eval_res = evaluate_response(resp, gold, min_steps=1)
        is_corr = eval_res["correct"]
        if is_corr:
            correct += 1

        n_tok = len(tokenizer.encode(resp, add_special_tokens=False))
        total_tokens += n_tok

        item = {
            "idx": idx,
            "question": q,
            "gold": gold,
            "response": resp,
            "extracted": eval_res["extracted"],
            "correct": is_corr,
            "steps": eval_res["steps"],
            "has_answer": eval_res["has_answer"],
            "time_sec": elapsed,
            "tokens": n_tok,
        }
        results.append(item)

        if idx < display_examples:
            icon = "✅" if is_corr else "❌"
            print(f"[{idx+1}/{len(ds)}] {icon} Expected: {gold} | Got: {eval_res['extracted']} ({elapsed:.2f}s)")

    acc = 100.0 * correct / max(1, len(results))
    ans_present = 100.0 * sum(r["has_answer"] for r in results) / max(1, len(results))
    step_cov = 100.0 * sum(r["steps"] > 0 for r in results) / max(1, len(results))

    summary = {
        "stage": stage_name,
        "n_examples": len(results),
        "n_shot": n_shot,
        "correct": correct,
        "accuracy": acc,
        "answer_present": ans_present,
        "step_coverage": step_cov,
        "avg_time_sec": total_time / max(1, len(results)),
        "tokens_per_sec": total_tokens / max(1e-5, total_time),
    }

    print(f"\n{'-' * 80}")
    print(f"Summary [{stage_name} - {n_shot} shot]:")
    print(f"Accuracy:       {acc:.2f}% ({correct}/{len(results)})")
    print(f"Answer Present: {ans_present:.2f}%")
    print(f"Tokens/sec:     {summary['tokens_per_sec']:.2f}")
    print(f"{'-' * 80}\n")

    return {"summary": summary, "results": results}
