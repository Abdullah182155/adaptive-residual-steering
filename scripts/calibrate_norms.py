"""
Per-layer norm calibration for steering envelope (N_l).

Phase 0.6: Compute median non-sink ‖h‖ per candidate layer, outlier-dimension profile,
and sink-token norm ratio.

Usage:
    python scripts/calibrate_norms.py --n_samples 500 --output norms_calibration.json
"""

import argparse
import json
import random
import time
import sys
from pathlib import Path
from typing import Dict, List

# ── Silence HF weight-loading bars and INFO spam BEFORE importing transformers ──
_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root))
from src.utils.logging_utils import setup_quiet_logging
setup_quiet_logging()  # default: "quiet" — set level="verbose" for full HF output

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import numpy as np
from datasets import load_dataset


def load_calibration_questions(
    n_samples: int,
    *,
    rollout_files: List[str] | None = None,
    seed: int = 42,
) -> List[str]:
    """Select unique prompts from saved RFT shards or GSM8K train, never test."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if rollout_files:
        questions = []
        seen = set()
        for file_name in rollout_files:
            with Path(file_name).open(encoding="utf-8") as stream:
                for line in stream:
                    question = json.loads(line).get("question")
                    if question and question not in seen:
                        seen.add(question)
                        questions.append(question)
        if len(questions) < n_samples:
            raise ValueError(
                f"rollout shards contain {len(questions)} unique questions; "
                f"need {n_samples} for norm calibration"
            )
        return random.Random(seed).sample(questions, n_samples)

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    questions = [row["question"] for row in dataset]
    return random.Random(seed).sample(questions, min(n_samples, len(questions)))


def collect_layer_norms(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    candidate_layers: List[int],
    sink_tokens: List[int],
    max_length: int = 512,
) -> Dict[int, List[float]]:
    """
    Collect hidden state norms for each candidate layer using output_hidden_states.
    
    Returns:
        Dict mapping layer -> list of norms (excluding sink tokens)
    """
    model.eval()
    layer_norms = {layer: [] for layer in candidate_layers}
    
    print(f"Using output_hidden_states to collect norms")
    print(f"Sink tokens: {sink_tokens}")
    
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Collecting norms")):
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        
        # outputs.hidden_states is a tuple of (embeddings + all layer outputs)
        # Index 0 is embeddings, index 1 is layer 0, etc.
        if outputs.hidden_states is None:
            print("Warning: model returned no hidden_states")
            continue
            
        seq_len = input_ids.shape[1]
        
        for layer in candidate_layers:
            # layer + 1 because hidden_states[0] is embeddings
            if layer + 1 >= len(outputs.hidden_states):
                print(f"Warning: layer {layer} + 1 >= len(hidden_states) {len(outputs.hidden_states)}")
                continue
                
            hs = outputs.hidden_states[layer + 1]  # (batch, seq_len, hidden_dim)
            if hs is None:
                print(f"Warning: layer {layer} hs is None")
                continue
            if len(hs.shape) < 2:
                print(f"Warning: layer {layer} hs shape has <2 dims: {hs.shape}")
                continue
            if hs.shape[0] == 0:
                print(f"Warning: layer {layer} hs batch dim is 0: {hs.shape}")
                continue
            
            # hs shape: (batch, seq_len, hidden_dim) or (seq_len, hidden_dim)
            if len(hs.shape) == 3:
                hs = hs[0]  # (seq_len, hidden_dim)
            
            norms = torch.norm(hs, dim=-1).cpu()  # (seq_len,)
            if not isinstance(norms, torch.Tensor) or len(norms.shape) == 0:
                continue
            
            # Count valid positions
            valid_count = 0
            # Exclude sink tokens (first token and padding)
            norms_len = norms.shape[0] if len(norms.shape) > 0 else 0
            for pos in range(1, seq_len):
                if pos >= norms_len:
                    continue
                token_id = input_ids[0, pos].item()
                if token_id not in sink_tokens and attention_mask[0, pos] == 1:
                    layer_norms[layer].append(norms[pos].item())
                    valid_count += 1
            
            if prompt_idx == 0 and valid_count == 0:
                print(f"Warning: First prompt collected 0 valid tokens for layer {layer}")
                print(f"  seq_len={seq_len}, norms_len={norms_len}")
                print(f"  hs shape before squeeze: {outputs.hidden_states[layer + 1].shape}")
                print(f"  hs shape after squeeze: {hs.shape}")
                print(f"  Sample tokens: {input_ids[0, :10].tolist()}")
                print(f"  Sample attention: {attention_mask[0, :10].tolist()}")
    
    return layer_norms


def compute_calibration_stats(
    layer_norms: Dict[int, List[float]],
    candidate_layers: List[int],
) -> Dict[str, any]:
    """Compute median norms and outlier statistics."""
    stats = {
        "layers": {},
        "metadata": {
            "n_samples_per_layer": {layer: len(layer_norms[layer]) for layer in candidate_layers},
        }
    }
    
    for layer in candidate_layers:
        norms = np.array(layer_norms[layer])
        if len(norms) == 0:
            raise ValueError(f"No norm samples were collected for layer {layer}")
        
        median_norm = np.median(norms)
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        # Outlier analysis (top 1% norms)
        p99 = np.percentile(norms, 99)
        outlier_ratio = (norms > p99).sum() / len(norms)
        
        stats["layers"][layer] = {
            "median_norm": float(median_norm),
            "mean_norm": float(mean_norm),
            "std_norm": float(std_norm),
            "p99_norm": float(p99),
            "outlier_ratio": float(outlier_ratio),
            "n_tokens": len(norms),
        }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Calibrate per-layer norms for steering envelope")
    parser.add_argument("--model", default="microsoft/phi-2", help="Model name or path")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rollout_files",
        nargs="+",
        help="Reuse question prompts from existing RFT shard JSONL files",
    )
    parser.add_argument("--output", default="norms_calibration.json", help="Output JSON path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--candidate_layers", nargs="+", type=int, default=[12, 15, 18, 21])
    parser.add_argument("--max_length", type=int, default=768, help="Max token length (should match max_tok_len)")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Loading model {args.model} on {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model_config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=model_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    # Identify sink tokens (common non-content tokens)
    sink_tokens = set()
    sink_tokens.add(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    sink_tokens.add(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
    sink_tokens.update(tokenizer.encode("\n", add_special_tokens=False))
    sink_tokens = list(sink_tokens)
    
    print(f"Sink tokens: {sink_tokens}")
    print(f"Candidate layers: {args.candidate_layers}")
    
    print(f"Loading {args.n_samples} prompts for norm calibration")
    questions = load_calibration_questions(
        args.n_samples,
        rollout_files=args.rollout_files,
        seed=args.seed,
    )

    # Build 0-shot prompts from training questions or the existing RFT shards.
    from src.data.prompts import build_prompt_unified
    prompts = [build_prompt_unified(question, n_shot=0, include_preface=False) for question in questions]
    
    print("Collecting layer norms...")
    start_time = time.time()
    layer_norms = collect_layer_norms(model, tokenizer, prompts, device, args.candidate_layers, sink_tokens, max_length=args.max_length)
    elapsed = time.time() - start_time
    print(f"Collected norms in {elapsed:.2f}s")
    
    print("Computing calibration statistics...")
    stats = compute_calibration_stats(layer_norms, args.candidate_layers)
    
    # Add metadata
    stats["metadata"]["model"] = args.model
    stats["metadata"]["n_prompts"] = len(prompts)
    stats["metadata"]["source"] = "rft_rollout_questions" if args.rollout_files else "gsm8k_train"
    stats["metadata"]["seed"] = args.seed
    stats["metadata"]["elapsed_seconds"] = elapsed
    stats["metadata"]["candidate_layers"] = args.candidate_layers
    
    # Save calibration
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nCalibration saved to {output_path}")
    print("\nPer-layer median norms (N_l):")
    for layer in args.candidate_layers:
        if layer in stats["layers"]:
            print(f"  Layer {layer}: {stats['layers'][layer]['median_norm']:.4f}")
    
    print("\nThese N_l values will be used as the per-layer cap basis for steering envelope.")


if __name__ == "__main__":
    main()
