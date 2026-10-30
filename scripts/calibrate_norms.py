"""
Per-layer norm calibration for steering envelope (N_l).

Phase 0.6: Compute median non-sink ‖h‖ per candidate layer, outlier-dimension profile,
and sink-token norm ratio.

Usage:
    python scripts/calibrate_norms.py --n_samples 500 --output norms_calibration.json
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import numpy as np
from datasets import load_dataset

# Add repo root to path for imports
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))


def collect_layer_norms(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    candidate_layers: List[int],
    sink_tokens: List[int],
) -> Dict[int, List[float]]:
    """
    Collect hidden state norms for each candidate layer.
    
    Returns:
        Dict mapping layer -> list of norms (excluding sink tokens)
    """
    model.eval()
    layer_norms = {layer: [] for layer in candidate_layers}
    
    # Hook to capture hidden states
    hooks = {}
    hidden_states = {}
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            # output is tuple (hidden_state, present, ...)
            hidden_states[layer_idx] = output[0].detach()
        return hook
    
    # Register hooks for candidate layers
    for i, layer in enumerate(model.model.layers):
        if i in candidate_layers:
            hooks[i] = layer.register_forward_hook(make_hook(i))
    
    try:
        for prompt in tqdm(prompts, desc="Collecting norms"):
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Extract norms for each layer
            seq_len = input_ids.shape[1]
            for layer in candidate_layers:
                if layer not in hidden_states:
                    continue
                
                hs = hidden_states[layer][0]  # (1, seq_len, hidden_dim)
                if hs is None or len(hs.shape) < 2 or hs.shape[0] == 0:
                    continue
                norms = torch.norm(hs, dim=-1).cpu()  # (seq_len,)
                if not isinstance(norms, torch.Tensor) or len(norms.shape) == 0:
                    continue
                
                # Exclude sink tokens (first token and padding)
                norms_len = norms.shape[0] if len(norms.shape) > 0 else 0
                for pos in range(1, seq_len):
                    if pos >= norms_len:
                        continue
                    token_id = input_ids[0, pos].item()
                    if token_id not in sink_tokens and attention_mask[0, pos] == 1:
                        layer_norms[layer].append(norms[pos].item())
    
    finally:
        # Remove hooks
        for hook in hooks.values():
            hook.remove()
    
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
            continue
        
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
    parser.add_argument("--output", default="norms_calibration.json", help="Output JSON path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--candidate_layers", nargs="+", type=int, default=[13, 15, 18, 20, 22])
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Loading model {args.model} on {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).to(device)
    
    # Identify sink tokens (common non-content tokens)
    sink_tokens = set()
    sink_tokens.add(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    sink_tokens.add(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
    sink_tokens.update(tokenizer.encode("\n", add_special_tokens=False))
    sink_tokens = list(sink_tokens)
    
    print(f"Sink tokens: {sink_tokens}")
    print(f"Candidate layers: {args.candidate_layers}")
    
    # Load GSM8K for calibration prompts
    print(f"Loading {args.n_samples} GSM8K questions for calibration")
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=42)
    ds = ds.select(range(min(args.n_samples, len(ds))))
    
    # Build prompts (0-shot for calibration)
    from src.data.prompts import build_prompt_unified
    prompts = [build_prompt_unified(ex["question"], n_shot=0, include_preface=False) for ex in ds]
    
    print("Collecting layer norms...")
    start_time = time.time()
    layer_norms = collect_layer_norms(model, tokenizer, prompts, device, args.candidate_layers, sink_tokens)
    elapsed = time.time() - start_time
    print(f"Collected norms in {elapsed:.2f}s")
    
    print("Computing calibration statistics...")
    stats = compute_calibration_stats(layer_norms, args.candidate_layers)
    
    # Add metadata
    stats["metadata"]["model"] = args.model
    stats["metadata"]["n_prompts"] = len(prompts)
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
