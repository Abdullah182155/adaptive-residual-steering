"""Diagnostic script for ISSUE-9: Attention-sink token dominance in router inputs.

Run before changing anything. Check if:
1. Mean pairwise cosine similarity across prompt embeddings is > 0.95
2. Or median ||h_0|| / median ||h_t|| is >> 10.
If confirmed, drop the first token from router prompt pooling.
"""

import itertools
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.configs.config import RSCConfig
from src.data.dataset import gsm8k_to_cot


def run_diagnostic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running router input diagnostic on device: {device}")
    
    cfg = RSCConfig(n_samples=200)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    tok.pad_token = tok.pad_token or tok.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype).to(device).eval()
    
    _, ev = gsm8k_to_cot(tok, cfg)
    layer = RSCConfig().rsc_layers_for(len(model.model.layers))[0]
    grab = {}
    model.model.layers[layer].register_forward_hook(
        lambda m, a, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o).float())
    )
    
    pooled_all, pooled_nofirst, first_ratio = [], [], []
    n_screen = min(64, len(ev))
    for i in range(n_screen):
        row = ev[i]
        m = row["attention_mask"].bool() & row["labels"].eq(-100)
        ids = row["input_ids"].unsqueeze(0).to(device)
        attn = row["attention_mask"].unsqueeze(0).to(device)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=attn)
        h = grab["h"][0][m.to(device)]
        if len(h) == 0:
            continue
        pooled_all.append(h.mean(0))
        if len(h) > 1:
            pooled_nofirst.append(h[1:].mean(0))
            first_ratio.append((h[0].norm() / h[1:].norm(dim=-1).median()).item())
        else:
            pooled_nofirst.append(h.mean(0))
            first_ratio.append(1.0)
            
    def mean_pair_cos(v):
        if len(v) < 2:
            return 0.0
        v_norm = torch.nn.functional.normalize(torch.stack(v), dim=-1)
        c = v_norm @ v_norm.T
        return ((c.sum() - c.diag().sum()) / (c.numel() - len(v_norm))).item()
        
    cos_all = mean_pair_cos(pooled_all)
    cos_nofirst = mean_pair_cos(pooled_nofirst)
    median_ratio = sorted(first_ratio)[len(first_ratio) // 2] if first_ratio else 0.0
    
    print("mean pairwise cosine, all tokens   :", f"{cos_all:.4f}")
    print("mean pairwise cosine, drop 1st tok :", f"{cos_nofirst:.4f}")
    print("median ||h_0|| / median ||h_t||    :", f"{median_ratio:.2f}")
    
    if cos_all > 0.95 or median_ratio > 10.0:
        print("=> ISSUE-9 CONFIRMED: Outlier/first token dominates router input features.")
    else:
        print("=> ISSUE-9 NOT CONFIRMED: First token dominance is within normal range.")


if __name__ == "__main__":
    run_diagnostic()
