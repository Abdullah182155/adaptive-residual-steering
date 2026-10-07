import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from src.data.dataset import PromptTemplateBank

class LRReducer:
    def __init__(
        self,
        optimizer,
        total_steps,
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        delta=1e-4,
        warmup_steps=0,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.delta = delta
        self.best_loss = float("inf")
        self.wait = 0
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.original_base_lrs = list(self.base_lrs)
        self.n_reductions = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.n_reductions += 1
                self.base_lrs = [
                    max(lr * (self.factor**self.n_reductions), self.min_lr) for lr in self.original_base_lrs
                ]
                self.current_step = self.warmup_steps
                self.wait = 0
                print(
                    f"  ⚠️ LR reduced (×{self.factor}^{self.n_reductions}): "
                    f"{[f'{lr:.2e}' for lr in self.base_lrs]}"
                )

    def scheduler_step(self):
        self.current_step += 1
        step = max(0, self.current_step - self.warmup_steps)
        remaining = max(1, self.total_steps - self.warmup_steps)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * step / remaining))
        for i, pg in enumerate(self.optimizer.param_groups):
            pg["lr"] = max(self.min_lr, self.base_lrs[i] * cosine_factor)

class CheckpointManager:
    def __init__(self, checkpoint_dir, keep_last=3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last = keep_last
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_loss = float("inf")
        self.best_path = None
        self.recent = []

    def save(self, model, optimizer, epoch, val_loss, phase="p1"):
        state = {
            "steer_state": {
                k: v.cpu() for k, v in model.state_dict().items() if "steer_net" in k or "gate" in k or "router" in k
            },
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        }
        path = os.path.join(self.checkpoint_dir, f"{phase}_epoch{epoch}.pt")
        torch.save(state, path)
        self.recent.append(path)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_path = os.path.join(self.checkpoint_dir, f"{phase}_best.pt")
            torch.save(state, self.best_path)

        while len(self.recent) > self.keep_last:
            old = self.recent.pop(0)
            if old != self.best_path and os.path.exists(old):
                os.remove(old)

class GradientMonitor:
    def __init__(self, warn_threshold=2.0):
        self.warn_threshold = warn_threshold
        self.norms = []
        self.explosions = 0
        self.vanishes = 0

    def check(self, params, step):
        total_norm = 0.0
        has_nan = False
        for p in params:
            if p.grad is not None:
                g = p.grad.data
                if not torch.isfinite(g).all():
                    has_nan = True
                    continue
                total_norm += g.norm(2).item() ** 2
        total_norm = total_norm**0.5
        if has_nan:
            return
        self.norms.append(total_norm)
        if total_norm > self.warn_threshold:
            self.explosions += 1
            if self.explosions <= 3:
                print(f"    ⚠️ Grad explosion at step {step}: {total_norm:.4f}")
        elif total_norm < 1e-6:
            self.vanishes += 1

    def summary(self):
        if not self.norms:
            return
        print(f"  Gradient summary: mean={np.mean(self.norms):.4f}, explosions={self.explosions}")

def compute_l1_gate(model) -> torch.Tensor:
    """L1 sparsity on gate outputs."""
    device = next(iter(p for g in model.gates.values() for p in g.parameters()), torch.tensor(0.0)).device
    total = torch.tensor(0.0, device=device)
    count = 0
    for gate in model.gates.values():
        alpha = gate._alpha_for_aux
        if alpha is None:
            continue
        a = alpha.float().squeeze(-1)
        vm = gate._valid_mask
        if vm is not None and vm.shape == a.shape:
            denom = vm.sum().clamp(min=1)
            total = total + (a * vm.float()).sum() / denom
        else:
            total = total + a.mean()
        count += 1
    return total / max(1, count)

def compute_gate_diversity_loss(model, floor: float = 0.05) -> torch.Tensor:
    """Penalize low per-sequence gate variance, excluding single max token."""
    device = next(iter(p for g in model.gates.values() for p in g.parameters()), torch.tensor(0.0)).device
    total = torch.tensor(0.0, device=device)
    count = 0
    for gate in model.gates.values():
        alpha = gate._alpha_for_aux
        if alpha is None:
            continue
        a = alpha.float().squeeze(-1)
        vm = gate._valid_mask
        for b in range(a.size(0)):
            a_b = a[b]
            valid_b = vm[b] if (vm is not None and vm.shape == a.shape) else torch.ones_like(a_b, dtype=torch.bool)
            if valid_b.sum() < 3:
                continue
            vals = a_b[valid_b]
            max_pos = vals.argmax()
            vals_excl_max = torch.cat([vals[:max_pos], vals[max_pos + 1:]])
            std_b = vals_excl_max.std(unbiased=False)
            total = total + F.relu(floor - std_b) ** 2
            count += 1
    return total / max(1, count)

def compute_antisat_gate(model, threshold: float = 0.92) -> torch.Tensor:
    """Penalise gate values above threshold to prevent saturation."""
    device = next(iter(p for g in model.gates.values() for p in g.parameters()), torch.tensor(0.0)).device
    total = torch.tensor(0.0, device=device)
    count = 0
    layer_keys = sorted(model.gates.keys(), key=lambda k: int(k))
    n_layers = max(1, len(layer_keys))
    for rank_idx, key in enumerate(layer_keys):
        gate = model.gates[key]
        alpha = gate._alpha_for_aux
        if alpha is None:
            continue
        a = alpha.float().squeeze(-1)
        layer_threshold = threshold - 0.30 * (1.0 - rank_idx / (n_layers - 1 + 1e-8))
        layer_threshold = max(0.50, layer_threshold)
        over = F.relu(a - layer_threshold) ** 2
        vm = gate._valid_mask
        if vm is not None and vm.shape == over.shape:
            denom = vm.sum().clamp(min=1)
            total = total + (over * vm.float()).sum() / denom
        else:
            total = total + over.mean()
        count += 1
    return total / max(1, count)

def compute_invariance_loss(model, tokenizer, questions, steps_lists, final_answers, cfg, device, seed_offset=0):
    """Paraphrase-invariance loss pulling steered representations together across two template views."""
    last_layer_idx = max(int(k) for k in model.steer_nets.keys())
    pooled = {}

    def grab_hook(module, args, output):
        pooled["cur"] = output[0]

    h = model.base_model.model.layers[last_layer_idx].register_forward_hook(grab_hook)
    try:
        def encode(texts):
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_tok_len).to(device)
            model._attn_mask_cache = enc["attention_mask"]
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            hs = pooled["cur"].float()
            mask = enc["attention_mask"].unsqueeze(-1).float()
            mean_pool = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
            return F.normalize(mean_pool, dim=-1)

        view1, view2 = [], []
        for j, (q, steps, ans) in enumerate(zip(questions, steps_lists, final_answers)):
            v1, v2 = PromptTemplateBank.two_views(q, steps, ans, seed=cfg.seed + seed_offset + j)
            view1.append(v1)
            view2.append(v2)

        z1 = encode(view1)
        z2 = encode(view2)
        return (1.0 - (z1 * z2).sum(-1)).mean()
    finally:
        h.remove()
