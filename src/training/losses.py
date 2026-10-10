import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
import re
import numpy as np
from src.data.dataset import PromptTemplateBank
from src.evaluation.verifier import extract_final_answer

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
            std_b = vals.std(unbiased=False)
            span_b = vals.max() - vals.min()
            # Penalize both low variance and collapsed dynamic range (< 0.15 span)
            total = total + F.relu(floor - std_b) ** 2 + 0.5 * F.relu(0.15 - span_b) ** 2
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

def log_gpu_memory(label: str, cfg=None, reset_peak: bool = True):
    """Memory diagnostic reporting CUDA peak allocated/reserved."""
    if cfg is not None and not getattr(cfg, "log_gpu_memory", True):
        return
    if not torch.cuda.is_available():
        return
    alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
    print(f"    [mem] {label}: peak_allocated={alloc_mb:,.0f}MB  peak_reserved={reserved_mb:,.0f}MB")
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()

def _extract_final_number(text: str) -> Optional[float]:
    """Helper using verifier extract_final_answer with fallback."""
    num = extract_final_answer(text)
    if num is not None:
        return num
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            return None
    return None

def compute_pg_loss(
    model,
    tokenizer,
    input_ids,
    labels,
    device,
    cfg,
    baseline_ema,
    attention_masks=None,
    train_batch_ids=None,
):
    """Outcome-based policy gradient loss measuring answer correctness under full steering vs no steering."""
    N = min(cfg.pg_n_samples, input_ids.size(0))
    advantages = []

    model.eval()
    with torch.no_grad():
        for i in range(N):
            ids_i = input_ids[i : i + 1]
            lbl_i = labels[i : i + 1]
            pad_id = tokenizer.pad_token_id

            q_mask = (lbl_i[0] == -100) & (ids_i[0] != pad_id)
            if q_mask.sum() == 0:
                continue
            q_ids = ids_i[0][q_mask].unsqueeze(0)
            q_attn = torch.ones_like(q_ids)

            gold_mask = lbl_i[0] != -100
            if gold_mask.sum() == 0:
                continue
            gold_text = tokenizer.decode(ids_i[0][gold_mask], skip_special_tokens=True)
            gold_num = _extract_final_number(gold_text)
            if gold_num is None:
                continue

            # Gate ON (1.0)
            model.set_gate_freeze(True, value=1.0)
            gen_on = model.generate(
                input_ids=q_ids,
                attention_mask=q_attn,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_id,
                use_cache=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            text_on = tokenizer.decode(gen_on[0][q_ids.size(1) :], skip_special_tokens=True)
            pred_on = _extract_final_number(text_on)
            r_on = 1.0 if (pred_on is not None and abs(pred_on - gold_num) < 0.01) else 0.0

            # Gate OFF (0.0)
            model.set_gate_freeze(True, value=0.0)
            gen_off = model.generate(
                input_ids=q_ids,
                attention_mask=q_attn,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_id,
                use_cache=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            text_off = tokenizer.decode(gen_off[0][q_ids.size(1) :], skip_special_tokens=True)
            pred_off = _extract_final_number(text_off)
            r_off = 1.0 if (pred_off is not None and abs(pred_off - gold_num) < 0.01) else 0.0

            advantages.append(r_on - r_off)

    model.set_gate_freeze(False)
    model.train()
    log_gpu_memory("compute_pg_loss after generation rounds", cfg)

    if not advantages:
        return torch.tensor(0.0, device=device)

    mean_adv = float(np.mean(advantages))
    baseline_ema[0] = (
        cfg.pg_baseline_momentum * baseline_ema[0] + (1 - cfg.pg_baseline_momentum) * mean_adv
    )
    centred_adv = mean_adv - baseline_ema[0]

    if abs(centred_adv) < 1e-5:
        return torch.tensor(0.0, device=device)

    # Forward on same batch to get live alphas
    attn = (
        attention_masks
        if attention_masks is not None
        else (input_ids != tokenizer.pad_token_id).long()
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        _ = model(input_ids=input_ids, attention_mask=attn)

    pg_total = torch.tensor(0.0, device=device)
    n_terms = 0
    for gate in model.gates.values():
        alpha = gate._alpha_for_aux
        if alpha is None:
            continue
        a = alpha.float().clamp(1e-8, 1 - 1e-8).squeeze(-1)
        vm = gate._valid_mask
        if vm is not None and vm.shape == a.shape:
            answer_mask = labels[:, :a.size(1)].ne(-100) if labels is not None else None
            if answer_mask is not None:
                answer_mask = answer_mask[:a.size(0), :a.size(1)]
                vm = vm & answer_mask
            weight_sum = vm.float().sum().clamp(min=1)
            if centred_adv > 0:
                pg_layer = (-centred_adv) * (torch.log(a) * vm.float()).sum() / weight_sum
            else:
                pg_layer = centred_adv * (torch.log(1 - a) * vm.float()).sum() / weight_sum
        else:
            if centred_adv > 0:
                pg_layer = (-centred_adv) * torch.log(a).mean()
            else:
                pg_layer = centred_adv * torch.log(1 - a).mean()
        pg_total = pg_total + pg_layer
        n_terms += 1

    return pg_total / max(1, n_terms)

@torch.no_grad()
def _span_logprob(model, input_ids, attention_mask, labels, gate_value):
    """Mean log-prob of the labelled span under a fixed gate value via teacher-forced forward pass."""
    model.set_gate_freeze(True, value=gate_value)
    with torch.amp.autocast("cuda", enabled=input_ids.is_cuda):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :].float()
    tgt = labels[:, 1:]
    valid = tgt != -100
    logp = F.log_softmax(logits, dim=-1)
    tgt_safe = tgt.clamp(min=0)
    token_logp = torch.gather(logp, 2, tgt_safe.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * valid.float()
    denom = valid.float().sum(-1).clamp(min=1)
    return token_logp.sum(-1) / denom

def compute_pg_loss_lite(
    model,
    tokenizer,
    input_ids,
    labels,
    device,
    cfg,
    baseline_ema,
    attention_masks=None,
    train_batch_ids=None,
):
    """Teacher-forced counterfactual PG alternative (no autoregressive generation)."""
    attn = (
        attention_masks
        if attention_masks is not None
        else (input_ids != tokenizer.pad_token_id).long()
    )

    lp_on = _span_logprob(model, input_ids, attn, labels, 1.0)
    lp_off = _span_logprob(model, input_ids, attn, labels, 0.0)
    advantage = lp_on - lp_off

    model.train()
    model.set_gate_freeze(False)
    log_gpu_memory("compute_pg_loss_lite after span-logprob probes", cfg)

    mean_adv = advantage.mean().item()
    baseline_ema[0] = (
        cfg.pg_baseline_momentum * baseline_ema[0] + (1 - cfg.pg_baseline_momentum) * mean_adv
    )
    centred_adv = mean_adv - baseline_ema[0]

    if abs(centred_adv) < 1e-5:
        return torch.tensor(0.0, device=device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        _ = model(input_ids=input_ids, attention_mask=attn)

    pg_total = torch.tensor(0.0, device=device)
    n_terms = 0
    for gate in model.gates.values():
        alpha = gate._alpha_for_aux
        if alpha is None:
            continue
        a = alpha.float().clamp(1e-8, 1 - 1e-8).squeeze(-1)
        vm = gate._valid_mask
        if vm is not None and vm.shape == a.shape:
            weight_sum = vm.float().sum().clamp(min=1)
            if centred_adv > 0:
                pg_layer = (-centred_adv) * (torch.log(a) * vm.float()).sum() / weight_sum
            else:
                pg_layer = centred_adv * (torch.log(1 - a) * vm.float()).sum() / weight_sum
        else:
            if centred_adv > 0:
                pg_layer = (-centred_adv) * torch.log(a).mean()
            else:
                pg_layer = centred_adv * torch.log(1 - a).mean()
        pg_total = pg_total + pg_layer
        n_terms += 1

    return pg_total / max(1, n_terms)

def run_pg_loss(
    model,
    tokenizer,
    input_ids,
    labels,
    device,
    cfg,
    baseline_ema,
    attention_masks=None,
    train_batch_ids=None,
):
    """Dispatches to generation-based or teacher-forced counterfactual PG loss."""
    if getattr(cfg, "pg_mode", "generation") == "teacher_forced":
        return compute_pg_loss_lite(
            model,
            tokenizer,
            input_ids,
            labels,
            device,
            cfg,
            baseline_ema,
            attention_masks,
            train_batch_ids,
        )
    return compute_pg_loss(
        model,
        tokenizer,
        input_ids,
        labels,
        device,
        cfg,
        baseline_ema,
        attention_masks,
        train_batch_ids,
    )
