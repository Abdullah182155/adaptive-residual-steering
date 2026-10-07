import gc
import os
import random
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

from src.configs.config import RSCConfig
from src.evaluation.verifier import epoch_end_accuracy
from src.routing.rloo import compute_router_rloo_loss
from src.training.losses import (
    LRReducer,
    CheckpointManager,
    GradientMonitor,
    compute_l1_gate,
    compute_antisat_gate,
    compute_gate_diversity_loss,
    compute_invariance_loss,
)
from src.training.contrastive import compute_contrastive_gate_loss

def evaluate_loss(model, data, cfg: RSCConfig, device: torch.device) -> float:
    """Compute per-token cross-entropy loss on eval set."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for start in range(0, len(data), cfg.batch_size):
        batch_idx = list(range(start, min(start + cfg.batch_size, len(data))))
        ids = torch.stack([data[i]["input_ids"] for i in batch_idx]).to(device)
        mask = torch.stack([data[i]["attention_mask"] for i in batch_idx]).to(device)
        lbls = torch.stack([data[i]["labels"] for i in batch_idx]).to(device)

        if hasattr(model, "_attn_mask_cache"):
            model._attn_mask_cache = mask

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(input_ids=ids, attention_mask=mask, labels=lbls)

        n_tokens = (lbls != -100).sum().item()
        if n_tokens > 0:
            total_nll += out.loss.item() * n_tokens
            total_tokens += n_tokens

    model.train()
    return total_nll / max(1, total_tokens)

def evaluate_and_maybe_checkpoint(
    model, tokenizer, eval_data, cfg: RSCConfig, device: torch.device, save_path: str,
    best_acc: float, best_val: float, patience_ctr: int, train_router: bool, label: str,
):
    if train_router:
        model.reset_routing_diagnostics()
    val_loss = evaluate_loss(model, eval_data, cfg, device)
    acc = epoch_end_accuracy(model, tokenizer, eval_data, device, n_samples=cfg.accuracy_eval_n, seed=cfg.seed)
    improved = acc > best_acc + 0.005
    best_state = None
    if improved:
        best_acc, best_val, patience_ctr = acc, val_loss, 0
        best_state = {
            k: v.clone() for k, v in model.state_dict().items()
            if "steer_net" in k or "gate" in k or "router" in k
        }
        torch.save(best_state, save_path)
    else:
        patience_ctr += 1
    return best_acc, best_val, best_state, patience_ctr, improved, val_loss, acc

def train_router_bootstrap(model, train_data, eval_data, cfg: RSCConfig, device: torch.device):
    """Phase 1.5: Router Bootstrap with frozen SteerNet & Gate."""
    for sn in model.steer_nets.values():
        for p in sn.parameters():
            p.requires_grad = False
    model.set_gate_freeze(True, 1.0)
    model.set_router_warmup(False)
    model.set_routing_override(None)

    router_params = list(model.router.parameters())
    for p in router_params:
        p.requires_grad = True

    optimizer = torch.optim.AdamW(router_params, lr=cfg.router_bootstrap_lr, weight_decay=cfg.weight_decay)
    model.reset_routing_diagnostics()

    pool_size = min(cfg.router_bootstrap_n_examples, len(train_data))
    rng = random.Random(cfg.seed)
    subset_idx = rng.sample(range(len(train_data)), pool_size)
    best_state, best_loss, patience_ctr = None, float("inf"), 0

    for epoch in range(cfg.router_bootstrap_epochs):
        random.shuffle(subset_idx)
        model.train()
        running, n_steps = 0.0, 0
        pbar = tqdm(subset_idx, desc=f"RouterBootstrap {epoch + 1}/{cfg.router_bootstrap_epochs}")
        for i in pbar:
            ids = train_data[i]["input_ids"].unsqueeze(0).to(device)
            mask = train_data[i]["attention_mask"].unsqueeze(0).to(device)
            lbls = train_data[i]["labels"].unsqueeze(0).to(device)
            model._attn_mask_cache = mask

            prompt_mask = mask.bool() & lbls.eq(-100)
            loss, rloo_diag = compute_router_rloo_loss(model, ids, mask, lbls, cfg, prompt_mask=prompt_mask)
            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router_params, cfg.clip_grad)
            optimizer.step()

            running += loss.item()
            n_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running / max(1, n_steps)
        if avg_loss < best_loss - 1e-4:
            best_loss, patience_ctr = avg_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items() if "router" in k}
        else:
            patience_ctr += 1
        if patience_ctr >= cfg.router_bootstrap_patience:
            break

    if best_state:
        model.load_state_dict(best_state, strict=False)
    model.set_routing_override(None)
    torch.save(
        {k: v.cpu() for k, v in model.state_dict().items() if "router" in k},
        cfg.router_bootstrap_save_path,
    )
    return best_state, best_loss
