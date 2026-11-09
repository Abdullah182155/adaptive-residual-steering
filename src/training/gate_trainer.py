"""Phase 2B: Gate training via RLOO + teacher-forced NLL.

The gate (RSCUsefulnessGate) learns per-token steering weights that estimate
whether applying the SteerNet correction at a given position helps or hurts.

Training approach:
- SteerNet weights: frozen (from Phase 1)
- Router weights:   frozen (from Phase 2) — provides routing context
- Gate weights:     trained via RLOO policy gradient + task NLL

Reward signal: difference in answer-weighted NLL between
  gate=learned (alpha from gate)  vs  gate=fixed (alpha=0.5 baseline)
Using teacher-forced NLL (no generation needed) keeps training fast.

Checkpoint outputs:
  cfg.gate_save_path          — gate weights only
  cfg.steernet_gated_save_path — steernet fine-tuned alongside gate (if enabled)
"""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.configs.config import RSCConfig
from src.gating.gate import RSCUsefulnessGate
from src.routing.rloo import per_example_answer_nll_weighted
from src.training.losses import T_FORMAT, T_OPERAND, T_OPERATOR, T_PROSE, T_RESULT, T_FINAL
from src.training.trainer import (
    _epoch_batches,
    _prepare,
    save_steer_weights,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _nll_with_gate_alpha(
    model,
    ids: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    gate_value: Optional[float],       # None = use learned gate
    answer_weight: float = 2.0,
) -> torch.Tensor:
    """Teacher-forced NLL with gate fixed at `gate_value` or learned.

    When gate_value is not None the gate is frozen at that constant α so we
    can compute a counterfactual baseline without extra gradient tracking.
    """
    for gate in model.gates.values():
        gate._freeze_gate = gate_value is not None
        gate._freeze_gate_value = gate_value if gate_value is not None else 1.0

    try:
        with torch.amp.autocast("cuda", enabled=ids.is_cuda):
            logits = model(input_ids=ids, attention_mask=mask).logits
    finally:
        # Always restore gate to learned mode, even on OOM or NaN
        for gate in model.gates.values():
            gate._freeze_gate = False

    return per_example_answer_nll_weighted(logits, labels, answer_weight=answer_weight)


def _contrastive_gate_loss(
    gates: nn.ModuleDict,
    token_types: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Push gate alpha UP on math tokens and DOWN on format tokens.

    Targets (from the old RSC notebook that achieved +18pp):
      math/operator/result/final → 0.85
      format                     → 0.15
      prose/discourse            → 0.65
    """
    valid = labels[:, 1:].ne(-100)
    if not valid.any():
        return labels.new_zeros((), dtype=torch.float32)

    tt = token_types[:, 1:]
    loss = labels.new_zeros((), dtype=torch.float32)
    n_gates = 0
    for gate in gates.values():
        alpha = gate._alpha_for_aux  # (B, T, 1) — grad-carrying
        if alpha is None:
            continue
        T_use = min(alpha.shape[1], tt.shape[1])
        a = alpha[:, :T_use, 0]
        t = tt[:, :T_use]
        v = valid[:, :T_use]

        math_mask = v & ((t == T_OPERAND) | (t == T_OPERATOR) | (t == T_RESULT) | (t == T_FINAL))
        fmt_mask = v & (t == T_FORMAT)
        prose_mask = v & (t == T_PROSE)

        if math_mask.any():
            loss = loss + (a[math_mask] - 0.85).pow(2).mean()
        if fmt_mask.any():
            loss = loss + (a[fmt_mask] - 0.15).pow(2).mean()
        if prose_mask.any():
            loss = loss + (a[prose_mask] - 0.65).pow(2).mean()
        n_gates += 1

    return loss / max(n_gates, 1)


# ── Main training function ──────────────────────────────────────────────────

def train_phase2b_gate(
    model,
    train_data,
    eval_data,
    cfg: RSCConfig,
    device: torch.device,
    tokenizer=None,
) -> Dict[str, Any]:
    """Train RSCUsefulnessGate while SteerNet (and optionally Router) are frozen.

    Returns history dict with loss / val_nll curves.
    """
    if not getattr(cfg, "use_gate", False):
        print("[gate] use_gate=False — skipping Phase 2B gate training")
        return {}

    gate_epochs    = getattr(cfg, "gate_epochs", 2)
    gate_lr        = getattr(cfg, "gate_lr", 1e-4)
    gate_l1_weight = getattr(cfg, "gate_l1_weight", 0.01)
    gate_rloo_w    = getattr(cfg, "gate_rloo_weight", 1.0)
    gate_contrastive_w = getattr(cfg, "gate_contrastive_weight", 0.5)
    finetune_sn    = getattr(cfg, "gate_finetune_steernet", True)
    sn_lr          = getattr(cfg, "gate_steernet_lr", 1e-5)
    answer_w       = getattr(cfg, "router_answer_nll_weight", 2.0)
    baseline_alpha = 0.5   # fixed alpha for RLOO baseline comparison

    print(f"\n{'='*70}")
    print(f"Phase 2B — Gate Training  ({gate_epochs} epochs, lr={gate_lr:.1e})")
    print(f"  answer_weight={answer_w}  l1={gate_l1_weight}  rloo_w={gate_rloo_w}  contrastive_w={gate_contrastive_w}")
    print(f"  finetune_steernet={'YES lr=' + str(sn_lr) if finetune_sn else 'NO'}")
    print(f"{'='*70}")

    # ── Create gates if they don't exist yet on the model ─────────────────
    if not hasattr(model, "gates") or not model.gates:
        hidden_size = model.base_model.config.hidden_size
        model.gates = nn.ModuleDict({
            str(layer): RSCUsefulnessGate(hidden_size, cfg).to(device)
            for layer in model.target_layers
        })
        print(f"[gate] Created {len(model.gates)} gate objects for layers {model.target_layers}")

    # Install gate provider: routes (position, hidden, delta) → alpha per layer
    target_layers = model.target_layers
    gates_ref = model.gates
    def _gate_provider(position: int, hidden: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        layer_key = str(target_layers[position])
        return gates_ref[layer_key](hidden, delta)
    model.set_gate_provider(_gate_provider)

    # ── Freeze backbone and SteerNets (unless fine-tuning SteerNets) ────────
    for p in model.base_model.parameters():
        p.requires_grad = False

    steer_params = list(model.steer_nets.parameters())
    for p in steer_params:
        p.requires_grad = finetune_sn

    gate_params = list(model.gates.parameters())
    for p in gate_params:
        p.requires_grad = True

    # ── Optimizer ──────────────────────────────────────────────────────────
    param_groups = [{"params": gate_params, "lr": gate_lr, "weight_decay": 0.01}]
    if finetune_sn and steer_params:
        param_groups.append({"params": steer_params, "lr": sn_lr, "weight_decay": 0.01})
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    rows = train_data if isinstance(train_data, list) else [train_data[i] for i in range(len(train_data))]
    lengths = [int(r["attention_mask"].sum()) for r in rows]
    n_batches = math.ceil(len(rows) / cfg.batch_size)
    total_steps = math.ceil(n_batches / cfg.grad_accum) * gate_epochs
    warm_steps = max(1, int(0.08 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warm_steps:
            return step / max(warm_steps, 1)
        t = (step - warm_steps) / max(total_steps - warm_steps, 1)
        return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_nll = float("inf")
    best_gate_state = None
    best_sn_state = None
    history: Dict[str, Any] = {"gate_train_loss": [], "gate_val_nll": [], "epochs": []}

    model.train()
    model.base_model.eval()

    for epoch in range(gate_epochs):
        batches = _epoch_batches(lengths, cfg.batch_size, epoch, cfg.seed)
        window_steps = list(range(0, len(batches), cfg.grad_accum))
        n_windows = len(window_steps)
        print_every = max(1, n_windows // 4)

        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        step_count = 0

        for wi, ws in enumerate(window_steps):
            window_batches = batches[ws : ws + cfg.grad_accum]
            window_loss_sum = torch.tensor(0.0, device=device)

            for bi in window_batches:
                # _epoch_batches returns List[List[int]] — each bi is a list of row indices
                batch_rows = [rows[i] for i in bi] if isinstance(bi, list) else rows[bi : bi + cfg.batch_size]
                if not batch_rows:
                    continue
                batch, extras = _prepare(batch_rows, device)
                ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
                token_types = extras.get("token_types")
                model.set_prompt_shield_mask(extras.get("prompt_shield_mask"))
                if ids.numel() == 0:
                    continue

                # ── Baseline NLL (fixed alpha=0.5, no gradient) ────────────
                with torch.no_grad():
                    nll_baseline = _nll_with_gate_alpha(
                        model, ids, mask, labels,
                        gate_value=baseline_alpha, answer_weight=answer_w,
                    )

                # ── Learned NLL (gate active, gradient flows through gate) ──
                for gate in model.gates.values():
                    gate._freeze_gate = False
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits_learned = model(input_ids=ids, attention_mask=mask).logits
                nll_learned = per_example_answer_nll_weighted(
                    logits_learned, labels, answer_weight=answer_w,
                )

                # ── RLOO-style advantage ───────────────────────────────────
                advantage = (nll_baseline - nll_learned).detach()   # positive = gate helps
                mean_alpha_log = torch.stack([
                    gate._alpha_for_aux.clamp(1e-6, 1 - 1e-6).log().mean()
                    for gate in model.gates.values()
                    if gate._alpha_for_aux is not None
                ]).mean()
                rloo_loss = -gate_rloo_w * (advantage.mean() * mean_alpha_log)

                # ── Task NLL (direct supervision) ─────────────────────────
                task_loss = nll_learned.mean()

                # ── L1 sparsity on gate activations ───────────────────────
                l1_loss = torch.stack([
                    gate.last_alpha.float().mean()
                    for gate in model.gates.values()
                    if gate.last_alpha is not None
                ]).mean()

                # ── Contrastive gate loss (math↑ format↓) ────────────────
                cg_loss = ids.new_zeros((), dtype=torch.float32)
                if token_types is not None and gate_contrastive_w > 0:
                    cg_loss = _contrastive_gate_loss(model.gates, token_types, labels)

                step_loss = task_loss + rloo_loss + gate_l1_weight * l1_loss + gate_contrastive_w * cg_loss
                step_loss = step_loss / len(window_batches)

                if torch.isfinite(step_loss):
                    scaler.scale(step_loss).backward()
                    window_loss_sum = window_loss_sum + step_loss.detach()

            # ── Optimizer step ─────────────────────────────────────────────
            scaler.unscale_(optimizer)
            all_params = gate_params + (steer_params if finetune_sn else [])
            for p in all_params:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    p.grad.zero_()
            torch.nn.utils.clip_grad_norm_(all_params, cfg.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(window_loss_sum)
            step_count += 1
            if (wi + 1) % print_every == 0 or wi == n_windows - 1:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  S2B epoch {epoch+1}/{gate_epochs} [{wi+1}/{n_windows}] "
                      f"loss={float(window_loss_sum):.3f} lr={lr_now:.1e}")

        # ── Validation NLL ─────────────────────────────────────────────────
        model.eval()
        val_nlls: List[float] = []
        eval_rows = eval_data if isinstance(eval_data, list) else [eval_data[i] for i in range(len(eval_data))]
        for start in range(0, len(eval_rows), cfg.batch_size):
            batch_rows = eval_rows[start : start + cfg.batch_size]
            if not batch_rows:
                continue
            batch, _ = _prepare(batch_rows, device)
            ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
            with torch.no_grad():
                for gate in model.gates.values():
                    gate._freeze_gate = False
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(input_ids=ids, attention_mask=mask).logits
                val_nlls.append(float(per_example_answer_nll_weighted(logits, labels, answer_weight=answer_w).mean()))
        val_nll = sum(val_nlls) / max(len(val_nlls), 1)
        avg_train = epoch_loss / max(step_count, 1)
        print(f"S2B epoch {epoch+1}/{gate_epochs}: train_loss={avg_train:.4f}  val_nll={val_nll:.4f}")

        # Checkpoint selection by val_nll
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_gate_state = {k: v.cpu().clone() for k, v in model.gates.state_dict().items()}
            if finetune_sn:
                best_sn_state = {k: v.cpu().clone() for k, v in model.steer_nets.state_dict().items()}

        history["gate_train_loss"].append(avg_train)
        history["gate_val_nll"].append(val_nll)
        history["epochs"].append({"epoch": epoch + 1, "val_nll": val_nll, "train_loss": avg_train})

        model.train()
        model.base_model.eval()

        # Free cache between epochs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Restore best state and save ────────────────────────────────────────
    if best_gate_state is not None:
        model.gates.load_state_dict(best_gate_state)
    if best_sn_state is not None:
        model.steer_nets.load_state_dict(best_sn_state)

    # Save gate
    gate_path = Path(getattr(cfg, "gate_save_path", "checkpoints/gate.pt"))
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"gates": {k: v.cpu() for k, v in model.gates.state_dict().items()},
         "fixed_layers": cfg.fixed_layers,
         "gate_epochs": gate_epochs,
         "best_val_nll": best_val_nll},
        gate_path,
    )
    print(f"[gate] Saved gate checkpoint: {gate_path}")

    # Save gated SteerNet (if fine-tuned)
    if finetune_sn:
        sn_gated_path = Path(getattr(cfg, "steernet_gated_save_path", "checkpoints/steernet_gated.pt"))
        save_steer_weights(model, str(sn_gated_path), cfg)
        print(f"[gate] Saved gated SteerNet checkpoint: {sn_gated_path}")

    history["best_val_nll"] = best_val_nll
    return history


def load_gate_weights(model, path: str, device: torch.device) -> None:
    """Load gate checkpoint into model.gates (non-strict — ignores missing keys)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("gates", ckpt)
    model.gates.load_state_dict(state, strict=False)
    print(f"[gate] Loaded gate checkpoint from {path}")
