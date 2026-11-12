"""Phase 2B: Gate training — V19 overhaul.

V18 changes (architecture):
- Task NLL removed from gate loss (overwhelmed contrastive)
- Hinge contrastive replaces MSE
- Anti-saturation loss
- RLOO baseline=0.0, answer_weight=1.0
- Per-layer LR scaling

V19 changes (methods):
- Generation-based accuracy reward every N steps (teacher-forced NLL ≠ accuracy)
- Checkpoint selection by NLL improvement (val_nll_base - val_nll_gate), not absolute NLL
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
    _generate_onpolicy_batch,
    _prepare,
    save_steer_weights,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _nll_with_gate_alpha(
    model,
    ids: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    gate_value: Optional[float],
    answer_weight: float = 1.0,
) -> torch.Tensor:
    """Teacher-forced NLL with gate fixed at `gate_value` (None = learned)."""
    for gate in model.gates.values():
        gate._freeze_gate = gate_value is not None
        gate._freeze_gate_value = gate_value if gate_value is not None else 1.0
    try:
        with torch.amp.autocast("cuda", enabled=ids.is_cuda):
            logits = model(input_ids=ids, attention_mask=mask).logits
    finally:
        for gate in model.gates.values():
            gate._freeze_gate = False
    return per_example_answer_nll_weighted(logits, labels, answer_weight=answer_weight)


@torch.no_grad()
def _grpo_accuracy_reward(
    model,
    tokenizer,
    sample_rows: List[dict],
    device: torch.device,
    max_new_tokens: int = 150,
) -> torch.Tensor:
    """GRPO-style gate reward: generate k examples, normalize advantages within group.

    REINFORCE (V19): reward ∈ {-1, 0, +1} with 2 samples → very high variance.
    GRPO (V20): 4 samples, normalize within group → continuous advantages, 4-5× lower variance.
    Gate can converge in 2-3 epochs instead of 5.

    advantages[i] = (reward[i] - mean(rewards)) / (std(rewards) + ε)
    """
    from src.data.prompts import build_prompt_unified
    from src.evaluation.answers import score_tiers

    questions = [r.get("question", "") for r in sample_rows]
    golds = [r.get("final_num", "") for r in sample_rows]

    valid = [(i, q, g) for i, (q, g) in enumerate(zip(questions, golds)) if q and g]
    if not valid:
        return torch.zeros(1, device=device)

    _, valid_questions, valid_golds = zip(*valid)
    prompts = [build_prompt_unified(q, n_shot=0) for q in valid_questions]

    model.eval()
    model.set_routing_override("only", model.target_layers)

    # Generate with gate OFF (alpha=0 → no steering)
    for gate in model.gates.values():
        gate._freeze_gate = True
        gate._freeze_gate_value = 0.0
    completions_off = _generate_onpolicy_batch(
        model, tokenizer, list(prompts), device,
        max_new_tokens=max_new_tokens, num_return_sequences=1, temperature=0.0,
    )
    for gate in model.gates.values():
        gate._freeze_gate = False
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Generate with gate ON (learned alpha)
    completions_on = _generate_onpolicy_batch(
        model, tokenizer, list(prompts), device,
        max_new_tokens=max_new_tokens, num_return_sequences=1, temperature=0.0,
    )

    # Score both and compute raw rewards
    raw_rewards = []
    for text_off, text_on, gold in zip(completions_off, completions_on, valid_golds):
        try:
            gold_val = float(str(gold).replace(",", "").replace("$", "").strip())
        except (ValueError, TypeError):
            raw_rewards.append(0.0)
            continue
        correct_off = float(score_tiers(text_off, gold_val).get("lenient", False))
        correct_on = float(score_tiers(text_on, gold_val).get("lenient", False))
        raw_rewards.append(correct_on - correct_off)

    model.train()
    model.base_model.eval()

    rewards = torch.tensor(raw_rewards, dtype=torch.float32, device=device)

    # GRPO normalization: continuous advantages within the group
    if rewards.numel() > 1 and rewards.std() > 0.01:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    else:
        advantages = rewards - rewards.mean()

    return advantages


def _entropy_reg(gates: nn.ModuleDict) -> torch.Tensor:
    """Binary entropy regularization: minimize H(alpha) to encourage crisp 0/1 decisions.

    H(alpha) = -alpha*log(alpha) - (1-alpha)*log(1-alpha)
    Minimizing H pushes alpha toward 0 or 1 (selective, not stuck at 0.5).
    Works WITH contrastive loss (math→1, format→0) unlike mean-alpha L1 which fights it.
    """
    device = next(iter(gates.parameters())).device
    total = torch.tensor(0.0, device=device)
    count = 0
    for gate in gates.values():
        alpha = gate._alpha_for_aux
        if alpha is None:
            continue
        a = alpha.float().squeeze(-1).clamp(1e-7, 1.0 - 1e-7)
        h = -(a * a.log() + (1.0 - a) * (1.0 - a).log())
        total = total + h.mean()
        count += 1
    return total / max(count, 1)


def _contrastive_gate_loss(
    gates: nn.ModuleDict,
    token_types: torch.Tensor,
    labels: torch.Tensor,
    mode: str = "hinge",
) -> torch.Tensor:
    """Push gate alpha UP on math tokens and DOWN on format tokens.

    mode="hinge": ReLU one-sided — zero gradient once target is met.
    mode="mse": symmetric MSE (old behavior).
    """
    valid = labels[:, 1:].ne(-100)
    if not valid.any():
        return labels.new_zeros((), dtype=torch.float32)

    tt = token_types[:, 1:]
    loss = labels.new_zeros((), dtype=torch.float32)
    n_gates = 0
    for gate in gates.values():
        alpha = gate._alpha_for_aux
        if alpha is None:
            continue
        T_use = min(alpha.shape[1], tt.shape[1])
        a = alpha[:, :T_use, 0]
        t = tt[:, :T_use]
        v = valid[:, :T_use]

        math_mask = v & ((t == T_OPERAND) | (t == T_OPERATOR) | (t == T_RESULT) | (t == T_FINAL))
        fmt_mask = v & (t == T_FORMAT)
        prose_mask = v & (t == T_PROSE)

        if mode == "hinge":
            if math_mask.any():
                loss = loss + F.relu(0.85 - a[math_mask]).mean()
            if fmt_mask.any():
                loss = loss + F.relu(a[fmt_mask] - 0.15).mean()
            if prose_mask.any():
                loss = loss + F.relu(0.65 - a[prose_mask]).mean()
        else:
            if math_mask.any():
                loss = loss + (a[math_mask] - 0.85).pow(2).mean()
            if fmt_mask.any():
                loss = loss + (a[fmt_mask] - 0.15).pow(2).mean()
            if prose_mask.any():
                loss = loss + (a[prose_mask] - 0.65).pow(2).mean()
        n_gates += 1

    return loss / max(n_gates, 1)


def _antisat_loss(
    gates: nn.ModuleDict,
    threshold: float = 0.85,
) -> torch.Tensor:
    """Penalize gate alpha above threshold. Earlier layers get tighter ceiling."""
    device = next(iter(gates.parameters())).device
    total = torch.tensor(0.0, device=device)
    layer_keys = sorted(gates.keys(), key=lambda k: int(k))
    n_layers = max(1, len(layer_keys))
    count = 0
    for rank_idx, key in enumerate(layer_keys):
        gate = gates[key]
        alpha = gate._alpha_for_aux
        if alpha is None:
            continue
        a = alpha.float().squeeze(-1)
        layer_thresh = threshold - 0.15 * (1.0 - rank_idx / (n_layers - 1 + 1e-8))
        layer_thresh = max(0.50, layer_thresh)
        over = F.relu(a - layer_thresh).pow(2)
        vm = gate._valid_mask
        if vm is not None and vm.shape == over.shape:
            denom = vm.sum().clamp(min=1)
            total = total + (over * vm.float()).sum() / denom
        else:
            total = total + over.mean()
        count += 1
    return total / max(count, 1)


# ── Main training function ──────────────────────────────────────────────────

def train_phase2b_gate(
    model,
    train_data,
    eval_data,
    cfg: RSCConfig,
    device: torch.device,
    tokenizer=None,
) -> Dict[str, Any]:
    """Train RSCUsefulnessGate with SteerNet jointly fine-tuned.

    V19 loss: RLOO + gen-accuracy reward + hinge contrastive + L1 + anti-saturation.
    Checkpoint selected by max NLL improvement (val_nll_base - val_nll_gate).
    """
    if not getattr(cfg, "use_gate", False):
        print("[gate] use_gate=False — skipping Phase 2B gate training")
        return {}

    gate_epochs        = getattr(cfg, "gate_epochs", 5)
    gate_lr            = getattr(cfg, "gate_lr", 1e-4)
    gate_l1_weight     = getattr(cfg, "gate_l1_weight", 0.0)      # V20: replaced by entropy
    gate_entropy_w     = getattr(cfg, "gate_entropy_weight", 0.1)
    gate_rloo_w        = getattr(cfg, "gate_rloo_weight", 1.0)
    gate_contrastive_w = getattr(cfg, "gate_contrastive_weight", 1.0)
    contrastive_mode   = getattr(cfg, "gate_contrastive_mode", "hinge")
    use_task_nll       = getattr(cfg, "gate_use_task_nll", False)
    antisat_w          = getattr(cfg, "gate_antisat_weight", 0.25)
    antisat_thresh     = getattr(cfg, "gate_antisat_threshold", 0.85)
    finetune_sn        = getattr(cfg, "gate_finetune_steernet", True)
    sn_lr              = getattr(cfg, "gate_steernet_lr", 1e-5)
    baseline_alpha     = getattr(cfg, "gate_rloo_baseline", 0.0)
    gen_reward_every   = getattr(cfg, "gate_gen_reward_every", 4)
    gen_subset         = getattr(cfg, "gate_gen_subset", 2)
    gen_weight         = getattr(cfg, "gate_gen_weight", 2.0)
    gen_max_tokens     = getattr(cfg, "gate_gen_max_tokens", 150)

    use_gen_reward = gen_reward_every > 0 and tokenizer is not None and gen_subset > 0

    print(f"\n{'='*70}")
    print(f"Phase 2B — Gate Training V19  ({gate_epochs} epochs, lr={gate_lr:.1e})")
    print(f"  contrastive={contrastive_mode} w={gate_contrastive_w}  entropy_w={gate_entropy_w}")
    print(f"  antisat w={antisat_w} thresh={antisat_thresh}  rloo baseline={baseline_alpha}")
    print(f"  gen_reward={'YES every=' + str(gen_reward_every) + ' subset=' + str(gen_subset) + ' w=' + str(gen_weight) if use_gen_reward else 'NO (tokenizer not passed)'}")
    print(f"  checkpoint_by=nll_improvement  finetune_sn={'YES lr=' + str(sn_lr) if finetune_sn else 'NO'}")
    print(f"{'='*70}")

    # ── Create gates if needed ─────────────────────────────────────────────
    if not hasattr(model, "gates") or not model.gates:
        hidden_size = model.base_model.config.hidden_size
        model.gates = nn.ModuleDict({
            str(layer): RSCUsefulnessGate(hidden_size, cfg).to(device)
            for layer in model.target_layers
        })
        print(f"[gate] Created {len(model.gates)} gate objects for layers {model.target_layers}")

    target_layers = model.target_layers
    gates_ref = model.gates
    def _gate_provider(position: int, hidden: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return gates_ref[str(target_layers[position])](hidden, delta)
    model.set_gate_provider(_gate_provider)

    # ── Freeze backbone ───────────────────────────────────────────────────
    for p in model.base_model.parameters():
        p.requires_grad = False

    steer_params = list(model.steer_nets.parameters())
    for p in steer_params:
        p.requires_grad = finetune_sn

    gate_params = list(model.gates.parameters())
    for p in gate_params:
        p.requires_grad = True

    # ── Optimizer: per-layer gate LR ─────────────────────────────────────
    layer_keys_sorted = sorted(model.gates.keys(), key=lambda k: int(k))
    n_gate_layers = len(layer_keys_sorted)
    param_groups = []
    for rank_idx, key in enumerate(layer_keys_sorted):
        g_params = list(model.gates[key].parameters())
        lr_scale = 1.0 + 0.5 * (1.0 - rank_idx / (n_gate_layers - 1 + 1e-8))
        param_groups.append({"params": g_params, "lr": gate_lr * lr_scale, "weight_decay": 0.01})
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

    best_val_improvement = -float("inf")  # nll_base - nll_gate; higher = gate helps more
    best_gate_state = None
    best_sn_state = None
    history: Dict[str, Any] = {
        "gate_train_loss": [], "gate_val_nll": [], "gate_val_improvement": [], "epochs": []
    }

    model.train()
    model.base_model.eval()

    global_opt_step = 0

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
                batch_rows = [rows[i] for i in bi] if isinstance(bi, list) else rows[bi : bi + cfg.batch_size]
                if not batch_rows:
                    continue
                batch, extras = _prepare(batch_rows, device)
                ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
                token_types = extras.get("token_types")
                model.set_prompt_shield_mask(extras.get("prompt_shield_mask"))
                if ids.numel() == 0:
                    continue

                # ── Baseline NLL (gate=0.0, no gradient) ──────────────────
                with torch.no_grad():
                    nll_baseline = _nll_with_gate_alpha(
                        model, ids, mask, labels, gate_value=baseline_alpha, answer_weight=1.0,
                    )

                # ── Learned NLL (gate active, gradient through gate) ───────
                for gate in model.gates.values():
                    gate._freeze_gate = False
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits_learned = model(input_ids=ids, attention_mask=mask).logits
                nll_learned = per_example_answer_nll_weighted(
                    logits_learned, labels, answer_weight=1.0,
                )

                # ── RLOO advantage ─────────────────────────────────────────
                advantage = (nll_baseline - nll_learned).detach()
                mean_alpha_log = torch.stack([
                    gate._alpha_for_aux.clamp(1e-6, 1 - 1e-6).log().mean()
                    for gate in model.gates.values()
                    if gate._alpha_for_aux is not None
                ]).mean()
                rloo_loss = -gate_rloo_w * (advantage.mean() * mean_alpha_log)

                # ── Entropy regularization (V20: replaces L1 mean-alpha) ──
                # Binary entropy H(alpha) encourages crisp 0/1 decisions.
                # Works WITH contrastive (math→1, format→0) unlike mean-alpha L1.
                entropy_loss = _entropy_reg(model.gates)

                # ── Hinge contrastive ──────────────────────────────────────
                cg_loss = ids.new_zeros((), dtype=torch.float32)
                if token_types is not None and gate_contrastive_w > 0:
                    cg_loss = _contrastive_gate_loss(
                        model.gates, token_types, labels, mode=contrastive_mode,
                    )

                # ── Anti-saturation ────────────────────────────────────────
                asat_loss = _antisat_loss(model.gates, threshold=antisat_thresh)

                # ── Compose gate loss ──────────────────────────────────────
                step_loss = (
                    rloo_loss
                    + gate_entropy_w * entropy_loss
                    + gate_contrastive_w * cg_loss
                    + antisat_w * asat_loss
                )
                if use_task_nll:
                    step_loss = step_loss + nll_learned.mean()

                step_loss = step_loss / len(window_batches)

                if torch.isfinite(step_loss):
                    scaler.scale(step_loss).backward()
                    window_loss_sum = window_loss_sum + step_loss.detach()

            # ── Generation-based accuracy reward (every N opt steps) ───────
            gen_pg_loss_val = 0.0
            if use_gen_reward and global_opt_step > 0 and global_opt_step % gen_reward_every == 0:
                gen_indices = torch.randint(len(rows), (gen_subset,), generator=torch.Generator().manual_seed(global_opt_step)).tolist()
                gen_rows = [rows[i] for i in gen_indices]
                gen_reward = _grpo_accuracy_reward(
                    model, tokenizer, gen_rows, device, max_new_tokens=gen_max_tokens,
                )
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                model.train()
                model.base_model.eval()
                if gen_reward.abs().sum().item() > 0:
                    # Fresh forward pass to get live alpha values with gradients.
                    # mean_alpha_log from the micro-batch loop is stale — its
                    # computation graph was freed by the earlier backward() calls.
                    for gate in model.gates.values():
                        gate._freeze_gate = False
                    fresh_batch, fresh_extras = _prepare(gen_rows, device)
                    model.set_prompt_shield_mask(fresh_extras.get("prompt_shield_mask"))
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        _logits = model(
                            input_ids=fresh_batch["input_ids"],
                            attention_mask=fresh_batch["attention_mask"],
                        ).logits
                    del _logits, fresh_batch, fresh_extras
                    fresh_alpha_log = torch.stack([
                        gate._alpha_for_aux.clamp(1e-6, 1 - 1e-6).log().mean()
                        for gate in model.gates.values()
                        if gate._alpha_for_aux is not None
                    ]).mean()
                    gen_pg_loss = -(gen_weight * gen_reward.mean() * fresh_alpha_log)
                    if torch.isfinite(gen_pg_loss):
                        scaler.scale(gen_pg_loss).backward()
                        gen_pg_loss_val = float(gen_pg_loss.detach())
                        window_loss_sum = window_loss_sum + gen_pg_loss.detach()

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

            global_opt_step += 1
            epoch_loss += float(window_loss_sum)
            step_count += 1

            if (wi + 1) % print_every == 0 or wi == n_windows - 1:
                lr_now = optimizer.param_groups[0]["lr"]
                gen_str = f" gen_pg={gen_pg_loss_val:.3f}" if gen_pg_loss_val != 0.0 else ""
                print(f"  S2B epoch {epoch+1}/{gate_epochs} [{wi+1}/{n_windows}] "
                      f"loss={float(window_loss_sum):.3f} lr={lr_now:.1e}{gen_str}")

        # ── Validation: measure NLL with gate AND without gate ─────────────
        model.eval()
        val_nlls_gate: List[float] = []
        val_nlls_base: List[float] = []
        eval_rows = eval_data if isinstance(eval_data, list) else [eval_data[i] for i in range(len(eval_data))]
        for start in range(0, len(eval_rows), cfg.batch_size):
            batch_rows = eval_rows[start : start + cfg.batch_size]
            if not batch_rows:
                continue
            batch, extras = _prepare(batch_rows, device)
            ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
            model.set_prompt_shield_mask(extras.get("prompt_shield_mask"))
            with torch.no_grad():
                # With gate
                for gate in model.gates.values():
                    gate._freeze_gate = False
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits_g = model(input_ids=ids, attention_mask=mask).logits
                val_nlls_gate.append(float(per_example_answer_nll_weighted(logits_g, labels).mean()))

                # Without gate (baseline)
                for gate in model.gates.values():
                    gate._freeze_gate = True
                    gate._freeze_gate_value = 0.0
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits_b = model(input_ids=ids, attention_mask=mask).logits
                val_nlls_base.append(float(per_example_answer_nll_weighted(logits_b, labels).mean()))
                for gate in model.gates.values():
                    gate._freeze_gate = False

        val_nll = sum(val_nlls_gate) / max(len(val_nlls_gate), 1)
        val_nll_base = sum(val_nlls_base) / max(len(val_nlls_base), 1)
        val_improvement = val_nll_base - val_nll  # positive = gate helps
        avg_train = epoch_loss / max(step_count, 1)

        print(f"S2B epoch {epoch+1}/{gate_epochs}: train={avg_train:.4f}  "
              f"val_nll={val_nll:.4f}  val_base={val_nll_base:.4f}  "
              f"improvement={val_improvement:+.4f}")

        # Select checkpoint by maximum NLL improvement (gate benefit), not min NLL
        if val_improvement > best_val_improvement:
            best_val_improvement = val_improvement
            best_gate_state = {k: v.cpu().clone() for k, v in model.gates.state_dict().items()}
            if finetune_sn:
                best_sn_state = {k: v.cpu().clone() for k, v in model.steer_nets.state_dict().items()}
            print(f"  ** New best: improvement={val_improvement:+.4f} (epoch {epoch+1})")
            # Save checkpoint immediately so it survives OOM/crash in later epochs
            _gate_path = Path(getattr(cfg, "gate_save_path", "checkpoints/gate.pt"))
            _gate_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"gates": {k: v.cpu() for k, v in model.gates.state_dict().items()},
                 "fixed_layers": cfg.fixed_layers,
                 "gate_epochs": gate_epochs,
                 "best_val_improvement": best_val_improvement,
                 "history": history},
                _gate_path,
            )
            if finetune_sn:
                _sn_path = Path(getattr(cfg, "steernet_gated_save_path", "checkpoints/steernet_gated.pt"))
                save_steer_weights(model, str(_sn_path), cfg)

        history["gate_train_loss"].append(avg_train)
        history["gate_val_nll"].append(val_nll)
        history["gate_val_improvement"].append(val_improvement)
        history["epochs"].append({
            "epoch": epoch + 1,
            "val_nll": val_nll,
            "val_nll_base": val_nll_base,
            "val_improvement": val_improvement,
            "train_loss": avg_train,
        })

        model.train()
        model.base_model.eval()

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Restore best state and save ────────────────────────────────────────
    if best_gate_state is not None:
        model.gates.load_state_dict(best_gate_state)
    if best_sn_state is not None:
        model.steer_nets.load_state_dict(best_sn_state)

    gate_path = Path(getattr(cfg, "gate_save_path", "checkpoints/gate.pt"))
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"gates": {k: v.cpu() for k, v in model.gates.state_dict().items()},
         "fixed_layers": cfg.fixed_layers,
         "gate_epochs": gate_epochs,
         "best_val_improvement": best_val_improvement,
         "history": history},
        gate_path,
    )
    print(f"[gate] Saved gate checkpoint: {gate_path}  (best_improvement={best_val_improvement:+.4f})")

    report_dir = Path(getattr(cfg, "report_dir", "report"))
    report_dir.mkdir(parents=True, exist_ok=True)
    gate_log_path = report_dir / "gate_training_log.json"
    gate_log_path.write_text(json.dumps(history, indent=2))
    print(f"[gate] Saved gate training log: {gate_log_path}")

    if finetune_sn:
        sn_gated_path = Path(getattr(cfg, "steernet_gated_save_path", "checkpoints/steernet_gated.pt"))
        save_steer_weights(model, str(sn_gated_path), cfg)
        print(f"[gate] Saved gated SteerNet checkpoint: {sn_gated_path}")

    history["best_val_improvement"] = best_val_improvement
    return history


def load_gate_weights(model, path: str, device: torch.device) -> None:
    """Load gate checkpoint into model.gates (non-strict — ignores missing keys)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("gates", ckpt)
    model.gates.load_state_dict(state, strict=False)
    print(f"[gate] Loaded gate checkpoint from {path}")
