"""Fixed-route SteerNet training utilities and the legacy typed Phase 1 trainer."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any, Dict

import torch

from src.configs.config import RSCConfig
from src.training.batching import (
    extract_batch_rows as _extract_batch_rows,
    epoch_batches as _epoch_batches,
    prepare_batch as _prepare,
    trim_to_attended_tokens as _trim_to_attended_tokens,
)
from src.training.optimization import (
    steering_parameter_groups as _param_groups,
    warmup_cosine_lr as _lr_lambda,
)
from src.training.on_policy import (
    collect_onpolicy_data,
    generate_onpolicy_batch as _generate_onpolicy_batch,
    tokenize_onpolicy as _tokenize_onpolicy,
    verify_arithmetic_steps as _verify_arithmetic_steps,
)
from src.training.checkpoints import checkpoint_metadata, load_steer_weights, save_steer_weights
from src.training.metrics import (
    _validate_generation,
    evaluate_loss,
    evaluate_loss_with_hooks,
    evaluate_typed,
)

from src.training.losses import (
    LearnedLossWeights,
    T_FINAL,
    T_FORMAT,
    T_OPERAND,
    T_OPERATOR,
    T_PROSE,
    T_RESULT,
    target_weights,
    token_weighted_nll,
    typed_objective,
    window_counts,
)

def train_phase1_steernet(
    model, train_data, eval_data, cfg: RSCConfig, device: torch.device,
    tokenizer=None,
) -> Dict[str, Any]:
    """Optimise SteerNet with V2 typed reasoning objective, focal hardness, and KL anchor."""
    cfg.validate()
    model.set_routing_override("only", cfg.fixed_layers)
    model.set_steer_scale(cfg.steer_scale)

    # ── Memory optimisation 1: expandable segments reduces fragmentation OOM ──
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # ── Memory optimisation 2: gradient checkpointing on the frozen backbone ──
    # Backbone parameters have requires_grad=False, so checkpointing only
    # costs recompute time, not accuracy. It cuts activation VRAM ~30-40%.
    if device.type == "cuda" and hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    rows = train_data if isinstance(train_data, list) else [train_data[i] for i in range(len(train_data))]
    lengths = [int(r["attention_mask"].sum()) for r in rows]

    learned_weights = None
    if cfg.loss_mode == "typed" and getattr(cfg, "typed_learned_weights", True):
        learned_weights = LearnedLossWeights().to(device)

    groups = _param_groups(model, cfg)
    if learned_weights is not None:
        groups.append({
            "params": list(learned_weights.parameters()),
            "lr": cfg.phase1_lr,
            "weight_decay": 0.0,
        })
    # Old code used betas=(0.9, 0.98), eps=1e-6; V5 uses (0.9, 0.999)
    _betas = (0.9, 0.98) if cfg.loss_mode in ("legacy", "legacy_pure") else (0.9, 0.999)
    optimizer = torch.optim.AdamW(groups, betas=_betas, eps=1e-6)
    n_batches = math.ceil(len(rows) / cfg.batch_size)
    total_steps = math.ceil(n_batches / cfg.grad_accum) * cfg.phase1_epochs
    warm_steps = max(1, int(cfg.phase1_warmup_frac * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_lambda(s, total_steps, warm_steps, cfg.phase1_min_lr_frac)
    )
    # Manual plateau tracker — mutates cosine scheduler's base_lrs so the
    # reduction persists across subsequent per-step cosine updates.
    _plateau_best_nll = float("inf")
    _plateau_wait = 0

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and next(model.base_model.parameters()).dtype == torch.float16,
    )

    backbone_eval = evaluate_typed(model, eval_data, device, cfg, steered=False)
    adapter_init = evaluate_typed(model, eval_data, device, cfg, steered=True)

    history: Dict[str, Any] = {
        "backbone": backbone_eval,
        "adapter_init": adapter_init,
        "train_nll": [],
        "validation_nll": [],   # uniform average NLL (absolute value)
        "crit_history": [],     # reasoning-focused weighted loss (checkpoint metric)
        "epochs": [],
        "steps": [],
    }

    best_crit = float("inf")       # kept for history logging
    best_result_top1 = -1.0        # V20: checkpoint by result_top1 (better proxy for test accuracy)
    best_gen_acc = -1.0            # V21: checkpoint by generation accuracy (actual lenient acc)
    best_state = None

    model.set_telemetry(False)

    for epoch in range(cfg.phase1_epochs):
        model.train()
        model.base_model.eval()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        step_count = 0

        batches = _epoch_batches(lengths, cfg.batch_size, epoch, cfg.seed)
        window_steps = list(range(0, len(batches), cfg.grad_accum))
        n_windows = len(window_steps)
        print_every = max(1, n_windows // 6)

        for wi, w0 in enumerate(window_steps):
            micro_indices = batches[w0 : w0 + cfg.grad_accum]
            micro = [_prepare([rows[i] for i in idx], device) for idx in micro_indices]
            counts = window_counts(
                [b["labels"] for b, _ in micro],
                [e.get("token_types", torch.zeros_like(b["labels"])) for b, e in micro],
            )
            window_loss = 0.0
            parts: Dict[str, float] = {}

            for batch, extras in micro:
                model.set_prompt_shield_mask(extras.get("prompt_shield_mask"))
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    if cfg.loss_mode == "typed":
                        base_logits = None
                        _need_base = (cfg.kl_weight > 0 or learned_weights is not None
                                      or getattr(cfg, "focal_gamma", 0) > 0)
                        if _need_base:
                            # ── KL pass: frozen backbone, hooks disabled ──────────────
                            with torch.no_grad(), model.hooks_disabled():
                                base_logits = model(
                                    input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                ).logits
                            # Move to CPU immediately to free VRAM before the steered pass.
                            # It will be moved back inside typed_objective only if needed.
                            base_logits = base_logits.cpu()
                        # ── Steered pass ─────────────────────────────────────────────
                        logits = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        ).logits
                        _tt = extras.get("token_types", torch.zeros_like(batch["labels"]))
                        loss, parts = typed_objective(
                            logits,
                            base_logits,
                            batch["labels"],
                            _tt,
                            cfg,
                            counts,
                            extras.get("neg_ids"),
                            learned_weights=learned_weights,
                        )
                        # ── Magnitude L1: global penalty + per-type multipliers ──
                        # V23 redesign: global downward pressure on ALL tokens (counteracts
                        # NLL-driven saturation). Math tokens get a low multiplier (0.1-0.3)
                        # so the NLL objective can still earn them high magnitude; non-math
                        # tokens get strong multipliers (1.0-4.0) so the mag_head must work
                        # hard to suppress them. With mag_l1_weight=0.10 (10× V22), the
                        # gradient balance tilts toward selectivity instead of saturation.
                        l1_loss = logits.new_zeros(())
                        mag_penalty = logits.new_zeros(())
                        n_nets = 0
                        tt = extras.get("token_types")
                        for sn in model.steer_nets.values():
                            if sn.last_pre_cap_q is None:
                                continue
                            q = sn.last_pre_cap_q
                            if tt is not None:
                                T_use = min(q.shape[1], tt.shape[1])
                                q_flat = q[:, :T_use, 0]
                                tt_use = tt[:, :T_use]
                                type_mult = torch.ones_like(q_flat)
                                # V24 redesign of type multipliers:
                                # OPERAND/OPERATOR → 0.0: let NLL maximize magnitude freely
                                #   (high amplitude on PLAN tokens improves approach selection)
                                # RESULT           → 3.0: suppress magnitude on = tokens
                                #   (high amplitude corrupts the specific numeric output)
                                # FINAL            → 5.0: suppress magnitude on answer tokens
                                #   (highest priority: arithmetic errors on FINAL = wrong answer)
                                # PROSE/FORMAT: keep existing — these are suppressed to prevent
                                #   the mag_head from saturating on non-math tokens.
                                type_mult[tt_use == T_OPERAND]  = 0.0
                                type_mult[tt_use == T_OPERATOR] = 0.0
                                type_mult[tt_use == T_RESULT]   = 3.0
                                type_mult[tt_use == T_FINAL]    = 5.0
                                # Non-math: moderate pressure (baseline already handles these)
                                type_mult[tt_use == T_PROSE]    = 1.5
                                type_mult[tt_use == T_FORMAT]   = 4.0
                                active_mask = q_flat > 0
                                weighted_q = (q_flat * type_mult)[active_mask]
                            else:
                                weighted_q = q[:, :, 0][q[:, :, 0] > 0]
                            if weighted_q.numel() > 0:
                                mag_penalty = mag_penalty + weighted_q.mean()
                                n_nets += 1
                        if n_nets > 0:
                            l1_loss = mag_penalty / (n_nets * len(micro))
                            parts["mag_l1"] = float(l1_loss.detach())
                        if learned_weights is not None:
                            loss = learned_weights(
                                parts.pop("_nll"), parts.pop("_kl"), l1_loss
                            )
                        elif n_nets > 0 and cfg.mag_l1_weight > 0:
                            loss = loss + cfg.mag_l1_weight * l1_loss
                        # typed_objective is already window-normalized across all micro-batches;
                        # do NOT divide by len(micro) again (double-normalization bug).
                    elif cfg.loss_mode == "legacy_pure":
                        # Pure legacy: model's built-in uniform NLL (matches old RSC_Training.py)
                        # V20: support answer region upweighting when s1_answer_weight > 1.0
                        _aw = getattr(cfg, "s1_answer_weight", 1.0)
                        if _aw > 1.0:
                            from src.routing.rloo import per_example_answer_nll_weighted
                            _logits = model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                            ).logits
                            loss = per_example_answer_nll_weighted(
                                _logits, batch["labels"],
                                answer_end_frac=getattr(cfg, "s1_answer_region_frac", 0.30),
                                answer_weight=_aw,
                            ).mean() / len(micro)
                        else:
                            output = model(**batch)
                            loss = output.loss / len(micro)
                        if cfg.mag_l1_weight > 0:
                            mag_pen = loss.new_zeros(())
                            n_sn = 0
                            for sn in model.steer_nets.values():
                                if sn.last_pre_cap_q is None:
                                    continue
                                q = sn.last_pre_cap_q[:, :, 0]
                                active = q[q > 0]
                                if active.numel() > 0:
                                    mag_pen = mag_pen + active.mean()
                                    n_sn += 1
                            if n_sn > 0:
                                l1 = mag_pen / (n_sn * len(micro))
                                loss = loss + cfg.mag_l1_weight * l1
                                parts["mag_l1"] = float(l1.detach())
                    else:
                        # Legacy mode with calc-token weighting
                        output = model(**batch)
                        weights = target_weights(
                            batch["labels"], extras.get("calc_mask"), cfg.calc_weight_alpha
                        )
                        loss = token_weighted_nll(output.logits, batch["labels"], weights) / len(micro)

                scaler.scale(loss).backward()
                window_loss += float(loss.detach())

            scaler.unscale_(optimizer)
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.steer_nets.parameters(), cfg.clip_grad))
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            # Only step scheduler if optimizer.step() was not skipped by GradScaler
            if scale_before <= scale_after:
                scheduler.step()
            # V23: clamp Kendall log_var to [-3, 3] to prevent one task (e.g. large kl)
            # from suppressing another task entirely via unbounded precision weighting.
            if learned_weights is not None:
                with torch.no_grad():
                    learned_weights.log_var.clamp_(-3.0, 3.0)
            optimizer.zero_grad(set_to_none=True)

            step_count += 1
            running_loss += window_loss
            lr_curr = optimizer.param_groups[0]["lr"]
            if learned_weights is not None:
                with torch.no_grad():
                    learned_weights.log_var.clamp_(-3.0, 3.0)
                    cw = learned_weights.category_weights()
                    prec = torch.exp(-learned_weights.log_var)
                parts.update({
                    "lw_fmt": float(cw[0]), "lw_op": float(cw[1]),
                    "lw_opd": float(cw[2]), "lw_res": float(cw[3]),
                    "lw_fin": float(cw[4]),
                    "prec_nll": float(prec[0]), "prec_kl": float(prec[1]),
                    "prec_l1": float(prec[2]),
                })
            history["steps"].append(
                {"epoch": epoch + 1, "loss": window_loss, "grad_norm": gnorm, "lr": lr_curr, **parts}
            )
            if (wi + 1) % print_every == 0 or wi == n_windows - 1:
                print(f"  S1 epoch {epoch+1}/{cfg.phase1_epochs} [{wi+1}/{n_windows}] loss={window_loss:.3f} gnorm={gnorm:.2f} lr={lr_curr:.1e}")

        # Checkpoint selection metric: critical loss on result/final tokens
        val_eval = evaluate_typed(model, eval_data, device, cfg, steered=True)
        history["epochs"].append(val_eval)
        history["train_nll"].append(running_loss / max(step_count, 1))
        history["validation_nll"].append(val_eval["validation_nll"])
        history["crit_history"].append(val_eval["crit"])
        # V21: generation-based validation for checkpoint selection
        gen_acc = -1.0
        eval_rows_list = eval_data if isinstance(eval_data, list) else [eval_data[i] for i in range(len(eval_data))]
        has_questions = eval_rows_list and "question" in eval_rows_list[0] and "final_num" in eval_rows_list[0]
        if tokenizer is not None and has_questions:
            gen_acc = _validate_generation(model, tokenizer, eval_rows_list, device, cfg, n=50)
            model.train()
            model.base_model.eval()

        _gen_str = f" gen_acc={gen_acc:.3f}" if gen_acc >= 0 else ""
        print(
            f"S1 epoch {epoch + 1}/{cfg.phase1_epochs}: crit={val_eval['crit']:.4f} "
            f"result_top1={val_eval['result_top1']:.3f} val_nll={val_eval['validation_nll']:.4f}"
            f"{_gen_str}"
        )

        if val_eval["crit"] <= best_crit:
            best_crit = val_eval["crit"]
        # V21: prefer generation accuracy for checkpoint; fall back to result_top1
        select_by_gen = gen_acc >= 0 and tokenizer is not None
        improved = (gen_acc > best_gen_acc) if select_by_gen else (val_eval["result_top1"] >= best_result_top1)
        if improved:
            if select_by_gen:
                best_gen_acc = gen_acc
            best_result_top1 = val_eval["result_top1"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.steer_nets.state_dict().items()
            }

        # Plateau-aware LR reduction — monitors same metric as checkpoint selection (crit)
        _val_nll_now = val_eval["crit"]
        if _val_nll_now < _plateau_best_nll - 1e-4:
            _plateau_best_nll = _val_nll_now
            _plateau_wait = 0
        else:
            _plateau_wait += 1
            if _plateau_wait >= getattr(cfg, "phase1_plateau_patience", 2):
                _factor = getattr(cfg, "phase1_plateau_factor", 0.5)
                _min_lr = getattr(cfg, "phase1_plateau_min_lr", 5e-6)
                scheduler.base_lrs = [max(lr * _factor, _min_lr) for lr in scheduler.base_lrs]
                _plateau_wait = 0
                print(f"  [plateau] LR reduced: {[f'{lr:.1e}' for lr in scheduler.base_lrs]}")

        # Per-epoch checkpoint for ablation analysis
        epoch_ckpt_path = Path(cfg.checkpoint_dir) / f"steernet_epoch{epoch + 1}.pt"
        epoch_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "steer_nets": {n: v.detach().cpu().clone() for n, v in model.steer_nets.state_dict().items()},
                "epoch": epoch + 1,
                "crit": val_eval["crit"],
                "val_nll": val_eval["validation_nll"],
            },
            epoch_ckpt_path,
        )

        # Free GPU cache between epochs to reduce fragmentation
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        on_policy_interval = getattr(cfg, "on_policy_interval", 1)
        is_last_epoch = (epoch + 1) >= cfg.phase1_epochs
        on_policy_due = (
            (epoch + 1) >= cfg.on_policy_start_epoch
            and not is_last_epoch
            and (epoch + 1 - cfg.on_policy_start_epoch) % on_policy_interval == 0
        )
        if (
            cfg.on_policy
            and tokenizer is not None
            and on_policy_due
            and len(rows) > 0
            and "question" in rows[0]
            and "final_num" in rows[0]
        ):
            new_rows = collect_onpolicy_data(model, tokenizer, rows, device, cfg, epoch=epoch)
            if new_rows:
                new_lengths = [int(r["attention_mask"].sum()) for r in new_rows]
                rows.extend(new_rows)
                lengths.extend(new_lengths)
                remaining_epochs = cfg.phase1_epochs - (epoch + 1)
                if remaining_epochs > 0:
                    new_n_batches = math.ceil(len(rows) / cfg.batch_size)
                    new_steps = math.ceil(new_n_batches / cfg.grad_accum) * remaining_epochs
                    total_steps = step_count + new_steps
                    warm_steps = 0
                    scheduler = torch.optim.lr_scheduler.LambdaLR(
                        optimizer,
                        lambda s, _ts=total_steps, _sc=step_count: _lr_lambda(
                            s + _sc, _ts, 0, cfg.phase1_min_lr_frac
                        ),
                    )
                print(f"[on-policy] Training data: {len(rows)} rows (added {len(new_rows)})")

    if best_state is not None:
        model.steer_nets.load_state_dict(best_state)
    model.set_telemetry(True)
    # Disable gradient checkpointing after training — not needed for inference/eval
    if device.type == "cuda" and hasattr(model.base_model, "gradient_checkpointing_disable"):
        model.base_model.gradient_checkpointing_disable()
    save_steer_weights(model, cfg.lora_save_path, cfg)
    report_path = Path(cfg.report_dir) / "training_log.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return {
        **history,
        "best_crit": best_crit,
        **checkpoint_metadata(cfg),
    }
