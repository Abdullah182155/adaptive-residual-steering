import gc
import os
import random
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

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
    run_pg_loss,
    log_gpu_memory,
)
from src.training.contrastive import compute_contrastive_gate_loss


def save_steer_weights(model, path: str):
    """Save only RSC-specific parameters (SteerNet, Gate, and Router weights)."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    state = {
        k: v.cpu()
        for k, v in model.state_dict().items()
        if "steer_net" in k or "gate" in k or "router" in k
    }
    torch.save(state, path)
    print(f"✅ Saved steering weights to {path} ({len(state)} tensors)")


def load_steer_weights(model, path: str, device: Optional[torch.device] = None):
    """Load RSC-specific weights into model."""
    if device is None:
        device = next(model.parameters()).device
    state = torch.load(path, map_location=device)
    if "steer_state" in state:
        model.load_state_dict(state["steer_state"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    print(f"✅ Loaded steering weights from {path}")


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


def _evaluate_and_maybe_checkpoint(
    model, tokenizer, eval_data, cfg: RSCConfig, device: torch.device, save_path: str,
    best_acc: float, best_val: float, patience_ctr: int, train_router: bool, label: str,
):
    """Evaluates val CE loss + accuracy and saves checkpoint if improved."""
    if train_router:
        model.reset_routing_diagnostics()
    val_loss = evaluate_loss(model, eval_data, cfg, device)
    acc = epoch_end_accuracy(model, tokenizer, eval_data, device, n_samples=cfg.accuracy_eval_n, seed=cfg.seed)
    improved = acc > best_acc + 0.005
    note = ""
    if train_router:
        diag = model.routing_diagnostics()
        n_candidates = max(1, len(diag))
        dead = sum(1 for s in diag.values() if s["selected_forwards"] / max(1, s["seen_forwards"]) < 0.02)
        k_dist = model.routing_k_distribution()
        k_str = ", ".join(f"K={k}:{v:.2f}" for k, v in k_dist.items() if v > 0.01)
        note = f"  (active layers: {n_candidates - dead}/{n_candidates} | {k_str})"
    print(f"    [checkpoint check] {label}: val={val_loss:.4f}  acc={acc:.3f}  {'[SAVED]' if improved else ''}{note}")
    if train_router and hasattr(model, "router_k_cap_gap_summary"):
        print(model.router_k_cap_gap_summary())
    best_state = None
    if improved:
        best_acc, best_val, patience_ctr = acc, val_loss, 0
        best_state = {
            k: v.clone() for k, v in model.state_dict().items()
            if "steer_net" in k or "gate" in k or "router" in k
        }
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        torch.save(best_state, save_path)
    else:
        patience_ctr += 1
    return best_acc, best_val, best_state, patience_ctr, improved, val_loss, acc


def train_phase1_steernet(model, train_data, eval_data, cfg: RSCConfig, device: torch.device):
    """Phase 1: Pure SteerNet training with gates pinned at 1.0 and router in coverage warmup."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log_gpu_memory("train_phase1 start", cfg)

    print("\n" + "=" * 80)
    print("PHASE 1: Pure SteerNet Training (Hidden States)")
    print("         Gates FIXED at 1.0 (disabled)")
    print(f"         lr={cfg.phase1_lr}, epochs={cfg.phase1_epochs}")
    print("=" * 80)

    model.set_gate_freeze(True, 1.0)
    model.set_router_warmup(True)
    model.set_router_exploration(False)
    for p in model.router.parameters():
        p.requires_grad = False

    steer_params = []
    for sn in model.steer_nets.values():
        for p in sn.parameters():
            p.requires_grad = True
            steer_params.append(p)
    print(f"  SteerNet params: {sum(p.numel() for p in steer_params):,}")

    optimizer_p1 = torch.optim.AdamW(
        steer_params,
        lr=cfg.phase1_lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-6,
    )

    steps_per_epoch = len(train_data) // (cfg.effective_batch_size)
    total_p1 = max(1, steps_per_epoch * cfg.phase1_epochs)
    warmup_p1 = int(total_p1 * cfg.phase1_warmup_ratio)

    lr_reducer_p1 = LRReducer(
        optimizer_p1,
        total_p1,
        factor=cfg.lr_reduce_factor,
        patience=cfg.lr_reduce_patience,
        min_lr=cfg.lr_reduce_min_lr,
        delta=cfg.early_stop_delta,
        warmup_steps=warmup_p1,
    )
    lr_reducer_p1.base_lrs = [cfg.phase1_lr]
    optimizer_p1.param_groups[0]["lr"] = cfg.phase1_lr
    ckpt_p1 = CheckpointManager(cfg.checkpoint_dir)
    grad_mon_p1 = GradientMonitor(cfg.grad_warn_threshold)

    best_val_p1 = float("inf")
    best_state_p1 = None
    p1_patience_ctr = 0
    logs = {"p1_train": [], "p1_val": []}
    scaler_p1 = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(max(0, cfg.phase1_epochs)):
        model.train()
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        epoch_loss, n_optimizer_steps = 0.0, 0
        accum_count_p1 = 0
        optimizer_p1.zero_grad(set_to_none=True)

        pbar = tqdm(
            range(0, len(indices), cfg.batch_size), desc=f"Phase1 {epoch+1}/{cfg.phase1_epochs}"
        )
        for start in pbar:
            batch_idx = indices[start : start + cfg.batch_size]
            ids = torch.stack([train_data[i]["input_ids"] for i in batch_idx]).to(device)
            mask = torch.stack([train_data[i]["attention_mask"] for i in batch_idx]).to(device)
            lbls = torch.stack([train_data[i]["labels"] for i in batch_idx]).to(device)
            model._attn_mask_cache = mask
            model.set_router_context_mask(mask.bool() & (lbls == -100))

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(input_ids=ids, attention_mask=mask, labels=lbls)
                loss = out.loss / cfg.grad_accum

            if not torch.isfinite(loss):
                continue

            scaler_p1.scale(loss).backward()
            accum_count_p1 += 1

            is_last_p1 = start + cfg.batch_size >= len(indices)
            if accum_count_p1 == cfg.grad_accum or is_last_p1:
                scaler_p1.unscale_(optimizer_p1)
                for p in steer_params:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        p.grad.zero_()
                torch.nn.utils.clip_grad_norm_(steer_params, cfg.clip_grad)
                if cfg.use_gradient_monitor:
                    grad_mon_p1.check(steer_params, n_optimizer_steps)
                scaler_p1.step(optimizer_p1)
                scaler_p1.update()
                lr_reducer_p1.scheduler_step()
                optimizer_p1.zero_grad(set_to_none=True)
                accum_count_p1 = 0
                n_optimizer_steps += 1

            epoch_loss += loss.item() * cfg.grad_accum
            pbar.set_postfix(loss=f"{loss.item() * cfg.grad_accum:.4f}")

        avg_loss = epoch_loss / max(1, n_optimizer_steps * cfg.grad_accum)
        val_loss = evaluate_loss(model, eval_data, cfg, device)
        logs["p1_train"].append(avg_loss)
        logs["p1_val"].append(val_loss)

        improved = "✅" if val_loss < best_val_p1 - cfg.early_stop_delta else "  "
        if val_loss < best_val_p1 - cfg.early_stop_delta:
            best_val_p1 = val_loss
            p1_patience_ctr = 0
            best_state_p1 = {
                k: v.clone()
                for k, v in model.state_dict().items()
                if "steer_net" in k or "router" in k
            }
        else:
            p1_patience_ctr += 1

        lr_reducer_p1.step(val_loss)
        lr_now = optimizer_p1.param_groups[0]["lr"]
        print(
            f"  Phase1 Epoch {epoch+1}: train={avg_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}  {improved}"
        )

        if (epoch + 1) % cfg.checkpoint_every_n == 0:
            ckpt_p1.save(model, optimizer_p1, epoch, val_loss, "p1")

        if p1_patience_ctr >= cfg.phase1_patience:
            print(f"  Phase1 early stopping (no improvement for {cfg.phase1_patience} epochs)")
            break

    if best_state_p1:
        model.load_state_dict(best_state_p1, strict=False)
    save_steer_weights(model, cfg.lora_save_path)
    grad_mon_p1.summary()
    print(f"\n✅ Phase 1 complete — best_val = {best_val_p1:.4f}")

    model.set_router_warmup(False)
    model.reset_routing_diagnostics()
    log_gpu_memory("after Phase 1", cfg)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_state_p1, best_val_p1, logs


def train_router_bootstrap(model, train_data, eval_data, cfg: RSCConfig, device: torch.device):
    """Phase 1.5: Router Bootstrap with frozen SteerNet & Gate using RLOO counterfactual loss."""
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

    print("\n" + "=" * 80)
    print("PHASE 1.5: Router Bootstrap (RLOO Counterfactual Loss)")
    print(f"           epochs={cfg.router_bootstrap_epochs}, lr={cfg.router_bootstrap_lr}")
    print("=" * 80)

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
            postfix_dict = {"loss": f"{loss.item():.4f}"}
            if rloo_diag:
                postfix_dict["mean_k"] = f"{rloo_diag.get('mean_k_sampled', 0.0):.2f}"
                if "win_rate_vs_base" in rloo_diag:
                    postfix_dict["win_base"] = f"{rloo_diag['win_rate_vs_base'] * 100:.0f}%"
            pbar.set_postfix(**postfix_dict)

        avg_loss = running / max(1, n_steps)
        k_dist = model.routing_k_distribution()
        diag = model.routing_diagnostics()
        active_cnt = sum(1 for s in diag.values() if s["selected_forwards"] > 0)
        coop = model.get_cooperation_analysis() if hasattr(model, "get_cooperation_analysis") else {}
        top_syn_str = ", ".join(f"L{a}+L{b}:{v:+.2f}" for a, b, v in coop.get("top_synergy_pairs", [])[:2])
        win_str = f" | win_base={rloo_diag.get('win_rate_vs_base', 0.0) * 100:.1f}%" if rloo_diag else ""
        syn_str = f" | top synergy: [{top_syn_str}]" if top_syn_str else ""
        print(f"  RouterBootstrap Epoch {epoch+1}: loss={avg_loss:.4f}{win_str} | active layers={active_cnt}/{len(diag)}{syn_str} | K-dist={k_dist}")
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
    save_path = cfg.router_bootstrap_save_path
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    torch.save(
        {k: v.cpu() for k, v in model.state_dict().items() if "router" in k},
        save_path,
    )
    print(f"✅ Router Bootstrap complete — best_probe_loss = {best_loss:.4f}")
    return best_state, best_loss


def run_staged_phase(
    model, tokenizer, train_data, eval_data, cfg: RSCConfig, device: torch.device,
    stage_name: str, epochs: int, param_groups: List[Dict[str, Any]], save_path: str,
    train_steer: bool, train_gate: bool, train_router: bool,
    global_step: int = 0, baseline_ema: Optional[List[float]] = None, patience: int = 3,
    init_best_acc: float = 0.0, init_best_val: float = float("inf"),
) -> Tuple[int, float, float]:
    """Shared Phase 2 epoch loop for SteerNet fine-tune, Gate-only, Joint fine-tune, and Gate re-tune."""
    if baseline_ema is None:
        baseline_ema = [0.0]
    print("\n" + "=" * 80)
    print(f"{stage_name}")
    print(
        f"         epochs={epochs}  train_steer={train_steer}  "
        f"train_gate={train_gate}  train_router={train_router}"
    )
    for g in param_groups:
        print(f"         group size={sum(p.numel() for p in g['params']):,}  lr={g['lr']:.2e}")
    print("=" * 80)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.98), eps=1e-8)
    all_params = [p for g in param_groups for p in g["params"]]
    steps_per_epoch = len(train_data) // cfg.effective_batch_size
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * cfg.phase2_warmup_ratio)
    lr_reducer = LRReducer(
        optimizer, total_steps, factor=cfg.lr_reduce_factor, patience=cfg.lr_reduce_patience,
        min_lr=cfg.lr_reduce_min_lr, warmup_steps=warmup_steps,
    )
    lr_reducer.base_lrs = [g["lr"] for g in param_groups]
    for i, g in enumerate(optimizer.param_groups):
        g["lr"] = lr_reducer.base_lrs[i]
    grad_mon = GradientMonitor(cfg.grad_warn_threshold)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_acc, best_val, best_state, patience_ctr = init_best_acc, init_best_val, None, 0

    for epoch in range(epochs):
        model.train()
        model.set_gate_freeze(not train_gate, 1.0)
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        epoch_task, n_opt_steps, accum_count = 0.0, 0, 0

        epoch_gate_l1, epoch_gate_div, epoch_gate_antisat, epoch_gate_contrast, n_gate_aux_steps = 0.0, 0.0, 0.0, 0.0, 0
        epoch_policy_loss, epoch_mean_k, n_router_rloo_steps = 0.0, 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(range(0, len(indices), cfg.batch_size), desc=f"{stage_name.split(':')[0]} {epoch + 1}/{epochs}")

        for start in pbar:
            batch_idx = indices[start : start + cfg.batch_size]
            ids = torch.stack([train_data[i]["input_ids"] for i in batch_idx]).to(device)
            mask = torch.stack([train_data[i]["attention_mask"] for i in batch_idx]).to(device)
            lbls = torch.stack([train_data[i]["labels"] for i in batch_idx]).to(device)
            model._attn_mask_cache = mask
            model.set_router_context_mask(mask.bool() & (lbls == -100))

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(input_ids=ids, attention_mask=mask, labels=lbls)
                task_loss = out.loss

            micro_loss = (task_loss / cfg.grad_accum) if (torch.isfinite(task_loss) and task_loss.requires_grad) else None

            if train_gate:
                l1_loss = compute_l1_gate(model)
                asat_loss = compute_antisat_gate(model, threshold=0.85)
                contrast_loss = compute_contrastive_gate_loss(model, ids, lbls, tokenizer, device)
                diversity_loss = compute_gate_diversity_loss(model, floor=cfg.gate_diversity_floor)
                epoch_gate_l1 += l1_loss.item()
                epoch_gate_div += diversity_loss.item()
                epoch_gate_antisat += asat_loss.item()
                epoch_gate_contrast += contrast_loss.item()
                n_gate_aux_steps += 1
                aux_loss = (
                    cfg.lambda_l1_gate * l1_loss
                    + cfg.lambda_gate_antisat * asat_loss
                    + cfg.lambda_gate_usefulness * contrast_loss
                    + cfg.lambda_gate_diversity * diversity_loss
                ) / cfg.grad_accum
                if torch.isfinite(aux_loss) and aux_loss.requires_grad:
                    micro_loss = aux_loss if micro_loss is None else (micro_loss + aux_loss)

            if micro_loss is not None:
                scaler.scale(micro_loss).backward()

            if train_gate and not train_steer and accum_count == 0 and global_step % cfg.pg_every_n_steps == 0:
                pg_idx = random.sample(range(len(train_data)), min(cfg.pg_n_samples, len(train_data)))
                pg_ids = torch.stack([train_data[j]["input_ids"] for j in pg_idx]).to(device)
                pg_lbls = torch.stack([train_data[j]["labels"] for j in pg_idx]).to(device)
                pg_loss = run_pg_loss(model, tokenizer, pg_ids, pg_lbls, device, cfg, baseline_ema)
                if torch.isfinite(pg_loss) and pg_loss.requires_grad:
                    scaler.scale(cfg.lambda_pg * pg_loss).backward()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if train_steer and accum_count == 0 and global_step % cfg.invariance_every_n_steps == 0:
                inv_loss = compute_invariance_loss(
                    model, tokenizer,
                    [train_data[i]["question"] for i in batch_idx],
                    [train_data[i]["steps_list"] for i in batch_idx],
                    [train_data[i]["final_num"] for i in batch_idx],
                    cfg, device, seed_offset=global_step,
                )
                if torch.isfinite(inv_loss) and inv_loss.requires_grad:
                    scaler.scale(cfg.lambda_invariance * inv_loss).backward()

            if train_router and accum_count == 0 and global_step % cfg.router_probe_every_n_steps == 0:
                prompt_mask = mask.bool() & lbls.eq(-100)
                router_loss, rloo_diag = compute_router_rloo_loss(model, ids, mask, lbls, cfg, prompt_mask=prompt_mask)
                if torch.isfinite(router_loss) and router_loss.requires_grad:
                    scaler.scale(router_loss).backward()
                if rloo_diag:
                    epoch_policy_loss += rloo_diag["policy_loss"]
                    epoch_mean_k += rloo_diag["mean_k_sampled"]
                    n_router_rloo_steps += 1
            accum_count += 1
            is_last = start + cfg.batch_size >= len(indices)
            if accum_count == cfg.grad_accum or is_last:
                scaler.unscale_(optimizer)
                for p in all_params:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        p.grad.zero_()
                torch.nn.utils.clip_grad_norm_(all_params, cfg.clip_grad)
                if cfg.use_gradient_monitor:
                    grad_mon.check(all_params, global_step)
                scaler.step(optimizer)
                scaler.update()
                lr_reducer.scheduler_step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                global_step += 1
                n_opt_steps += 1

                if global_step % 50 == 0:
                    gc.collect()
                    log_gpu_memory(f"run_staged_phase step {global_step}", cfg)

                if (
                    train_router
                    and cfg.router_checkpoint_every_n_steps > 0
                    and global_step % cfg.router_checkpoint_every_n_steps == 0
                ):
                    best_acc, best_val, sub_state, patience_ctr, _, _, _ = _evaluate_and_maybe_checkpoint(
                        model, tokenizer, eval_data, cfg, device, save_path,
                        best_acc, best_val, patience_ctr, train_router,
                        label=f"step {global_step} (mid-epoch {epoch + 1})",
                    )
                    if sub_state is not None:
                        best_state = sub_state
                    if patience_ctr >= patience:
                        print(f"  {stage_name.split(':')[0]} early stopping (mid-epoch).")
                        break

            epoch_task += task_loss.item()
            postfix_kwargs = {"task": f"{task_loss.item():.4f}"}
            if train_router and n_router_rloo_steps > 0:
                postfix_kwargs["mean_k"] = f"{epoch_mean_k / n_router_rloo_steps:.2f}"
            pbar.set_postfix(**postfix_kwargs)

        if patience_ctr >= patience:
            break
        avg_task = epoch_task / max(1, n_opt_steps * cfg.grad_accum)
        best_acc, best_val, epoch_state, patience_ctr, improved, cur_val_loss, cur_acc = _evaluate_and_maybe_checkpoint(
            model, tokenizer, eval_data, cfg, device, save_path,
            best_acc, best_val, patience_ctr, train_router,
            label=f"epoch {epoch + 1} boundary",
        )
        if epoch_state is not None:
            best_state = epoch_state
        print(f"  {stage_name.split(':')[0]} Epoch {epoch + 1}: task={avg_task:.4f}  val={cur_val_loss:.4f}  acc={cur_acc:.3f}  {'✅' if improved else ''}")

    return global_step, best_val, best_acc


def train_rsc(model, tokenizer, train_data, eval_data, cfg: RSCConfig, device: torch.device):
    """Complete multi-phase staged training pipeline:
    Phase 1: Pure SteerNet (Warmup Rotation, Gates=1.0)
    Phase 1.5: Router Bootstrap (RLOO Counterfactual Loss)
    Phase 2A: SteerNet Fine-Tune (Exhaustive Rotation)
    Phase 2B: Gate-Only Training (Dynamic Routing, Aux & PG Losses)
    Phase 2C: Light Joint Fine-Tuning (SteerNet + Gate + Router)
    Phase 2D: Gate Re-Tune (Against Final Router)
    """
    logs = {}

    # PHASE 1: Pure SteerNet
    best_state_p1, best_val_p1, p1_logs = train_phase1_steernet(model, train_data, eval_data, cfg, device)
    logs.update(p1_logs)

    # PHASE 1.5: Router Bootstrap
    _, best_p15 = train_router_bootstrap(model, train_data, eval_data, cfg, device)
    logs["router_bootstrap_loss"] = best_p15
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # PHASE 2A: SteerNet Fine-tune
    for p in model.router.parameters():
        p.requires_grad = False
    model.set_gate_freeze(True, 1.0)
    model.set_router_warmup(True)
    steer_params_ft = []
    for sn in model.steer_nets.values():
        for p in sn.parameters():
            p.requires_grad = True
            steer_params_ft.append(p)

    global_step, best_p2a_val, best_p2a_acc = run_staged_phase(
        model, tokenizer, train_data, eval_data, cfg, device,
        stage_name="PHASE 2A: SteerNet Fine-tune (exhaustive rotation, gate=1.0)",
        epochs=cfg.steer_finetune_epochs,
        param_groups=[{"params": steer_params_ft, "lr": cfg.steer_finetune_lr}],
        save_path=cfg.steer_finetune_save_path,
        train_steer=True, train_gate=False, train_router=False,
        global_step=0, baseline_ema=[0.0], patience=cfg.phase2_patience,
    )
    logs["p2a_val"] = best_p2a_val
    model.reset_routing_diagnostics()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # PHASE 2B: Gate-only Training
    for p in steer_params_ft:
        p.requires_grad = False
    model.set_gate_freeze(False)
    model.set_router_warmup(False)
    model.set_router_exploration(True)
    gate_params_ft = []
    for key in sorted(model.gates.keys(), key=lambda k: int(k)):
        for p in model.gates[key].parameters():
            p.requires_grad = True
            gate_params_ft.append(p)

    global_step, best_p2b_val, best_p2b_acc = run_staged_phase(
        model, tokenizer, train_data, eval_data, cfg, device,
        stage_name="PHASE 2B: Gate-only Training (SteerNet + Router frozen)",
        epochs=cfg.gate_only_epochs,
        param_groups=[{"params": gate_params_ft, "lr": cfg.gate_only_lr}],
        save_path=cfg.gate_only_save_path,
        train_steer=False, train_gate=True, train_router=False,
        global_step=global_step, baseline_ema=[0.0], patience=cfg.phase2_patience,
    )
    logs["p2b_val"] = best_p2b_val
    model.reset_routing_diagnostics()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # PHASE 2C: Light Joint Fine-tune
    for p in steer_params_ft:
        p.requires_grad = True
    for p in gate_params_ft:
        p.requires_grad = True
    for p in model.router.parameters():
        p.requires_grad = True
    model.set_router_exploration(True)

    joint_groups = [
        {"params": steer_params_ft, "lr": cfg.steer_finetune_lr * cfg.joint_finetune_lr_scale},
        {"params": gate_params_ft, "lr": cfg.gate_only_lr * cfg.joint_finetune_lr_scale},
        {
            "params": list(model.router.parameters()),
            "lr": cfg.router_bootstrap_lr * cfg.joint_finetune_router_lr_scale,
        },
    ]

    global_step, best_val_p2, best_acc_p2 = run_staged_phase(
        model, tokenizer, train_data, eval_data, cfg, device,
        stage_name="PHASE 2C: Light Joint Fine-tune",
        epochs=cfg.joint_finetune_epochs,
        param_groups=joint_groups,
        save_path=cfg.phase2_save_path,
        train_steer=True, train_gate=True, train_router=True,
        global_step=global_step, baseline_ema=[0.0], patience=max(1, cfg.phase2_patience),
    )
    logs["p2_val"] = [best_val_p2]

    # PHASE 2D: Gate Re-tune
    for p in model.router.parameters():
        p.requires_grad = False
    for p in steer_params_ft:
        p.requires_grad = False
    model.set_gate_freeze(False)
    model.set_router_exploration(False)
    for p in gate_params_ft:
        p.requires_grad = True

    global_step, best_p2d_val, best_p2d_acc = run_staged_phase(
        model, tokenizer, train_data, eval_data, cfg, device,
        stage_name="PHASE 2D: Gate Re-tune (against final router)",
        epochs=cfg.gate_retune_epochs,
        param_groups=[{"params": gate_params_ft, "lr": cfg.gate_retune_lr}],
        save_path=cfg.phase2_save_path,
        train_steer=False, train_gate=True, train_router=False,
        global_step=global_step, baseline_ema=[0.0], patience=cfg.phase2_patience,
        init_best_acc=best_acc_p2, init_best_val=best_val_p2,
    )
    logs["p2d_val"] = best_p2d_val
    model.set_router_exploration(False)

    if os.path.exists(cfg.phase2_save_path):
        _p2d_state = torch.load(cfg.phase2_save_path, map_location=device)
        model.load_state_dict(_p2d_state, strict=False)
        print(f"  ✅ Loaded final joint+gate-retune checkpoint from {cfg.phase2_save_path}")

    print("\n" + "=" * 80)
    print("🎉 FULL ARS TRAINING COMPLETE!")
    print(f"   Phase 1 best validation loss           : {best_val_p1:.4f}")
    print(f"   Router bootstrap best probe loss       : {best_p15:.4f}")
    print(f"   Phase 2A (SteerNet ft) best val        : {best_p2a_val:.4f}")
    print(f"   Phase 2B (Gate-only) best val          : {best_p2b_val:.4f}")
    print(f"   Phase 2C (joint ft) best val           : {best_val_p2:.4f}")
    print(f"   Phase 2D (gate re-tune) best val       : {best_p2d_val:.4f}")
    print(f"   Final saved checkpoint                 : {cfg.phase2_save_path}")
    print("=" * 80)

    return logs, best_val_p1, best_val_p2
