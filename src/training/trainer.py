"""Fixed-route SteerNet V5 trainer for ARS Phase 1 with V2 typed reasoning loss."""

from __future__ import annotations

from collections import defaultdict
import contextlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.configs.config import RSCConfig
from src.training.losses import (
    LearnedLossWeights,
    T_FINAL,
    T_FORMAT,
    T_NONE,
    T_OPERAND,
    T_OPERATOR,
    T_PROSE,
    T_RESULT,
    annotate_char_types,
    lambdas,
    target_weights,
    token_types_from_offsets,
    token_weighted_nll,
    typed_objective,
    window_counts,
)


def _stack_batch(rows: List[dict], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move the standard token tensors for a contiguous batch to a device."""
    return {
        name: torch.stack([row[name] for row in rows]).to(device)
        for name in ("input_ids", "attention_mask", "labels")
        if name in rows[0]
    }


def _extract_batch_rows(data, offset: int, batch_size: int) -> List[dict]:
    """Extract a batch of sample dicts from either a list or a dataset object."""
    if isinstance(data, list):
        return data[offset : offset + batch_size]
    end = min(offset + batch_size, len(data))
    return [data[i] for i in range(offset, end)]


def _trim_to_attended_tokens(
    batch: Dict[str, torch.Tensor],
    extras: Dict[str, torch.Tensor] | None = None,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Drop all-padding columns while preserving every aligned auxiliary mask."""
    extras = extras or {}
    attention_mask = batch["attention_mask"].bool()
    live_columns = attention_mask.any(dim=0).nonzero(as_tuple=False).flatten()
    if live_columns.numel() == 0:
        raise ValueError("batch contains no attended tokens")
    start = int(live_columns.min())
    stop = int(live_columns.max()) + 1
    trimmed_batch = {name: value[:, start:stop] for name, value in batch.items()}
    trimmed_extras = {
        name: value[:, start:stop]
        for name, value in extras.items()
        if value.ndim >= 2 and (value.shape[1] == attention_mask.shape[1] or value.shape[-1] == attention_mask.shape[-1])
    }
    return trimmed_batch, trimmed_extras


def _prepare(rows: List[dict], device: torch.device):
    batch = {
        name: torch.stack([row[name] for row in rows]).to(device)
        for name in ("input_ids", "attention_mask", "labels")
        if name in rows[0]
    }
    extras = {
        name: torch.stack([row[name] for row in rows]).to(device)
        for name in ("prompt_shield_mask", "token_types", "calc_mask", "neg_ids")
        if name in rows[0]
    }
    return _trim_to_attended_tokens(batch, extras)


def _param_groups(model, cfg: RSCConfig) -> List[Dict[str, Any]]:
    """Configure optimizer parameter groups: exclude LayerNorms and scale-invariant up from weight decay."""
    decay, no_decay, up = [], [], []
    for name, p in model.steer_nets.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("up.weight"):
            up.append(p)
        elif name.endswith("mag_head.bias"):
            decay.append(p)  # L2 prevents bias from growing unbounded → keeps sigmoid < 1
        elif p.ndim >= 2 and ".conv." not in name:
            decay.append(p)
        else:
            no_decay.append(p)
    lr = cfg.phase1_lr
    return [
        {"params": decay, "lr": lr, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        {"params": up, "lr": lr, "weight_decay": 0.0},
    ]


def _lr_lambda(step: int, total: int, warm: int, floor: float) -> float:
    """Linear warmup followed by cosine decay to floor."""
    if step < warm:
        return (step + 1) / max(warm, 1)
    prog = (step - warm) / max(1, total - warm)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))


def _epoch_batches(lengths: List[int], bs: int, epoch: int, seed: int, mega: int = 16) -> List[List[int]]:
    """Dynamic length-bucketing in mega-chunks followed by batch shuffle."""
    g = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(len(lengths), generator=g).tolist()
    batches = []
    for i in range(0, len(order), bs * mega):
        chunk = sorted(order[i : i + bs * mega], key=lambda j: lengths[j])
        batches += [chunk[k : k + bs] for k in range(0, len(chunk), bs)]
    return [batches[i] for i in torch.randperm(len(batches), generator=g).tolist()]


@torch.no_grad()
def evaluate_typed(
    model, rows: Iterable[dict], device: torch.device, cfg: RSCConfig, steered: bool = True
) -> Dict[str, float]:
    """Compute per-category NLL, critical loss, and top-1 accuracy on result tokens."""
    row_list = [rows[i] for i in range(len(rows))] if hasattr(rows, "select") else list(rows)
    if not row_list:
        return {"crit": 0.0, "result_top1": 0.0, "validation_nll": 0.0}
    model.eval()
    s = defaultdict(float)
    n = defaultdict(int)
    hit = defaultdict(int)
    ctx = contextlib.nullcontext() if steered else model.hooks_disabled()
    with ctx:
        for row in row_list:
            batch, extras = _prepare([row], device)
            model.set_prompt_shield_mask(extras.get("prompt_shield_mask"))
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            sel = batch["labels"][:, 1:].ne(-100)
            if not sel.any():
                continue
            z = logits[:, :-1][sel].float()
            y = batch["labels"][:, 1:][sel]
            c = extras["token_types"][:, 1:][sel] if "token_types" in extras else torch.zeros_like(y)
            nll = F.cross_entropy(z, y, reduction="none")
            ok = z.argmax(-1).eq(y)
            for cid in c.unique().tolist():
                m = c == cid
                s[cid] += float(nll[m].sum())
                n[cid] += int(m.sum())
                hit[cid] += int(ok[m].sum())

    out = {f"nll_{c}": s[c] / max(n[c], 1) for c in n if n[c]}
    lam_map = lambdas(cfg)
    out["crit"] = sum(lam * out.get(f"nll_{c}", 0.0) for c, lam in lam_map.items())
    rn = n[T_RESULT] + n[T_FINAL]
    out["result_top1"] = (hit[T_RESULT] + hit[T_FINAL]) / max(rn, 1)
    total_tokens = sum(n.values())
    total_nll = sum(s.values())
    out["validation_nll"] = total_nll / max(total_tokens, 1)
    return out


@torch.no_grad()
def evaluate_loss(model, rows: Iterable[dict], device: torch.device) -> float:
    """Return validation NLL for backward compatibility."""
    cfg = getattr(model, "cfg", RSCConfig())
    typed = evaluate_typed(model, rows, device, cfg, steered=True)
    return float(typed.get("validation_nll", typed.get("crit", 0.0)))


def evaluate_loss_with_hooks(model, rows, device: torch.device, enabled: bool) -> float:
    context = torch.enable_grad() if enabled else model.hooks_disabled()
    with context:
        return evaluate_loss(model, rows, device)


def checkpoint_metadata(cfg: RSCConfig) -> Dict[str, Any]:
    return {
        "route": list(cfg.fixed_layers),
        "steer_scale": cfg.steer_scale,
        "steer_kappa": cfg.steer_kappa,
        "k_scale_mode": cfg.k_scale_mode,
        "loss_mode": cfg.loss_mode,
    }


def save_steer_weights(model, path: str, cfg: RSCConfig) -> None:
    """Persist Phase 1 adapter state with enough metadata to reject mismatches."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": checkpoint_metadata(cfg),
            "steer_nets": model.steer_nets.state_dict(),
        },
        path,
    )


def load_steer_weights(
    model, path: str, cfg: RSCConfig, device: Optional[torch.device] = None
) -> Dict[str, Any]:
    state = torch.load(path, map_location=device or "cpu", weights_only=True)
    metadata = state["metadata"]
    if metadata["route"] != list(cfg.fixed_layers):
        raise ValueError("checkpoint route does not match the active fixed route")
    model.steer_nets.load_state_dict(state["steer_nets"])
    return metadata


@torch.no_grad()
def _generate_onpolicy_batch(model, tokenizer, prompts, device, max_new_tokens=256):
    enc = tokenizer(prompts, return_tensors="pt", truncation=True,
                    max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in enc.items()}
    model.set_prompt_shield_mask(None)
    tokens = model.generate(**inputs, max_new_tokens=max_new_tokens,
                            do_sample=False, temperature=0.0)
    prompt_lens = inputs["attention_mask"].sum(dim=1)
    outputs = []
    for i in range(len(prompts)):
        plen = int(prompt_lens[i])
        text = tokenizer.decode(tokens[i, plen:], skip_special_tokens=True)
        outputs.append(text)
    return outputs


def _tokenize_onpolicy(tokenizer, question, completion, cfg):
    prompt_text = f"Question: {question}\nAnswer: Let's think step by step.\n"
    full_text = prompt_text + completion
    target_start = len(prompt_text)

    enc = tokenizer(full_text, truncation=True, max_length=cfg.max_tok_len,
                    padding="max_length", return_offsets_mapping=True)

    ids, offsets, am = enc["input_ids"], enc["offset_mapping"], enc["attention_mask"]
    labels = [tid if (m and e > s and s >= target_start) else -100
              for tid, (s, e), m in zip(ids, offsets, am)]
    shield = [0] * len(ids)
    char_labels = annotate_char_types(full_text, target_start)
    tt = token_types_from_offsets(full_text, char_labels, offsets)
    tt = [t if l != -100 else T_NONE for t, l in zip(tt, labels)]
    calc = [1 if t in (T_OPERAND, T_OPERATOR, T_RESULT, T_FINAL) else 0 for t in tt]

    return {
        "input_ids": torch.tensor(ids),
        "attention_mask": torch.tensor(am),
        "labels": torch.tensor(labels),
        "prompt_shield_mask": torch.tensor(shield),
        "token_types": torch.tensor(tt),
        "calc_mask": torch.tensor(calc),
    }


@torch.no_grad()
def collect_onpolicy_data(model, tokenizer, rows, device, cfg):
    from src.evaluation.answers import extract_numeric_answer

    n_gen = max(1, int(len(rows) * cfg.on_policy_frac))
    indices = torch.randperm(
        len(rows), generator=torch.Generator().manual_seed(cfg.seed)
    )[:n_gen].tolist()

    questions = [rows[i]["question"] for i in indices]
    golds = [rows[i]["final_num"] for i in indices]
    prompts = [f"Question: {q}\nAnswer: Let's think step by step.\n" for q in questions]

    gc_enabled = getattr(model.base_model, "is_gradient_checkpointing", False)
    if gc_enabled and hasattr(model.base_model, "gradient_checkpointing_disable"):
        model.base_model.gradient_checkpointing_disable()

    model.eval()
    model.set_routing_override("only", cfg.fixed_layers)

    completions: List[str] = []
    gen_bs = cfg.on_policy_gen_batch
    for i in tqdm(range(0, len(prompts), gen_bs), desc="on-policy gen", leave=False):
        batch = prompts[i : i + gen_bs]
        completions.extend(
            _generate_onpolicy_batch(model, tokenizer, batch, device)
        )

    if gc_enabled and hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    new_rows: List[Dict[str, Any]] = []
    n_correct = 0
    for q, comp, gold in zip(questions, completions, golds):
        try:
            gold_num = float(str(gold).replace(",", "").replace("$", "").strip())
        except (ValueError, AttributeError):
            continue
        pred = extract_numeric_answer(comp)
        if pred is None:
            continue
        if abs(pred - gold_num) > max(0.01, abs(gold_num) * 0.005):
            continue
        n_correct += 1
        row = _tokenize_onpolicy(tokenizer, q, comp, cfg)
        new_rows.append(row)

    print(
        f"[on-policy] {n_gen} generated → {n_correct} correct "
        f"({100 * n_correct / max(n_gen, 1):.1f}%) → {len(new_rows)} new rows"
    )

    model.train()
    model.base_model.eval()
    return new_rows


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
    if cfg.loss_mode == "typed":
        learned_weights = LearnedLossWeights().to(device)

    groups = _param_groups(model, cfg)
    if learned_weights is not None:
        groups.append({
            "params": list(learned_weights.parameters()),
            "lr": cfg.phase1_lr,
            "weight_decay": 0.0,
        })
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.99))
    n_batches = math.ceil(len(rows) / cfg.batch_size)
    total_steps = math.ceil(n_batches / cfg.grad_accum) * cfg.phase1_epochs
    warm_steps = max(1, int(cfg.phase1_warmup_frac * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_lambda(s, total_steps, warm_steps, cfg.phase1_min_lr_frac)
    )

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
        "validation_nll": [],
        "epochs": [],
        "steps": [],
    }

    best_crit = float("inf")
    best_state = None

    model.set_telemetry(False)

    for epoch in range(cfg.phase1_epochs):
        model.train()
        model.base_model.eval()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        step_count = 0

        batches = _epoch_batches(lengths, cfg.batch_size, epoch, cfg.seed)
        bar = tqdm(range(0, len(batches), cfg.grad_accum), desc=f"S1 {epoch + 1}/{cfg.phase1_epochs}", disable=True)

        for w0 in bar:
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
                        if cfg.kl_weight > 0 or learned_weights is not None:
                            # ── KL pass: frozen backbone, hooks disabled ──────────────
                            with torch.no_grad(), model.hooks_disabled():
                                base_logits = model(
                                    input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                ).logits
                            # Move to CPU immediately to free VRAM before the steered pass.
                            # It will be moved back inside typed_objective only if needed.
                            base_logits = base_logits.cpu()
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                        # ── Steered pass ─────────────────────────────────────────────
                        logits = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        ).logits
                        loss, parts = typed_objective(
                            logits,
                            base_logits,
                            batch["labels"],
                            extras["token_types"],
                            cfg,
                            counts,
                            extras.get("neg_ids"),
                            learned_weights=learned_weights,
                        )
                        # ── Magnitude L1 on prose tokens ──
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
                                prose_pos = (tt[:, :T_use] == T_PROSE)
                                q_prose = q[:, :T_use, 0][prose_pos]
                                active = q_prose[q_prose > 0]
                            else:
                                active = q[:, :, 0][q[:, :, 0] > 0]
                            if active.numel() > 0:
                                mag_penalty = mag_penalty + active.mean()
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
                    else:
                        # Legacy mode
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
            optimizer.zero_grad(set_to_none=True)

            step_count += 1
            running_loss += window_loss
            lr_curr = optimizer.param_groups[0]["lr"]
            if learned_weights is not None:
                with torch.no_grad():
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
            bar.set_postfix(loss=f"{window_loss:.3f}", gnorm=f"{gnorm:.2f}", lr=f"{lr_curr:.1e}")

        # Checkpoint selection metric: critical loss on result/final tokens
        val_eval = evaluate_typed(model, eval_data, device, cfg, steered=True)
        history["epochs"].append(val_eval)
        history["train_nll"].append(running_loss / max(step_count, 1))
        history["validation_nll"].append(val_eval["crit"])
        print(
            f"S1 epoch {epoch + 1}/{cfg.phase1_epochs}: crit={val_eval['crit']:.4f} "
            f"result_top1={val_eval['result_top1']:.3f} val_nll={val_eval['validation_nll']:.4f}"
        )

        if val_eval["crit"] <= best_crit:
            best_crit = val_eval["crit"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.steer_nets.state_dict().items()
            }

        if (
            cfg.on_policy
            and tokenizer is not None
            and (epoch + 1) == cfg.on_policy_start_epoch
            and len(rows) > 0
            and "question" in rows[0]
            and "final_num" in rows[0]
        ):
            new_rows = collect_onpolicy_data(model, tokenizer, rows, device, cfg)
            if new_rows:
                new_lengths = [int(r["attention_mask"].sum()) for r in new_rows]
                rows.extend(new_rows)
                lengths.extend(new_lengths)
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
