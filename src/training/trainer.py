"""Fixed-route SteerNet V5 trainer for ARS Phase 1 with V2 typed reasoning loss."""

from __future__ import annotations

from collections import defaultdict
import contextlib
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F


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
        {"params": up, "lr": lr, "weight_decay": cfg.weight_decay * 0.1},
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
    bs = max(1, getattr(cfg, "batch_size", 2))
    with ctx:
        for start in range(0, len(row_list), bs):
            chunk = row_list[start : start + bs]
            batch, extras = _prepare(chunk, device)
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


@torch.no_grad()
def _validate_generation(model, tokenizer, rows, device, cfg, n=50):
    """Generate on val questions, compute lenient accuracy for checkpoint selection."""
    from src.data.prompts import build_prompt_unified
    from src.evaluation.answers import score_tiers

    model.eval()
    model.set_routing_override("only", cfg.fixed_layers)

    gc_on = getattr(model.base_model, "is_gradient_checkpointing", False)
    if gc_on and hasattr(model.base_model, "gradient_checkpointing_disable"):
        model.base_model.gradient_checkpointing_disable()

    correct, total = 0, 0
    gen_bs = max(1, getattr(cfg, "on_policy_gen_batch", 1))
    sample = rows[: min(n, len(rows))]
    for i in range(0, len(sample), gen_bs):
        batch_rows = sample[i : i + gen_bs]
        prompts = [build_prompt_unified(r["question"], n_shot=0) for r in batch_rows]
        texts = _generate_onpolicy_batch(
            model, tokenizer, prompts, device,
            max_new_tokens=256, temperature=0.0,
        )
        for r, text in zip(batch_rows, texts):
            try:
                gold_val = float(
                    str(r.get("final_num", "")).replace(",", "").replace("$", "").strip()
                )
            except (ValueError, TypeError):
                continue
            total += 1
            if score_tiers(text, gold_val).get("lenient", False):
                correct += 1

    if gc_on and hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    acc = correct / max(total, 1)
    print(f"  [gen-val] {correct}/{total} = {100 * acc:.1f}% lenient (0-shot)")
    return acc


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
def _generate_onpolicy_batch(model, tokenizer, prompts, device, max_new_tokens=256,
                              num_return_sequences=1, temperature=0.0, max_prompt_length=512):
    enc = tokenizer(prompts, return_tensors="pt", truncation=True,
                    max_length=max_prompt_length, padding=True)
    inputs = {k: v.to(device) for k, v in enc.items()}
    model.set_prompt_shield_mask(None)
    do_sample = (num_return_sequences > 1 or temperature > 0.0)
    tokens = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        num_return_sequences=num_return_sequences,
    )
    B = len(prompts)
    # prompt lengths broadcast across the k sequences per prompt
    prompt_lens = inputs["attention_mask"].sum(dim=1).repeat_interleave(num_return_sequences)
    outputs = []
    for i in range(B * num_return_sequences):
        plen = int(prompt_lens[i])
        text = tokenizer.decode(tokens[i, plen:], skip_special_tokens=True)
        outputs.append(text)
    return outputs  # length = B * num_return_sequences, grouped as [prompt0_k0, prompt0_k1, ..., prompt1_k0, ...]


def _tokenize_onpolicy(tokenizer, question, completion, cfg, n_shot=0):
    from src.data.prompts import build_prompt_with_shield
    prompt_text, enc_full, shield_mask = build_prompt_with_shield(
        tokenizer, question, n_shot=n_shot, policy=getattr(cfg, "shield_policy", "context")
    )
    full_text = prompt_text + completion
    target_start = len(prompt_text)

    enc = tokenizer(full_text, truncation=True, max_length=cfg.max_tok_len,
                    padding="max_length", return_offsets_mapping=True)

    ids, offsets, am = enc["input_ids"], enc["offset_mapping"], enc["attention_mask"]
    labels = [tid if (m and e > s and s >= target_start) else -100
              for tid, (s, e), m in zip(ids, offsets, am)]
    # Compute shield mask aligned to the full tokenization
    shield_char_end = target_start if n_shot > 0 else 0
    shield = [1 if (m and e > 0 and e <= shield_char_end) else 0
              for (s, e), m in zip(offsets, am)]
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
def collect_onpolicy_data(model, tokenizer, rows, device, cfg, epoch=0):
    import random
    from src.evaluation.answers import extract_numeric_answer, score_tiers

    orig_indices = [i for i, r in enumerate(rows) if "question" in r]
    n_gen = max(1, int(len(orig_indices) * cfg.on_policy_frac))
    perm = torch.randperm(
        len(orig_indices), generator=torch.Generator().manual_seed(cfg.seed + epoch)
    )[:n_gen].tolist()
    indices = [orig_indices[j] for j in perm]

    from src.data.prompts import build_prompt_unified

    rng = random.Random(cfg.seed + epoch + 1000)
    questions = [rows[i]["question"] for i in indices]
    golds = [rows[i]["final_num"] for i in indices]
    gold_cots = [rows[i].get("answer", "") for i in indices]
    _on_shot_probs = cfg.train_shot_probs[:3]
    shot_counts = [rng.choices([0, 1, 2], weights=_on_shot_probs)[0] for _ in questions]
    prompts = [
        build_prompt_unified(q, n_shot=ns) for q, ns in zip(questions, shot_counts)
    ]

    gc_enabled = getattr(model.base_model, "is_gradient_checkpointing", False)
    if gc_enabled and hasattr(model.base_model, "gradient_checkpointing_disable"):
        model.base_model.gradient_checkpointing_disable()

    model.eval()
    model.set_routing_override("only", cfg.fixed_layers)

    # Pass@k: generate k completions per question
    on_policy_k = getattr(cfg, "on_policy_k", 1)
    on_policy_temperature = getattr(cfg, "on_policy_temperature", 0.0)

    completions_flat: List[str] = []
    gen_bs = cfg.on_policy_gen_batch
    total_batches = math.ceil(len(prompts) / gen_bs)
    for bi, i in enumerate(range(0, len(prompts), gen_bs)):
        batch = prompts[i : i + gen_bs]
        completions_flat.extend(
            _generate_onpolicy_batch(
                model, tokenizer, batch, device,
                num_return_sequences=on_policy_k,
                temperature=on_policy_temperature,
                max_prompt_length=cfg.max_tok_len,
            )
        )
        if (bi + 1) % 25 == 0 or (bi + 1) == total_batches:
            print(f"  on-policy gen: {bi + 1}/{total_batches} batches", flush=True)

    if gc_enabled and hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # completions_flat is grouped: [q0_k0, q0_k1, ..., q1_k0, q1_k1, ...]
    # For each question: keep first correct completion, or fall back to gold CoT
    new_rows: List[Dict[str, Any]] = []
    n_tried = 0
    n_correct = 0
    n_gold_fallback = 0
    for qi, (q, gold, gold_cot, ns) in enumerate(zip(questions, golds, gold_cots, shot_counts)):
        try:
            gold_num = float(str(gold).replace(",", "").replace("$", "").strip())
        except (ValueError, AttributeError):
            continue
        base = qi * on_policy_k
        found = False
        for ki in range(on_policy_k):
            n_tried += 1
            comp = completions_flat[base + ki]
            scored = score_tiers(comp, gold_num)
            if scored["lenient"]:
                n_correct += 1
                row = _tokenize_onpolicy(tokenizer, q, comp, cfg, n_shot=ns)
                row["question"] = q
                row["final_num"] = gold
                new_rows.append(row)
                found = True
                break
        if not found and gold_cot and getattr(cfg, "on_policy_gold_fallback", True):
            row = _tokenize_onpolicy(tokenizer, q, gold_cot, cfg, n_shot=ns)
            row["question"] = q
            row["final_num"] = gold
            new_rows.append(row)
            n_gold_fallback += 1

    print(
        f"[on-policy] {n_gen} questions × {on_policy_k} samples = {n_tried} tried "
        f"→ {n_correct} correct ({100 * n_correct / max(n_gen, 1):.1f}%) "
        f"+ {n_gold_fallback} gold fallback → {len(new_rows)} new rows"
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
                                q_flat = q[:, :T_use, 0]
                                tt_use = tt[:, :T_use]
                                type_mult = torch.ones_like(q_flat)
                                type_mult[tt_use == T_RESULT] = 0.1
                                type_mult[tt_use == T_FINAL] = 0.1
                                type_mult[tt_use == T_OPERAND] = 0.3
                                type_mult[tt_use == T_OPERATOR] = 0.3
                                type_mult[tt_use == T_PROSE] = 1.0
                                type_mult[tt_use == T_FORMAT] = 3.0
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
