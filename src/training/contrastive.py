"""Phase 1C: Contrastive Residual Training for reasoning-targeted steering.

Trains SteerNet deltas to align with the direction between correct and
incorrect reasoning traces in activation space, focusing on questions
the model currently gets wrong (hard-example curriculum).
"""

from __future__ import annotations

import gc
import math
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from src.configs.config import RSCConfig
from src.training.losses import (
    annotate_char_types,
    token_types_from_offsets,
    typed_objective,
    window_counts,
    T_NONE,
)


def _tokenize_pair_item(tokenizer, question, completion, cfg, n_shot=0):
    """Tokenize a question+completion for direction extraction."""
    from src.training.trainer import _tokenize_onpolicy
    return _tokenize_onpolicy(tokenizer, question, completion, cfg, n_shot=n_shot)


def collect_contrastive_pairs(
    model, tokenizer, rows, device, cfg: RSCConfig,
) -> List[Dict[str, Any]]:
    """Generate greedy completions and collect pairs where the model is wrong."""
    from src.data.prompts import build_prompt_unified
    from src.evaluation.answers import score_tiers
    from src.training.trainer import _generate_onpolicy_batch
    import random

    rng = random.Random(cfg.seed + 999)
    candidates = [r for r in rows if "question" in r and "final_num" in r]

    frac = getattr(cfg, "contrastive_frac", 1.0)
    if frac < 1.0:
        n_use = max(1, int(len(candidates) * frac))
        perm = torch.randperm(len(candidates), generator=torch.Generator().manual_seed(cfg.seed))
        candidates = [candidates[i] for i in perm[:n_use].tolist()]

    questions = [r["question"] for r in candidates]
    golds = [r["final_num"] for r in candidates]
    gold_cots = [r.get("answer", "") for r in candidates]
    shot_counts = [rng.choices([0, 1, 2], weights=[0.6, 0.2, 0.2])[0] for _ in questions]
    prompts = [build_prompt_unified(q, n_shot=ns) for q, ns in zip(questions, shot_counts)]

    gc_was_on = getattr(model.base_model, "is_gradient_checkpointing", False)
    if gc_was_on and hasattr(model.base_model, "gradient_checkpointing_disable"):
        model.base_model.gradient_checkpointing_disable()

    model.eval()
    model.set_routing_override("only", cfg.fixed_layers)

    k = getattr(cfg, "contrastive_n_samples", 1)
    temp = getattr(cfg, "contrastive_temperature", 0.0)
    gen_bs = getattr(cfg, "contrastive_gen_batch", 4)

    completions_flat: List[str] = []
    total_batches = math.ceil(len(prompts) / gen_bs)
    for bi, i in enumerate(range(0, len(prompts), gen_bs)):
        batch = prompts[i : i + gen_bs]
        completions_flat.extend(
            _generate_onpolicy_batch(
                model, tokenizer, batch, device,
                num_return_sequences=k,
                temperature=temp if temp > 0 else 1.0,
            )
            if temp > 0
            else _generate_onpolicy_batch(
                model, tokenizer, batch, device,
                num_return_sequences=1,
                temperature=0.0,
            )
        )
        if (bi + 1) % 50 == 0 or (bi + 1) == total_batches:
            print(f"  contrastive gen: {bi + 1}/{total_batches} batches", flush=True)

    if gc_was_on and hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    pairs: List[Dict[str, Any]] = []
    n_correct = 0
    for qi, (q, gold, gold_cot, ns) in enumerate(zip(questions, golds, gold_cots, shot_counts)):
        if not gold_cot:
            continue
        try:
            gold_num = float(str(gold).replace(",", "").replace("$", "").strip())
        except (ValueError, AttributeError):
            continue

        base = qi * k
        wrong_comp = None
        for ki in range(k):
            idx = base + ki
            if idx >= len(completions_flat):
                break
            comp = completions_flat[idx]
            scored = score_tiers(comp, gold_num)
            if not scored["lenient"]:
                wrong_comp = comp
                break

        if wrong_comp is None:
            n_correct += 1
            continue

        pairs.append({
            "question": q,
            "gold_completion": gold_cot,
            "wrong_completion": wrong_comp,
            "n_shot": ns,
        })

    print(
        f"[contrastive] {len(candidates)} questions: "
        f"{n_correct} correct (skipped), {len(pairs)} wrong (contrastive pairs)"
    )
    return pairs


@torch.no_grad()
def compute_layer_directions(
    model, tokenizer, pairs, device, cfg: RSCConfig,
) -> List[Dict[str, Any]]:
    """Extract hidden states and compute per-layer direction vectors."""
    result = []
    total = len(pairs)

    for pi, pair in enumerate(pairs):
        q = pair["question"]
        ns = pair["n_shot"]

        gold_row = _tokenize_pair_item(tokenizer, q, pair["gold_completion"], cfg, n_shot=ns)
        wrong_row = _tokenize_pair_item(tokenizer, q, pair["wrong_completion"], cfg, n_shot=ns)

        gold_ids = gold_row["input_ids"].unsqueeze(0).to(device)
        gold_mask = gold_row["attention_mask"].unsqueeze(0).to(device)
        gold_labels = gold_row["labels"].unsqueeze(0)

        wrong_ids = wrong_row["input_ids"].unsqueeze(0).to(device)
        wrong_mask = wrong_row["attention_mask"].unsqueeze(0).to(device)
        wrong_labels = wrong_row["labels"].unsqueeze(0)

        gold_comp_mask = gold_labels[:, 1:].ne(-100).squeeze(0)
        wrong_comp_mask = wrong_labels[:, 1:].ne(-100).squeeze(0)

        with model.capture_base_hidden() as gold_hidden:
            model(input_ids=gold_ids, attention_mask=gold_mask)

        with model.capture_base_hidden() as wrong_hidden:
            model(input_ids=wrong_ids, attention_mask=wrong_mask)

        layer_dirs = {}
        for layer_idx in model.target_layers:
            if layer_idx not in gold_hidden or layer_idx not in wrong_hidden:
                continue
            h_g = gold_hidden[layer_idx]
            h_w = wrong_hidden[layer_idx]

            T_g = min(gold_comp_mask.shape[0], h_g.shape[1] - 1)
            T_w = min(wrong_comp_mask.shape[0], h_w.shape[1] - 1)
            m_g = gold_comp_mask[:T_g]
            m_w = wrong_comp_mask[:T_w]

            if m_g.any() and m_w.any():
                avg_g = h_g[0, 1 : T_g + 1][m_g].mean(dim=0)
                avg_w = h_w[0, 1 : T_w + 1][m_w].mean(dim=0)
                direction = avg_g - avg_w
                dn = direction.norm().clamp_min(1e-6)
                layer_dirs[layer_idx] = (direction / dn).cpu()
            else:
                H = h_g.shape[-1]
                layer_dirs[layer_idx] = torch.zeros(H)

        result.append({
            **pair,
            "directions": layer_dirs,
            "tokenized_gold": gold_row,
        })

        if (pi + 1) % 200 == 0 or (pi + 1) == total:
            print(f"  directions: {pi + 1}/{total}", flush=True)

    return result


def contrastive_cosine_loss(
    captured_deltas: Dict[int, torch.Tensor],
    target_directions: Dict[int, torch.Tensor],
    comp_mask: torch.BoolTensor,
    device: torch.device,
) -> torch.Tensor:
    """Negative cosine similarity between captured deltas and target directions."""
    loss = torch.zeros((), device=device)
    n = 0

    for layer_idx, delta in captured_deltas.items():
        if layer_idx not in target_directions:
            continue
        td = target_directions[layer_idx].to(device)
        B, T, H = delta.shape
        T_m = min(comp_mask.shape[1], T)
        if T_m == 0 or not comp_mask[:, :T_m].any():
            continue

        mask_f = comp_mask[:, :T_m].unsqueeze(-1).float()
        avg = (delta[:, :T_m] * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1)

        if td.ndim == 1:
            td = td.unsqueeze(0).expand_as(avg)

        cos = F.cosine_similarity(avg, td, dim=-1)
        loss = loss - cos.mean()
        n += 1

    return loss / max(n, 1)


def train_phase1c_contrastive(
    model,
    train_data,
    eval_data,
    cfg: RSCConfig,
    device: torch.device,
    tokenizer,
) -> Dict[str, Any]:
    """Phase 1C: Contrastive Residual Training on hard examples."""
    from src.training.trainer import (
        evaluate_typed,
        save_steer_weights,
        _param_groups,
        _prepare,
        _lr_lambda,
    )

    print("=" * 60)
    print("Phase 1C: Contrastive Residual Training")
    print("=" * 60)

    # ── Step 1: Generate contrastive pairs ──
    print("\n[1/3] Generating contrastive pairs...")
    pairs = collect_contrastive_pairs(model, tokenizer, train_data, device, cfg)
    if not pairs:
        print("[contrastive] No wrong completions — model is already correct. Skipping.")
        return {"contrastive_pairs": 0}

    # ── Step 2: Compute direction vectors ──
    print(f"\n[2/3] Computing direction vectors ({len(pairs)} pairs)...")
    contrastive_data = compute_layer_directions(model, tokenizer, pairs, device, cfg)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Step 3: Train ──
    n_data = len(contrastive_data)
    print(f"\n[3/3] Training: {cfg.contrastive_epochs} epochs, {n_data} pairs")

    model.set_routing_override("only", cfg.fixed_layers)

    groups = _param_groups(model, cfg)
    for g in groups:
        g["lr"] = cfg.contrastive_lr
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999))

    bs = cfg.batch_size
    n_batches = math.ceil(n_data / bs)
    total_steps = n_batches * cfg.contrastive_epochs
    warm_steps = max(1, int(0.1 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: _lr_lambda(s, total_steps, warm_steps, 0.05),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
            and next(model.base_model.parameters()).dtype == torch.float16
        ),
    )

    best_val_nll = float("inf")
    best_state = None
    history: Dict[str, Any] = {"contrastive_pairs": n_data, "epochs": []}

    for epoch in range(cfg.contrastive_epochs):
        model.train()
        model.base_model.eval()
        optimizer.zero_grad(set_to_none=True)

        g = torch.Generator().manual_seed(cfg.seed + epoch)
        order = torch.randperm(n_data, generator=g).tolist()

        sum_cos = 0.0
        sum_nll = 0.0
        n_steps = 0

        for bi in range(0, len(order), bs):
            items = [contrastive_data[j] for j in order[bi : bi + bs]]
            micro_n = len(items)

            for item in items:
                gold_row = item["tokenized_gold"]
                directions = item["directions"]

                batch_dict, extras = _prepare([gold_row], device)
                model.set_prompt_shield_mask(extras.get("prompt_shield_mask"))
                comp_mask = batch_dict["labels"][:, 1:].ne(-100)

                with torch.autocast(
                    device_type=device.type, enabled=device.type == "cuda"
                ):
                    model._capture_deltas = True
                    model._captured_deltas = {}

                    logits = model(
                        input_ids=batch_dict["input_ids"],
                        attention_mask=batch_dict["attention_mask"],
                    ).logits

                    captured = dict(model._captured_deltas)
                    model._capture_deltas = False
                    model._captured_deltas = {}

                    cos_loss = contrastive_cosine_loss(
                        captured, directions, comp_mask, device
                    )

                    nll_loss = torch.zeros((), device=device)
                    if cfg.contrastive_nll_weight > 0:
                        types = extras.get(
                            "token_types",
                            torch.zeros_like(batch_dict["labels"]),
                        )
                        counts = window_counts([batch_dict["labels"]], [types])
                        nll_loss, _ = typed_objective(
                            logits,
                            None,
                            batch_dict["labels"],
                            types,
                            cfg,
                            counts,
                        )

                    loss = (
                        cfg.contrastive_weight * cos_loss
                        + cfg.contrastive_nll_weight * nll_loss
                    ) / micro_n

                scaler.scale(loss).backward()
                sum_cos += float(cos_loss.detach())
                sum_nll += float(nll_loss.detach())

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.steer_nets.parameters(), cfg.clip_grad
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            n_steps += 1

        val_eval = evaluate_typed(model, eval_data, device, cfg, steered=True)
        avg_cos = sum_cos / max(n_steps * bs, 1)
        avg_nll = sum_nll / max(n_steps * bs, 1)

        print(
            f"S1C epoch {epoch + 1}/{cfg.contrastive_epochs}: "
            f"cos={avg_cos:.4f} nll_reg={avg_nll:.4f} "
            f"val_nll={val_eval['validation_nll']:.4f} "
            f"result_top1={val_eval['result_top1']:.3f}"
        )
        history["epochs"].append(
            {"cos_loss": avg_cos, "nll_reg": avg_nll, **val_eval}
        )

        if val_eval["validation_nll"] <= best_val_nll:
            best_val_nll = val_eval["validation_nll"]
            best_state = {
                n: v.detach().cpu().clone()
                for n, v in model.steer_nets.state_dict().items()
            }

        epoch_path = Path(cfg.checkpoint_dir) / f"steernet_s1c_epoch{epoch + 1}.pt"
        epoch_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "steer_nets": {
                    n: v.detach().cpu().clone()
                    for n, v in model.steer_nets.state_dict().items()
                },
                "epoch": epoch + 1,
                "val_nll": val_eval["validation_nll"],
            },
            epoch_path,
        )

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best_state is not None:
        model.steer_nets.load_state_dict(best_state)

    if device.type == "cuda" and hasattr(
        model.base_model, "gradient_checkpointing_disable"
    ):
        model.base_model.gradient_checkpointing_disable()

    save_steer_weights(model, cfg.lora_save_path, cfg)
    model.set_telemetry(True)

    report_path = Path(cfg.report_dir) / "contrastive_log.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    history["best_val_nll"] = best_val_nll
    print(f"[S1C] best_val_nll={best_val_nll:.4f}")
    return history
