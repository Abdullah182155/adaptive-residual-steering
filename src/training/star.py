"""Phase 1R: Self-Taught Reasoning (STaR) for reasoning-targeted steering.

Trains SteerNet on the model's own correct solutions via rejection sampling,
with rationalization (hint-guided generation) for hard questions.
"""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from src.configs.config import RSCConfig
from src.training.losses import typed_objective, window_counts


def collect_star_completions(
    model, tokenizer, rows, device, cfg: RSCConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate k completions per question; return correct ones and hard questions."""
    from src.data.prompts import build_prompt_unified
    from src.evaluation.answers import score_tiers
    from src.training.trainer import _generate_onpolicy_batch
    import random

    rng = random.Random(cfg.seed + 777)
    candidates = [r for r in rows if "question" in r and "final_num" in r]

    questions = [r["question"] for r in candidates]
    golds = [r["final_num"] for r in candidates]
    shot_counts = [rng.choices([0, 1, 2], weights=[0.6, 0.2, 0.2])[0] for _ in questions]
    prompts = [build_prompt_unified(q, n_shot=ns) for q, ns in zip(questions, shot_counts)]

    gc_was_on = getattr(model.base_model, "is_gradient_checkpointing", False)
    if gc_was_on and hasattr(model.base_model, "gradient_checkpointing_disable"):
        model.base_model.gradient_checkpointing_disable()

    model.eval()
    model.set_routing_override("only", cfg.fixed_layers)

    k = cfg.star_k
    temp = cfg.star_temperature
    gen_bs = cfg.star_gen_batch

    completions_flat: List[str] = []
    total_batches = math.ceil(len(prompts) / gen_bs)
    for bi, i in enumerate(range(0, len(prompts), gen_bs)):
        batch = prompts[i : i + gen_bs]
        completions_flat.extend(
            _generate_onpolicy_batch(
                model, tokenizer, batch, device,
                num_return_sequences=k,
                temperature=temp,
            )
        )
        if (bi + 1) % 50 == 0 or (bi + 1) == total_batches:
            print(f"  STaR gen: {bi + 1}/{total_batches} batches", flush=True)

    if gc_was_on and hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    correct_data: List[Dict[str, Any]] = []
    hard_questions: List[Dict[str, Any]] = []
    n_correct_q = 0

    for qi, (q, gold, ns) in enumerate(zip(questions, golds, shot_counts)):
        try:
            gold_num = float(str(gold).replace(",", "").replace("$", "").strip())
        except (ValueError, AttributeError):
            continue

        base = qi * k
        found = False
        for ki in range(k):
            idx = base + ki
            if idx >= len(completions_flat):
                break
            comp = completions_flat[idx]
            if score_tiers(comp, gold_num)["lenient"]:
                correct_data.append({
                    "question": q,
                    "completion": comp,
                    "n_shot": ns,
                    "final_num": gold,
                    "source": "self",
                })
                found = True
                n_correct_q += 1
                break

        if not found:
            hard_questions.append({
                "question": q,
                "gold_answer": gold,
                "n_shot": ns,
            })

    print(
        f"[STaR] {len(candidates)} questions × {k} samples: "
        f"{n_correct_q} correct ({100 * n_correct_q / max(len(candidates), 1):.1f}%), "
        f"{len(hard_questions)} hard (need hint)"
    )
    return correct_data, hard_questions


def rationalize_hard_questions(
    model, tokenizer, hard_questions, device, cfg: RSCConfig,
) -> List[Dict[str, Any]]:
    """For hard questions, provide the gold answer as a hint and regenerate."""
    if not hard_questions:
        return []

    from src.data.prompts import build_prompt_unified
    from src.evaluation.answers import score_tiers
    from src.training.trainer import _generate_onpolicy_batch

    gc_was_on = getattr(model.base_model, "is_gradient_checkpointing", False)
    if gc_was_on and hasattr(model.base_model, "gradient_checkpointing_disable"):
        model.base_model.gradient_checkpointing_disable()

    model.eval()
    model.set_routing_override("only", cfg.fixed_layers)

    hint_k = cfg.star_hint_k
    temp = cfg.star_temperature
    gen_bs = cfg.star_gen_batch

    hinted_prompts = []
    for hq in hard_questions:
        hinted_q = f"{hq['question']}\n(Hint: The final answer is {hq['gold_answer']}.)"
        hinted_prompts.append(build_prompt_unified(hinted_q, n_shot=0))

    completions_flat: List[str] = []
    total_batches = math.ceil(len(hinted_prompts) / gen_bs)
    for bi, i in enumerate(range(0, len(hinted_prompts), gen_bs)):
        batch = hinted_prompts[i : i + gen_bs]
        completions_flat.extend(
            _generate_onpolicy_batch(
                model, tokenizer, batch, device,
                num_return_sequences=hint_k,
                temperature=temp,
            )
        )
        if (bi + 1) % 50 == 0 or (bi + 1) == total_batches:
            print(f"  STaR hint: {bi + 1}/{total_batches} batches", flush=True)

    if gc_was_on and hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    rationalized: List[Dict[str, Any]] = []
    for qi, hq in enumerate(hard_questions):
        try:
            gold_num = float(
                str(hq["gold_answer"]).replace(",", "").replace("$", "").strip()
            )
        except (ValueError, AttributeError):
            continue

        base = qi * hint_k
        for ki in range(hint_k):
            idx = base + ki
            if idx >= len(completions_flat):
                break
            comp = completions_flat[idx]
            if score_tiers(comp, gold_num)["lenient"]:
                rationalized.append({
                    "question": hq["question"],
                    "completion": comp,
                    "n_shot": hq["n_shot"],
                    "final_num": hq["gold_answer"],
                    "source": "hint",
                })
                break

    print(
        f"[STaR hint] {len(hard_questions)} hard → "
        f"{len(rationalized)} rationalized "
        f"({100 * len(rationalized) / max(len(hard_questions), 1):.1f}%)"
    )
    return rationalized


def train_phase1r_star(
    model,
    train_data,
    eval_data,
    cfg: RSCConfig,
    device: torch.device,
    tokenizer,
) -> Dict[str, Any]:
    """Phase 1R: Self-Taught Reasoning via rejection sampling + rationalization."""
    from src.training.trainer import (
        evaluate_typed,
        save_steer_weights,
        _param_groups,
        _prepare,
        _lr_lambda,
        _epoch_batches,
        _tokenize_onpolicy,
    )

    print("=" * 60)
    print("Phase 1R: Self-Taught Reasoning (STaR)")
    print("=" * 60)

    history: Dict[str, Any] = {"rounds": []}
    best_val_nll = float("inf")
    best_state = None

    for round_idx in range(cfg.star_rounds):
        print(f"\n{'─' * 40}")
        print(f"STaR Round {round_idx + 1}/{cfg.star_rounds}")
        print(f"{'─' * 40}")

        # ── Step 1: Collect self-generated correct completions ──
        print(
            f"\n[1/3] Generating completions "
            f"(k={cfg.star_k}, temp={cfg.star_temperature})..."
        )
        correct_data, hard_questions = collect_star_completions(
            model, tokenizer, train_data, device, cfg
        )

        # ── Step 2: Rationalize hard questions ──
        print(f"\n[2/3] Rationalizing {len(hard_questions)} hard questions...")
        rationalized = rationalize_hard_questions(
            model, tokenizer, hard_questions, device, cfg
        )

        all_data = correct_data + rationalized
        if not all_data:
            print("[STaR] No training data collected. Skipping round.")
            continue

        print(
            f"\n[3/3] Training: {cfg.star_epochs} epochs on {len(all_data)} examples "
            f"({len(correct_data)} self + {len(rationalized)} hint)"
        )

        # ── Tokenize ──
        tokenized_rows: List[dict] = []
        for item in all_data:
            row = _tokenize_onpolicy(
                tokenizer, item["question"], item["completion"],
                cfg, n_shot=item["n_shot"],
            )
            row["question"] = item["question"]
            row["final_num"] = item["final_num"]
            tokenized_rows.append(row)

        # ── Setup training ──
        model.set_routing_override("only", cfg.fixed_layers)
        if device.type == "cuda" and hasattr(
            model.base_model, "gradient_checkpointing_enable"
        ):
            model.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        groups = _param_groups(model, cfg)
        for g in groups:
            g["lr"] = cfg.star_lr
        _betas = (0.9, 0.98) if cfg.loss_mode in ("legacy", "legacy_pure") else (0.9, 0.999)
        optimizer = torch.optim.AdamW(groups, betas=_betas, eps=1e-6)

        lengths = [int(r["attention_mask"].sum()) for r in tokenized_rows]
        n_batches = math.ceil(len(tokenized_rows) / cfg.batch_size)
        total_steps = math.ceil(n_batches / cfg.grad_accum) * cfg.star_epochs
        warm_steps = max(1, int(0.08 * total_steps))
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

        round_history: Dict[str, Any] = {
            "round": round_idx + 1,
            "n_self": len(correct_data),
            "n_hint": len(rationalized),
            "n_total": len(all_data),
            "epochs": [],
        }

        for epoch in range(cfg.star_epochs):
            model.train()
            model.base_model.eval()
            optimizer.zero_grad(set_to_none=True)

            batches = _epoch_batches(
                lengths, cfg.batch_size, epoch, cfg.seed + round_idx * 100
            )
            window_steps = list(range(0, len(batches), cfg.grad_accum))
            n_windows = len(window_steps)

            sum_loss = 0.0
            n_steps = 0

            for wi, w0 in enumerate(window_steps):
                micro_indices = batches[w0 : w0 + cfg.grad_accum]
                micro = [
                    _prepare([tokenized_rows[i] for i in idx], device)
                    for idx in micro_indices
                ]
                counts = window_counts(
                    [b["labels"] for b, _ in micro],
                    [
                        e.get("token_types", torch.zeros_like(b["labels"]))
                        for b, e in micro
                    ],
                )

                for batch, extras in micro:
                    model.set_prompt_shield_mask(extras.get("prompt_shield_mask"))
                    with torch.autocast(
                        device_type=device.type, enabled=device.type == "cuda"
                    ):
                        base_logits = None
                        _need_base = (cfg.kl_weight > 0
                                      or getattr(cfg, "focal_gamma", 0) > 0)
                        if _need_base:
                            with torch.no_grad(), model.hooks_disabled():
                                base_logits = model(
                                    input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                ).logits.cpu()

                        logits = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        ).logits

                        _tt = extras.get(
                            "token_types", torch.zeros_like(batch["labels"])
                        )
                        loss, _ = typed_objective(
                            logits, base_logits, batch["labels"],
                            _tt, cfg, counts,
                        )
                        loss = loss / len(micro)

                    scaler.scale(loss).backward()
                    sum_loss += float(loss.detach())

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.steer_nets.parameters(), cfg.clip_grad
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                n_steps += 1

                if (wi + 1) % max(1, n_windows // 4) == 0 or wi == n_windows - 1:
                    avg = sum_loss / max(n_steps, 1)
                    print(
                        f"  S1R r{round_idx + 1} e{epoch + 1}/{cfg.star_epochs} "
                        f"[{wi + 1}/{n_windows}] loss={avg:.4f}"
                    )

            val_eval = evaluate_typed(model, eval_data, device, cfg, steered=True)
            avg_loss = sum_loss / max(n_steps, 1)
            print(
                f"S1R round {round_idx + 1} epoch {epoch + 1}/{cfg.star_epochs}: "
                f"train_loss={avg_loss:.4f} "
                f"val_nll={val_eval['validation_nll']:.4f} "
                f"result_top1={val_eval['result_top1']:.3f}"
            )
            round_history["epochs"].append({"train_loss": avg_loss, **val_eval})

            if val_eval["crit"] <= best_val_nll:
                best_val_nll = val_eval["crit"]
                best_state = {
                    n: v.detach().cpu().clone()
                    for n, v in model.steer_nets.state_dict().items()
                }

            ckpt_path = (
                Path(cfg.checkpoint_dir)
                / f"steernet_s1r_r{round_idx + 1}_e{epoch + 1}.pt"
            )
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "steer_nets": {
                        n: v.detach().cpu().clone()
                        for n, v in model.steer_nets.state_dict().items()
                    },
                    "round": round_idx + 1,
                    "epoch": epoch + 1,
                    "val_nll": val_eval["validation_nll"],
                },
                ckpt_path,
            )

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        history["rounds"].append(round_history)

    if best_state is not None:
        model.steer_nets.load_state_dict(best_state)

    if device.type == "cuda" and hasattr(
        model.base_model, "gradient_checkpointing_disable"
    ):
        model.base_model.gradient_checkpointing_disable()

    save_steer_weights(model, cfg.lora_save_path, cfg)
    model.set_telemetry(True)

    report_path = Path(cfg.report_dir) / "star_log.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    history["best_val_nll"] = best_val_nll
    print(f"[S1R] best_val_nll={best_val_nll:.4f}")
    return history
