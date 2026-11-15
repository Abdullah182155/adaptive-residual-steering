"""Plain token-mean NLL trainer for positive rejection-sampled rollouts."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.evaluation.answers import score_tiers
from src.evaluation.generation import build_prompts, generate_batch
from src.training.trainer import (
    _epoch_batches,
    _lr_lambda,
    _param_groups,
    _prepare,
    checkpoint_metadata,
)

_STAGE1_ROUTE = [17, 20, 23, 26]


def _plain_nll_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return summed causal NLL over completion labels, excluding prompt/padding."""
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.size(-1)),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
        reduction="sum",
    )


@torch.no_grad()
def evaluate_generation(
    model,
    tokenizer,
    rows: Sequence[dict],
    cfg,
    device: torch.device | str,
    *,
    n_shot: int = 0,
    batch_size: int = 2,
    limit: Optional[int] = None,
    steered: bool,
) -> tuple[list[bool], dict[str, float]]:
    """Evaluate paired greedy generations and return lenient flags and diagnostics."""
    selected = list(rows[:limit] if limit is not None else rows)
    if not selected:
        raise ValueError("generation evaluation requires at least one row")
    model.eval()
    model.set_routing_override("only" if steered else "none", cfg.fixed_layers)
    flags: list[bool] = []
    strict: list[bool] = []
    lengths: list[int] = []
    for start in range(0, len(selected), batch_size):
        batch_rows = selected[start : start + batch_size]
        indices = list(range(start, start + len(batch_rows)))
        prompts, shields = build_prompts(
            tokenizer,
            [row["question"] for row in batch_rows],
            indices,
            n_shot,
            policy=cfg.shield_policy,
        )
        results = generate_batch(
            model,
            tokenizer,
            prompts,
            device,
            prompt_shields=shields,
            max_new_tokens=cfg.generation_max_new_tokens,
            max_prompt_len=max(cfg.max_tok_len, 1024),
            repetition_penalty=1.0,
        )
        for row, result in zip(batch_rows, results):
            gold = float(
                str(row.get("final_num", row.get("answer")))
                .replace(",", "")
                .replace("$", "")
                .strip()
            )
            score = score_tiers(result.text, gold)
            flags.append(bool(score["lenient"]))
            strict.append(bool(score["strict"]))
            lengths.append(result.n_new_tokens)
    metrics = {
        "lenient_accuracy": 100 * sum(flags) / len(flags),
        "strict_accuracy": 100 * sum(strict) / len(strict),
        "mean_tokens": sum(lengths) / len(lengths),
    }
    return flags, metrics


def _save_rft_checkpoint(model, path: str | Path, cfg) -> None:
    """Persist the trained adapter with RFT objective and route provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = checkpoint_metadata(cfg)
    metadata.update(
        {
            "loss_mode": "rft_token_mean_nll",
            "route": list(cfg.fixed_layers),
            "steernet_use_mag_head": False,
            "phase1_lr": cfg.phase1_lr,
            "rho_max": cfg.rho_max,
            "lora_rank": cfg.lora_rank,
            "shield_policy": cfg.shield_policy,
        }
    )
    torch.save(
        {
            "metadata": metadata,
            "steer_nets": {
                name: value.detach().cpu().clone()
                for name, value in model.steer_nets.state_dict().items()
            },
        },
        path,
    )


def train_phase1_rft(
    model,
    train_rows: Sequence[dict[str, torch.Tensor]],
    dev_rows: Sequence[dict],
    cfg,
    device: torch.device | str,
    tokenizer,
    *,
    epochs: int = 2,
    dev_limit: int = 100,
    save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Train frozen-backbone SteerNet on positive RFT completions using plain NLL."""
    cfg.validate()
    if not train_rows:
        raise ValueError("train_rows cannot be empty")
    if list(cfg.fixed_layers) != _STAGE1_ROUTE:
        raise ValueError(f"Phase 1 RFT route is locked to {_STAGE1_ROUTE}")
    if cfg.use_gate:
        raise ValueError("Phase 1 RFT requires use_gate=False")
    if cfg.steernet_use_mag_head:
        raise ValueError("Phase 1 RFT pins magnitude to identity (m=1.0)")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if any(parameter.requires_grad for parameter in model.base_model.parameters()):
        raise ValueError("Phi-2 backbone must remain fully frozen during RFT")
    if any(network.use_mag_head for network in model.steer_nets.values()):
        raise ValueError("SteerNet magnitude heads must be disabled for Phase 1 RFT")
    trainable_parameters = [
        parameter
        for parameter in model.steer_nets.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("Phase 1 RFT requires trainable SteerNet parameters")

    target_device = torch.device(device)
    model.set_routing_override("only", cfg.fixed_layers)
    if target_device.type == "cuda" and hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    eval_rows = list(dev_rows[:dev_limit])
    baseline_flags, baseline_metrics = evaluate_generation(
        model, tokenizer, eval_rows, cfg, target_device, n_shot=0, steered=False
    )
    baseline_2shot_flags, baseline_2shot_metrics = evaluate_generation(
        model, tokenizer, eval_rows, cfg, target_device, n_shot=2, steered=False
    )
    # Baseline evaluation disables steering; restore the training route before
    # the first loss forward so adapter parameters remain connected to the graph.
    model.set_routing_override("only", cfg.fixed_layers)
    optimizer = torch.optim.AdamW(
        _param_groups(model, cfg), betas=(0.9, 0.999), eps=1e-6
    )
    lengths = [int(row["attention_mask"].sum()) for row in train_rows]
    batch_count = math.ceil(len(train_rows) / cfg.batch_size)
    steps_per_epoch = math.ceil(batch_count / cfg.grad_accum)
    total_steps = max(1, steps_per_epoch * epochs)
    warm_steps = max(1, int(cfg.phase1_warmup_frac * total_steps))
    state = {"step": 0}
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda _: _lr_lambda(
            state["step"], total_steps, warm_steps, cfg.phase1_min_lr_frac
        ),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=target_device.type == "cuda"
        and next(model.base_model.parameters()).dtype == torch.float16,
    )

    best_delta = float("-inf")
    best_state = None
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        model.train()
        model.base_model.eval()
        optimizer.zero_grad(set_to_none=True)
        batches = _epoch_batches(lengths, cfg.batch_size, epoch, cfg.seed)
        loss_sum = 0.0
        optimizer_steps = 0
        progress = tqdm(
            total=len(batches),
            desc=f"RFT epoch {epoch + 1}/{epochs}",
            unit="batch",
            file=sys.stdout,
            disable=False,
            mininterval=0.5,
        )
        for window_start in range(0, len(batches), cfg.grad_accum):
            window = batches[window_start : window_start + cfg.grad_accum]
            micro_batches = [
                _prepare([train_rows[index] for index in batch_indices], target_device)
                for batch_indices in window
            ]
            token_count = sum(
                int(batch["labels"][:, 1:].ne(-100).sum())
                for batch, _ in micro_batches
            )
            if token_count == 0:
                raise ValueError("RFT batch window has no completion tokens")
            for batch, extras in micro_batches:
                model.set_prompt_shield_mask(extras.get("prompt_shield_mask"))
                with torch.autocast(
                    device_type=target_device.type,
                    enabled=target_device.type == "cuda",
                ):
                    logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).logits
                    loss = _plain_nll_sum(logits, batch["labels"]) / token_count
                scaler.scale(loss).backward()
                loss_sum += float(loss.detach())
                progress.update(1)
                progress.set_postfix(
                    nll=f"{float(loss.detach()):.4f}",
                    updates=optimizer_steps,
                )
                sys.stdout.flush()
            scaler.unscale_(optimizer)
            if not any(parameter.grad is not None for parameter in trainable_parameters):
                raise RuntimeError(
                    "RFT produced no gradients for trainable SteerNet parameters; "
                    "verify that the fixed steering route is active"
                )
            torch.nn.utils.clip_grad_norm_(model.steer_nets.parameters(), cfg.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            state["step"] += 1
            scheduler.step()
            optimizer_steps += 1
        progress.close()

        current_flags, diagnostics = evaluate_generation(
            model, tokenizer, eval_rows, cfg, target_device, n_shot=0, steered=True
        )
        delta_pp = 100 * sum(
            int(current) - int(baseline)
            for current, baseline in zip(current_flags, baseline_flags)
        ) / len(baseline_flags)
        length_ok = diagnostics["mean_tokens"] >= 0.8 * baseline_metrics["mean_tokens"]
        row = {
            "epoch": epoch + 1,
            "mean_window_nll": loss_sum / max(optimizer_steps, 1),
            "delta_lenient_pp": delta_pp,
            "length_ok": length_ok,
            **diagnostics,
        }
        history.append(row)
        print(
            f"[rft] epoch={epoch + 1}/{epochs} nll={row['mean_window_nll']:.4f} "
            f"delta={delta_pp:+.2f}pp mean_tokens={diagnostics['mean_tokens']:.1f} "
            f"length_ok={length_ok}"
        )
        if length_ok and delta_pp > best_delta:
            best_delta = delta_pp
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.steer_nets.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("no RFT checkpoint met the 0.8x completion-length veto")
    model.steer_nets.load_state_dict(best_state)
    if target_device.type == "cuda" and hasattr(model.base_model, "gradient_checkpointing_disable"):
        model.base_model.gradient_checkpointing_disable()
    checkpoint_path = save_path or cfg.lora_save_path
    _save_rft_checkpoint(model, checkpoint_path, cfg)
    two_shot_flags, two_shot_metrics = evaluate_generation(
        model, tokenizer, eval_rows, cfg, target_device, n_shot=2, steered=True
    )
    two_shot_metrics["delta_lenient_pp"] = 100 * sum(
        int(current) - int(baseline)
        for current, baseline in zip(two_shot_flags, baseline_2shot_flags)
    ) / len(baseline_2shot_flags)
    result = {
        "route": list(cfg.fixed_layers),
        "best_delta_lenient_pp": best_delta,
        "baseline_0shot": baseline_metrics,
        "baseline_2shot": baseline_2shot_metrics,
        "selected_2shot": two_shot_metrics,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    report_path = Path(cfg.report_dir) / "rft_training_log.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
