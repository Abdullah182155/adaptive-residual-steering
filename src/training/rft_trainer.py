"""Plain token-mean NLL trainer for positive rejection-sampled rollouts."""

from __future__ import annotations

import gc
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
from src.training.batching import prepare_batch as _prepare, epoch_batches as _epoch_batches
from src.training.optimization import (
    steering_parameter_groups as _param_groups,
    warmup_cosine_lr as _lr_lambda,
)
from src.training.checkpoints import checkpoint_metadata

_STAGE1_ROUTE = [17, 20, 23, 26]
_STAGE2_ROUTE = [12, 15, 18, 21]
_FEW_SHOT_SELECTION_TOLERANCE_PP = 2.0


def _plain_nll_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return summed causal NLL over completion labels, excluding prompt/padding."""
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.size(-1)),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
        reduction="sum",
    )


def _run_rft_window(
    model: torch.nn.Module,
    train_rows: Sequence[dict[str, torch.Tensor]],
    microbatches: Sequence[Sequence[int]],
    token_count: int,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> float:
    """Run one accumulation window while materializing only one GPU batch at a time."""
    window_loss = 0.0
    for indices in microbatches:
        batch, extras = _prepare([train_rows[index] for index in indices], device)
        model.set_prompt_shield_mask(extras.get("prompt_shield_mask"))
        with torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
            loss = _plain_nll_sum(logits, batch["labels"]) / token_count
        scaler.scale(loss).backward()
        window_loss += float(loss.detach())
        del loss, logits, batch, extras
    model.set_prompt_shield_mask(None)
    return window_loss


def _attempt_rft_window(
    model: torch.nn.Module,
    train_rows: Sequence[dict[str, torch.Tensor]],
    microbatches: Sequence[Sequence[int]],
    token_count: int,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> tuple[Optional[float], bool]:
    """Catch OOM in a short-lived frame so its traceback releases GPU tensors."""
    try:
        return (
            _run_rft_window(
                model, train_rows, microbatches, token_count, device, scaler
            ),
            False,
        )
    except torch.cuda.OutOfMemoryError:
        return None, True


def _enable_checkpointing_without_dropout(backbone: torch.nn.Module) -> int:
    """Keep the frozen backbone deterministic while activating its checkpointed layers."""
    backbone.eval()
    checkpointed_modules = [
        module
        for module in backbone.modules()
        if bool(getattr(module, "gradient_checkpointing", False))
    ]
    # Transformers only checkpoints layers whose own training flag is true.
    # Flip that flag without recursively enabling dropout in their children.
    for module in checkpointed_modules:
        module.training = True
    return len(checkpointed_modules)


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
    intervention_gain: float = 1.0,
) -> tuple[list[bool], dict[str, float]]:
    """Evaluate paired greedy generations and return lenient flags and diagnostics."""
    selected = list(rows[:limit] if limit is not None else rows)
    if not selected:
        raise ValueError("generation evaluation requires at least one row")
    model.eval()
    model.set_routing_override("only" if steered else "none", cfg.fixed_layers)
    if hasattr(model, "set_intervention_gain"):
        model.set_intervention_gain(1.0)
    flags: list[bool] = []
    strict: list[bool] = []
    lengths: list[int] = []
    for start in range(0, len(selected), batch_size):
        batch_rows = selected[start : start + batch_size]
        if steered and hasattr(model, "set_intervention_gains"):
            model.set_intervention_gains(
                torch.full((len(batch_rows),), float(intervention_gain), dtype=torch.float32)
            )
        indices = list(range(start, start + len(batch_rows)))
        prompts, shields = build_prompts(
            tokenizer,
            [row["question"] for row in batch_rows],
            indices,
            n_shot,
            policy=cfg.shield_policy,
            template_ids=(
                [index % 3 for index in indices]
                if getattr(cfg, "diversify_prompts", False)
                else None
            ),
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
    steernet_arch = getattr(cfg, "steernet_arch", "v5")
    expected_route = _STAGE2_ROUTE if steernet_arch == "v2" else _STAGE1_ROUTE
    if list(cfg.fixed_layers) != expected_route:
        raise ValueError(f"{steernet_arch} RFT route is locked to {expected_route}")
    if cfg.use_gate:
        raise ValueError("Phase 1 RFT requires use_gate=False")
    if getattr(cfg, "steernet_use_mag_head", False):
        raise ValueError("Phase 1 RFT pins magnitude to identity (m=1.0)")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if any(parameter.requires_grad for parameter in model.base_model.parameters()):
        raise ValueError("Phi-2 backbone must remain fully frozen during RFT")
    if any(getattr(network, "use_mag_head", False) for network in model.steer_nets.values()):
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
    best_2shot_delta = float("-inf")
    best_selection_score = float("-inf")
    best_selection_feasible = False
    best_epoch: Optional[int] = None
    best_length_ratio: Optional[float] = None
    best_state = None
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        model.train()
        checkpointed_modules = _enable_checkpointing_without_dropout(model.base_model)
        if target_device.type == "cuda" and checkpointed_modules == 0:
            raise RuntimeError(
                "CUDA RFT requires active gradient-checkpointed backbone layers; "
                "the installed Transformers model exposes none"
            )
        if epoch == 0 and target_device.type == "cuda":
            print(
                f"[rft] gradient checkpointing active on {checkpointed_modules} "
                "backbone module(s); backbone dropout remains disabled",
                flush=True,
            )
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
            token_count = sum(
                int(train_rows[index]["labels"][1:].ne(-100).sum())
                for batch_indices in window
                for index in batch_indices
            )
            if token_count == 0:
                raise ValueError("RFT batch window has no completion tokens")
            window_loss, paired_oom = _attempt_rft_window(
                model,
                train_rows,
                window,
                token_count,
                target_device,
                scaler,
            )
            if paired_oom:
                # A single unusually long pair can exceed the T4 activation peak.
                # Discard this partial accumulation window and replay its examples
                # individually after the failed forward/backward traceback is gone.
                optimizer.zero_grad(set_to_none=True)
                model.set_prompt_shield_mask(None)
                model.reset_steernet_cache()
                gc.collect()
                if target_device.type == "cuda":
                    torch.cuda.empty_cache()
                singleton_microbatches = [
                    [index] for batch_indices in window for index in batch_indices
                ]
                print(
                    "[rft] CUDA OOM in paired microbatch; replaying this accumulation "
                    f"window as {len(singleton_microbatches)} single-example batches",
                    flush=True,
                )
                window_loss, singleton_oom = _attempt_rft_window(
                    model,
                    train_rows,
                    singleton_microbatches,
                    token_count,
                    target_device,
                    scaler,
                )
                if singleton_oom:
                    raise torch.cuda.OutOfMemoryError(
                        "RFT still exceeds CUDA memory with single-example microbatches"
                    )
            assert window_loss is not None
            loss_sum += window_loss
            progress.update(len(window))
            progress.set_postfix(
                nll=f"{window_loss:.4f}",
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
        current_2shot_flags, two_shot_diagnostics = evaluate_generation(
            model, tokenizer, eval_rows, cfg, target_device, n_shot=2, steered=True
        )
        delta_2shot_pp = 100 * sum(
            int(current) - int(baseline)
            for current, baseline in zip(current_2shot_flags, baseline_2shot_flags)
        ) / len(baseline_2shot_flags)
        baseline_mean_tokens = baseline_metrics["mean_tokens"]
        length_ratio = (
            diagnostics["mean_tokens"] / baseline_mean_tokens
            if baseline_mean_tokens > 0
            else 1.0
        )
        length_ok = length_ratio >= 0.8
        row = {
            "epoch": epoch + 1,
            "mean_window_nll": loss_sum / max(optimizer_steps, 1),
            "delta_lenient_pp": delta_pp,
            "delta_2shot_lenient_pp": delta_2shot_pp,
            "length_ok": length_ok,
            "completion_length_ratio": length_ratio,
            **diagnostics,
            "two_shot_mean_tokens": two_shot_diagnostics["mean_tokens"],
        }
        history.append(row)
        print(
            f"[rft] epoch={epoch + 1}/{epochs} nll={row['mean_window_nll']:.4f} "
            f"delta={delta_pp:+.2f}pp mean_tokens={diagnostics['mean_tokens']:.1f} "
            f"length_ratio={length_ratio:.2f} length_ok={length_ok} "
            f"delta_2shot={delta_2shot_pp:+.2f}pp"
        )
        # Generation stops as soon as a valid final answer is produced, so shorter
        # correct completions are not evidence of collapse. Keep length as a
        # diagnostic and select the checkpoint by the paired primary metric.
        # Prefer 0-shot gains while rejecting checkpoints with a material
        # few-shot regression; if every epoch violates the tolerance, choose
        # the best combined score and surface that fact in the run report.
        feasible = delta_2shot_pp >= -_FEW_SHOT_SELECTION_TOLERANCE_PP
        selection_score = delta_pp if feasible else delta_pp + min(delta_2shot_pp, 0.0)
        should_select = (
            (feasible and not best_selection_feasible)
            or (feasible == best_selection_feasible and selection_score > best_selection_score)
        )
        if should_select:
            best_selection_score = selection_score
            best_selection_feasible = feasible
            best_delta = delta_pp
            best_2shot_delta = delta_2shot_pp
            best_epoch = epoch + 1
            best_length_ratio = length_ratio
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.steer_nets.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("RFT completed without a selectable validation checkpoint")
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
        "selected_epoch": best_epoch,
        "best_delta_lenient_pp": best_delta,
        "selected_2shot_delta_lenient_pp": best_2shot_delta,
        "selected_2shot_noninferiority_passed": best_2shot_delta >= -_FEW_SHOT_SELECTION_TOLERANCE_PP,
        "selection_2shot_tolerance_pp": _FEW_SHOT_SELECTION_TOLERANCE_PP,
        "selected_completion_length_ratio": best_length_ratio,
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
