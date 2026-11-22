"""Plain token-mean NLL trainer for positive rejection-sampled rollouts."""

from __future__ import annotations

import gc
import json
import math
import random
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

_LEGACY_FEATURE_ROUTE = [17, 20, 23, 26]
_FIXED_LOW_RANK_ROUTE = [12, 15, 18, 21]
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
        if hasattr(model, "set_prompt_conditioning_mask"):
            prompt_conditioning_mask = batch["attention_mask"].bool() & batch[
                "labels"
            ].eq(-100)
            model.set_prompt_conditioning_mask(prompt_conditioning_mask)
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
    if hasattr(model, "set_prompt_conditioning_mask"):
        model.set_prompt_conditioning_mask(None)
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
    prompt_strength_override: Optional[float] = None,
    progress_desc: Optional[str] = None,
) -> tuple[list[bool], dict[str, float]]:
    """Evaluate paired greedy generations and return lenient flags and diagnostics."""
    selected = list(rows[:limit] if limit is not None else rows)
    if not selected:
        raise ValueError("generation evaluation requires at least one row")
    model.eval()
    model.set_routing_override("only" if steered else "none", cfg.fixed_layers)
    if hasattr(model, "set_intervention_gain"):
        model.set_intervention_gain(1.0)
    if hasattr(model, "set_prompt_strength_override"):
        model.set_prompt_strength_override(prompt_strength_override)
    flags: list[bool] = []
    strict: list[bool] = []
    lengths: list[int] = []
    prompt_strengths: dict[int, list[float]] = {}
    progress = tqdm(
        total=len(selected),
        desc=progress_desc or "Generation evaluation",
        unit="question",
        file=sys.stdout,
        disable=progress_desc is None,
        mininterval=0.5,
        leave=False,
    )
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
        if steered and hasattr(model, "get_prompt_strengths"):
            for layer, values in model.get_prompt_strengths().items():
                prompt_strengths.setdefault(layer, []).extend(
                    float(value) for value in values.flatten().tolist()
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
        progress.update(len(batch_rows))
    progress.close()
    metrics = {
        "lenient_accuracy": 100 * sum(flags) / len(flags),
        "strict_accuracy": 100 * sum(strict) / len(strict),
        "mean_tokens": sum(lengths) / len(lengths),
    }
    if prompt_strengths:
        strength_summary = {}
        for layer, values in sorted(prompt_strengths.items()):
            if not values:
                continue
            mean_strength = sum(values) / len(values)
            variance = sum((value - mean_strength) ** 2 for value in values) / len(values)
            strength_summary[str(layer)] = {
                "mean": mean_strength,
                "std": variance**0.5,
                "min": min(values),
                "max": max(values),
            }
        metrics["prompt_strength_by_layer"] = strength_summary
    # Per-example gains are only valid for the evaluation batch that installed
    # them. Restore the scalar default before another batch size is used.
    if hasattr(model, "set_intervention_gain"):
        model.set_intervention_gain(1.0)
    if hasattr(model, "set_prompt_strength_override"):
        model.set_prompt_strength_override(None)
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
            "prompt_strength_head": cfg.steernet_arch == "prompt_conditioned_residual",
            "prompt_strength_max": cfg.prompt_strength_max,
            "phase1_lr": cfg.phase1_lr,
            "rho_max": cfg.rho_max,
            "lora_rank": cfg.lora_rank,
            "steering_rank": cfg.lora_rank,
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
    expected_route = (
        _FIXED_LOW_RANK_ROUTE
        if steernet_arch in {"v2", "prompt_conditioned_residual"}
        else _LEGACY_FEATURE_ROUTE
    )
    if list(cfg.fixed_layers) != expected_route:
        raise ValueError(f"{steernet_arch} RFT route is locked to {expected_route}")
    if cfg.use_gate:
        raise ValueError("Phase 1 RFT requires use_gate=False")
    if getattr(cfg, "steernet_use_mag_head", False):
        raise ValueError("RFT does not use the legacy token-wise magnitude head")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if any(parameter.requires_grad for parameter in model.base_model.parameters()):
        raise ValueError("Phi-2 backbone must remain fully frozen during RFT")
    if any(getattr(network, "use_mag_head", False) for network in model.steer_nets.values()):
        raise ValueError("Legacy SteerNet magnitude heads must be disabled for RFT")
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
    print(
        f"[rft-eval] held-out questions={len(eval_rows)}; "
        "baseline and checkpoint selection evaluate both 0-shot and 2-shot",
        flush=True,
    )
    baseline_flags, baseline_metrics = evaluate_generation(
        model, tokenizer, eval_rows, cfg, target_device, n_shot=0, steered=False,
        progress_desc="RFT eval baseline 0-shot",
    )
    baseline_2shot_flags, baseline_2shot_metrics = evaluate_generation(
        model, tokenizer, eval_rows, cfg, target_device, n_shot=2, steered=False,
        progress_desc="RFT eval baseline 2-shot",
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
        # Clear any per-example gain left by validation before training forwards.
        if hasattr(model, "set_intervention_gain"):
            model.set_intervention_gain(1.0)
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
                if hasattr(model, "set_prompt_conditioning_mask"):
                    model.set_prompt_conditioning_mask(None)
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
            model, tokenizer, eval_rows, cfg, target_device, n_shot=0, steered=True,
            progress_desc=f"RFT eval epoch {epoch + 1} 0-shot",
        )
        delta_pp = 100 * sum(
            int(current) - int(baseline)
            for current, baseline in zip(current_flags, baseline_flags)
        ) / len(baseline_flags)
        current_2shot_flags, two_shot_diagnostics = evaluate_generation(
            model, tokenizer, eval_rows, cfg, target_device, n_shot=2, steered=True,
            progress_desc=f"RFT eval epoch {epoch + 1} 2-shot",
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
        model, tokenizer, eval_rows, cfg, target_device, n_shot=2, steered=True,
        progress_desc="RFT eval selected checkpoint 2-shot",
    )
    two_shot_metrics["delta_lenient_pp"] = 100 * sum(
        int(current) - int(baseline)
        for current, baseline in zip(two_shot_flags, baseline_2shot_flags)
    ) / len(baseline_2shot_flags)
    result = {
        "route": list(cfg.fixed_layers),
        "validation_questions": len(eval_rows),
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


@torch.no_grad()
def _extract_prompt_strength_features(
    model,
    tokenizer,
    prompts: Sequence[str],
    cfg,
    device: torch.device,
    *,
    batch_size: int = 4,
) -> dict[int, torch.Tensor]:
    """Extract masked prompt summaries from the frozen backbone at each route layer."""
    if not prompts:
        raise ValueError("at least one prompt is required for strength feature extraction")
    model.eval()
    model.set_routing_override("only", cfg.fixed_layers)
    tokenizer.padding_side = "left"
    features: dict[int, list[torch.Tensor]] = {
        layer: [] for layer in cfg.fixed_layers
    }
    for start in range(0, len(prompts), batch_size):
        prompt_batch = list(prompts[start : start + batch_size])
        encoded = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max(cfg.max_tok_len, 1024),
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).bool()
        model.set_prompt_conditioning_mask(attention_mask)
        model.reset_steernet_cache()
        try:
            with model.capture_steering_inputs() as captured:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            del outputs
        finally:
            model.set_prompt_conditioning_mask(None)
            model.reset_steernet_cache()
        for layer in cfg.fixed_layers:
            if layer not in captured:
                raise RuntimeError(
                    f"prompt strength feature extraction missed active layer {layer}"
                )
            summary = model.steer_nets[str(layer)].strength_features(
                captured[layer], attention_mask
            )
            features[layer].append(summary.detach().cpu())
        del encoded, input_ids, attention_mask, captured
    model.set_routing_override("only", cfg.fixed_layers)
    return {layer: torch.cat(values, dim=0) for layer, values in features.items()}


@torch.no_grad()
def collect_prompt_strength_teacher(
    model,
    tokenizer,
    rows: Sequence[dict],
    cfg,
    device: torch.device | str,
    *,
    n_questions: int = 256,
    candidate_strengths: Sequence[float] = (0.0, 0.5, 0.75, 1.0, 1.25),
    shot_weights: Sequence[float] = (0.60, 0.25, 0.15),
    seed: int = 42,
) -> dict[str, Any]:
    """Choose bounded strength targets from training-question answer outcomes."""
    if getattr(cfg, "steernet_arch", None) != "prompt_conditioned_residual":
        raise ValueError("strength teacher requires prompt_conditioned_residual architecture")
    if n_questions < 1 or not rows:
        raise ValueError("strength teacher requires non-empty rows and a positive question count")
    if len(candidate_strengths) < 2 or any(
        not math.isfinite(float(value))
        or float(value) < 0
        or float(value) > cfg.prompt_strength_max
        for value in candidate_strengths
    ):
        raise ValueError("candidate strengths must be finite values within the configured bound")
    if len(shot_weights) != 3 or any(weight < 0 for weight in shot_weights) or sum(shot_weights) <= 0:
        raise ValueError("shot_weights must contain three nonnegative values with positive sum")

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), min(n_questions, len(rows))))
    selected = [rows[index] for index in indices]
    shot_counts = {0: 0, 2: 0, 4: 0}
    shots = []
    for _ in selected:
        shot = rng.choices((0, 2, 4), weights=shot_weights, k=1)[0]
        shots.append(shot)
        shot_counts[shot] += 1
    question_ids = [
        int(row.get("source_index", row.get("question_index", index)))
        for index, row in zip(indices, selected)
    ]
    prompts = []
    shields = []
    for row, question_id, shot in zip(selected, question_ids, shots):
        prompt_batch, shield_batch = build_prompts(
            tokenizer,
            [str(row["question"])],
            [question_id],
            shot,
            policy=cfg.shield_policy,
            template_ids=[question_id % 3]
            if getattr(cfg, "diversify_prompts", False)
            else None,
        )
        prompts.append(prompt_batch[0])
        shields.append(shield_batch[0])
    targets_by_candidate: dict[str, list[bool]] = {}
    model.eval()
    model.set_routing_override("only", cfg.fixed_layers)
    model.set_intervention_gain(1.0)
    try:
        for strength in candidate_strengths:
            candidate_flags = []
            for index, (row, prompt, shield) in enumerate(zip(selected, prompts, shields)):
                model.set_prompt_strength_override(float(strength))
                generated = generate_batch(
                    model,
                    tokenizer,
                    [prompt],
                    device,
                    prompt_shields=[shield],
                    max_new_tokens=cfg.generation_max_new_tokens,
                    max_prompt_len=max(cfg.max_tok_len, 1024),
                    repetition_penalty=1.0,
                )[0]
                raw_gold = str(row.get("final_num", row.get("answer", "")))
                if "####" in raw_gold:
                    raw_gold = raw_gold.split("####")[-1].strip()
                try:
                    gold = float(raw_gold.replace(",", "").replace("$", "").strip())
                except ValueError as error:
                    raise ValueError(
                        f"could not parse GSM8K target for source row {question_ids[index]}"
                    ) from error
                candidate_flags.append(bool(score_tiers(generated.text, gold)["lenient"]))
            targets_by_candidate[str(float(strength))] = candidate_flags
    finally:
        model.set_prompt_strength_override(None)

    teacher_targets = []
    teacher_records = []
    for offset, question_id in enumerate(question_ids):
        successful_strengths = [
            float(strength)
            for strength in candidate_strengths
            if targets_by_candidate[str(float(strength))][offset]
        ]
        target = (
            min(successful_strengths, key=lambda value: (abs(value - 1.0), value))
            if successful_strengths
            else 1.0
        )
        teacher_targets.append(target)
        teacher_records.append(
            {
                "question_index": question_id,
                "n_shot": shots[offset],
                "correct_strengths": successful_strengths,
                "target_strength": target,
                "target_reason": (
                    "nearest_correct_strength_to_neutral"
                    if successful_strengths
                    else "neutral_fallback_no_candidate_correct"
                ),
            }
        )

    return {
        "question_indices": question_ids,
        "prompts": prompts,
        "shots": shots,
        "targets": teacher_targets,
        "records": teacher_records,
        "shot_counts": {str(key): value for key, value in shot_counts.items()},
        "candidate_results": {
            strength: {
                "correct": sum(flags),
                "accuracy_pct": 100.0 * sum(flags) / len(flags),
            }
            for strength, flags in targets_by_candidate.items()
        },
        "all_candidates_wrong": sum(
            not any(targets_by_candidate[str(float(strength))][offset] for strength in candidate_strengths)
            for offset in range(len(selected))
        ),
    }


def train_prompt_strength_heads(
    model,
    layer_features: dict[int, torch.Tensor],
    targets: Sequence[float],
    device: torch.device | str,
    *,
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 2e-3,
    seed: int = 42,
    patience: int = 6,
) -> dict[str, Any]:
    """Fit the per-layer strength heads to held-out-outcome teacher targets."""
    if not layer_features or len(targets) < 10:
        raise ValueError("strength-head fitting requires features for at least ten questions")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or patience < 1:
        raise ValueError("invalid strength-head optimization settings")
    count = len(targets)
    if any(values.shape[0] != count for values in layer_features.values()):
        raise ValueError("each layer feature matrix must have one row per teacher target")

    target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(count, generator=generator).tolist()
    validation_count = max(1, int(round(count * 0.20)))
    validation_indices = torch.tensor(permutation[:validation_count], dtype=torch.long, device=device)
    training_indices = torch.tensor(permutation[validation_count:], dtype=torch.long, device=device)
    if training_indices.numel() == 0:
        raise ValueError("strength-head fitting requires at least one training question")

    heads = {
        layer: model.steer_nets[str(layer)].strength_head
        for layer in sorted(layer_features)
    }
    for parameter in model.steer_nets.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for head in heads.values():
        for parameter in head.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)
    device_features = {
        layer: values.to(device=device, dtype=torch.float32)
        for layer, values in layer_features.items()
    }
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []
    max_strength = next(iter(model.steer_nets.values())).max_strength

    model.eval()
    for epoch in range(epochs):
        order = training_indices[torch.randperm(training_indices.numel(), generator=generator, device="cpu").to(device)]
        for start in range(0, order.numel(), batch_size):
            batch_indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for layer, head in heads.items():
                features = device_features[layer].index_select(0, batch_indices)
                prediction = max_strength * torch.sigmoid(head(features)).squeeze(-1)
                losses.append(F.smooth_l1_loss(prediction, target_tensor.index_select(0, batch_indices)))
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

        with torch.no_grad():
            validation_losses = []
            for layer, head in heads.items():
                features = device_features[layer].index_select(0, validation_indices)
                prediction = max_strength * torch.sigmoid(head(features)).squeeze(-1)
                validation_losses.append(
                    F.smooth_l1_loss(prediction, target_tensor.index_select(0, validation_indices))
                )
            validation_loss = float(torch.stack(validation_losses).mean())
        history.append({"epoch": epoch + 1, "validation_smooth_l1": validation_loss})
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                f"{layer}.{name}": value.detach().cpu().clone()
                for layer, head in heads.items()
                for name, value in head.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("strength-head fitting completed without a validation checkpoint")
    for layer, head in heads.items():
        prefix = f"{layer}."
        head.load_state_dict(
            {
                name[len(prefix) :]: value
                for name, value in best_state.items()
                if name.startswith(prefix)
            }
        )
    return {
        "questions": count,
        "train_questions": int(training_indices.numel()),
        "validation_questions": int(validation_indices.numel()),
        "selected_epoch": best_epoch,
        "best_validation_smooth_l1": best_validation_loss,
        "history": history,
    }


def fit_prompt_strength_controller(
    model,
    tokenizer,
    train_rows: Sequence[dict],
    cfg,
    device: torch.device | str,
    *,
    n_questions: int = 256,
    candidate_strengths: Sequence[float] = (0.0, 0.5, 0.75, 1.0, 1.25),
    shot_weights: Sequence[float] = (0.60, 0.25, 0.15),
    epochs: int = 40,
    batch_size: int = 32,
    seed: int = 42,
    save_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Collect correctness-based strength targets, fit heads, and save the checkpoint."""
    if getattr(cfg, "steernet_arch", None) != "prompt_conditioned_residual":
        raise ValueError("prompt strength fitting requires prompt_conditioned_residual architecture")
    teacher = collect_prompt_strength_teacher(
        model,
        tokenizer,
        train_rows,
        cfg,
        device,
        n_questions=n_questions,
        candidate_strengths=candidate_strengths,
        shot_weights=shot_weights,
        seed=seed,
    )
    optimization = {}
    # Fit in transformer order. Updating an earlier layer's head changes the
    # hidden states seen by later layers, so refresh their features after each
    # upstream head has been fitted.
    for layer in cfg.fixed_layers:
        layer_features = _extract_prompt_strength_features(
            model, tokenizer, teacher["prompts"], cfg, torch.device(device)
        )
        optimization[str(layer)] = train_prompt_strength_heads(
            model,
            {layer: layer_features[layer]},
            teacher["targets"],
            device,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
    checkpoint_path = save_path or cfg.lora_save_path
    _save_rft_checkpoint(model, checkpoint_path, cfg)
    report = {
        "architecture_id": "ars.prompt_conditioned_residual",
        "architecture_version": "1.0.0",
        "teacher_objective": "greedy_exact_answer_outcome",
        "candidate_strengths": [float(value) for value in candidate_strengths],
        "teacher": {
            key: value
            for key, value in teacher.items()
            if key not in {"prompts", "targets", "records"}
        },
        "teacher_records": teacher["records"],
        "optimization": optimization,
        "checkpoint": str(checkpoint_path),
    }
    return report
