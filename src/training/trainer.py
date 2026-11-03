"""Fixed-route SteerNet V5 trainer for ARS Phase 1."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, Iterable, List, Optional

import torch
from tqdm.auto import tqdm

from src.training.losses import steer_norm_penalty, target_weights, token_weighted_nll


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
    """Drop all-padding columns while preserving every aligned auxiliary mask.

    Phi-2's current rotary implementation requires position tensors to match
    the live sequence length. Fixed-width dataset padding otherwise supplies a
    384-token position tensor for a much shorter hidden-state sequence.
    """
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
        if value.ndim >= 2 and value.shape[-1] == attention_mask.shape[-1]
    }
    return trimmed_batch, trimmed_extras


def _set_prompt_shield(model, rows: List[dict], device: torch.device) -> None:
    if "prompt_shield_mask" not in rows[0]:
        model.set_prompt_shield_mask(None)
        return
    mask = torch.stack([row["prompt_shield_mask"] for row in rows]).to(device)
    model.set_prompt_shield_mask(mask)


@torch.no_grad()
def evaluate_loss(model, rows: Iterable[dict], device: torch.device) -> float:
    """Return mean teacher-forced NLL using the same fixed route as training."""
    if not isinstance(rows, (list, tuple)) and hasattr(rows, "select"):
        row_list = [rows[i] for i in range(len(rows))]
    else:
        row_list = list(rows)
    if not row_list:
        return 0.0
    model.eval()
    losses = []
    for row in row_list:
        batch = _stack_batch([row], device)
        extras = {}
        if "prompt_shield_mask" in row:
            extras["prompt_shield_mask"] = row["prompt_shield_mask"].unsqueeze(0).to(device)
        batch, extras = _trim_to_attended_tokens(batch, extras)
        model.set_prompt_shield_mask(extras.get("prompt_shield_mask"))
        output = model(**batch)
        losses.append(float(token_weighted_nll(output.logits, batch["labels"])))
    return sum(losses) / len(losses)


def checkpoint_metadata(cfg) -> Dict[str, Any]:
    return {
        "route": list(cfg.fixed_layers),
        "steer_scale": cfg.steer_scale,
        "steer_kappa": cfg.steer_kappa,
        "k_scale_mode": cfg.k_scale_mode,
    }


def save_steer_weights(model, path: str, cfg) -> None:
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
    model, path: str, cfg, device: Optional[torch.device] = None
) -> Dict[str, Any]:
    state = torch.load(path, map_location=device or "cpu", weights_only=True)
    metadata = state["metadata"]
    if metadata["route"] != list(cfg.fixed_layers):
        raise ValueError("checkpoint route does not match the active fixed route")
    model.steer_nets.load_state_dict(state["steer_nets"])
    return metadata


def train_phase1_steernet(
    model, train_data, eval_data, cfg, device: torch.device
) -> Dict[str, Any]:
    """Optimise only the three fixed-route SteerNets with NLL and norm control."""
    cfg.validate()
    model.set_routing_override("only", cfg.fixed_layers)
    model.set_steer_scale(cfg.steer_scale)
    parameters = list(model.steer_nets.parameters())
    decay = [parameter for parameter in parameters if parameter.ndim >= 2]
    no_decay = [parameter for parameter in parameters if parameter.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=min(cfg.phase1_lr, 5e-4),
        betas=(0.9, 0.99),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and next(model.base_model.parameters()).dtype == torch.float16
    )

    backbone_nll = evaluate_loss_with_hooks(model, eval_data, device, enabled=False)
    adapter_init_nll = evaluate_loss(model, eval_data, device)

    history = {"train_nll": [], "validation_nll": [], "backbone_nll": backbone_nll, "adapter_init_nll": adapter_init_nll, "steps": []}
    best_validation = float("inf")
    best_state = None
    for epoch in range(cfg.phase1_epochs):
        model.train()
        model.base_model.eval()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        batch_count = 0
        progress = tqdm(range(0, len(train_data), cfg.batch_size), desc=f"S1 {epoch + 1}/{cfg.phase1_epochs}")
        for offset in progress:
            rows = _extract_batch_rows(train_data, offset, cfg.batch_size)
            if not rows:
                continue
            batch = _stack_batch(rows, device)
            extras = {}
            for name in ("prompt_shield_mask", "calc_mask"):
                if name in rows[0]:
                    extras[name] = torch.stack([row[name] for row in rows]).to(device)
            batch, extras = _trim_to_attended_tokens(batch, extras)
            model.set_prompt_shield_mask(extras.get("prompt_shield_mask"))
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(**batch)
                calc_mask = None
                if "calc_mask" in extras:
                    calc_mask = extras["calc_mask"]
                weights = target_weights(batch["labels"], calc_mask, cfg.calc_weight_alpha)
                task_loss = token_weighted_nll(output.logits, batch["labels"], weights)
                penalty = steer_norm_penalty(model, batch["attention_mask"].bool(), cfg.steer_penalty_margin)
                loss = task_loss + cfg.steer_penalty_weight * penalty
            scaler.scale(loss / cfg.grad_accum).backward()
            batch_count += 1
            if batch_count % cfg.grad_accum == 0 or offset + cfg.batch_size >= len(
                train_data
            ):
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, cfg.clip_grad))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                q_values = [net.last_pre_cap_q.mean().detach() for net in model.steer_nets.values() if net.last_pre_cap_q is not None]
                q_mean = float(torch.stack(q_values).mean()) if q_values else 0.0
                history["steps"].append({"epoch": epoch + 1, "loss": float(loss.detach()), "grad_norm": grad_norm, "q_mean": q_mean})
                progress.set_postfix(nll_ema=f"{float(task_loss):.3f}", gnorm=f"{grad_norm:.2f}", q=f"{q_mean:.2f}")
            running_loss += float(loss.detach())

        validation = evaluate_loss(model, eval_data, device)
        history["train_nll"].append(running_loss / max(batch_count, 1))
        history["validation_nll"].append(validation)
        if validation <= best_validation:
            best_validation = validation
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.steer_nets.state_dict().items()
            }
        print(
            f"S1 epoch {epoch + 1}/{cfg.phase1_epochs}: validation_nll={validation:.4f}"
        )

    if best_state is not None:
        model.steer_nets.load_state_dict(best_state)
    save_steer_weights(model, cfg.lora_save_path, cfg)
    report_path = Path(cfg.report_dir) / "training_log.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return {
        **history,
        "best_validation_nll": best_validation,
        **checkpoint_metadata(cfg),
    }


def evaluate_loss_with_hooks(model, rows, device: torch.device, enabled: bool) -> float:
    context = torch.enable_grad() if enabled else model.hooks_disabled()
    with context:
        return evaluate_loss(model, rows, device)
