"""Fixed-route SteerNet V5 trainer for ARS Phase 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from src.training.losses import steer_norm_penalty, token_weighted_nll


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
        _set_prompt_shield(model, [row], device)
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
    optimizer = torch.optim.AdamW(
        parameters, lr=cfg.phase1_lr, weight_decay=cfg.weight_decay
    )

    history = {"train_nll": [], "validation_nll": []}
    best_validation = float("inf")
    best_state = None
    for epoch in range(cfg.phase1_epochs):
        model.train()
        model.base_model.eval()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        batch_count = 0
        for offset in range(0, len(train_data), cfg.batch_size):
            rows = _extract_batch_rows(train_data, offset, cfg.batch_size)
            if not rows:
                continue
            batch = _stack_batch(rows, device)
            _set_prompt_shield(model, rows, device)
            output = model(**batch)
            task_loss = token_weighted_nll(output.logits, batch["labels"])
            loss = task_loss + 0.30 * steer_norm_penalty(model)
            (loss / cfg.grad_accum).backward()
            batch_count += 1
            if batch_count % cfg.grad_accum == 0 or offset + cfg.batch_size >= len(
                train_data
            ):
                torch.nn.utils.clip_grad_norm_(parameters, cfg.clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
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
    return {
        **history,
        "best_validation_nll": best_validation,
        **checkpoint_metadata(cfg),
    }
