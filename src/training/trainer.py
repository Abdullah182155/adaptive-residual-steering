"""Focused S1 trainer for the fixed [13, 20, 22] SteerNet V5 route."""
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import torch
from src.training.losses import steer_norm_penalty, token_weighted_nll

def _batch(rows: Iterable[dict], device: torch.device):
    rows = list(rows)
    return {key: torch.stack([row[key] for row in rows]).to(device)
            for key in ("input_ids", "attention_mask", "labels") if key in rows[0]}

@torch.no_grad()
def evaluate_loss(model, rows, device: torch.device) -> float:
    if not rows: return 0.0
    model.eval()
    return sum(float(token_weighted_nll(model(**_batch([row], device)).logits, _batch([row], device)["labels"]))
               for row in rows) / len(rows)

def save_steer_weights(model, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"fixed_layers": model.target_layers, "steer_nets": model.steer_nets.state_dict()}, path)

def load_steer_weights(model, path: str, device: Optional[torch.device] = None) -> None:
    state = torch.load(path, map_location=device or "cpu", weights_only=True)
    if list(state["fixed_layers"]) != list(model.target_layers): raise ValueError("checkpoint route does not match active route")
    model.steer_nets.load_state_dict(state["steer_nets"])

def train_phase1_steernet(model, train_data, eval_data, cfg, device: torch.device, tokenizer=None) -> Dict[str, Any]:
    del tokenizer
    params = list(model.steer_nets.parameters())
    optimiser = torch.optim.AdamW(params, lr=cfg.phase1_lr, weight_decay=cfg.weight_decay)
    history, best_state, best_validation = {"train": [], "validation": []}, None, float("inf")
    model.set_routing_override("only", model.target_layers)
    for _ in range(cfg.phase1_epochs):
        model.train(); model.base_model.eval(); total = steps = 0; optimiser.zero_grad(set_to_none=True)
        for start in range(0, len(train_data), cfg.batch_size):
            rows, batch = train_data[start:start + cfg.batch_size], _batch(train_data[start:start + cfg.batch_size], device)
            if "prompt_shield_mask" in rows[0]: model.set_prompt_shield_mask(torch.stack([r["prompt_shield_mask"] for r in rows]).to(device))
            loss = token_weighted_nll(model(**batch).logits, batch["labels"]) + 0.30 * steer_norm_penalty(model)
            (loss / cfg.grad_accum).backward(); steps += 1
            if steps % cfg.grad_accum == 0 or start + cfg.batch_size >= len(train_data):
                torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad); optimiser.step(); optimiser.zero_grad(set_to_none=True)
            total += float(loss.detach())
        validation = evaluate_loss(model, eval_data, device)
        history["train"].append(total / max(steps, 1)); history["validation"].append(validation)
        if validation <= best_validation:
            best_validation = validation; best_state = {k: v.detach().cpu().clone() for k, v in model.steer_nets.state_dict().items()}
    if best_state is not None: model.steer_nets.load_state_dict(best_state)
    save_steer_weights(model, cfg.lora_save_path)
    return {**history, "best_validation": best_validation, "route": list(model.target_layers)}
