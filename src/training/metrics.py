"""Validation metrics and generation-based checkpoint-selection helpers."""

from __future__ import annotations

from collections import defaultdict
import contextlib
from typing import Any, Dict, Iterable, Sequence

import torch
import torch.nn.functional as F

from src.configs.config import RSCConfig
from src.training.batching import prepare_batch as _prepare
from src.training.losses import T_FINAL, T_RESULT, lambdas
from src.training.on_policy import generate_onpolicy_batch as _generate_onpolicy_batch


@torch.no_grad()
def evaluate_typed(
    model: torch.nn.Module,
    rows: Iterable[dict[str, Any]],
    device: torch.device,
    cfg: RSCConfig,
    steered: bool = True,
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
def evaluate_loss(
    model: torch.nn.Module, rows: Iterable[dict[str, Any]], device: torch.device
) -> float:
    """Return validation NLL for backward compatibility."""
    cfg = getattr(model, "cfg", RSCConfig())
    typed = evaluate_typed(model, rows, device, cfg, steered=True)
    return float(typed.get("validation_nll", typed.get("crit", 0.0)))

def evaluate_loss_with_hooks(
    model: torch.nn.Module,
    rows: Iterable[dict[str, Any]],
    device: torch.device,
    enabled: bool,
) -> float:
    context = torch.enable_grad() if enabled else model.hooks_disabled()
    with context:
        return evaluate_loss(model, rows, device)

@torch.no_grad()
def _validate_generation(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    device: torch.device,
    cfg: RSCConfig,
    n: int = 50,
) -> float:
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
