"""Loss functions used by fixed-route Phase 1 training."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F


# Token category IDs
T_NONE = 0
T_FORMAT = 1
T_PROSE = 2
T_OPERATOR = 3
T_OPERAND = 4
T_RESULT = 5
T_FINAL = 6

_PRIORITY = {
    T_PROSE: 0,
    T_OPERAND: 1,
    T_OPERATOR: 2,
    T_RESULT: 3,
    T_FINAL: 4,
    T_FORMAT: 5,
}

_NUM = r"\$?-?\d[\d,]*(?:\.\d+)?"
_PASSES = (
    # Ascending priority: later passes overwrite earlier ones
    (re.compile(_NUM), T_OPERAND, 0),
    (re.compile(r"(?<=\d)\s?[+\-*/=×÷]\s?(?=[\$\-]?\d)"), T_OPERATOR, 0),
    (re.compile(r"=\s*(" + _NUM + ")"), T_RESULT, 1),
    (re.compile(r"(?:Final answer:|####)\s*(" + _NUM + ")"), T_FINAL, 1),
    (re.compile(r"Step \d+:|Final answer:|####"), T_FORMAT, 0),
)


def annotate_char_types(text: str, start: int) -> List[int]:
    """Tag character spans starting at `start` with token categories."""
    labels = [T_PROSE] * len(text)
    seg = text[start:]
    for rx, tid, grp in _PASSES:
        for m in rx.finditer(seg):
            a, b = m.span(grp)
            for i in range(start + a, start + b):
                labels[i] = tid
    return labels


def token_types_from_offsets(
    text: str, char_labels: List[int], offsets: Sequence[Tuple[int, int]]
) -> List[int]:
    """Map character categories to token categories by priority."""
    out = []
    for s, e in offsets:
        best = None
        for i in range(s, e):
            if i >= len(text) or text[i].isspace():
                continue
            t = char_labels[i]
            if best is None or _PRIORITY[t] > _PRIORITY[best]:
                best = t
        out.append(T_NONE if best is None else best)
    return out


def lambdas(cfg: Any) -> Dict[int, float]:
    """Category loss weights lambda_c."""
    return {
        T_FORMAT: getattr(cfg, "lambda_format", 0.05),
        T_OPERATOR: getattr(cfg, "lambda_operator", 0.10),
        T_OPERAND: getattr(cfg, "lambda_operand", 0.20),
        T_RESULT: getattr(cfg, "lambda_result", 0.35),
        T_FINAL: getattr(cfg, "lambda_final", 0.30),
    }


def window_counts(
    labels_list: Sequence[torch.Tensor], types_list: Sequence[torch.Tensor]
) -> Dict[int, int]:
    """Token counts per category over one gradient-accumulation window."""
    counts = {c: 0 for c in range(7)}
    for labels, types in zip(labels_list, types_list):
        sel = labels[:, 1:].ne(-100)
        t = types[:, 1:][sel]
        for c in counts:
            counts[c] += int((t == c).sum())
    return counts


def typed_objective(
    logits: torch.Tensor,
    base_logits: Optional[torch.Tensor],
    labels: torch.Tensor,
    types: torch.Tensor,
    cfg: Any,
    counts: Dict[int, int],
    neg_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return window-normalized typed reasoning loss and telemetry parts."""
    sel = labels[:, 1:].ne(-100)
    if not sel.any():
        return logits.new_zeros(()), {}

    z = logits[:, :-1][sel].float()  # (N, V) fp32 for stability
    y = labels[:, 1:][sel]
    c = types[:, 1:][sel]
    logp = F.log_softmax(z, dim=-1)
    nll = -logp.gather(-1, y[:, None]).squeeze(-1)

    with torch.no_grad():
        p = (-nll).exp()
        gamma = getattr(cfg, "focal_gamma", 1.0)
        w = (1.0 - p).clamp_min(0).pow(gamma) if gamma > 0 else torch.ones_like(p)

    loss = z.new_zeros(())
    parts = {}
    lam_map = lambdas(cfg)
    for cid, lam in lam_map.items():
        m = c == cid
        if lam == 0.0 or not bool(m.any()):
            continue
        # Window normalization prevents micro-batch composition bias
        w_norm = max(counts.get(cid, 0), 1)
        term = lam * (w[m] * nll[m]).sum() / w_norm
        loss = loss + term
        parts[f"nll_{cid}"] = float(nll[m].detach().mean())

    kl_weight = getattr(cfg, "kl_weight", 1.0)
    if kl_weight > 0 and base_logits is not None:
        m = c == T_PROSE
        if bool(m.any()):
            # base_logits may have been pinned to CPU to save VRAM; move it back only now.
            bl = base_logits.to(logits.device) if base_logits.device != logits.device else base_logits
            lq = F.log_softmax(bl[:, :-1][sel][m].float(), dim=-1)
            kl = (lq.exp() * (lq - logp[m])).sum(-1).sum() / max(counts.get(T_PROSE, 0), 1)
            loss = loss + kl_weight * kl
            parts["kl_prose"] = float(kl.detach()) * max(counts.get(T_PROSE, 0), 1) / int(m.sum())

    return loss, parts


# Legacy loss functions preserved for backward compatibility
def target_weights(
    labels: torch.Tensor,
    calc_mask: torch.Tensor | None = None,
    calc_alpha: float = 1.0,
) -> torch.Tensor:
    """Weight supervised calculation tokens by ``1 + calc_alpha``."""
    weights = labels.ne(-100).float()
    if calc_mask is not None:
        weights = weights * (1.0 + calc_alpha * calc_mask.to(weights.device, weights.dtype))
    return weights


def weighted_nll_sum(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fp32 weighted next-token NLL sum and its exact token weight."""
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:].contiguous()
    losses = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shifted_labels)
    if weights is None:
        weights = shifted_labels.ne(-100).float()
    else:
        weights = weights[:, 1:].to(losses.device, losses.dtype)
    return (losses * weights).sum(), weights.sum().clamp_min(1.0)


def token_weighted_nll(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute next-token NLL across all non-ignored target tokens."""
    total, weight = weighted_nll_sum(logits, labels, weights)
    return total / weight
