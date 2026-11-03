"""Loss functions used by fixed-route Phase 1 training."""

import torch
import torch.nn.functional as functional


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
    losses = functional.cross_entropy(
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


def steer_norm_penalty(model, mask=None, margin: float = 1.0) -> torch.Tensor:
    """Apply a zero-inside-margin hinge penalty to saturation telemetry."""
    penalties = []
    for steer_net in model.steer_nets.values():
        ratio = getattr(steer_net, "last_pre_cap_q", None)
        if ratio is not None:
            if mask is not None and mask.shape == ratio.squeeze(-1).shape:
                ratio = ratio[mask.unsqueeze(-1)]
            penalties.append(torch.relu(ratio - margin).square().mean())
    if penalties:
        return torch.stack(penalties).mean()
    return torch.zeros((), device=next(model.parameters()).device)
