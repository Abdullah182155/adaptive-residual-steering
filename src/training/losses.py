"""Loss functions used by fixed-route Phase 1 training."""

import torch
import torch.nn.functional as functional


def token_weighted_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute next-token NLL across all non-ignored target tokens."""
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=-100,
    )


def steer_norm_penalty(model) -> torch.Tensor:
    """Penalize raw residuals before their smooth norm clamp saturates."""
    penalties = []
    for steer_net in model.steer_nets.values():
        ratio = getattr(steer_net, "last_pre_cap_q", None)
        if ratio is not None:
            penalties.append(ratio.square().mean())
    if penalties:
        return torch.stack(penalties).mean()
    return torch.zeros((), device=next(model.parameters()).device)
