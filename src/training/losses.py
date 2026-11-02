"""Small, explicit losses used by Phase 1."""
import torch
import torch.nn.functional as F

def token_weighted_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    targets, scores = labels[:, 1:].contiguous(), logits[:, :-1].contiguous()
    return F.cross_entropy(scores.view(-1, scores.size(-1)), targets.view(-1), ignore_index=-100)

def steer_norm_penalty(model) -> torch.Tensor:
    values = [net.last_pre_cap_q.square().mean() for net in model.steer_nets.values()
              if getattr(net, "last_pre_cap_q", None) is not None]
    return torch.stack(values).mean() if values else torch.zeros((), device=next(model.parameters()).device)
