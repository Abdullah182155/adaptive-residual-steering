import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

def per_example_answer_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean next-token NLL for each example, excluding masked prompt tokens."""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    token_nll = F.cross_entropy(
        shift_logits.transpose(1, 2), shift_labels, ignore_index=-100, reduction="none"
    )
    valid = shift_labels.ne(-100)
    return (token_nll * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

def sample_subset_plackett_luce(
    logits: torch.Tensor,
    k: torch.Tensor,
    synergy_matrix: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample without replacement a size-k[b] subset using Cooperative Plackett-Luce Gumbel-max draws."""
    B, N = logits.shape
    device = logits.device
    K_max = int(k.max().item()) if k.numel() else 0
    remaining = torch.ones(B, N, dtype=torch.bool, device=device)
    selected = torch.zeros(B, N, dtype=torch.bool, device=device)
    logprob = torch.zeros(B, device=device, dtype=logits.dtype)
    det_logits = logits.detach()
    for step in range(K_max):
        active = k > step
        # Dynamically modulate logits with cross-layer synergy from previously selected layers
        if step > 0 and synergy_matrix is not None:
            syn_boost = torch.bmm(selected.float().unsqueeze(1), synergy_matrix).squeeze(1)
            step_logits = logits + syn_boost
            step_det_logits = det_logits + syn_boost.detach()
        else:
            step_logits = logits
            step_det_logits = det_logits

        log_p_grad = F.log_softmax(step_logits.masked_fill(~remaining, float("-inf")), dim=-1)
        with torch.no_grad():
            log_p_det = F.log_softmax(step_det_logits.masked_fill(~remaining, float("-inf")), dim=-1)
            u_noise = torch.rand(B, N, device=device).clamp(min=1e-20, max=1 - 1e-20)
            gumbel = -torch.log(-torch.log(u_noise))
            perturbed = torch.where(remaining, log_p_det + gumbel, torch.full_like(log_p_det, float("-inf")))
            choice = perturbed.argmax(dim=-1)
        step_logprob = log_p_grad.gather(1, choice.unsqueeze(-1)).squeeze(-1)
        logprob = logprob + torch.where(active, step_logprob, torch.zeros_like(step_logprob))
        newly = torch.zeros(B, N, dtype=torch.bool, device=device)
        newly.scatter_(1, choice.unsqueeze(-1), True)
        newly = newly & active.unsqueeze(-1)
        selected = selected | newly
        remaining = remaining & ~newly
    return selected, logprob

def sample_k(k_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample K in {1, ..., K_max} via Gumbel-max on k_logits (B, K_max)."""
    B, K_max = k_logits.shape
    log_p_grad = F.log_softmax(k_logits, dim=-1)
    with torch.no_grad():
        log_p_det = F.log_softmax(k_logits.detach(), dim=-1)
        u_noise = torch.rand_like(log_p_det).clamp(min=1e-20, max=1 - 1e-20)
        gumbel = -torch.log(-torch.log(u_noise))
        choice = (log_p_det + gumbel).argmax(dim=-1)
    logprob = log_p_grad.gather(1, choice.unsqueeze(-1)).squeeze(-1)
    return choice + 1, logprob

def policy_entropy_bonus(logits: torch.Tensor, k_logits: torch.Tensor, k_max: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute decoupled entropy for K distribution and Plackett-Luce sequential subset policy."""
    log_p_k = F.log_softmax(k_logits, dim=-1)
    p_k = log_p_k.exp()
    h_k = -(p_k * log_p_k).sum(dim=-1).mean()

    B, N = logits.shape
    remaining = torch.ones(B, N, dtype=torch.bool, device=logits.device)
    step_entropies = []
    for _ in range(min(k_max, N)):
        log_p = F.log_softmax(logits.masked_fill(~remaining, float("-inf")), dim=-1)
        p = log_p.exp()
        h_step = -(p * log_p.clamp(min=-30)).sum(dim=-1)
        step_entropies.append(h_step.mean())
        with torch.no_grad():
            top = log_p.masked_fill(~remaining, float("-inf")).argmax(dim=-1)
            newly = torch.zeros(B, N, dtype=torch.bool, device=logits.device)
            newly.scatter_(1, top.unsqueeze(-1), True)
        remaining = remaining & ~newly
    h_subset = torch.stack(step_entropies).mean()
    return h_k, h_subset

def compute_router_rloo_loss(model, ids: torch.Tensor, mask: torch.Tensor,
                              labels: torch.Tensor, cfg, prompt_mask: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """RLOO subset selection loss for training cooperative router policy with leave-one-out advantage."""
    if prompt_mask is None:
        prompt_mask = mask.bool() & labels.eq(-100)
    model.set_router_context_mask(prompt_mask)
    if model.router._last_pooled is None:
        return torch.zeros((), device=ids.device), {}

    u_fresh, k_logits_fresh = model.router.recompute_from_cache()
    synergy = getattr(model.router, "_last_synergy_matrix", None)
    K_max = model.cfg.max_active_layers

    k_a, logprob_k_a = sample_k(k_logits_fresh)
    k_b, logprob_k_b = sample_k(k_logits_fresh)
    subset_a_mask, logprob_subset_a = sample_subset_plackett_luce(u_fresh, k_a, synergy_matrix=synergy)
    subset_b_mask, logprob_subset_b = sample_subset_plackett_luce(u_fresh, k_b, synergy_matrix=synergy)
    logprob_a = logprob_k_a + logprob_subset_a
    logprob_b = logprob_k_b + logprob_subset_b

    old_override = model._routing_override
    try:
        model.set_routing_override("matrix", subset_a_mask.detach())
        model.set_router_context_mask(prompt_mask)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=ids.is_cuda):
                nll_a = per_example_answer_nll(model(input_ids=ids, attention_mask=mask).logits, labels)
        model.set_routing_override("matrix", subset_b_mask.detach())
        model.set_router_context_mask(prompt_mask)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=ids.is_cuda):
                nll_b = per_example_answer_nll(model(input_ids=ids, attention_mask=mask).logits, labels)
    finally:
        if old_override is None:
            model.set_routing_override(None)
        else:
            model.set_routing_override(old_override[0], old_override[1])

    lambda_cost = getattr(cfg, "lambda_cost", 0.005)
    reward_a = (-nll_a - lambda_cost * (k_a.float() - 1.0)).detach()
    reward_b = (-nll_b - lambda_cost * (k_b.float() - 1.0)).detach()
    advantage_a = reward_a - reward_b

    policy_loss = -((logprob_a - logprob_b) * advantage_a).mean()
    h_k, h_subset = policy_entropy_bonus(u_fresh, k_logits_fresh, K_max)

    beta_k = getattr(cfg, "beta_entropy_k", 0.15)
    beta_subset = getattr(cfg, "beta_entropy_subset", 0.15)

    # Anti-monopoly balance loss: prevents single-layer monopoly
    mean_selected = (subset_a_mask.float() + subset_b_mask.float()).mean(dim=0) / 2.0
    target_sel = (k_a.float() + k_b.float()).mean() / (2.0 * model.router.n_candidates)
    balance_loss = F.mse_loss(mean_selected, target_sel.expand_as(mean_selected))
    lambda_balance = getattr(cfg, "router_balance_weight", 0.10)

    loss = policy_loss - beta_k * h_k - beta_subset * h_subset + lambda_balance * balance_loss

    diag = {
        "policy_loss": float(policy_loss.detach().item()),
        "h_k_raw": float(h_k.detach().item()),
        "h_k_weighted": float((beta_k * h_k).detach().item()),
        "h_subset_raw": float(h_subset.detach().item()),
        "h_subset_weighted": float((beta_subset * h_subset).detach().item()),
        "mean_k_sampled": float(((k_a.float() + k_b.float()) / 2).mean().item()),
    }
    return loss, diag
