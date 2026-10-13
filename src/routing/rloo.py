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
        # Dynamically modulate logits with damped cross-layer synergy from previously selected layers
        if step > 0 and synergy_matrix is not None:
            syn_boost = 0.5 * torch.bmm(selected.float().unsqueeze(1), synergy_matrix).squeeze(1)
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

def sample_k(k_logits: torch.Tensor, k_max: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample K via Gumbel-max on k_logits.
    Supports K in {0, ..., K_max} if logits has K_max+1 dim, or {1, ..., K_max} for legacy K_max dim.
    """
    B, n_classes = k_logits.shape
    log_p_grad = F.log_softmax(k_logits, dim=-1)
    with torch.no_grad():
        log_p_det = F.log_softmax(k_logits.detach(), dim=-1)
        u_noise = torch.rand_like(log_p_det).clamp(min=1e-20, max=1 - 1e-20)
        gumbel = -torch.log(-torch.log(u_noise))
        choice = (log_p_det + gumbel).argmax(dim=-1)
    logprob = log_p_grad.gather(1, choice.unsqueeze(-1)).squeeze(-1)
    # If logits dimension includes K=0 (e.g. 5 classes for max 4), choice is already 0..4
    if n_classes > (k_max or 4):
        return choice, logprob
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


def mutual_information_diversity_loss(
    k_logits: torch.Tensor, layer_logits: torch.Tensor, temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return negative I(prompt; K) and negative I(prompt; first layer).

    A per-example entropy bonus alone can leave every prompt with the same
    slightly-preferred argmax.  With a real RLOO minibatch, mutual information
    rewards broad *marginal* route coverage while penalising indecisive routes.
    It does not prescribe a particular K or layer for any example.
    """
    def negative_mi(logits: torch.Tensor) -> torch.Tensor:
        # Using T=1 rewarded harmless changes in soft probability while the
        # actual deployment argmax remained permanently K=3.  A low but finite
        # temperature makes this a surrogate for diversity of discrete routes.
        probs = F.softmax(logits / max(float(temperature), 1e-3), dim=-1)
        marginal = probs.mean(dim=0)
        h_marginal = -(marginal * marginal.clamp_min(1e-8).log()).sum()
        h_conditional = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return h_conditional - h_marginal
    return negative_mi(k_logits), negative_mi(layer_logits)


def counterfactual_k_preference_loss(
    k_logits: torch.Tensor,
    k_a: torch.Tensor,
    k_b: torch.Tensor,
    reward_0: torch.Tensor,
    reward_a: torch.Tensor,
    reward_b: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Distil observed {K=0, K_a, K_b} counterfactual rewards into K logits.

    Each row supervises only actions actually evaluated for that prompt.  This
    is not a fixed K schedule: K=0 wins only when its unsteered reward exceeds
    the sampled intervention routes.  Duplicate sampled K values are averaged
    before the soft preference target is formed.
    """
    B, n_actions = k_logits.shape
    rewards_sum = torch.zeros(B, n_actions, device=k_logits.device, dtype=k_logits.dtype)
    counts = torch.zeros_like(rewards_sum)
    zero = torch.zeros_like(k_a)
    for actions, rewards in ((zero, reward_0), (k_a, reward_a), (k_b, reward_b)):
        actions = actions.long().clamp(0, n_actions - 1)
        rewards_sum.scatter_add_(1, actions.unsqueeze(1), rewards.to(k_logits.dtype).unsqueeze(1))
        counts.scatter_add_(1, actions.unsqueeze(1), torch.ones_like(rewards, dtype=k_logits.dtype).unsqueeze(1))
    observed = counts > 0
    mean_reward = rewards_sum / counts.clamp_min(1)
    target_scores = (mean_reward / max(float(temperature), 1e-4)).masked_fill(~observed, float("-inf"))
    target = F.softmax(target_scores, dim=-1)
    return -(target * F.log_softmax(k_logits, dim=-1)).sum(dim=-1).mean()

def compute_router_rloo_loss(model, ids: torch.Tensor, mask: torch.Tensor,
                              labels: torch.Tensor, cfg, prompt_mask: torch.Tensor = None,
                              shield_mask: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """RLOO subset selection loss for training cooperative router policy with leave-one-out advantage."""
    if prompt_mask is None:
        prompt_mask = mask.bool() & labels.eq(-100)
    # The policy state must come from *this* example.  In bootstrap training
    # there is otherwise no ordinary task forward before this call, so relying
    # on _last_pooled trains current rewards against the previous example.
    # Populate the detached router cache with a fresh, same-example prefill.
    old_override = model._routing_override
    # Counterfactual reward is a comparison between routes.  It must not draw
    # a different dropout mask for each arm, otherwise RLOO learns dropout
    # noise instead of route utility.  Preserve every module's mode because
    # this function is called from a mixed-mode staged curriculum.
    module_modes = {module: module.training for module in model.modules()}
    try:
        model.set_routing_override(None)
        model.set_router_context_mask(prompt_mask)
        if hasattr(model, "set_prompt_shield_mask"):
            model.set_prompt_shield_mask(shield_mask)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=ids.is_cuda):
                _ = model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        if old_override is None:
            model.set_routing_override(None)
        else:
            model.set_routing_override(old_override[0], old_override[1])

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

    def counterfactual_nll(route_mask: torch.Tensor) -> torch.Tensor:
        """Evaluate one fixed route with the same prompt/shield contract."""
        model.set_routing_override("matrix", route_mask.detach())
        model.set_router_context_mask(prompt_mask)
        if hasattr(model, "set_prompt_shield_mask"):
            model.set_prompt_shield_mask(shield_mask)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=ids.is_cuda):
                return per_example_answer_nll(model(input_ids=ids, attention_mask=mask, use_cache=False).logits, labels)

    def deterministic_counterfactual_nll(route_mask: torch.Tensor) -> torch.Tensor:
        """Score one auxiliary counterfactual without dropout noise."""
        modes = {module: module.training for module in model.modules()}
        prior_override = model._routing_override
        model.eval()
        try:
            return counterfactual_nll(route_mask)
        finally:
            if prior_override is None:
                model.set_routing_override(None)
            else:
                model.set_routing_override(prior_override[0], prior_override[1])
            for module, was_training in modes.items():
                module.train(was_training)

    try:
        model.eval()
        nll_a = counterfactual_nll(subset_a_mask)
        nll_b = counterfactual_nll(subset_b_mask)
        # Arm 0: Ground-truth unsteered baseline anchor
        model.set_routing_override("none")
        model.set_router_context_mask(prompt_mask)
        if hasattr(model, "set_prompt_shield_mask"):
            model.set_prompt_shield_mask(shield_mask)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=ids.is_cuda):
                nll_0 = per_example_answer_nll(model(input_ids=ids, attention_mask=mask, use_cache=False).logits, labels)
    finally:
        if old_override is None:
            model.set_routing_override(None)
        else:
            model.set_routing_override(old_override[0], old_override[1])
        for module, was_training in module_modes.items():
            module.train(was_training)

    reward_0 = (-nll_0).detach()
    lambda_cost = getattr(cfg, "lambda_cost", 0.005)
    reward_a = (-nll_a - lambda_cost * k_a.float()).detach()
    reward_b = (-nll_b - lambda_cost * k_b.float()).detach()

    # Anchor-Grounded 3-Arm Leave-One-Out (RLOO)
    baseline_for_a = (reward_b + reward_0) / 2.0
    baseline_for_b = (reward_a + reward_0) / 2.0
    advantage_a = reward_a - baseline_for_a
    advantage_b = reward_b - baseline_for_b

    policy_loss = -0.5 * (logprob_a * advantage_a + logprob_b * advantage_b).mean()

    # Direct pairwise *interaction* signal.  A route's gain over the base is
    # not synergy: two individually useful layers would otherwise make every
    # co-selected pair look positive.  For a sampled pair (i,j), estimate
    # I_ij = R({i,j}) - R({i}) - R({j}) + R(empty), then align only M_ij to it.
    # This preserves the cooperative-router hypothesis while giving destructive
    # pairs a negative target instead of allowing universal +1 saturation.
    delta_r_a = (reward_a - reward_0).detach()
    delta_r_b = (reward_b - reward_0).detach()
    pairwise_interaction_loss = torch.zeros((), device=ids.device)
    mean_pair_interaction = torch.zeros((), device=ids.device)
    if synergy is not None:
        pair_source = subset_a_mask
        first = pair_source.float().argmax(dim=-1)
        without_first = pair_source.clone()
        without_first.scatter_(1, first.unsqueeze(-1), False)
        second = without_first.float().argmax(dim=-1)
        valid_pair = pair_source.sum(dim=-1).ge(2)
        probe = torch.rand_like(valid_pair.float()) < float(getattr(cfg, "router_pairwise_probe_probability", 0.25))
        valid_pair = valid_pair & probe
        if valid_pair.any():
            singleton_i = torch.zeros_like(pair_source)
            singleton_j = torch.zeros_like(pair_source)
            singleton_i.scatter_(1, first.unsqueeze(-1), True)
            singleton_j.scatter_(1, second.unsqueeze(-1), True)
            pair_mask = singleton_i | singleton_j
            nll_i = deterministic_counterfactual_nll(singleton_i)
            nll_j = deterministic_counterfactual_nll(singleton_j)
            nll_pair = deterministic_counterfactual_nll(pair_mask)
            interaction = (-nll_pair + nll_i + nll_j - nll_0).detach()
            pair_syn = synergy[torch.arange(ids.shape[0], device=ids.device), first, second]
            pairwise_interaction_loss = -(interaction[valid_pair] * pair_syn[valid_pair]).mean()
            mean_pair_interaction = interaction[valid_pair].mean()

    h_k, h_subset = policy_entropy_bonus(u_fresh, k_logits_fresh, K_max)
    diversity_temp = float(getattr(cfg, "router_diversity_temperature", 1.0))
    k_mi_loss, layer_mi_loss = mutual_information_diversity_loss(
        k_logits_fresh, u_fresh, temperature=diversity_temp,
    )

    beta_k = getattr(cfg, "beta_entropy_k", 0.15)
    beta_subset = getattr(cfg, "beta_entropy_subset", 0.15)

    # Balance loss removed: old MSE loss forced all layers to ~mean(K)/8 (40% each),
    # preventing specialization. Layer dropout (0.15) already breaks monopoly naturally.
    balance_loss = torch.tensor(0.0, device=ids.device)
    lambda_balance = 0.0
    lambda_synergy = getattr(cfg, "lambda_synergy", 0.10)
    lambda_k_mi = getattr(cfg, "lambda_router_k_mi", 0.0)
    lambda_layer_mi = getattr(cfg, "lambda_router_layer_mi", 0.0)
    k_preference_loss = counterfactual_k_preference_loss(
        k_logits_fresh, k_a, k_b, reward_0, reward_a, reward_b,
        temperature=float(getattr(cfg, "router_k_preference_temperature", 0.05)),
    )
    lambda_k_preference = float(getattr(cfg, "lambda_router_k_preference", 0.0))

    # Target-K Centering Loss: penalizes mean K deviating from balanced sweet spot K* = 2.2
    mean_sampled_k = ((k_a.float() + k_b.float()) / 2.0).mean()
    k_center_loss = (mean_sampled_k - 2.2).pow(2)
    lambda_k_center = getattr(cfg, "lambda_k_center", 0.05)

    loss = (
        policy_loss
        - beta_k * h_k
        - beta_subset * h_subset
        + lambda_balance * balance_loss
        + lambda_synergy * pairwise_interaction_loss
        + lambda_k_center * k_center_loss
        + lambda_k_mi * k_mi_loss
        + lambda_layer_mi * layer_mi_loss
        + lambda_k_preference * k_preference_loss
    )

    win_rate = ((reward_a > reward_0).float() + (reward_b > reward_0).float()).mean().item() / 2.0
    diag = {
        "policy_loss": float(policy_loss.detach().item()),
        "synergy_align_loss": float(pairwise_interaction_loss.detach().item()),
        "pairwise_interaction": float(mean_pair_interaction.detach().item()),
        "win_rate_vs_base": float(win_rate),
        "mean_reward_delta": float((0.5 * (delta_r_a + delta_r_b)).mean().item()),
        "h_k_raw": float(h_k.detach().item()),
        "h_k_weighted": float((beta_k * h_k).detach().item()),
        "h_subset_raw": float(h_subset.detach().item()),
        "h_subset_weighted": float((beta_subset * h_subset).detach().item()),
        "mean_k_sampled": float(((k_a.float() + k_b.float()) / 2).mean().item()),
        "k_mi_loss": float(k_mi_loss.detach().item()),
        "layer_mi_loss": float(layer_mi_loss.detach().item()),
        "k_preference_loss": float(k_preference_loss.detach().item()),
        "k0_probability": float(F.softmax(k_logits_fresh.detach(), dim=-1)[:, 0].mean().item()),
        "argmax_k_distribution": {
            int(k): float((k_logits_fresh.detach().argmax(dim=-1) == k).float().mean().item())
            for k in range(k_logits_fresh.shape[-1])
        },
        "sampled_k": [int(k.item()) for k in torch.cat([k_a.view(-1), k_b.view(-1)])],
    }
    return loss, diag
