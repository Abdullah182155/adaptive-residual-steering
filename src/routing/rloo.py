import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional


class PairEvidenceTable:
    """Count-weighted empirical interaction evidence for candidate layer pairs."""

    def __init__(self, shrinkage: float = 8.0):
        self.shrinkage = max(0.0, float(shrinkage))
        self._rows: Dict[Tuple[int, int], Dict[str, float]] = {}

    def update(self, first: int, second: int, interaction: float) -> None:
        key = tuple(sorted((int(first), int(second))))
        if key[0] == key[1]:
            return
        row = self._rows.setdefault(key, {"count": 0.0, "sum": 0.0})
        row["count"] += 1.0
        row["sum"] += float(interaction)

    def estimate(self, first: int, second: int) -> float:
        row = self._rows.get(tuple(sorted((int(first), int(second)))))
        if not row or row["count"] <= 0:
            return 0.0
        mean = row["sum"] / row["count"]
        return float((row["count"] / (row["count"] + self.shrinkage)) * mean)

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            f"L{first}+L{second}": {
                "observations": int(row["count"]),
                "mean_interaction": row["sum"] / max(1.0, row["count"]),
                "shrunken_interaction": self.estimate(first, second),
            }
            for (first, second), row in sorted(self._rows.items())
        }


def generation_anchor_reward(*args, **kwargs) -> torch.Tensor:
    """Optional generated-answer reward hook; disabled unless a runner supplies it.

    RLOO remains teacher-forced by default so its memory profile is unchanged.
    The training loop may later inject a detached exact-answer anchor estimate
    at a sparse interval without coupling autoregressive graphs to bootstrap.
    """
    device = kwargs.get("device", torch.device("cpu"))
    return torch.zeros((), device=device)

def per_example_answer_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    answer_tail_tokens: int = 0,
    answer_tail_weight: float = 1.0,
    calc_mask: Optional[torch.Tensor] = None,
    calc_target_mode: str = "all",
    calc_weight: float = 1.0,
) -> torch.Tensor:
    """Mean answer NLL, optionally emphasizing calculation tokens and completion tail.

    Engine v2: When calc_target_mode == 'calculation', non-calculation tokens
    are zero-weighted so that SteerNets optimize solely for numerical and
    arithmetic correctness rather than conversational English prose.
    """
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    token_nll = F.cross_entropy(
        shift_logits.transpose(1, 2), shift_labels, ignore_index=-100, reduction="none"
    )
    valid = shift_labels.ne(-100)
    weights = valid.to(token_nll.dtype)

    if calc_mask is not None:
        shift_calc = calc_mask[:, 1:].bool() if calc_mask.shape[1] > 1 else calc_mask.bool()
        min_len = min(weights.shape[1], shift_calc.shape[1])
        if calc_target_mode == "calculation":
            weights[:, :min_len] = weights[:, :min_len] * shift_calc[:, :min_len].to(weights.dtype)
            if weights.shape[1] > min_len:
                weights[:, min_len:] = 0.0
            zero_weights = weights.sum(dim=1).eq(0)
            if zero_weights.any():
                weights[zero_weights] = valid[zero_weights].to(token_nll.dtype)
        elif calc_target_mode == "calculation_weighted":
            weights[:, :min_len] = torch.where(
                shift_calc[:, :min_len] & valid[:, :min_len],
                weights[:, :min_len] * float(calc_weight),
                weights[:, :min_len],
            )

    tail_count = max(0, int(answer_tail_tokens))
    if tail_count and float(answer_tail_weight) != 1.0:
        for row in range(valid.shape[0]):
            positions = valid[row].nonzero(as_tuple=False).flatten()
            if positions.numel():
                weights[row, positions[-tail_count:]] = weights[row, positions[-tail_count:]] * float(answer_tail_weight)
    return (token_nll * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)


def per_example_execution_reward(
    logits: torch.Tensor,
    labels: torch.Tensor,
    calc_mask: Optional[torch.Tensor] = None,
    answer_tail_tokens: int = 8,
    outcome_weight: float = 1.0,
    nll_weight: float = 0.2,
) -> torch.Tensor:
    """Execution outcome reward for RLOO counterfactual evaluation (Engine v2).

    Instead of purely minimizing prose NLL (which caused router collapse), rewards
    exact token prediction on calculation steps and the final answer tail.
    A route only wins if it achieves correct execution on math/outcome tokens.
    """
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(-100)
    preds = shift_logits.argmax(dim=-1)
    tok_correct = (preds == shift_labels) & valid

    if calc_mask is not None:
        shift_calc = calc_mask[:, 1:].bool() if calc_mask.shape[1] > 1 else calc_mask.bool()
        min_len = min(valid.shape[1], shift_calc.shape[1])
        target_mask = valid.clone()
        target_mask[:, :min_len] = target_mask[:, :min_len] & shift_calc[:, :min_len]
        zero_rows = target_mask.sum(dim=1).eq(0)
        if zero_rows.any():
            for row in range(valid.shape[0]):
                if zero_rows[row]:
                    pos = valid[row].nonzero(as_tuple=False).flatten()
                    if pos.numel():
                        target_mask[row, pos[-answer_tail_tokens:]] = True
    else:
        target_mask = torch.zeros_like(valid)
        for row in range(valid.shape[0]):
            pos = valid[row].nonzero(as_tuple=False).flatten()
            if pos.numel():
                target_mask[row, pos[-answer_tail_tokens:]] = True

    denom = target_mask.float().sum(dim=1).clamp(min=1.0)
    acc = (tok_correct & target_mask).float().sum(dim=1) / denom

    token_nll = F.cross_entropy(
        shift_logits.transpose(1, 2), shift_labels, ignore_index=-100, reduction="none"
    )
    nll = (token_nll * target_mask.float()).sum(dim=1) / denom

    acc_reward = (2.0 * acc - 1.0) * float(outcome_weight)
    return acc_reward - float(nll_weight) * nll

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
    h_k = -torch.where(p_k > 0, p_k * log_p_k, torch.zeros_like(log_p_k)).sum(dim=-1).mean()

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
    k_actions: torch.Tensor,
    reward_0: torch.Tensor,
    action_rewards: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Distil measured K-action rewards into deterministic K logits.

    ``k_actions`` has shape ``(B, A)`` and may contain any evaluated subset of
    ``{1, ..., K_max}``; the unsteered K=0 reward is always included.
    K=0 competes symmetrically with every measured steered action.  The
    steered rewards already include their proportional route tariff, so no
    additional baseline margin or aggregate-logit hinge is applied here.
    """
    if not torch.isfinite(k_logits).all():
        raise FloatingPointError("non-finite K logits in counterfactual preference loss")
    if not torch.isfinite(reward_0).all() or not torch.isfinite(action_rewards).all():
        raise FloatingPointError("non-finite counterfactual reward in K preference loss")
    B, n_actions = k_logits.shape
    rewards_sum = torch.zeros(B, n_actions, device=k_logits.device, dtype=k_logits.dtype)
    counts = torch.zeros_like(rewards_sum)
    zero = torch.zeros(B, 1, device=k_logits.device, dtype=torch.long)
    actions = torch.cat([zero, k_actions.long()], dim=1).clamp(0, n_actions - 1)
    rewards = torch.cat([reward_0.to(k_logits.dtype).unsqueeze(1), action_rewards.to(k_logits.dtype)], dim=1)
    rewards_sum.scatter_add_(1, actions, rewards)
    counts.scatter_add_(1, actions, torch.ones_like(rewards))
    observed = counts > 0
    mean_reward = rewards_sum / counts.clamp_min(1)
    target_scores = (mean_reward / max(float(temperature), 1e-4)).masked_fill(~observed, float("-inf"))
    target = F.softmax(target_scores, dim=-1)
    if not torch.isfinite(target).all():
        raise FloatingPointError("non-finite K preference target")
    # Legal-action masking may put ``-inf`` in an unavailable K logit.  A
    # zero target probability multiplied by that value is mathematically zero
    # but numerically NaN, so form only the observed cross-entropy terms.
    log_probs = F.log_softmax(k_logits, dim=-1)
    if not torch.isfinite(log_probs).all():
        raise FloatingPointError("non-finite K preference log probabilities")
    terms = torch.where(target > 0, target * log_probs, torch.zeros_like(log_probs))
    return -terms.sum(dim=-1).mean()


def capped_route_tariff(
    per_layer_cost: torch.Tensor,
    k_actions: torch.Tensor,
    raw_delta: torch.Tensor,
    tariff_scale: float,
    max_positive_gain_fraction: float,
) -> torch.Tensor:
    """Price route size without reversing a measured positive utility.

    The first bootstrap epoch linearly ramps ``tariff_scale`` from zero.  For
    a route that already improves on K=0, its total tariff is bounded by a
    configured fraction of that measured improvement.  This keeps compute
    pressure proportional while preventing a fixed K tariff from making a
    genuinely useful K=2 route look worse than the baseline.
    """
    if not torch.isfinite(per_layer_cost).all() or not torch.isfinite(raw_delta).all():
        raise FloatingPointError("non-finite route tariff inputs")
    scale = max(0.0, min(1.0, float(tariff_scale)))
    fraction = max(0.0, min(1.0, float(max_positive_gain_fraction)))
    base_tariff = per_layer_cost.unsqueeze(1) * k_actions.to(per_layer_cost.dtype) * scale
    return torch.minimum(base_tariff, raw_delta.clamp_min(0.0) * fraction)

def compute_router_rloo_loss(
    model,
    ids: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    cfg,
    prompt_mask: torch.Tensor = None,
    shield_mask: torch.Tensor = None,
    calc_mask: Optional[torch.Tensor] = None,
    tariff_scale: float = 1.0,
    k_preference_temperature: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
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
        # Train against the same deterministic representation used at
        # deployment.  Keeping only the router in eval mode left a frozen
        # backbone's dropout (or future stochastic modules) active while
        # constructing the detached policy state, even though every reward
        # arm below was evaluated deterministically.
        model.eval()
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
    # Phase 1B certifies a finite layer pool.  Counterfactual training must
    # use that exact pool; otherwise RLOO can reward layers that never passed
    # acceptance or pair co-adaptation and deployment no longer matches the
    # curriculum.  Lightweight test doubles may not expose the helper.
    if hasattr(model, "mask_routing_logits"):
        u_policy = model.mask_routing_logits(u_fresh)
        k_logits_policy = model.mask_routing_k_logits(k_logits_fresh)
        if hasattr(model, "routing_eligibility_mask"):
            eligible_count = int(model.routing_eligibility_mask(ids.device).sum().item())
        else:
            eligible_count = u_fresh.shape[-1]
    else:
        u_policy = u_fresh
        k_logits_policy = k_logits_fresh
        eligible_count = u_fresh.shape[-1]
    synergy = getattr(model.router, "_last_synergy_matrix", None)
    # Do this before Plackett--Luce sampling: a NaN synergy bonus otherwise
    # poisons categorical logits before the later pairwise diagnostic can
    # observe it.  Bootstrap treats this as an explicit invalid-numerics
    # event and returns to the certified fixed-SteerNet fallback.
    if synergy is not None and not torch.isfinite(synergy).all():
        raise FloatingPointError("non-finite router synergy matrix before subset sampling")
    K_max = min(model.cfg.max_active_layers, eligible_count)
    if K_max < 1:
        raise RuntimeError("RLOO requires at least one eligible routing layer")

    # K is a tiny discrete action space (0..4), so it is both cheaper in
    # variance and more scientifically defensible to score every non-zero K
    # than to hope two random samples cover it.  This is *not* a fixed route:
    # each K still samples a prompt-conditioned Plackett-Luce subset, and the
    # K head is supervised only by these counterfactual rewards.
    k_actions = torch.arange(1, K_max + 1, device=ids.device, dtype=torch.long)
    k_actions = k_actions.unsqueeze(0).expand(ids.shape[0], -1)
    subset_masks = []
    subset_logprobs = []
    for arm in range(K_max):
        route_mask, route_logprob = sample_subset_plackett_luce(
            u_policy, k_actions[:, arm], synergy_matrix=synergy
        )
        subset_masks.append(route_mask)
        subset_logprobs.append(route_logprob)

    old_override = model._routing_override
    use_execution_reward = (getattr(cfg, "router_reward_type", "nll") == "execution")
    outcome_weight = float(getattr(cfg, "router_outcome_reward_weight", 1.0))
    calc_target_mode = getattr(cfg, "steer_loss_target", "calculation")
    calc_weight = float(getattr(cfg, "calc_token_weight", 3.0))

    def counterfactual_score(route_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate one route (or baseline if None). Returns (reward, nll)."""
        if route_mask is None:
            model.set_routing_override("none")
        else:
            model.set_routing_override("matrix", route_mask.detach())
        model.set_router_context_mask(prompt_mask)
        if hasattr(model, "set_prompt_shield_mask"):
            model.set_prompt_shield_mask(shield_mask)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=ids.is_cuda):
                out = model(input_ids=ids, attention_mask=mask, use_cache=False)
                nll = per_example_answer_nll(
                    out.logits,
                    labels,
                    answer_tail_tokens=getattr(cfg, "answer_tail_tokens", 0),
                    answer_tail_weight=getattr(cfg, "answer_tail_weight", 1.0),
                    calc_mask=calc_mask,
                    calc_target_mode=calc_target_mode,
                    calc_weight=calc_weight,
                )
                if use_execution_reward:
                    reward = per_example_execution_reward(
                        out.logits,
                        labels,
                        calc_mask=calc_mask,
                        answer_tail_tokens=getattr(cfg, "answer_tail_tokens", 8),
                        outcome_weight=outcome_weight,
                    )
                else:
                    reward = -nll
                return reward, nll

    def deterministic_counterfactual_score(route_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score one auxiliary counterfactual without dropout noise."""
        modes = {module: module.training for module in model.modules()}
        prior_override = model._routing_override
        model.eval()
        try:
            return counterfactual_score(route_mask)
        finally:
            if prior_override is None:
                model.set_routing_override(None)
            else:
                model.set_routing_override(prior_override[0], prior_override[1])
            for module, was_training in modes.items():
                module.train(was_training)

    try:
        model.eval()
        arm_rewards_list = []
        arm_nlls_list = []
        for route_mask in subset_masks:
            rew, nll_val = counterfactual_score(route_mask)
            arm_rewards_list.append(rew)
            arm_nlls_list.append(nll_val)
        arm_rewards = torch.stack(arm_rewards_list, dim=1)
        arm_nll = torch.stack(arm_nlls_list, dim=1)

        # Arm 0: Ground-truth unsteered baseline anchor
        reward_0, nll_0 = counterfactual_score(None)
    finally:
        if old_override is None:
            model.set_routing_override(None)
        else:
            model.set_routing_override(old_override[0], old_override[1])
        for module, was_training in module_modes.items():
            module.train(was_training)

    reward_0 = reward_0.detach()
    # Price compute without reversing a measured positive utility.  The
    # tariff is linearly warmed by the bootstrap caller and capped to a
    # fraction of a route's raw gain over K=0.
    absolute_cost = float(getattr(cfg, "lambda_cost", 0.005))
    relative_cost = float(getattr(cfg, "router_relative_k_cost", 0.0))
    per_layer_cost = torch.maximum(
        torch.full_like(nll_0, absolute_cost),
        nll_0.detach() * relative_cost,
    )
    raw_delta = arm_rewards - reward_0.unsqueeze(1)
    max_fraction = float(getattr(cfg, "router_tariff_max_positive_gain_fraction", 0.50))
    tariff = capped_route_tariff(
        per_layer_cost, k_actions, raw_delta, tariff_scale, max_fraction,
    )
    action_rewards = (arm_rewards - tariff).detach()

    # Anchor-grounded leave-one-out advantage for each sampled subset.  K
    # itself is evaluated exhaustively and receives direct action-value
    # supervision below; this policy term learns which *layers* are useful at
    # a given K without an off-policy, forced-K gradient.
    reward_sum = action_rewards.sum(dim=1, keepdim=True) + reward_0.unsqueeze(1)
    advantages = action_rewards - (reward_sum - action_rewards) / float(K_max)
    subset_logprobs_t = torch.stack(subset_logprobs, dim=1)
    policy_loss = -(subset_logprobs_t * advantages).mean()

    # Direct pairwise *interaction* signal.  A route's gain over the base is
    # not synergy: two individually useful layers would otherwise make every
    # co-selected pair look positive.  For a sampled pair (i,j), estimate
    # I_ij = R({i,j}) - R({i}) - R({j}) + R(empty), then align only M_ij to it.
    # This preserves the cooperative-router hypothesis while giving destructive
    # pairs a negative target instead of allowing universal +1 saturation.
    reward_deltas = (action_rewards - reward_0.unsqueeze(1)).detach()
    pairwise_interaction_loss = torch.zeros((), device=ids.device)
    mean_pair_interaction = torch.zeros((), device=ids.device)
    pair_utility = {}
    if synergy is not None:
        if not torch.isfinite(synergy).all():
            raise FloatingPointError("non-finite router synergy matrix")
        # K=2 is the first non-trivial pair route and therefore measures a
        # clean second-order interaction instead of an arbitrary pair inside
        # a much larger selected set.
        pair_source = subset_masks[1] if K_max >= 2 else subset_masks[0]
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
            rew_i, _ = deterministic_counterfactual_score(singleton_i)
            rew_j, _ = deterministic_counterfactual_score(singleton_j)
            rew_pair, _ = deterministic_counterfactual_score(pair_mask)
            interaction = (rew_pair - rew_i - rew_j + reward_0).detach()
            # Batch-local pair evidence is deliberately reported before it is
            # collapsed into a scalar loss.  This makes positive synergy
            # auditable against both its raw reward and the K=2 tariff.
            for b in torch.nonzero(valid_pair, as_tuple=False).flatten().tolist():
                first_layer = int(model.target_layers[int(first[b].item())])
                second_layer = int(model.target_layers[int(second[b].item())])
                evidence = getattr(model.router, "pair_evidence", None)
                if evidence is None:
                    evidence = PairEvidenceTable()
                    model.router.pair_evidence = evidence
                evidence.update(first_layer, second_layer, float(interaction[b].item()))
                key = f"L{int(model.target_layers[int(first[b].item())])}+L{int(model.target_layers[int(second[b].item())])}"
                row = pair_utility.setdefault(key, {"observations": 0, "raw_reward_sum": 0.0, "net_reward_sum": 0.0, "interaction_sum": 0.0})
                row["observations"] += 1
                row["raw_reward_sum"] += float(rew_pair[b].item())
                row["net_reward_sum"] += float((rew_pair[b] - tariff[b, 1]).item())
                row["interaction_sum"] += float(interaction[b].item())
            pair_syn = synergy[torch.arange(ids.shape[0], device=ids.device), first, second]
            target_scale = max(float(getattr(cfg, "router_synergy_target_scale", 0.10)), 1e-4)
            interaction_target = torch.tanh(interaction / target_scale)
            if not (
                torch.isfinite(pair_syn[valid_pair]).all()
                and torch.isfinite(interaction_target[valid_pair]).all()
            ):
                raise FloatingPointError("non-finite pairwise synergy interaction")
            pairwise_interaction_loss = F.smooth_l1_loss(
                pair_syn[valid_pair], interaction_target[valid_pair], beta=0.25,
            )
            mean_pair_interaction = interaction[valid_pair].mean()

    for row in pair_utility.values():
        n_obs = max(1, int(row.pop("observations")))
        row["observations"] = n_obs
        row["raw_reward"] = row.pop("raw_reward_sum") / n_obs
        row["cost_adjusted_reward"] = row.pop("net_reward_sum") / n_obs
        row["interaction"] = row.pop("interaction_sum") / n_obs
    if hasattr(model.router, "pair_evidence"):
        pair_utility["evidence_table"] = model.router.pair_evidence.as_dict()

    h_k, h_subset = policy_entropy_bonus(u_policy, k_logits_policy, K_max)
    diversity_temp = float(getattr(cfg, "router_diversity_temperature", 1.0))
    k_mi_loss, layer_mi_loss = mutual_information_diversity_loss(
        k_logits_policy, u_policy, temperature=diversity_temp,
    )

    beta_k = getattr(cfg, "beta_entropy_k", 0.15)
    beta_subset = getattr(cfg, "beta_entropy_subset", 0.15)

    # Balance loss removed: old MSE loss forced all layers to ~mean(K)/8 (40% each),
    # preventing specialization. Layer dropout (0.15) already breaks monopoly naturally.
    balance_loss = torch.tensor(0.0, device=ids.device)
    lambda_balance = 0.0
    lambda_synergy = getattr(cfg, "lambda_synergy", 0.10)
    # RC-1 fix: previously hardcoded to default=0.0, silently disabling all three
    # diversity losses regardless of config. Config sets lambda_router_k_mi=0.10,
    # lambda_router_layer_mi=0.10, lambda_router_k_preference=0.25 -- those must
    # actually be used. The old getattr(..., 0.0) was a backwards-compat stub
    # that was never updated when these hyperparams were added to config.
    lambda_k_mi = float(getattr(cfg, "lambda_router_k_mi", 0.10))
    lambda_layer_mi = float(getattr(cfg, "lambda_router_layer_mi", 0.10))
    preference_temperature = (
        float(k_preference_temperature) if k_preference_temperature is not None
        else float(getattr(cfg, "router_k_preference_temperature", 0.10))
    )
    k_preference_loss = counterfactual_k_preference_loss(
        k_logits_policy, k_actions, reward_0, action_rewards,
        temperature=preference_temperature,
    )
    lambda_k_preference = float(getattr(cfg, "lambda_router_k_preference", 0.25))
    synergy_prior_loss = model.router.synergy_prior_loss(synergy) if synergy is not None else torch.zeros((), device=ids.device)
    lambda_synergy_prior = float(getattr(cfg, "lambda_synergy_prior", 0.0))

    # Kept at zero by default for backwards-compatible configuration.  The
    # reported mean is exhaustive K coverage, not a biased random sample.
    mean_sampled_k = k_actions.float().mean()
    k_center_loss = (mean_sampled_k - 2.2).pow(2)
    lambda_k_center = getattr(cfg, "lambda_k_center", 0.0)  # BUG-C1 fix: fallback 0.05→0.0 (matches config.py)

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
        + lambda_synergy_prior * synergy_prior_loss
    )

    win_rate = (action_rewards > reward_0.unsqueeze(1)).float().mean().item()
    # Keep reward accounting observable.  Without raw-versus-net values, a
    # K=0 or single-layer collapse is indistinguishable from a genuine utility
    # preference and cost hyperparameters become guesswork.
    k_utility = {
        0: {
            "raw_reward": float(reward_0.mean().item()),
            "cost_adjusted_reward": float(reward_0.mean().item()),
            "nll": float(nll_0.mean().item()),
            "cost": 0.0,
        }
    }
    layer_utility = {}
    for arm, k_value in enumerate(range(1, K_max + 1)):
        k_utility[k_value] = {
            "raw_reward": float(arm_rewards[:, arm].mean().item()),
            "cost_adjusted_reward": float(action_rewards[:, arm].mean().item()),
            "nll": float(arm_nll[:, arm].mean().item()),
            "cost": float(tariff[:, arm].mean().item()),
            "raw_delta_vs_k0": float(raw_delta[:, arm].mean().item()),
        }
        selected = subset_masks[arm]
        for pos, layer in enumerate(model.target_layers):
            chosen = selected[:, pos]
            if chosen.any():
                layer_utility[int(layer)] = {
                    "observations": int(chosen.sum().item()),
                    "raw_reward": float(arm_rewards[chosen, arm].mean().item()),
                    "cost_adjusted_reward": float(action_rewards[chosen, arm].mean().item()),
                }
    diag = {
        "policy_loss": float(policy_loss.detach().item()),
        "synergy_align_loss": float(pairwise_interaction_loss.detach().item()),
        "pairwise_interaction": float(mean_pair_interaction.detach().item()),
        "win_rate_vs_base": float(win_rate),
        "mean_reward_delta": float(reward_deltas.mean().item()),
        "h_k_raw": float(h_k.detach().item()),
        "h_k_weighted": float((beta_k * h_k).detach().item()),
        "h_subset_raw": float(h_subset.detach().item()),
        "h_subset_weighted": float((beta_subset * h_subset).detach().item()),
        "mean_k_sampled": float(mean_sampled_k.detach().item()),
        "k_mi_loss": float(k_mi_loss.detach().item()),
        "layer_mi_loss": float(layer_mi_loss.detach().item()),
        "k_preference_loss": float(k_preference_loss.detach().item()),
        "k_preference_temperature": preference_temperature,
        "synergy_prior_loss": float(synergy_prior_loss.detach().item()),
        "k0_probability": float(F.softmax(k_logits_policy.detach(), dim=-1)[:, 0].mean().item()),
        "mean_per_layer_cost": float(per_layer_cost.mean().item()),
        "tariff_scale": float(tariff_scale),
        "mean_tariff": float(tariff.mean().item()),
        "argmax_k_distribution": {
            int(k): float((k_logits_policy.detach().argmax(dim=-1) == k).float().mean().item())
            for k in range(k_logits_policy.shape[-1])
        },
        "counterfactual_k_actions": list(range(1, K_max + 1)),
        "counterfactual_utility_by_k": k_utility,
        "counterfactual_utility_by_layer": layer_utility,
        "counterfactual_utility_by_pair": pair_utility,
    }
    return loss, diag
