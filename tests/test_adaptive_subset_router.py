import torch

from src.routing.adaptive_subset_router import AdaptiveSubsetRouter, sample_route_actions


def _router() -> AdaptiveSubsetRouter:
    return AdaptiveSubsetRouter(
        candidate_layers=[8, 12, 15, 18, 21, 24],
        hidden_dim=16,
        n_backbone_layers=32,
        max_active_layers=4,
        router_dim=16,
    )


def test_router_can_choose_no_steering_and_alpha_stays_bounded():
    router = _router().eval()
    with torch.no_grad():
        router.k_head.weight.zero_()
        router.k_head.bias.zero_()
        router.k_head.bias[0] = 2.0
        router.alpha_head.weight.zero_()
        router.alpha_head.bias.zero_()
        router.alpha_head.bias[-1] = 2.0

    hidden = torch.randn(2, 5, 16)
    prompt_mask = torch.tensor([[True, True, True, False, False], [True] * 5])
    decision = router.choose(hidden, prompt_mask)

    assert decision.k.tolist() == [0, 0]
    assert not decision.mask.any()
    assert decision.alpha.tolist() == [1.0, 1.0]


def test_router_selects_requested_k_and_handles_pairwise_synergy():
    router = _router().eval()
    with torch.no_grad():
        router.k_head.weight.zero_()
        router.k_head.bias.zero_()
        router.k_head.bias[2] = 3.0
        router.utility_head.weight.zero_()
        router.utility_head.bias.zero_()
        router.synergy_query.weight.zero_()

    decision = router.choose(torch.randn(1, 4, 16), torch.ones(1, 4, dtype=torch.bool))

    assert decision.k.tolist() == [2]
    assert int(decision.mask.sum()) == 2
    assert 1.0 <= float(decision.alpha[0]) <= 2.0


def test_sampled_actions_include_zero_k_and_have_finite_policy_terms():
    router = _router().train()
    hidden = torch.randn(3, 6, 16)
    prompt_mask = torch.ones(3, 6, dtype=torch.bool)

    decision, log_prob, entropy = sample_route_actions(router, hidden, prompt_mask)

    assert decision.mask.shape == (3, 6)
    assert decision.k.min() >= 0 and decision.k.max() <= 4
    assert torch.equal(decision.mask.sum(dim=-1), decision.k)
    assert torch.all((decision.alpha >= 1.0) & (decision.alpha <= 2.0))
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()
    (log_prob.mean() - entropy.mean()).backward()
    assert router.k_head.weight.grad is not None
    assert torch.isfinite(router.k_head.weight.grad).all()


def test_no_steering_action_does_not_train_irrelevant_alpha_choice():
    router = _router().train()
    with torch.no_grad():
        router.k_head.weight.zero_()
        router.k_head.bias.fill_(-20.0)
        router.k_head.bias[0] = 20.0

    decision, log_prob, entropy = sample_route_actions(
        router, torch.randn(1, 4, 16), torch.ones(1, 4, dtype=torch.bool)
    )
    (log_prob.sum() + entropy.sum()).backward()

    assert decision.k.tolist() == [0]
    assert router.alpha_head.weight.grad is None or torch.count_nonzero(router.alpha_head.weight.grad) == 0
