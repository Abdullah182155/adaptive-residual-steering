"""Question-conditioned router for selecting zero to four steering layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RouteDecision:
    """A per-example layer subset and bounded steering strength."""

    mask: torch.BoolTensor
    k: torch.LongTensor
    alpha: torch.Tensor


class AdaptiveSubsetRouter(nn.Module):
    """Choose a context-dependent subset, including the no-steering action.

    The model scores layers jointly and predicts pairwise compatibility. Greedy
    selection uses both signals, while a separate head predicts cardinality in
    ``{0, ..., max_active_layers}`` and a discrete strength in ``[1, 2]``.
    """

    def __init__(
        self,
        candidate_layers: Sequence[int],
        hidden_dim: int = 2560,
        n_backbone_layers: int = 32,
        max_active_layers: int = 4,
        alpha_values: Sequence[float] = (1.0, 1.25, 1.5, 1.75, 2.0),
        router_dim: int = 128,
    ) -> None:
        super().__init__()
        layers = tuple(int(layer) for layer in candidate_layers)
        alphas = tuple(float(value) for value in alpha_values)
        if not layers or len(set(layers)) != len(layers):
            raise ValueError("candidate_layers must be a non-empty unique sequence")
        if any(layer < 0 or layer >= n_backbone_layers for layer in layers):
            raise ValueError("candidate layer index is outside the backbone")
        if max_active_layers < 1 or max_active_layers > len(layers):
            raise ValueError("max_active_layers must be between one and candidate count")
        if len(alphas) < 2 or any(value < 1.0 or value > 2.0 for value in alphas):
            raise ValueError("alpha_values must contain values in [1, 2]")

        self.candidate_layers = layers
        self.max_active_layers = int(max_active_layers)
        self.alpha_values = alphas
        self.n_backbone_layers = int(n_backbone_layers)
        self.n_candidates = len(layers)

        self.prompt_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, router_dim),
            nn.GELU(),
            nn.LayerNorm(router_dim),
        )
        self.layer_embedding = nn.Embedding(self.n_candidates, router_dim)
        depths = torch.tensor(
            [layer / max(1, n_backbone_layers - 1) for layer in layers],
            dtype=torch.float32,
        )
        self.register_buffer("layer_depths", depths)
        self.depth_embedding = nn.Sequential(
            nn.Linear(1, router_dim), nn.Tanh()
        )
        self.joint_layer = nn.TransformerEncoderLayer(
            d_model=router_dim,
            nhead=4,
            dim_feedforward=router_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.utility_head = nn.Linear(router_dim, 1)
        self.synergy_query = nn.Linear(router_dim, router_dim, bias=False)
        self.k_head = nn.Linear(router_dim, max_active_layers + 1)
        self.alpha_head = nn.Linear(router_dim, len(alphas))

    def pool_prompt(
        self, hidden: torch.Tensor, prompt_mask: torch.BoolTensor
    ) -> torch.Tensor:
        """Pool only valid prompt positions, excluding padding and completions."""
        if hidden.ndim != 3 or prompt_mask.shape != hidden.shape[:2]:
            raise ValueError("hidden must be (batch, time, dim) with a matching mask")
        mask = prompt_mask.to(device=hidden.device, dtype=hidden.dtype)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / count
        return self.prompt_encoder(pooled.float())

    def forward(
        self, hidden: torch.Tensor, prompt_mask: torch.BoolTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-layer utilities, pair synergy, k logits, and alpha logits."""
        pooled = self.pool_prompt(hidden, prompt_mask)
        layer_ids = torch.arange(self.n_candidates, device=hidden.device)
        embeddings = self.layer_embedding(layer_ids).unsqueeze(0)
        depths = self.depth_embedding(self.layer_depths[:, None]).unsqueeze(0)
        tokens = pooled[:, None, :] + embeddings + depths
        joint = self.joint_layer(tokens)

        utilities = self.utility_head(joint).squeeze(-1)
        queries = self.synergy_query(joint)
        synergy = torch.bmm(queries, joint.transpose(1, 2)) / (joint.shape[-1] ** 0.5)
        synergy = 0.5 * (synergy + synergy.transpose(1, 2))
        diagonal = torch.eye(self.n_candidates, device=hidden.device, dtype=torch.bool)
        synergy = synergy.masked_fill(diagonal.unsqueeze(0), 0.0)

        summary = joint.mean(dim=1) + pooled
        return utilities, synergy, self.k_head(summary), self.alpha_head(summary)

    @torch.no_grad()
    def choose(
        self, hidden: torch.Tensor, prompt_mask: torch.BoolTensor
    ) -> RouteDecision:
        """Select the highest scoring deterministic route for each prompt."""
        utilities, synergy, k_logits, alpha_logits = self(hidden, prompt_mask)
        k = k_logits.argmax(dim=-1)
        alpha_ids = alpha_logits.argmax(dim=-1)
        mask = torch.zeros(
            hidden.shape[0], self.n_candidates, dtype=torch.bool, device=hidden.device
        )
        for batch_idx in range(hidden.shape[0]):
            selected: list[int] = []
            for _ in range(int(k[batch_idx].item())):
                adjusted = utilities[batch_idx].clone()
                if selected:
                    adjusted += synergy[batch_idx, selected].mean(dim=0)
                if selected:
                    adjusted[selected] = float("-inf")
                next_layer = int(adjusted.argmax().item())
                selected.append(next_layer)
                mask[batch_idx, next_layer] = True
        alpha_lut = torch.tensor(self.alpha_values, device=hidden.device, dtype=hidden.dtype)
        alpha = alpha_lut[alpha_ids]
        alpha = torch.where(k.eq(0), torch.ones_like(alpha), alpha)
        return RouteDecision(mask=mask, k=k, alpha=alpha)


def sample_route_actions(
    router: AdaptiveSubsetRouter,
    hidden: torch.Tensor,
    prompt_mask: torch.BoolTensor,
) -> tuple[RouteDecision, torch.Tensor, torch.Tensor]:
    """Sample differentiable-policy actions for offline RLOO training.

    Returns a route, its joint log probability, and the sum of categorical
    entropies. ``k=0`` naturally yields an empty subset and unit alpha.
    """
    utilities, synergy, k_logits, alpha_logits = router(hidden, prompt_mask)
    batch_size = hidden.shape[0]
    k_dist = torch.distributions.Categorical(logits=k_logits)
    k = k_dist.sample()
    log_prob = k_dist.log_prob(k)
    entropy = k_dist.entropy()

    alpha_dist = torch.distributions.Categorical(logits=alpha_logits)
    alpha_ids = alpha_dist.sample()
    alpha_values = torch.tensor(
        router.alpha_values, device=hidden.device, dtype=hidden.dtype
    )
    alpha = alpha_values[alpha_ids]
    uses_steering = k.gt(0)
    log_prob = log_prob + torch.where(uses_steering, alpha_dist.log_prob(alpha_ids), 0.0)
    entropy = entropy + torch.where(uses_steering, alpha_dist.entropy(), 0.0)

    mask = torch.zeros(
        batch_size, router.n_candidates, dtype=torch.bool, device=hidden.device
    )
    for step in range(router.max_active_layers):
        active = k > step
        if not bool(active.any()):
            break
        adjusted = utilities.clone()
        if step:
            adjusted += torch.bmm(mask.float().unsqueeze(1), synergy).squeeze(1) / step
        adjusted = adjusted.masked_fill(mask.clone(), float("-inf"))
        layer_dist = torch.distributions.Categorical(logits=adjusted)
        selected = layer_dist.sample()
        step_log_prob = layer_dist.log_prob(selected)
        log_prob = log_prob + torch.where(active, step_log_prob, 0.0)
        entropy = entropy + torch.where(active, layer_dist.entropy(), 0.0)
        newly_selected = F.one_hot(selected, router.n_candidates).bool() & active[:, None]
        mask |= newly_selected

    alpha = torch.where(k.eq(0), torch.ones_like(alpha), alpha)
    return RouteDecision(mask=mask, k=k, alpha=alpha), log_prob, entropy
