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
        return self.encode_pooled(self.raw_prompt_pool(hidden, prompt_mask))

    @staticmethod
    def raw_prompt_pool(
        hidden: torch.Tensor, prompt_mask: torch.BoolTensor
    ) -> torch.Tensor:
        """Mean-pool prompt states before the trainable router encoder."""
        if hidden.ndim != 3 or prompt_mask.shape != hidden.shape[:2]:
            raise ValueError("hidden must be (batch, time, dim) with a matching mask")
        mask = prompt_mask.to(device=hidden.device, dtype=hidden.dtype)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / count

    def encode_pooled(self, pooled: torch.Tensor):
        """Return route heads from pre-pooled prompt representations."""
        pooled = self.prompt_encoder(pooled.float())
        layer_ids = torch.arange(self.n_candidates, device=pooled.device)
        embeddings = self.layer_embedding(layer_ids).unsqueeze(0)
        depths = self.depth_embedding(self.layer_depths[:, None]).unsqueeze(0)
        tokens = pooled[:, None, :] + embeddings + depths
        joint = self.joint_layer(tokens)

        utilities = self.utility_head(joint).squeeze(-1)
        queries = self.synergy_query(joint)
        synergy = torch.bmm(queries, joint.transpose(1, 2)) / (joint.shape[-1] ** 0.5)
        synergy = 0.5 * (synergy + synergy.transpose(1, 2))
        diagonal = torch.eye(self.n_candidates, device=pooled.device, dtype=torch.bool)
        synergy = synergy.masked_fill(diagonal.unsqueeze(0), 0.0)
        summary = joint.mean(dim=1) + pooled
        return utilities, synergy, self.k_head(summary), self.alpha_head(summary)

    def forward(
        self, hidden: torch.Tensor, prompt_mask: torch.BoolTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-layer utilities, pair synergy, k logits, and alpha logits."""
        return self.encode_pooled(self.raw_prompt_pool(hidden, prompt_mask))

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


class RouteValueRouter(nn.Module):
    """Predict a prompt-conditioned score for every catalog route action."""

    def __init__(
        self,
        candidate_layers: Sequence[int],
        route_catalog: Sequence[Sequence[int]],
        hidden_dim: int = 2560,
        max_active_layers: int = 4,
        alpha_values: Sequence[float] = (1.0, 1.5, 2.0),
        router_dim: int = 128,
    ) -> None:
        super().__init__()
        self.candidate_layers = tuple(int(layer) for layer in candidate_layers)
        self.max_active_layers = int(max_active_layers)
        self.alpha_values = tuple(float(value) for value in alpha_values)
        routes = [tuple(int(layer) for layer in route) for route in route_catalog]
        if () not in routes:
            routes.insert(0, ())
        if not routes or len(set(routes)) != len(routes):
            raise ValueError("route_catalog must contain unique routes")
        if any(
            len(route) > self.max_active_layers
            or not set(route).issubset(self.candidate_layers)
            for route in routes
        ):
            raise ValueError("route_catalog contains an invalid candidate-layer subset")
        if not self.alpha_values or any(not 1.0 <= alpha <= 2.0 for alpha in self.alpha_values):
            raise ValueError("alpha_values must lie in [1, 2]")

        self.route_catalog = tuple(routes)
        self.action_catalog = tuple(
            (route, alpha)
            for route in routes
            for alpha in ((1.0,) if not route else self.alpha_values)
        )
        self.prompt_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, router_dim),
            nn.GELU(),
            nn.LayerNorm(router_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(2, router_dim),
            nn.GELU(),
            nn.LayerNorm(router_dim),
        )
        action_features = []
        for route, alpha in self.action_catalog:
            layer_mask = [float(layer in route) for layer in self.candidate_layers]
            action_features.append(
                layer_mask + [len(route) / self.max_active_layers, (alpha - 1.0)]
            )
        self.register_buffer("action_features", torch.tensor(action_features, dtype=torch.float32))
        self.action_encoder = nn.Sequential(
            nn.Linear(len(self.candidate_layers) + 2, router_dim),
            nn.GELU(),
            nn.LayerNorm(router_dim),
        )
        self.action_score = nn.Sequential(
            nn.Linear(router_dim * 2, router_dim),
            nn.GELU(),
            nn.Linear(router_dim, 1),
        )

    @staticmethod
    def raw_prompt_pool(hidden: torch.Tensor, prompt_mask: torch.BoolTensor) -> torch.Tensor:
        """Mean-pool valid prompt tokens, excluding padding and completions."""
        return AdaptiveSubsetRouter.raw_prompt_pool(hidden, prompt_mask)

    def score_pooled(
        self, pooled_prompt: torch.Tensor, context_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict the cost-adjusted completion score for every route action."""
        if context_features is None:
            context_features = torch.zeros(
                pooled_prompt.shape[0], 2, device=pooled_prompt.device, dtype=torch.float32
            )
        context_features = torch.as_tensor(
            context_features, device=pooled_prompt.device, dtype=torch.float32
        )
        if context_features.shape != (pooled_prompt.shape[0], 2):
            raise ValueError("context_features must have shape (batch, 2)")
        prompt = self.prompt_encoder(pooled_prompt.float()) + self.context_encoder(context_features)
        action = self.action_encoder(self.action_features)
        prompt = prompt[:, None, :].expand(-1, action.shape[0], -1)
        action = action[None, :, :].expand(prompt.shape[0], -1, -1)
        return self.action_score(torch.cat((prompt, action), dim=-1)).squeeze(-1)

    def forward(
        self,
        hidden: torch.Tensor,
        prompt_mask: torch.BoolTensor,
        context_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score the catalog using prompt-position hidden states only."""
        pooled = self.raw_prompt_pool(hidden, prompt_mask)
        return self.score_pooled(pooled, context_features)

    @torch.no_grad()
    def choose(
        self,
        hidden: torch.Tensor,
        prompt_mask: torch.BoolTensor,
        context_features: torch.Tensor | None = None,
    ) -> RouteDecision:
        """Choose the lowest predicted-score route, including the k=0 action."""
        scores = self(hidden, prompt_mask, context_features)
        action_indices = scores.argmin(dim=-1).tolist()
        mask = torch.zeros(
            hidden.shape[0], len(self.candidate_layers), dtype=torch.bool, device=hidden.device
        )
        k_values = []
        alpha_values = []
        for batch_index, action_index in enumerate(action_indices):
            route, alpha = self.action_catalog[action_index]
            k_values.append(len(route))
            alpha_values.append(alpha)
            for layer in route:
                mask[batch_index, self.candidate_layers.index(layer)] = True
        return RouteDecision(
            mask=mask,
            k=torch.tensor(k_values, dtype=torch.long, device=hidden.device),
            alpha=torch.tensor(alpha_values, dtype=hidden.dtype, device=hidden.device),
        )


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
