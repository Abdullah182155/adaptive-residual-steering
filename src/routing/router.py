import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from src.configs.config import RSCConfig

class JointLayerRouter(nn.Module):
    """Scores candidate layers JOINTLY using self-attention and derives a trainable K.
    
    Operates over the candidate set using iterative refinement (Pass 1 -> Pass 2)
    and predicts utility logits u and K distribution logits without scalar cutoffs.
    """

    def __init__(self, target_layers: List[int], hidden_dim: int, n_layers: int,
                 max_active_layers: int = 4,
                 semantic_dim: int = 16, candidate_semantic_dim: int = 8,
                 d_model: int = 32, n_heads: int = 2,
                 layer_dropout: float = 0.15,
                 difficulty_aware: bool = True,
                 synergy_limit: float = 0.75):
        super().__init__()
        self.target_layers = list(target_layers)
        n_candidates = len(self.target_layers)
        self.n_candidates = n_candidates
        self.max_active_layers = max_active_layers
        self.candidate_semantic_dim = candidate_semantic_dim
        self.semantic_dim = semantic_dim
        self.layer_dropout = layer_dropout
        self.difficulty_aware = difficulty_aware
        self.synergy_limit = float(synergy_limit)
        depths = torch.tensor(
            [idx / max(1, n_layers - 1) for idx in self.target_layers], dtype=torch.float32
        )
        self.register_buffer("depths", depths)

        self.summary_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, semantic_dim), nn.Tanh()
        )
        self.candidate_semantic_read = nn.Linear(semantic_dim, n_candidates * candidate_semantic_dim)

        self.identity_embed = nn.Embedding(n_candidates, d_model)
        nn.init.normal_(self.identity_embed.weight, mean=0.0, std=0.1)
        
        n_scalar_dim = 5 if difficulty_aware else 3
        self.input_proj = nn.Linear(n_scalar_dim + 1 + candidate_semantic_dim, d_model)

        # Pass 1: initial joint scoring
        self.attend = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.0, activation="gelu", batch_first=True,
        )
        self.utility_head = nn.Linear(d_model, 1)

        # Pass 2: iterative refinement
        self.refine_proj = nn.Linear(1, d_model)
        self.attend_refine = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.0, activation="gelu", batch_first=True,
        )
        self.utility_head_refine = nn.Linear(d_model, 1)
        self.k_head = nn.Linear(d_model, max_active_layers + 1)
        self.d_model = d_model

        # Cross-layer cooperation & synergy projection
        self.synergy_proj = nn.Linear(d_model, d_model, bias=False)
        # RC-3 fix: was orthogonal_(gain=0.1) which made synergy_proj perpendicular to
        # input space, causing cosine_sim(proj(x), x) ~= 0 or negative from the first
        # forward pass. Every pair scored ~-0.54 (uniform negative synergy) regardless
        # of actual layer compatibility, so the router correctly learned K=1 is always
        # better. Fix: near-identity init (scaled eye + small noise) so synergy starts
        # at 0.0 (neutral) and learns from actual counterfactual interaction signals.
        d_syn = self.synergy_proj.weight.shape[0]
        nn.init.eye_(self.synergy_proj.weight)
        with torch.no_grad():
            self.synergy_proj.weight.add_(torch.randn_like(self.synergy_proj.weight) * 0.01)
        # A pair-specific residual prevents one shared projection from
        # broadcasting the sign of a single destructive pair to every pair.
        # It is deliberately zero-initialised: no cooperation is assumed
        # before an interaction counterfactual measures it.
        self.synergy_pair_bias = nn.Parameter(torch.zeros(n_candidates, n_candidates))

        # Initialize final decision heads near zero for conservative initial boundaries
        nn.init.zeros_(self.utility_head.weight)
        nn.init.zeros_(self.utility_head.bias)
        nn.init.zeros_(self.utility_head_refine.weight)
        nn.init.zeros_(self.utility_head_refine.bias)
        nn.init.zeros_(self.k_head.weight)
        nn.init.zeros_(self.k_head.bias)

        # Diagnostic tracking buffers (detached, real decisions only).
        # ISSUE-18 fix: persistent=False → not saved/loaded with checkpoints.
        # Prevents hash churn and eliminates stale EMAs across runs.
        self.register_buffer("logit_ema", torch.zeros(n_candidates), persistent=False)
        self.register_buffer("logit_sq_ema", torch.zeros(n_candidates), persistent=False)
        self.register_buffer("selection_rate_ema", torch.full((n_candidates,), 0.5), persistent=False)
        self.register_buffer("k_dist_ema", torch.full((max_active_layers + 1,), 1.0 / (max_active_layers + 1)), persistent=False)
        self.register_buffer("selected_forwards", torch.zeros(n_candidates, dtype=torch.long), persistent=False)
        self.register_buffer("seen_forwards", torch.zeros(n_candidates, dtype=torch.long), persistent=False)

        self._last_scalars = None
        self._last_pooled = None
        self._last_semantic_shared = None
        self._last_prediction = None

    def _semantic_from_pooled(self, pooled: torch.Tensor):
        semantic_shared = self.summary_proj(pooled)
        self._last_semantic_shared = semantic_shared.detach()
        return self.candidate_semantic_read(semantic_shared).view(
            pooled.shape[0], self.n_candidates, self.candidate_semantic_dim
        )

    def _base_features(self, h: torch.Tensor, valid_mask: torch.Tensor):
        h_det = h.detach().float()
        rms = h_det.square().mean(dim=-1).sqrt()
        valid = valid_mask.float()
        denom = valid.sum(dim=1).clamp(min=1)
        mean_rms = (rms * valid).sum(dim=1) / denom
        centered = (rms - mean_rms.unsqueeze(1)) * valid
        rms_std = centered.square().sum(dim=1).div(denom).sqrt()
        coverage = valid.mean(dim=1)
        pooled = (h_det * valid.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)

        if self.difficulty_aware:
            # Difficulty indicators:
            # 1. Normalized sequence reasoning length: log(1 + denom) / 6.24 (capped in [0, 1])
            norm_len = (torch.log1p(denom) / 6.24).clamp(0.0, 1.0)
            # 2. Hidden state dispersion (mean absolute token deviation from sequence mean)
            dispersion = (centered.abs().sum(dim=1) / denom).clamp(min=0.0)
            self._last_scalars = torch.stack([mean_rms, rms_std, coverage, norm_len, dispersion], dim=-1).detach()
        else:
            self._last_scalars = torch.stack([mean_rms, rms_std, coverage], dim=-1).detach()

        self._last_pooled = pooled.detach()
        per_candidate_semantic = self._semantic_from_pooled(pooled)
        return self._last_scalars, per_candidate_semantic

    def _score_from_base(self, scalars: torch.Tensor, per_candidate_semantic: torch.Tensor, return_synergy: bool = False):
        B = scalars.shape[0]
        N = self.n_candidates
        depth = self.depths.view(1, N, 1).expand(B, -1, -1)
        scalars_rep = scalars.unsqueeze(1).expand(-1, N, -1)
        tokens_in = torch.cat([scalars_rep, depth, per_candidate_semantic], dim=-1)
        tokens = self.input_proj(tokens_in) + self.identity_embed.weight.unsqueeze(0)

        # Pass 1
        attended_1 = self.attend(tokens)
        u_1 = self.utility_head(attended_1).squeeze(-1)

        # Pass 2
        soft_select_1 = F.softmax(u_1, dim=-1)
        tokens_2 = attended_1 + self.refine_proj(soft_select_1.unsqueeze(-1))
        attended_2 = self.attend_refine(tokens_2)
        u_final = self.utility_head_refine(attended_2).squeeze(-1)
        pooled_2 = attended_2.mean(dim=1)
        k_logits = self.k_head(pooled_2)

        # Stochastic Layer Dropout during training:
        # Prevents co-dependent clique stacking by randomly masking candidate layers,
        # forcing each candidate layer to learn an independent marginal steering utility.
        if self.training and self.layer_dropout > 0.0:
            drop_mask = torch.rand_like(u_final) < self.layer_dropout
            all_dropped = drop_mask.all(dim=-1, keepdim=True)
            drop_mask = drop_mask & (~all_dropped)
            u_final = u_final.masked_fill(drop_mask, -1e4)

        synergy = self._compute_synergy(attended_2)
        self._last_synergy_matrix = synergy

        if return_synergy:
            return u_final, k_logits, synergy
        return u_final, k_logits

    def _compute_synergy(self, candidate_states: torch.Tensor) -> torch.Tensor:
        """Bound context-aware pair interactions around a zero prior.

        The old unnormalised dot product could grow without bound and tanh
        therefore saturated every pair at -1 or +1.  Cosine similarity keeps
        the contextual term finite, while the symmetric pair table receives
        gradients only for the pair whose counterfactual was actually probed.
        """
        n_candidates = candidate_states.shape[1]
        query = F.normalize(self.synergy_proj(candidate_states), dim=-1, eps=1e-6)
        keys = F.normalize(candidate_states, dim=-1, eps=1e-6)
        contextual = torch.bmm(query, keys.transpose(1, 2))
        contextual = 0.5 * (contextual + contextual.transpose(1, 2))
        pair_bias = 0.5 * (self.synergy_pair_bias + self.synergy_pair_bias.transpose(0, 1))
        raw = contextual + pair_bias.unsqueeze(0)
        # Keep the router's subset bonus in a non-saturated, interpretable
        # range.  A measured interaction may still be negative or positive.
        synergy = self.synergy_limit * torch.tanh(raw)
        eye = torch.eye(n_candidates, device=candidate_states.device, dtype=torch.bool).unsqueeze(0)
        return synergy.masked_fill(eye, 0.0)

    def synergy_prior_loss(self, synergy: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Shrink unmeasured pair scores toward the neutral no-interaction prior."""
        matrix = synergy if synergy is not None else self._last_synergy_matrix
        if matrix is None:
            return self.synergy_pair_bias.square().mean()
        n_candidates = matrix.shape[-1]
        eye = torch.eye(n_candidates, device=matrix.device, dtype=torch.bool).unsqueeze(0)
        values = matrix.masked_select(~eye)
        return values.square().mean() if values.numel() else matrix.new_zeros(())

    def forward(self, h: torch.Tensor, valid_mask: torch.Tensor, return_synergy: bool = False):
        scalars, per_candidate_semantic = self._base_features(h, valid_mask)
        return self._score_from_base(scalars, per_candidate_semantic, return_synergy=return_synergy)

    def recompute_from_cache(self, return_synergy: bool = False):
        if self._last_pooled is None:
            raise RuntimeError("JointLayerRouter.recompute_from_cache called before any forward()")
        per_candidate_semantic = self._semantic_from_pooled(self._last_pooled)
        return self._score_from_base(self._last_scalars, per_candidate_semantic, return_synergy=return_synergy)

    def observe(
        self,
        u: torch.Tensor,
        selected_matrix: torch.Tensor,
        eligible_mask: Optional[torch.Tensor] = None,
    ):
        """Update diagnostics without treating masked candidates as -inf data."""
        if not self.training:
            return
        if eligible_mask is None:
            eligible_mask = torch.ones(u.shape[-1], dtype=torch.bool, device=u.device)
        eligible_mask = eligible_mask.to(device=u.device, dtype=torch.bool)
        if eligible_mask.shape != (u.shape[-1],):
            raise ValueError("eligible_mask must have shape (n_candidates,)")
        mean_u = u.detach().mean(dim=0)
        mean_u_sq = u.detach().pow(2).mean(dim=0)
        self.logit_ema[eligible_mask] = (
            self.logit_ema[eligible_mask] * 0.98 + 0.02 * mean_u[eligible_mask]
        )
        self.logit_sq_ema[eligible_mask] = (
            self.logit_sq_ema[eligible_mask] * 0.98 + 0.02 * mean_u_sq[eligible_mask]
        )
        realized = selected_matrix.detach().float().mean(dim=0)
        # BUG-I1 fix: selection_rate_ema was never updated — always stayed at init value 0.5.
        # Now properly tracks empirical per-layer selection rate via EMA.
        self.selection_rate_ema[eligible_mask] = (
            self.selection_rate_ema[eligible_mask] * 0.98 + 0.02 * realized[eligible_mask]
        )
        k_realized = selected_matrix.detach().sum(dim=-1).clamp(min=0, max=self.max_active_layers)
        k_onehot = F.one_hot(k_realized.long(), num_classes=self.max_active_layers + 1).float().mean(dim=0)
        self.k_dist_ema.mul_(0.98).add_(0.02 * k_onehot)

    def get_layer_synergy_matrix(self) -> Optional[torch.Tensor]:
        """Returns the latest learned cross-layer synergy matrix bounded in [-1, 1]."""
        if hasattr(self, "_last_synergy_matrix") and self._last_synergy_matrix is not None:
            return self._last_synergy_matrix.detach()
        tokens = self.identity_embed.weight.unsqueeze(0)
        return self._compute_synergy(tokens).detach()

    def get_cooperation_analysis(self, eligible_layers: Optional[List[int]] = None) -> dict:
        """Extract finite cooperative pairs, optionally restricted to a legal pool."""
        syn_mat = self.get_layer_synergy_matrix()
        if syn_mat is None:
            return {"top_synergy_pairs": [], "top_antagonistic_pairs": [], "mean_synergy": 0.0}
        mat = syn_mat[0] if syn_mat.ndim == 3 else syn_mat
        mat_cpu = mat.detach().cpu().numpy()
        N = len(self.target_layers)
        pairs = []
        allowed = set(self.target_layers if eligible_layers is None else eligible_layers)
        for i in range(N):
            for j in range(i + 1, N):
                if self.target_layers[i] not in allowed or self.target_layers[j] not in allowed:
                    continue
                val = float(mat_cpu[i, j])
                if not np.isfinite(val):
                    continue
                pairs.append((self.target_layers[i], self.target_layers[j], val))
        pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
        finite_values = [item[2] for item in pairs]
        return {
            "top_synergy_pairs": pairs_sorted[:3],
            "top_antagonistic_pairs": list(reversed(pairs_sorted[-3:])),
            "mean_synergy": float(np.mean(finite_values)) if finite_values else 0.0,
        }

