import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from src.configs.config import RSCConfig

class JointLayerRouter(nn.Module):
    """Scores candidate layers JOINTLY using self-attention and derives a trainable K.
    
    Operates over the candidate set using iterative refinement (Pass 1 -> Pass 2)
    and predicts utility logits u and K distribution logits without scalar cutoffs.
    """

    def __init__(self, target_layers: List[int], hidden_dim: int, n_layers: int,
                 max_active_layers: int = 4,
                 semantic_dim: int = 16, candidate_semantic_dim: int = 8,
                 d_model: int = 32, n_heads: int = 2):
        super().__init__()
        self.target_layers = list(target_layers)
        n_candidates = len(self.target_layers)
        self.n_candidates = n_candidates
        self.max_active_layers = max_active_layers
        self.candidate_semantic_dim = candidate_semantic_dim
        self.semantic_dim = semantic_dim
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
        self.input_proj = nn.Linear(4 + candidate_semantic_dim, d_model)

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
        self.k_head = nn.Linear(d_model, max_active_layers)

        # Initialize final decision heads near zero for conservative initial boundaries
        nn.init.zeros_(self.utility_head.weight)
        nn.init.zeros_(self.utility_head.bias)
        nn.init.zeros_(self.utility_head_refine.weight)
        nn.init.zeros_(self.utility_head_refine.bias)
        nn.init.zeros_(self.k_head.weight)
        nn.init.zeros_(self.k_head.bias)

        # Diagnostic tracking buffers (detached, real decisions only)
        self.register_buffer("logit_ema", torch.zeros(n_candidates))
        self.register_buffer("logit_sq_ema", torch.zeros(n_candidates))
        self.register_buffer("selection_rate_ema", torch.full((n_candidates,), 0.5))
        self.register_buffer("k_dist_ema", torch.full((max_active_layers,), 1.0 / max_active_layers))
        self.register_buffer("selected_forwards", torch.zeros(n_candidates, dtype=torch.long))
        self.register_buffer("seen_forwards", torch.zeros(n_candidates, dtype=torch.long))

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
        self._last_scalars = torch.stack([mean_rms, rms_std, coverage], dim=-1).detach()
        self._last_pooled = pooled.detach()
        per_candidate_semantic = self._semantic_from_pooled(pooled)
        return self._last_scalars, per_candidate_semantic

    def _score_from_base(self, scalars: torch.Tensor, per_candidate_semantic: torch.Tensor):
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
        return u_final, k_logits

    def forward(self, h: torch.Tensor, valid_mask: torch.Tensor):
        scalars, per_candidate_semantic = self._base_features(h, valid_mask)
        return self._score_from_base(scalars, per_candidate_semantic)

    def recompute_from_cache(self):
        if self._last_pooled is None:
            raise RuntimeError("JointLayerRouter.recompute_from_cache called before any forward()")
        per_candidate_semantic = self._semantic_from_pooled(self._last_pooled)
        return self._score_from_base(self._last_scalars, per_candidate_semantic)

    def observe(self, u: torch.Tensor, selected_matrix: torch.Tensor):
        if not self.training:
            return
        self.logit_ema.mul_(0.98).add_(0.02 * u.detach().mean(dim=0))
        self.logit_sq_ema.mul_(0.98).add_(0.02 * u.detach().pow(2).mean(dim=0))
        realized = selected_matrix.detach().float().mean(dim=0)
        self.selection_rate_ema.mul_(0.98).add_(0.02 * realized)
        k_realized = selected_matrix.detach().sum(dim=-1).clamp(min=1, max=self.max_active_layers)
        k_onehot = F.one_hot(k_realized.long() - 1, num_classes=self.max_active_layers).float().mean(dim=0)
        self.k_dist_ema.mul_(0.98).add_(0.02 * k_onehot)
