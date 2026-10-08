import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.configs.config import RSCConfig

class RSCUsefulnessGate(nn.Module):
    """Estimates P(steering improves accuracy | h_t, delta_t).

    Inputs:
        h              (B, T, d) - hidden state at this layer
        delta          (B, T, d) - SteerNet proposed correction (detached inside forward)
        global_context (B, semantic_dim) or None - detached router semantic summary

    Output:
        alpha          (B, T, 1) in (0, 1) - per-token steering weight
    """

    def __init__(self, hidden_dim: int, cfg: RSCConfig, global_context_dim: int = 16):
        super().__init__()
        gate_dim = cfg.gate_dim
        self.global_context_dim = global_context_dim
        self.h_norm = nn.LayerNorm(hidden_dim)
        self.d_norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(2 * hidden_dim + 2 + global_context_dim, gate_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(cfg.gate_dropout)
        self.fc2 = nn.Linear(gate_dim, 1)

        # Preserve the already-trained Phase-1 correction at the beginning of Phase 2
        nn.init.zeros_(self.fc2.weight)
        init_logit = math.log(cfg.gate_init_alpha / (1.0 - cfg.gate_init_alpha))
        self.fc2.bias.data.fill_(init_logit)

        self._freeze_gate = False
        self._freeze_gate_value = 1.0
        self.last_alpha = None      # (B, T, 1), detached - for diagnostics
        self._alpha_for_aux = None  # (B, T, 1), grad-carrying - for PG loss
        self._valid_mask = None     # (B, T) bool - non-padding positions, set by hook

    def forward(self, h: torch.Tensor, delta: torch.Tensor, global_context: torch.Tensor = None) -> torch.Tensor:
        if self._freeze_gate:
            val = torch.full(
                (h.size(0), h.size(1), 1),
                self._freeze_gate_value,
                dtype=h.dtype,
                device=h.device,
            )
            self.last_alpha = val.detach()
            self._alpha_for_aux = val
            return val

        h_fp = h.detach().float()
        d_det = delta.detach().float()  # No gradient from gate into SteerNet

        h_n = self.h_norm(h_fp)
        d_n = self.d_norm(d_det)
        mag = torch.log1p(d_det.norm(dim=-1, keepdim=True))
        cos = F.cosine_similarity(h_fp.detach(), d_det, dim=-1, eps=1e-6).unsqueeze(-1)

        if global_context is not None:
            ctx = global_context.detach().to(dtype=h_fp.dtype).unsqueeze(1).expand(-1, h_fp.shape[1], -1)
        else:
            ctx = torch.zeros(h_fp.shape[0], h_fp.shape[1], self.global_context_dim, dtype=h_fp.dtype, device=h_fp.device)

        feat = torch.cat([h_n, d_n, mag, cos, ctx], dim=-1)  # (B, T, 2d + 2 + global_context_dim)
        x = self.drop(self.act(self.fc1(feat)))
        gate_logit = torch.clamp(self.fc2(x), min=-10.0, max=10.0)
        alpha = torch.sigmoid(gate_logit)  # (B, T, 1)

        self.last_alpha = alpha.detach()
        self._alpha_for_aux = alpha
        return alpha.to(h.dtype)
