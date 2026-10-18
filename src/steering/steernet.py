import torch
import torch.nn as nn

class RSCSteerNet(nn.Module):
    """K-aware low-rank residual steering MLP with Bounded Relative Perturbation.
    
    Produces a correction delta from hidden state h, conditioned on the number of
    simultaneously active candidate layers (active_k).
    Enforces a strict relative perturbation envelope: ||delta|| <= beta * ||h||
    to prevent base representation collapse and destructive interference.
    Optionally projects delta onto the orthogonal complement of h.
    """

    def __init__(
        self,
        hidden_dim: int,
        rank: int,
        lora_alpha: float,
        dropout: float = 0.1,
        magnitude_mode: str = "direct",
        max_magnitude: float = 2.0,
        max_relative_norm: float = 0.15,
        orthogonal_projection: bool = False,
        k_max: int = 4,
    ):
        super().__init__()
        self.k_max = max(1, int(k_max))
        self.magnitude_mode = magnitude_mode
        self.max_magnitude = max_magnitude
        self.max_relative_norm = max_relative_norm
        self.orthogonal_projection = orthogonal_projection
        self._clip_stats = 0.0
        self._last_pre_cap_ratio = None
        self._pre_cap_sum = None
        self._pre_cap_n = 0
        self._pre_cap_max = None
        # Detached evaluation-only samples support an exact p95 safety check
        # during the compact Phase-1 generation probes.  They are cleared by
        # ``pre_cap_report(reset=True)`` and never retain an autograd graph.
        self._pre_cap_values = []
        self.last_pre_cap_q = None
        self._last_valid_mask = None
        self.norm = nn.LayerNorm(hidden_dim)
        self.down1 = nn.Linear(hidden_dim, rank, bias=False)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.mid = nn.Linear(rank, rank, bias=False)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        self.up = nn.Linear(rank, hidden_dim, bias=False)

        # Per-token magnitude head for 'direct' magnitude mode (ISSUE-5 / R2).
        # log_magnitude is retained for 'decoupled' mode backward compat.
        self.mag_head = nn.Linear(rank, 1)
        nn.init.zeros_(self.mag_head.weight)
        nn.init.zeros_(self.mag_head.bias)  # R2: 0 init (sigmoid=0.5); identity is guaranteed by up.weight=0

        # K-conditioning projection: input is active_k normalized to [0, 1]
        self.k_proj = nn.Linear(1, rank)
        nn.init.zeros_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)

        scaling_factor = lora_alpha / rank
        self.register_buffer("scale", torch.tensor(scaling_factor, dtype=torch.float32))

        # Learnable log-magnitude for decoupled mode
        self.log_magnitude = nn.Parameter(torch.zeros(1))

        for lin in [self.down1, self.mid]:
            nn.init.xavier_uniform_(lin.weight, gain=0.02)
        # Zero-init the projection-up layer so delta=0 at init (ISSUE-5 identity init).
        # This ensures the SteerNet starts as a perfect pass-through (baseline identity)
        # and the task loss must earn any nonzero intervention from scratch.
        nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor, active_k: torch.Tensor = None) -> torch.Tensor:
        h_flt = h.float()
        x = self.norm(h_flt)
        x = self.drop1(self.act1(self.down1(x)))
        if active_k is not None:
            k_norm = (active_k.to(dtype=x.dtype) / float(self.k_max)).view(-1, 1, 1)
            x = x + self.k_proj(k_norm)
        mid_pre_drop = self.act2(self.mid(x))
        mid = self.drop2(mid_pre_drop)
        x = x + mid
        delta = self.up(x)

        if self.magnitude_mode == "decoupled":
            d_norm = torch.norm(delta, dim=-1, keepdim=True) + 1e-7
            direction = delta / d_norm
            mag = torch.exp(self.log_magnitude).clamp(max=self.max_magnitude)
            delta = direction * mag * self.scale.to(x.dtype)
        elif self.magnitude_mode == "direct":
            # Per-token sigmoid gate on magnitude
            mag = torch.sigmoid(self.mag_head(mid_pre_drop))
            delta = delta * mag * self.scale.to(x.dtype)
        else:
            delta = delta * self.scale.to(x.dtype)

        # Smooth Saturation Envelope (R2 / R4 fix):
        # f(n) = n * c / sqrt(n^2 + c^2), bounded strictly by cap while preserving
        # relative token-wise magnitude dynamics.
        if self.max_relative_norm > 0:
            h_norm = torch.norm(h_flt, dim=-1, keepdim=True) + 1e-7
            delta_norm = torch.norm(delta, dim=-1, keepdim=True) + 1e-7
            cap = self.max_relative_norm * h_norm
            # Differentiable pre-cap norm ratio tensor [B, T, 1] (Engine v2.1)
            self.last_pre_cap_q = delta_norm / (cap + 1e-7)
            soft = cap / torch.sqrt(delta_norm.square() + cap.square())
            if not self.training:
                # Retain tensor on device — no .item() host sync during autoregressive decode (R4)
                r = self.last_pre_cap_q.detach().float().flatten()
                if self._pre_cap_sum is None:
                    self._pre_cap_sum = r.sum()
                    self._pre_cap_n = r.numel()
                    self._pre_cap_max = r.max()
                else:
                    self._pre_cap_sum = self._pre_cap_sum + r.sum()
                    self._pre_cap_n = self._pre_cap_n + r.numel()
                    self._pre_cap_max = torch.maximum(self._pre_cap_max, r.max())
                self._pre_cap_values.append(r)
                self._last_pre_cap_ratio = r.mean()
                self._clip_stats = (delta_norm > cap).detach().float().mean()
            delta = delta * soft
        else:
            self.last_pre_cap_q = None

        # Orthogonal Steering Projection (optional):
        # Projects delta onto the orthogonal complement of h so steering strictly
        # rotates/steers representation features without inflating radial norm.
        if self.orthogonal_projection:
            h_sq = torch.sum(h_flt * h_flt, dim=-1, keepdim=True) + 1e-7
            proj = (torch.sum(delta * h_flt, dim=-1, keepdim=True) / h_sq) * h_flt
            delta = delta - proj

        return delta

    def reset_pre_cap_stats(self):
        """Reset accumulated pre-cap norm saturation statistics."""
        self._pre_cap_sum = None
        self._pre_cap_n = 0
        self._pre_cap_max = None
        self._pre_cap_values = []


def pre_cap_report(model: torch.nn.Module, reset: bool = True) -> dict:
    """Read true token-aggregated pre-cap norm saturation statistics once per evaluation (R2)."""
    out = {}
    steer_nets = getattr(model, "steer_nets", {})
    if isinstance(steer_nets, torch.nn.ModuleDict):
        for k, sn in steer_nets.items():
            n = getattr(sn, "_pre_cap_n", 0)
            if n and getattr(sn, "_pre_cap_sum", None) is not None:
                mean_val = float((sn._pre_cap_sum / n).item())
                max_val = float(sn._pre_cap_max.item())
                values = getattr(sn, "_pre_cap_values", [])
                p95_val = (
                    float(torch.quantile(torch.cat(values), 0.95).item())
                    if values else mean_val
                )
                out[int(k) if str(k).isdigit() else k] = {
                    "mean": round(mean_val, 4),
                    "p95": round(p95_val, 4),
                    "max": round(max_val, 4),
                    "tokens": int(n),
                }
            if reset and hasattr(sn, "reset_pre_cap_stats"):
                sn.reset_pre_cap_stats()
    return out

