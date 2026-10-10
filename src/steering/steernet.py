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
    ):
        super().__init__()
        self.magnitude_mode = magnitude_mode
        self.max_magnitude = max_magnitude
        self.max_relative_norm = max_relative_norm
        self.orthogonal_projection = orthogonal_projection
        self.norm = nn.LayerNorm(hidden_dim)
        self.down1 = nn.Linear(hidden_dim, rank, bias=False)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.mid = nn.Linear(rank, rank, bias=False)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        
        # K-conditioning projection: input is active_k normalized to [0, 1]
        self.k_proj = nn.Linear(1, rank)
        nn.init.zeros_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)

        scaling_factor = lora_alpha / rank
        self.register_buffer("scale", torch.tensor(scaling_factor, dtype=torch.float32))

        # Learnable log-magnitude for decoupled mode
        self.log_magnitude = nn.Parameter(torch.zeros(1))

        for lin in [self.down1, self.mid, self.up]:
            nn.init.xavier_uniform_(lin.weight, gain=0.02)

    def forward(self, h: torch.Tensor, active_k: torch.Tensor = None) -> torch.Tensor:
        h_flt = h.float()
        x = self.norm(h_flt)
        x = self.drop1(self.act1(self.down1(x)))
        if active_k is not None:
            k_norm = (active_k.to(dtype=x.dtype) / 4.0).view(-1, 1, 1)
            x = x + self.k_proj(k_norm)
        mid = self.drop2(self.act2(self.mid(x)))
        x = x + mid
        delta = self.up(x)

        if self.magnitude_mode == "decoupled":
            d_norm = torch.norm(delta, dim=-1, keepdim=True) + 1e-7
            direction = delta / d_norm
            mag = torch.exp(self.log_magnitude).clamp(max=self.max_magnitude)
            delta = direction * mag * self.scale.to(x.dtype)
        else:
            delta = delta * self.scale.to(x.dtype)

        # Bounded Relative Perturbation Envelope:
        # Prevents residual explosion: ||delta|| <= beta * ||h||
        if self.max_relative_norm > 0:
            h_norm = torch.norm(h_flt, dim=-1, keepdim=True) + 1e-7
            delta_norm = torch.norm(delta, dim=-1, keepdim=True) + 1e-7
            max_allowed = self.max_relative_norm * h_norm
            scale_factor = torch.clamp(max_allowed / delta_norm, max=1.0)
            delta = delta * scale_factor

        # Orthogonal Steering Projection (optional):
        # Projects delta onto the orthogonal complement of h so steering strictly
        # rotates/steers representation features without inflating radial norm.
        if self.orthogonal_projection:
            h_sq = torch.sum(h_flt * h_flt, dim=-1, keepdim=True) + 1e-7
            proj = (torch.sum(delta * h_flt, dim=-1, keepdim=True) / h_sq) * h_flt
            delta = delta - proj

        return delta

