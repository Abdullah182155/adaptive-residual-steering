import torch
import torch.nn as nn

class RSCSteerNet(nn.Module):
    """K-aware low-rank residual steering MLP.
    
    Produces a correction delta from hidden state h, conditioned on the number of
    simultaneously active candidate layers (active_k) to prevent signal dilution
    when multiple layers steer concurrently.
    """

    def __init__(self, hidden_dim: int, rank: int, lora_alpha: float, dropout: float = 0.1):
        super().__init__()
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

        for lin in [self.down1, self.mid, self.up]:
            nn.init.xavier_uniform_(lin.weight, gain=0.02)

    def forward(self, h: torch.Tensor, active_k: torch.Tensor = None) -> torch.Tensor:
        x = self.norm(h.float())
        x = self.drop1(self.act1(self.down1(x)))
        if active_k is not None:
            k_norm = (active_k.to(dtype=x.dtype) / 4.0).view(-1, 1, 1)
            x = x + self.k_proj(k_norm)
        mid = self.drop2(self.act2(self.mid(x)))
        x = x + mid
        return self.up(x) * self.scale.to(x.dtype)
