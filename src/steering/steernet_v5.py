"""
RSCSteerNetV5: Steering engine with rank 64 and per-layer cap basis.

Phase 1.1: Build RSCSteerNetV5 with:
- Rank 64 (fixed)
- Per-layer cap basis N_l (from calibration)
- Smooth cap C_l = κ * N_l with κ ∈ {0.02, 0.04, 0.08}
- Position-0 shielding
- Zero-init W_u
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Dict


class RSCSteerNetV5(nn.Module):
    """RSCSteerNetV5: Steering engine with rank 64 and per-layer cap basis.
    
    Key differences from v4:
    - Fixed rank 64
    - Per-layer cap basis N_l loaded from calibration
    - Smooth cap C_l = κ * N_l
    - Position-0 shielding (no steering on first token)
    - Zero-init W_u for identity initialization
    """
    
    def __init__(
        self,
        hidden_dim: int,
        rank: int = 64,
        lora_alpha: float = 32.0,
        dropout: float = 0.1,
        kappa: float = 0.04,  # Multiplier for cap: C_l = κ * N_l
        layer_norms: Optional[Dict[int, float]] = None,  # N_l per layer
        layer_idx: int = 0,  # Which layer this steernet is for
        max_magnitude: float = 2.0,
        orthogonal_projection: bool = False,
        telemetry_reservoir_size: int = 8192,
    ):
        super().__init__()
        self.rank = rank
        self.kappa = kappa
        self.layer_idx = layer_idx
        self.layer_norms = layer_norms or {}
        self.max_magnitude = max_magnitude
        self.orthogonal_projection = orthogonal_projection
        
        # Per-layer cap basis N_l
        self.N_l = self.layer_norms.get(layer_idx, 10.0)  # Default if not calibrated
        
        # Smooth cap C_l = κ * N_l
        self.C_l = self.kappa * self.N_l
        
        # Telemetry (same as v4)
        self._pre_cap_sum = 0.0
        self._pre_cap_n = 0
        self._pre_cap_max = 0.0
        self._pre_cap_reservoir_size = max(1, int(telemetry_reservoir_size))
        self._pre_cap_values = []
        self._pre_cap_priorities = []
        self._pre_cap_generator = torch.Generator(device="cpu")
        self._pre_cap_generator.manual_seed(0)
        self.last_pre_cap_q = None
        self._clip_stats = 0.0
        
        # Network layers (same structure as v4, but fixed rank 64)
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        nn.init.xavier_uniform_(self.down.weight, gain=0.02)
        
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        
        # Zero-init up projection for identity
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.zeros_(self.up.weight)
        
        # Per-token magnitude head
        self.mag_head = nn.Linear(rank, 1)
        nn.init.zeros_(self.mag_head.weight)
        nn.init.zeros_(self.mag_head.bias)
        
        scaling_factor = lora_alpha / rank
        self.register_buffer("scale", torch.tensor(scaling_factor, dtype=torch.float32))
    
    def forward(
        self, 
        h: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None,
        active_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h: (B, T, hidden_dim) hidden states
            position_ids: (B, T) position indices for shielding
            active_k: (B,) active K count (for compatibility with v4 interface)
        
        Returns:
            delta: (B, T, hidden_dim) steering correction
        """
        B, T, hidden_dim = h.shape
        h_flt = h.float()
        
        # Position-0 shielding: no steering on first token
        if position_ids is not None:
            shield_mask = (position_ids > 0).float().unsqueeze(-1)  # (B, T, 1)
        else:
            # Default: shield first position
            shield_mask = torch.ones(B, T, 1, device=h.device)
            if T > 0:
                shield_mask[:, 0, :] = 0.0
        
        # Process through network
        x = self.norm(h_flt)
        x = self.drop(self.act(self.down(x)))
        
        # Magnitude gating
        mag = torch.sigmoid(self.mag_head(x))
        delta = self.up(x) * mag * self.scale.to(x.dtype)
        
        # Apply position shielding
        delta = delta * shield_mask
        
        # Smooth cap envelope: C_l = κ * N_l
        # f(n) = n * C_l / sqrt(n^2 + C_l^2)
        delta_norm = torch.norm(delta, dim=-1, keepdim=True) + 1e-7
        cap = self.C_l
        
        # Pre-cap ratio for telemetry
        self.last_pre_cap_q = delta_norm / (cap + 1e-7)
        
        # Smooth saturation
        soft = cap / torch.sqrt(delta_norm.square() + cap * cap)
        delta = delta * soft
        
        # Telemetry (same as v4)
        if not self.training:
            r = self.last_pre_cap_q.detach().float().flatten()
            self._pre_cap_sum += float(r.sum().item())
            self._pre_cap_n += int(r.numel())
            self._pre_cap_max = max(self._pre_cap_max, float(r.max().item()))
            self._update_pre_cap_reservoir(r)
            self._clip_stats = (delta_norm > cap).detach().float().mean()
        
        # Orthogonal projection (optional)
        if self.orthogonal_projection:
            h_sq = torch.sum(h_flt * h_flt, dim=-1, keepdim=True) + 1e-7
            proj = (torch.sum(delta * h_flt, dim=-1, keepdim=True) / h_sq) * h_flt
            delta = delta - proj
        
        return delta
    
    def _update_pre_cap_reservoir(self, values: torch.Tensor) -> None:
        """Merge CPU priority reservoir (same as v4)."""
        sample = values.detach().float().flatten().cpu()
        if sample.numel() == 0:
            return
        priorities = torch.rand(sample.numel(), generator=self._pre_cap_generator)
        if self._pre_cap_values:
            sample = torch.cat((self._pre_cap_values[0], sample))
            priorities = torch.cat((self._pre_cap_priorities[0], priorities))
        keep = min(self._pre_cap_reservoir_size, int(sample.numel()))
        if sample.numel() > keep:
            chosen = torch.topk(priorities, keep, largest=False, sorted=False).indices
            sample = sample.index_select(0, chosen)
            priorities = priorities.index_select(0, chosen)
        self._pre_cap_values = [sample]
        self._pre_cap_priorities = [priorities]
    
    def reset_pre_cap_stats(self):
        """Reset accumulated pre-cap norm saturation statistics."""
        self._pre_cap_sum = 0.0
        self._pre_cap_n = 0
        self._pre_cap_max = 0.0
        self._pre_cap_values = []
        self._pre_cap_priorities = []
        self._pre_cap_generator.manual_seed(0)


def load_layer_norms(calibration_path: str) -> Dict[int, float]:
    """Load per-layer N_l values from calibration JSON."""
    import json
    with open(calibration_path, "r") as f:
        data = json.load(f)
    
    layer_norms = {}
    for layer, stats in data["layers"].items():
        layer_norms[int(layer)] = stats["median_norm"]
    
    return layer_norms
