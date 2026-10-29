"""
RSCSteerNetV5: Steering engine with rank 64 and per-layer cap basis.

Phase 1.1: Build RSCSteerNetV5 with:
|- Rank 64 (fixed)
|- Per-layer cap basis N_l (from calibration)
|- Smooth cap C_l = κ * N_l with κ ∈ {0.02, 0.04, 0.08}
|- Position-0 shielding
|- Zero-init W_u
|- 1D causal depthwise convolution with k=4 (streaming cache for decode)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class CausalDWConv1d(nn.Module):
    """Causal depthwise 1D convolution with streaming cache for auto-regressive decode.
    
    - Prefill mode: full causal convolution with left padding
    - Decode mode: streaming O(1) memory using rolling buffer of last k-1 activations
    """
    
    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        # Depthwise conv: each channel has its own filter
        self.weight = nn.Parameter(torch.randn(channels, 1, kernel_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(channels))
        
        # Streaming cache for decode mode
        self._cache = None  # Rolling buffer of shape (B, C, k-1)
        self._cache_initialized = False
    
    def reset_cache(self):
        """Reset streaming cache (call at start of each decode sequence)."""
        self._cache = None
        self._cache_initialized = False
    
    def forward(self, x: torch.Tensor, is_decode: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) input hidden states
            is_decode: True for streaming decode, False for prefill
        
        Returns:
            (B, T, C) output after causal conv
        """
        B, T, C = x.shape
        x_1d = x.transpose(1, 2)  # (B, C, T)
        
        if not is_decode:
            # Prefill mode: full causal convolution with left padding
            # Pad left with (k-1) zeros to ensure causality
            padding = self.kernel_size - 1
            x_padded = F.pad(x_1d, (padding, 0), mode='constant', value=0)
            out = F.conv1d(x_padded, self.weight, self.bias, groups=self.channels)
            out = out.transpose(1, 2)  # (B, T, C)
        else:
            # Decode mode: streaming with rolling buffer
            if not self._cache_initialized or self._cache is None or self._cache.shape[0] != B:
                # Initialize cache with zeros
                self._cache = torch.zeros(B, C, self.kernel_size - 1, device=x.device, dtype=x.dtype)
                self._cache_initialized = True
            
            # Pad left with previous cache
            x_with_cache = torch.cat([self._cache, x_1d], dim=2)  # (B, C, k-1 + T)
            
            # Update cache to hold the last (kernel_size - 1) tokens for next step
            self._cache = x_with_cache[:, :, -(self.kernel_size - 1):]
            
            # Convolve
            out = F.conv1d(x_with_cache, self.weight, self.bias, groups=self.channels)
            out = out[:, :, -(T):]  # Take only the T outputs for new tokens
            out = out.transpose(1, 2)  # (B, T, C)
        
        return out


class RSCSteerNetV5(nn.Module):
    """RSCSteerNetV5: Steering engine with rank 64 and per-layer cap basis.
    
    Key differences from v4:
    - Fixed rank 64
    - Per-layer cap basis N_l loaded from calibration
    - Smooth cap C_l = κ * N_l
    - Position-0 shielding (no steering on first token)
    - Zero-init W_u for identity initialization
    - Causal 1D depthwise convolution (k=4) with streaming cache
    """
    
    def __init__(
        self,
        hidden_dim: int,
        rank: int = 64,
        lora_alpha: float = 32.0,
        dropout: float = 0.1,
        kappa: float = 0.08,  # Multiplier for cap: C_l = κ * N_l (increased from 0.04 to 0.08 for stronger steering)
        layer_norms: Optional[Dict[int, float]] = None,  # N_l per layer
        layer_idx: int = 0,  # Which layer this steernet is for
        max_magnitude: float = 2.0,
        orthogonal_projection: bool = False,
        telemetry_reservoir_size: int = 8192,
        use_causal_conv: bool = True,  # Use causal convolution
        conv_kernel_size: int = 4,  # Kernel size for causal conv
    ):
        super().__init__()
        self.rank = rank
        self.kappa = kappa
        self.layer_idx = layer_idx
        self.layer_norms = layer_norms or {}
        self.max_magnitude = max_magnitude
        self.orthogonal_projection = orthogonal_projection
        self.use_causal_conv = use_causal_conv
        self.conv_kernel_size = conv_kernel_size
        
        # Per-layer cap basis N_l
        self.N_l = self.layer_norms.get(layer_idx, 10.0)  # Default if not calibrated
        
        # Smooth cap C_l = κ * N_l
        self.C_l = self.kappa * self.N_l
        
        # Log the cap for debugging
        if layer_idx % 5 == 0:  # Log every 5th layer to avoid spam
            print(f"[SteerNetV5 L{layer_idx}] N_l={self.N_l:.3f}, kappa={self.kappa}, C_l={self.C_l:.3f}")
        
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
        
        # Network layers
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        # Use standard initialization (gain=1.0) instead of tiny 0.02
        nn.init.xavier_uniform_(self.down.weight, gain=1.0)
        
        # Causal convolution for sequence mixing
        if self.use_causal_conv:
            self.conv = CausalDWConv1d(rank, kernel_size=conv_kernel_size)
        
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        
        # Small init up projection (not zero) to allow steering to develop
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.up.weight, gain=0.1)  # Small but non-zero
        
        # Per-token magnitude head
        self.mag_head = nn.Linear(rank, 1)
        nn.init.zeros_(self.mag_head.weight)
        nn.init.zeros_(self.mag_head.bias)
        
        scaling_factor = lora_alpha / rank
        self.register_buffer("scale", torch.tensor(scaling_factor, dtype=torch.float32))
    
    def reset_cache(self):
        """Reset streaming cache for causal conv (call at start of decode)."""
        if self.use_causal_conv:
            self.conv.reset_cache()
    
    def forward(
        self, 
        h: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None,
        active_k: Optional[torch.Tensor] = None,
        is_decode: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            h: (B, T, hidden_dim) hidden states
            position_ids: (B, T) position indices for shielding
            active_k: (B,) active K count (for compatibility with v4 interface)
            is_decode: True for streaming decode mode
        
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
        
        # Apply causal convolution if enabled
        if self.use_causal_conv:
            x = self.conv(x, is_decode=is_decode)
        
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
