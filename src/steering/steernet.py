"""SteerNet V5: capped causal residual adapters for fixed-route ARS."""

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
        # Depthwise conv: Dirac-centered initialization
        # Tap on current token (index -1) = 1.0, history taps have small perturbation (0.01)
        # Preserves signal flow (gain ~1.0) while enabling temporal mixing from step 0
        w = torch.randn(channels, 1, kernel_size) * 0.01
        w[:, :, -1] = 1.0
        self.weight = nn.Parameter(w)
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
            x_padded = F.pad(x_1d, (padding, 0), mode="constant", value=0)
            out = F.conv1d(x_padded, self.weight, self.bias, groups=self.channels)
            out = out.transpose(1, 2)  # (B, T, C)
            # Prime cache with the last (kernel_size - 1) tokens so streaming decode starts with real context
            if T >= padding:
                self._cache = x_1d[:, :, -padding:].detach()
                self._cache_initialized = True
            elif T > 0:
                pad_needed = padding - T
                self._cache = F.pad(
                    x_1d, (pad_needed, 0), mode="constant", value=0
                ).detach()
                self._cache_initialized = True
        else:
            # Decode mode: streaming with rolling buffer
            if (
                not self._cache_initialized
                or self._cache is None
                or self._cache.shape[0] != B
            ):
                # Initialize cache with zeros
                self._cache = torch.zeros(
                    B, C, self.kernel_size - 1, device=x.device, dtype=x.dtype
                )
                self._cache_initialized = True

            # Pad left with previous cache
            x_with_cache = torch.cat([self._cache, x_1d], dim=2)  # (B, C, k-1 + T)

            # Convolve first — so cache is only updated after a successful conv
            out = F.conv1d(x_with_cache, self.weight, self.bias, groups=self.channels)

            # Update cache AFTER successful convolution
            self._cache = x_with_cache[:, :, -(self.kernel_size - 1) :]
            out = out[:, :, -(T):]  # Take only the T outputs for new tokens
            out = out.transpose(1, 2)  # (B, T, C)

        return out


class SteerNet(nn.Module):
    """Rank-64 causal residual adapter with a per-layer norm cap.

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
        kappa: float = 0.04,
        amplitude_mode: str = "relative",
        rho_max: float = 0.04,
        layer_norms: Optional[Dict[int, float]] = None,  # N_l per layer
        layer_idx: int = 0,  # Which layer this steernet is for
        telemetry_reservoir_size: int = 8192,
        use_causal_conv: bool = True,  # Use causal convolution
        conv_kernel_size: int = 4,  # Kernel size for causal conv
        mag_head_init_bias: float = -3.0,
        init_gain: float = 1.0,       # xavier gain for down/up; old code = 0.02
        use_mag_head: bool = True,     # per-token magnitude head; old code = False
        use_soft_cap: bool = True,     # smooth cap envelope; old code = False
    ):
        super().__init__()
        self.rank = rank
        self.kappa = kappa
        self.amplitude_mode = amplitude_mode
        self.rho_max = rho_max
        self.layer_idx = layer_idx
        self.layer_norms = layer_norms or {}
        self.use_causal_conv = use_causal_conv
        self.conv_kernel_size = conv_kernel_size
        self.use_mag_head = use_mag_head
        self.use_soft_cap = use_soft_cap

        # Per-layer cap basis N_l
        self.N_l = self.layer_norms.get(
            layer_idx, DEFAULT_PHI2_LAYER_NORMS.get(layer_idx, 10.0)
        )

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
        self.telemetry = True
        self._clip_stats = 0.0

        # Network layers
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        nn.init.xavier_uniform_(self.down.weight, gain=init_gain)

        # Causal convolution for sequence mixing
        if self.use_causal_conv:
            self.conv = CausalDWConv1d(rank, kernel_size=conv_kernel_size)

        self.act = nn.GELU()

        # Only create the dropout modules for the active path (avoid unused parameters)
        if use_mag_head:
            self.drop = nn.Dropout(dropout)       # V5 path: before up projection
        else:
            self.drop_down = nn.Dropout(dropout)  # Legacy path: after down
            self.drop_mid = nn.Dropout(dropout)   # Legacy path: after mid

        self.mid = nn.Linear(rank, rank, bias=False)
        nn.init.xavier_uniform_(self.mid.weight, gain=0.02)

        self.up = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.up.weight, gain=init_gain)

        # Per-token magnitude head (only when enabled)
        if self.use_mag_head:
            self.mag_head = nn.Linear(rank, 1)
            nn.init.zeros_(self.mag_head.weight)
            nn.init.constant_(self.mag_head.bias, mag_head_init_bias)

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
        injection_mask: Optional[torch.BoolTensor] = None,
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

        # Position-0 shielding: no steering on first prompt token (attention sink)
        if position_ids is not None:
            shield_mask = (position_ids > 0).float().unsqueeze(-1)  # (B, T, 1)
        elif is_decode:
            # Auto-regressive decode: all tokens are continuation tokens (position >= 1)
            shield_mask = torch.ones(B, T, 1, device=h.device, dtype=h.dtype)
        else:
            # Prompt prefill: shield position 0
            shield_mask = torch.ones(B, T, 1, device=h.device, dtype=h.dtype)
            if T > 0:
                shield_mask[:, 0, :] = 0.0

        # Process through network
        x = self.norm(h_flt)

        if self.use_mag_head:
            # V5 path: dropout only before up projection
            x = self.act(self.down(x))
            if self.use_causal_conv:
                x = self.conv(x, is_decode=is_decode)
            x = x + self.act(self.mid(x))
            magnitude = torch.sigmoid(self.mag_head(x))
            direction = self.up(self.drop(x))
        else:
            # Legacy path: dropout after down AND mid (matches old RSCSteerNet)
            x = self.drop_down(self.act(self.down(x)))
            if self.use_causal_conv:
                x = self.conv(x, is_decode=is_decode)
            mid = self.drop_mid(self.act(self.mid(x)))
            x = x + mid
            magnitude = 1.0
            direction = self.up(x)

        # Apply position shielding
        if self.amplitude_mode == "relative":
            centered_norm = (h_flt - h_flt.mean(dim=-1, keepdim=True)).norm(
                dim=-1, keepdim=True
            )
            direction_norm = direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            active_k = (
                torch.ones(B, device=h.device, dtype=h_flt.dtype)
                if active_k is None
                else active_k.to(h_flt)
            )
            radius = self.rho_max * active_k.clamp_min(1).rsqrt().view(B, 1, 1)
            delta = radius * magnitude * centered_norm * direction / direction_norm
            q = magnitude
        else:
            delta = direction * magnitude * self.scale.to(x.dtype)
            q = None

        delta = delta * shield_mask

        # Smooth cap envelope (optional — old code had none)
        if self.use_soft_cap:
            delta_norm = torch.norm(delta, dim=-1, keepdim=True) + 1e-7
            cap = self.C_l
            soft = cap / torch.sqrt(delta_norm.square() + cap * cap)
            delta = delta * soft
        else:
            delta_norm = torch.norm(delta, dim=-1, keepdim=True) + 1e-7
            cap = self.C_l

        if self.amplitude_mode == "relative":
            pre_cap_q = q * shield_mask
            if injection_mask is not None:
                pre_cap_q = pre_cap_q * injection_mask.unsqueeze(-1)
        else:
            pre_cap_q = delta_norm / (cap + 1e-7)
        self.last_pre_cap_q = pre_cap_q if self.training else pre_cap_q.detach()

        # Telemetry (same as v4)
        if self.telemetry and not self.training:
            r = self.last_pre_cap_q.detach().float().flatten()
            self._pre_cap_sum += float(r.sum().item())
            self._pre_cap_n += int(r.numel())
            self._pre_cap_max = max(self._pre_cap_max, float(r.max().item()))
            self._update_pre_cap_reservoir(r)
            self._clip_stats = (delta_norm > cap).detach().float().mean()

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


# The explicit compatibility alias preserves old checkpoints while imports use
# the stable publication-quality ``SteerNet`` name.
RSCSteerNetV5 = SteerNet


DEFAULT_PHI2_LAYER_NORMS: Dict[int, float] = {
    8:  52.0,    # estimated (run calibrate_norms.py for exact values)
    11: 57.0,    # estimated
    13: 60.4375, # calibrated
    15: 64.6250, # calibrated
    17: 70.0,    # estimated
    18: 72.4375, # calibrated (legacy candidate)
    19: 76.0,    # estimated
    20: 78.3125, # calibrated
    21: 81.0,    # estimated
    22: 85.1250, # calibrated
    24: 90.0,    # estimated
}


def load_layer_norms(calibration_path: Optional[str] = None) -> Dict[int, float]:
    """Load per-layer N_l values from calibration JSON, or fall back to certified phi-2 defaults."""
    import json
    import os

    if not calibration_path or not os.path.exists(calibration_path):
        if calibration_path:
            print(
                f"[SteerNetV5] Calibration file '{calibration_path}' not found on disk. "
                f"Falling back to certified phi-2 layer norms."
            )
        return dict(DEFAULT_PHI2_LAYER_NORMS)

    try:
        with open(calibration_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        layer_norms = {}
        for layer, stats in data.get("layers", {}).items():
            if isinstance(stats, dict) and "median_norm" in stats:
                layer_norms[int(layer)] = float(stats["median_norm"])
            elif isinstance(stats, (int, float)):
                layer_norms[int(layer)] = float(stats)

        return layer_norms
    except Exception as e:
        print(
            f"[SteerNetV5] Failed to parse '{calibration_path}' ({e}). "
            f"Falling back to certified phi-2 layer norms."
        )
        return dict(DEFAULT_PHI2_LAYER_NORMS)
