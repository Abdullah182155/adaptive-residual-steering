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

        # Causal Context Window (k=3) for local syntax & reasoning step awareness
        # True identity (current-token tap at index 2), causal alignment (ISSUE-17)
        self.causal_conv = nn.Conv1d(gate_dim, gate_dim, kernel_size=3, padding=2, groups=gate_dim)
        with torch.no_grad():
            self.causal_conv.weight.zero_()
            self.causal_conv.weight[:, 0, 2] = 1.0
            self.causal_conv.bias.zero_()

        self.fc2 = nn.Linear(gate_dim, 1)

        # Small non-zero initialization ensures gradient flow back into fc1 from step 0
        nn.init.normal_(self.fc2.weight, std=0.02)
        init_logit = math.log(cfg.gate_init_alpha / (1.0 - cfg.gate_init_alpha))
        self.fc2.bias.data.fill_(init_logit)

        self._freeze_gate = False
        self._freeze_gate_value = 1.0
        self.gating_mode = getattr(cfg, "gating_mode", "learned")
        self.gate_constant_value = getattr(cfg, "gate_constant_value", 0.80)
        self._conv_cache = None     # (B, 2, gate_dim) state for streaming decode
        self.last_alpha = None      # (B, T, 1), detached - for diagnostics
        self._alpha_for_aux = None  # (B, T, 1), grad-carrying - for PG loss
        self._valid_mask = None     # (B, T) bool - non-padding positions, set by hook
        self.reset_diagnostics()

    def reset_conv_cache(self):
        """Reset streaming state cache for new generation sequences."""
        self._conv_cache = None

    def reset_diagnostics(self):
        """Reset non-persistent telemetry for tokens that actually receive steering."""
        self._telemetry = {
            "count": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "sequence_count": 0,
            "sequence_std_sum": 0.0,
            "low_count": 0,
            "high_count": 0,
        }

    @torch.no_grad()
    def record_diagnostics(self, alpha: torch.Tensor, intervention_mask: torch.Tensor):
        """Accumulate token-level gate health only where a residual is injected."""
        a = alpha.detach().float().squeeze(-1)
        valid = intervention_mask.detach().to(device=a.device, dtype=torch.bool)
        if valid.shape != a.shape:
            raise ValueError(f"gate telemetry mask mismatch: alpha={tuple(a.shape)}, mask={tuple(valid.shape)}")
        values = a[valid]
        if values.numel() == 0:
            return
        stats = self._telemetry
        stats["count"] += int(values.numel())
        stats["sum"] += float(values.sum().item())
        stats["sum_sq"] += float(values.square().sum().item())
        stats["low_count"] += int((values < 0.05).sum().item())
        stats["high_count"] += int((values > 0.95).sum().item())
        for row, row_mask in zip(a, valid):
            row_values = row[row_mask]
            if row_values.numel() >= 2:
                stats["sequence_count"] += 1
                stats["sequence_std_sum"] += float(row_values.std(unbiased=False).item())

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

        if self.gating_mode == "constant":
            val = torch.full(
                (h.size(0), h.size(1), 1),
                self.gate_constant_value,
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

        # Local causal temporal context: streaming causal conv support for both prefill and decode
        B, T, D = x.shape
        if T > 1:
            x_conv = self.causal_conv(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
            # Cache *pre-convolution* features.  Caching the residual output
            # would feed an already-convolved value back into the next decode
            # step, making token-by-token generation differ from prefill.
            self._conv_cache = x[:, -2:, :].detach()
            x = x + self.drop(self.act(x_conv))
        else:
            # Autoregressive single-step decode (T == 1): maintain temporal context via streaming cache
            if self._conv_cache is not None and self._conv_cache.shape[0] == B:
                x_cat = torch.cat([self._conv_cache, x], dim=1)  # (B, 3, D)
                conv_out = self.causal_conv(x_cat.transpose(1, 2))[:, :, :3].transpose(1, 2)
                x = x + self.drop(self.act(conv_out[:, -1:, :]))
                self._conv_cache = x_cat[:, 1:, :].detach()
            else:
                pre = x  # pre-conv features of this token
                x_conv = self.causal_conv(pre.transpose(1, 2))[:, :, :1].transpose(1, 2)
                x = pre + self.drop(self.act(x_conv))
                self._conv_cache = torch.cat([torch.zeros_like(pre), pre], dim=1).detach()  # [pad, token0]

        gate_logit = torch.clamp(self.fc2(x), min=-10.0, max=10.0)
        alpha = torch.sigmoid(gate_logit)  # (B, T, 1)

        self.last_alpha = alpha.detach()
        self._alpha_for_aux = alpha
        return alpha.to(h.dtype)

    def get_saturation_metrics(self) -> dict:
        """Return aggregate intervention-token telemetry, not one decode token."""
        stats = self._telemetry
        # Direct module tests and lightweight callers may ask for a metric
        # before a wrapper has supplied an intervention mask.  Preserve the
        # useful last-forward fallback while ARS diagnostics use the aggregate
        # branch below.
        if stats["count"] == 0 and self.last_alpha is not None:
            a = self.last_alpha.float().squeeze(-1)
            valid = self._valid_mask
            if valid is not None and valid.shape == a.shape:
                a = a[valid]
            if a.numel() > 0:
                return {
                    "mean_alpha": round(float(a.mean().item()), 4),
                    "token_std": round(float(a.std(unbiased=False).item()), 4),
                    "mean_sequence_std": 0.0,
                    "low_saturation_pct": round(float((a < 0.05).float().mean().item() * 100.0), 2),
                    "high_saturation_pct": round(float((a > 0.95).float().mean().item() * 100.0), 2),
                    "intervened_tokens": 0,
                }
        total = max(1, int(stats["count"]))
        mean = stats["sum"] / total
        variance = max(0.0, stats["sum_sq"] / total - mean ** 2)
        return {
            "mean_alpha": round(mean, 4),
            "token_std": round(variance ** 0.5, 4),
            "mean_sequence_std": round(stats["sequence_std_sum"] / max(1, stats["sequence_count"]), 4),
            "low_saturation_pct": round(100.0 * stats["low_count"] / total, 2),
            "high_saturation_pct": round(100.0 * stats["high_count"] / total, 2),
            "intervened_tokens": int(stats["count"]),
        }
