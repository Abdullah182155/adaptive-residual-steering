import weakref
from typing import List, Optional, Tuple, Dict, Any, Set
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.configs.config import RSCConfig
from src.steering.steernet import RSCSteerNet
from src.gating.gate import RSCUsefulnessGate
from src.routing.router import JointLayerRouter
from src.routing.rloo import sample_k, sample_subset_plackett_luce

class Phi2WithRSC(nn.Module):
    """Wraps Phi-2 backbone with RSC forward hooks for steering, gating, and routing."""

    def __init__(self, base_model, cfg: RSCConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.base_model = base_model
        self.cfg = cfg
        self.hidden_dim = base_model.config.hidden_size
        self.device = device or next(base_model.parameters()).device

        n_layers = len(base_model.model.layers)
        self.target_layers = cfg.rsc_layers_for(n_layers)

        self.gates = nn.ModuleDict()
        self.steer_nets = nn.ModuleDict()
        self.router = JointLayerRouter(
            self.target_layers, self.hidden_dim, n_layers, cfg.max_active_layers
        ).to(self.device)

        self._hooks = []
        self._attn_mask_cache = None
        self._selected_counts = None
        self._desired_k = None
        self._precomputed_routes = None
        self._router_probabilities = None
        self._router_logits = None
        self._router_margins = None
        self._last_inferred_k = None
        self._router_context_mask_pending = None
        self._router_context_mask_active = None
        self._router_warmup = False
        self._warmup_cursor = 0
        self._warmup_width_visits = [0] * self.cfg.max_active_layers
        self._prompt_level_selected_count = [0] * len(self.target_layers)
        self._prompt_level_total = 0
        self._routing_override = None
        self._router_inference_mode = "argmax"

        # Cached decode attributes
        self._cached_decode_routes = None
        self._cached_decode_k = None
        self._cached_decode_probs = None
        self._cached_decode_logits = None
        self._cached_decode_margins = None
        self._cached_decode_active_k = None
        self._active_k_realized = None
        self._collect_routing_trace = False
        self.last_routing_trace = None

        self.register_buffer(
            "routing_k_hist",
            torch.zeros(cfg.max_active_layers + 1, dtype=torch.long, device=self.device),
        )
        self.register_buffer("routing_forwards", torch.tensor(0, dtype=torch.long, device=self.device))

        self._inject_rsc(n_layers)

        # Freeze backbone parameters strictly
        for p in self.base_model.parameters():
            p.requires_grad = False

    def _inject_rsc(self, n_layers: int):
        layers = self.base_model.model.layers
        self_ref = weakref.ref(self)

        for idx in self.target_layers:
            gate = RSCUsefulnessGate(self.hidden_dim, self.cfg).to(self.device)
            steer = RSCSteerNet(
                self.hidden_dim, self.cfg.lora_rank, self.cfg.lora_alpha, self.cfg.lora_dropout
            ).to(self.device)
            self.gates[str(idx)] = gate
            self.steer_nets[str(idx)] = steer
            layer_pos = self.target_layers.index(idx)

            def make_hook(layer_idx, layer_pos, g, s):
                def hook_fn(module, args, output):
                    wrapper = self_ref()
                    if wrapper is None:
                        return output
                    is_tuple = isinstance(output, tuple)
                    h = output[0] if is_tuple else output
                    orig_2d = False
                    if h.ndim == 2:
                        orig_2d = True
                        h = h.unsqueeze(0)  # Convert (T, d) to (1, T, d)
                    with torch.autocast(device_type="cuda", enabled=False):
                        h_fp = h.float()
                        mask = wrapper._valid_mask_for(h_fp)
                        selected, route_prob = wrapper._select_layer(layer_idx, layer_pos, h_fp, mask)
                        if not selected.any():
                            g._alpha_for_aux, g.last_alpha, g._valid_mask = None, None, None
                            return output
                        wrapper.router.selected_forwards[layer_pos] += selected.long().sum()
                        delta = s(h_fp, active_k=wrapper._active_k_realized)
                        alpha = g(h_fp, delta, global_context=wrapper.router._last_semantic_shared)
                        g._valid_mask = mask
                        route = selected.to(alpha.dtype).view(-1, 1, 1)
                        route_st = route
                        k_scale = wrapper._active_k_realized.to(dtype=alpha.dtype).sqrt().view(-1, 1, 1)
                        h_new = h_fp + route_st * alpha.detach() * delta * mask.to(alpha.dtype).unsqueeze(-1) / k_scale
                    h_out = h_new.to(h.dtype)
                    if orig_2d:
                        h_out = h_out.squeeze(0)
                    if is_tuple:
                        return (h_out,) + output[1:]
                    return h_out

                return hook_fn

            handle = layers[idx].register_forward_hook(make_hook(idx, layer_pos, gate, steer))
            self._hooks.append(handle)

        def _capture_mask_pre_hook(module, args, kwargs):
            mask = kwargs.get("attention_mask", None)
            if mask is None and len(args) > 1:
                mask = args[1]
            self._attn_mask_cache = mask
            inp = kwargs.get("input_ids") if "input_ids" in kwargs else (args[0] if args else None)
            inp_len = inp.shape[1] if inp is not None and inp.ndim >= 2 else 0
            mask_len = mask.shape[1] if mask is not None and mask.ndim == 2 else 0
            batch = inp.shape[0] if inp is not None else mask.shape[0]

            is_cached_decode = (inp_len == 1 and mask_len > 1)
            self._selected_counts = torch.zeros(batch, dtype=torch.long, device=next(self.gates.parameters()).device)

            if is_cached_decode and self._cached_decode_routes is not None:
                self._precomputed_routes = self._cached_decode_routes
                self._desired_k = self._cached_decode_k
                self._router_probabilities = self._cached_decode_probs
                self._router_logits = self._cached_decode_logits
                self._router_margins = self._cached_decode_margins
                self._active_k_realized = self._cached_decode_active_k
            else:
                self._desired_k = None
                self._precomputed_routes = None
                self._router_probabilities = None
                self._router_logits = None
                self._router_margins = None
                self._active_k_realized = None
                if not is_cached_decode:
                    self._cached_decode_routes = None
                    self._cached_decode_k = None
                    self._cached_decode_probs = None
                    self._cached_decode_logits = None
                    self._cached_decode_margins = None
                    self._cached_decode_active_k = None

            self._router_context_mask_active = self._router_context_mask_pending
            self._router_context_mask_pending = None
            return args, kwargs

        self._mask_capture_hook = self.base_model.register_forward_pre_hook(
            _capture_mask_pre_hook, with_kwargs=True
        )
        self._hooks.append(self._mask_capture_hook)

    def set_gate_freeze(self, freeze: bool, value: float = 1.0):
        for gate in self.gates.values():
            gate._freeze_gate = freeze
            gate._freeze_gate_value = value

    def set_router_warmup(self, enabled: bool):
        self._router_warmup = enabled

    def set_router_inference_mode(self, mode: str):
        if mode not in {"argmax", "sample"}:
            raise ValueError('router inference mode must be "argmax" or "sample"')
        self._router_inference_mode = mode

    def set_router_exploration(self, enabled: bool):
        return

    def set_router_context_mask(self, context_mask=None):
        self._router_context_mask_pending = context_mask

    def set_routing_override(self, mode=None, layers=None):
        if mode is None:
            self._routing_override = None
            return
        if mode == "all":
            mode = "only"
            layers = list(self.target_layers)
        if mode not in {"only", "exclude", "none", "force", "matrix"}:
            raise ValueError("routing override must be one of: only, exclude, none, force, matrix, all")
        if mode == "matrix":
            if not torch.is_tensor(layers) or layers.dtype != torch.bool or layers.dim() != 2:
                raise ValueError("matrix override requires a (B, n_candidates) bool tensor")
            self._routing_override = (mode, layers)
            return
        layers = set(layers or [])
        unknown = layers - set(self.target_layers)
        if unknown:
            raise ValueError(f"routing override contains non-candidate layers: {sorted(unknown)}")
        self._routing_override = (mode, layers)

    def set_routing_trace(self, enabled: bool):
        self._collect_routing_trace = enabled
        if not enabled:
            self.last_routing_trace = None

    def _valid_mask_for(self, h: torch.Tensor) -> torch.Tensor:
        mask = self._attn_mask_cache
        if mask is None:
            return torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        if mask.ndim != 2 or mask.shape[0] != h.shape[0] or mask.shape[1] < h.shape[1]:
            raise RuntimeError(f"RSC attention-mask mismatch: mask={tuple(mask.shape)}, hidden={tuple(h.shape)}")
        return mask[:, -h.shape[1] :].to(device=h.device, dtype=torch.bool)

    def _router_mask_for(self, h: torch.Tensor, fallback_mask: torch.Tensor) -> torch.Tensor:
        mask = self._router_context_mask_active
        if mask is None or mask.ndim != 2 or mask.shape[0] != h.shape[0] or mask.shape[1] < h.shape[1]:
            return fallback_mask
        aligned = mask[:, -h.shape[1] :].to(device=h.device, dtype=torch.bool)
        return torch.where(aligned.any(dim=1, keepdim=True), aligned, fallback_mask)

    def _finalize_selection(self, selected_matrix: torch.Tensor, margins: torch.Tensor) -> torch.Tensor:
        K = self.cfg.max_active_layers
        counts = selected_matrix.sum(dim=-1)
        over_budget = counts > K
        if over_budget.any():
            ranks = margins.argsort(dim=-1, descending=True).argsort(dim=-1)
            capped = ranks < K
            selected_matrix = torch.where(over_budget.unsqueeze(-1), selected_matrix & capped, selected_matrix)
        none_selected = ~selected_matrix.any(dim=-1)
        if none_selected.any():
            fallback_idx = margins.argmax(dim=-1)
            fb_mask = torch.zeros_like(selected_matrix)
            fb_mask[torch.arange(selected_matrix.shape[0], device=selected_matrix.device), fallback_idx] = True
            selected_matrix = torch.where(none_selected.unsqueeze(-1), fb_mask, selected_matrix)
        return selected_matrix

    def _prepare_routes(self, h: torch.Tensor, mask: torch.Tensor):
        if self._precomputed_routes is not None:
            self.router.seen_forwards += h.shape[0]
            return
        if self._selected_counts is None or self._selected_counts.shape[0] != h.shape[0]:
            raise RuntimeError("RSC router state was not initialised by the model forward pre-hook")

        router_mask = self._router_mask_for(h, mask)
        u, k_logits = self.router(h, router_mask)
        self._router_logits = u
        self._router_margins = u
        self._router_probabilities = F.softmax(u, dim=-1)
        self.router.seen_forwards += h.shape[0]

        n_candidates = len(self.target_layers)
        K_max = self.cfg.max_active_layers

        if self._router_warmup:
            width = 1 + (self._warmup_cursor % K_max)
            stride = max(1, n_candidates // width)
            pos = self._warmup_width_visits[width - 1]
            forced_indices = {self.target_layers[(pos + j * stride) % n_candidates] for j in range(width)}
            self._warmup_width_visits[width - 1] += 1
            selected_matrix = torch.tensor(
                [idx in forced_indices for idx in self.target_layers], device=h.device, dtype=torch.bool
            ).unsqueeze(0).expand(h.shape[0], -1)
            self._warmup_cursor += 1
            self._desired_k = selected_matrix.long().sum(dim=-1).clamp(min=1, max=K_max)
        elif self._routing_override is None:
            if self.training:
                with torch.no_grad():
                    k_real, _ = sample_k(k_logits)
                    selected_matrix, _ = sample_subset_plackett_luce(u, k_real)
                self._desired_k = k_real
            else:
                if self._router_inference_mode == "sample":
                    with torch.no_grad():
                        k_real, _ = sample_k(k_logits)
                    k_real = k_real.clamp(min=1, max=K_max)
                else:
                    k_real = (k_logits.argmax(dim=-1) + 1).clamp(min=1, max=K_max)
                ranks = u.argsort(dim=-1, descending=True)
                selected_matrix = torch.zeros(h.shape[0], n_candidates, dtype=torch.bool, device=h.device)
                for b in range(h.shape[0]):
                    kb = int(k_real[b].item())
                    selected_matrix[b, ranks[b, :kb]] = True
                self._desired_k = k_real
            self.router.observe(u, selected_matrix)
            any_selected_this_fwd = selected_matrix.any(dim=0)
            for i in range(n_candidates):
                if bool(any_selected_this_fwd[i]):
                    self._prompt_level_selected_count[i] += 1
            self._prompt_level_total += 1
        else:
            mode, layers = self._routing_override
            if mode == "none":
                selected_matrix = torch.zeros(h.shape[0], n_candidates, dtype=torch.bool, device=h.device)
            elif mode == "matrix":
                selected_matrix = layers.to(device=h.device, dtype=torch.bool)
            elif mode == "only":
                selected_matrix = torch.tensor(
                    [idx in layers for idx in self.target_layers], device=h.device, dtype=torch.bool
                ).unsqueeze(0).expand(h.shape[0], -1)
            elif mode == "force":
                base_ranks = u.argsort(dim=-1, descending=True)
                base_selected = torch.zeros(h.shape[0], n_candidates, dtype=torch.bool, device=h.device)
                base_selected.scatter_(1, base_ranks[:, :K_max], True)
                forced = torch.tensor(
                    [idx in layers for idx in self.target_layers], device=h.device, dtype=torch.bool
                ).unsqueeze(0)
                selected_matrix = base_selected | forced
            else:
                eligible = torch.tensor([idx not in layers for idx in self.target_layers], device=h.device)
                scores = u.masked_fill(~eligible.unsqueeze(0), float("-inf"))
                selected_matrix = self._finalize_selection((scores > float("-inf")) & eligible.unsqueeze(0), scores)
            self._desired_k = selected_matrix.long().sum(dim=-1).clamp(min=1, max=K_max)

        self.router._last_prediction = self._desired_k
        self._last_inferred_k = self._desired_k
        self._precomputed_routes = selected_matrix
        self._active_k_realized = selected_matrix.sum(dim=-1).clamp(min=1)

        if h.shape[1] > 1:
            self._cached_decode_routes = selected_matrix.detach()
            self._cached_decode_k = self._desired_k.detach()
            self._cached_decode_probs = self._router_probabilities.detach()
            self._cached_decode_logits = self._router_logits.detach()
            self._cached_decode_margins = self._router_margins.detach()
            self._cached_decode_active_k = self._active_k_realized.detach()

    def _select_layer(self, layer_idx: int, layer_pos: int, h: torch.Tensor, mask: torch.Tensor):
        if layer_idx == self.target_layers[0]:
            self._prepare_routes(h, mask)
        if self._precomputed_routes is None or self._desired_k is None:
            raise RuntimeError("RSC routes were not prepared at the first candidate layer")
        selected = self._precomputed_routes[:, layer_pos]
        probability = self._router_probabilities[:, layer_pos]
        margin = self._router_margins[:, layer_pos]
        fallback = torch.zeros_like(selected)
        if self._collect_routing_trace:
            if self.last_routing_trace is None:
                self.last_routing_trace = {}
            self.last_routing_trace[int(layer_idx)] = {
                "probability": probability.detach().float().cpu().tolist(),
                "logit": self._router_logits[:, layer_pos].detach().float().cpu().tolist(),
                "selection_rate_ema": float(self.router.selection_rate_ema[layer_pos].item()),
                "probability_ema": float(self.router.selection_rate_ema[layer_pos].item()),
                "logit_ema": float(self.router.logit_ema[layer_pos].item()),
                "margin_z": margin.detach().float().cpu().tolist(),
                "selected": selected.detach().cpu().tolist(),
                "fallback": fallback.detach().cpu().tolist(),
                "desired_k": self._desired_k.detach().cpu().tolist(),
                "predicted_k": self.router._last_prediction.detach().cpu().tolist(),
            }
        self._selected_counts = self._selected_counts + selected.long()
        if layer_idx == self.target_layers[-1] and not self._router_warmup:
            counts = self._selected_counts.clamp(max=self.cfg.max_active_layers)
            self.routing_k_hist.add_(torch.bincount(counts, minlength=self.cfg.max_active_layers + 1))
            self.routing_forwards.add_(counts.numel())
        return selected, probability

    def reset_routing_diagnostics(self):
        self.router.selected_forwards.zero_()
        self.router.seen_forwards.zero_()
        self.routing_k_hist.zero_()
        self.routing_forwards.zero_()

    def routing_diagnostics(self):
        return {
            int(layer): {
                "selection_rate_ema": float(self.router.selection_rate_ema[pos].item()),
                "logit_ema": float(self.router.logit_ema[pos].item()),
                "selected_forwards": int(self.router.selected_forwards[pos].item()),
                "seen_forwards": int(self.router.seen_forwards[pos].item()),
            }
            for pos, layer in enumerate(self.target_layers)
        }

    def routing_k_distribution(self):
        total = max(1, int(self.routing_forwards.item()))
        return {k: int(count.item()) / total for k, count in enumerate(self.routing_k_hist)}

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)
