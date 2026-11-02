"""Phi-2 wrapper with a constant, auditable Phase 1 steering route."""
from contextlib import contextmanager
import weakref
from typing import Dict, Optional, Sequence
import torch
import torch.nn as nn
from src.steering.steernet_v5 import RSCSteerNetV5, load_layer_norms

class Phi2WithRSC(nn.Module):
    def __init__(self, base_model, cfg, device: Optional[torch.device] = None):
        super().__init__()
        self.base_model, self.cfg = base_model, cfg
        self.device = device or next(base_model.parameters()).device
        layers = base_model.model.layers
        self.target_layers = list(cfg.fixed_layers)
        if not self.target_layers or any(layer < 0 or layer >= len(layers) for layer in self.target_layers):
            raise ValueError(f"fixed_layers must name valid backbone layers: {self.target_layers}")
        hidden_dim = getattr(base_model.config, "hidden_size", getattr(base_model.config, "n_embd", None))
        if hidden_dim is None: raise ValueError("backbone config has no hidden size")
        norms = load_layer_norms(cfg.layer_norms_calibration)
        self.steer_nets = nn.ModuleDict({str(layer): RSCSteerNetV5(
            hidden_dim, rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha, dropout=cfg.lora_dropout,
            kappa=cfg.steer_kappa, layer_norms=norms, layer_idx=layer).to(self.device)
            for layer in self.target_layers})
        self._hooks, self._hooks_disabled = [], False
        self._routing_override = ("only", set(self.target_layers))
        self._prompt_shield_mask_pending = self._prompt_shield_mask_active = None
        self._decode_only_steering = False
        self._shot_aware_scale = self._current_n_shot = None
        self._steer_scale = None
        self._trace_cached_decode = False
        self.guard_state, self.routing_step_log, self._collect_routing_trace = {}, [], False
        self._inject(layers)
        for parameter in self.base_model.parameters(): parameter.requires_grad = False

    def _inject(self, layers):
        ref = weakref.ref(self)
        for position, layer_idx in enumerate(self.target_layers):
            net = self.steer_nets[str(layer_idx)]
            def hook(_, __, output, position=position, layer_idx=layer_idx, net=net):
                wrapper = ref()
                if wrapper is None or wrapper._hooks_disabled or not wrapper._selected(position): return output
                h, suffix = (output[0], output[1:]) if isinstance(output, tuple) else (output, ())
                original_2d = h.ndim == 2
                if original_2d: h = h.unsqueeze(0)
                h_float = h.float(); mask = wrapper._valid_mask(h_float)
                if wrapper._decode_only_steering and not wrapper._trace_cached_decode: mask.zero_()
                if wrapper._prompt_shield_mask_active is not None:
                    shield = wrapper._prompt_shield_mask_active[:, -h.shape[1]:].to(h.device, torch.bool)
                    mask &= ~shield
                delta = net(h_float, is_decode=wrapper._trace_cached_decode)
                scale = wrapper._effective_scale()
                injected = delta * mask.unsqueeze(-1).to(delta.dtype) * scale
                wrapper._record_guard(injected, h_float, mask)
                result = (h_float + injected).to(h.dtype)
                if original_2d: result = result.squeeze(0)
                return (result, *suffix) if suffix else result
            self._hooks.append(layers[layer_idx].register_forward_hook(hook))
        def pre_hook(_, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            mask = kwargs.get("attention_mask")
            if mask is None and len(args) > 1: mask = args[1]
            self._attention_mask = mask
            self._trace_cached_decode = bool(ids is not None and ids.shape[1] == 1 and mask is not None and mask.shape[1] > 1)
            self._prompt_shield_mask_active = None if self._trace_cached_decode else self._prompt_shield_mask_pending
            if not self._trace_cached_decode:
                for net in self.steer_nets.values(): net.reset_cache()
            if self._collect_routing_trace:
                self.routing_step_log.append({"routes": [tuple(self._route()) for _ in range(ids.shape[0])]})
        self._hooks.append(self.base_model.register_forward_pre_hook(pre_hook, with_kwargs=True))

    def _route(self):
        if self._routing_override is None: return self.target_layers
        mode, value = self._routing_override
        if mode == "none": return []
        if mode == "only": return list(value)
        if mode == "matrix": return []
        raise ValueError(f"unsupported routing mode: {mode}")

    def _selected(self, position):
        if self._routing_override is None: return True
        mode, value = self._routing_override
        if mode == "none": return False
        if mode == "only": return self.target_layers[position] in value
        if mode == "matrix": return bool(value[:, position].any())
        raise ValueError(f"unsupported routing mode: {mode}")

    def _valid_mask(self, h):
        if getattr(self, "_attention_mask", None) is None: return torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        return self._attention_mask[:, -h.shape[1]:].to(h.device, torch.bool)

    def _effective_scale(self):
        scale = self.cfg.steer_scale if self._steer_scale is None else self._steer_scale
        if self._shot_aware_scale is not None and self._current_n_shot is not None:
            scale *= self._shot_aware_scale.get(self._current_n_shot, 1.0)
        return float(scale)

    def _record_guard(self, delta, hidden, mask):
        if not mask.any(): return
        ratio = (delta.norm(dim=-1) / hidden.norm(dim=-1).clamp_min(1e-6))[mask].max()
        self.guard_state["rel_max"] = max(float(self.guard_state.get("rel_max", 0.0)), float(ratio))
        self.guard_state["nonfinite"] = not bool(torch.isfinite(delta).all())

    def set_routing_override(self, mode=None, layers=None):
        if mode is None: self._routing_override = None; return
        if mode not in {"none", "only", "matrix"}: raise ValueError(f"unsupported routing mode: {mode}")
        if mode == "matrix":
            if not isinstance(layers, torch.Tensor) or layers.ndim != 2 or layers.shape[1] != len(self.target_layers): raise ValueError("matrix route has invalid shape")
        self._routing_override = (mode, layers if mode == "matrix" else set(layers or []))

    def subset_matrix(self, subsets: Sequence[Sequence[int]]):
        for subset in subsets:
            unknown = set(subset) - set(self.target_layers)
            if unknown: raise ValueError(f"not a candidate layer: {sorted(unknown)}")
        return torch.tensor([[layer in subset for layer in self.target_layers] for subset in subsets], dtype=torch.bool)
    def set_prompt_shield_mask(self, mask): self._prompt_shield_mask_pending = mask
    def set_decode_only_steering(self, enabled): self._decode_only_steering = bool(enabled)
    def set_shot_aware_scale(self, scales: Optional[Dict[int, float]]): self._shot_aware_scale = scales
    def set_current_n_shot(self, n_shot: Optional[int]): self._current_n_shot = n_shot
    def set_steer_scale(self, scale: Optional[float]): self._steer_scale = scale
    def set_gate_freeze(self, _, value=1.0): self._steer_scale = float(value)
    def set_gate_mode(self, _, scale=None): self._steer_scale = 1.0 if scale is None else float(scale)
    def set_routing_trace(self, enabled): self._collect_routing_trace = bool(enabled)
    def reset_steernet_cache(self):
        for net in self.steer_nets.values(): net.reset_cache()
    def remove_hooks(self):
        for handle in self._hooks: handle.remove()
        self._hooks.clear()
    @contextmanager
    def hooks_disabled(self):
        previous = self._hooks_disabled; self._hooks_disabled = True
        try: yield
        finally: self._hooks_disabled = previous
    def forward(self, *args, **kwargs): return self.base_model(*args, **kwargs)
    def generate(self, *args, **kwargs): self.reset_steernet_cache(); return self.base_model.generate(*args, **kwargs)
