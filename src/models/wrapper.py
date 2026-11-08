"""Phi-2 wrapper with extensible fixed-route residual steering hooks."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, Optional, Sequence
import weakref

import torch
import torch.nn as nn

from src.steering.steernet import SteerNet, load_layer_norms


RouteSelector = Callable[[int, torch.device], torch.BoolTensor]
GateProvider = Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]


class Phi2WithRSC(nn.Module):
    """Inject SteerNet residuals without modifying the backbone implementation.

    The fixed route is the Phase 1 selector. A future router can be installed
    through :meth:`set_route_selector`; a future gate can be installed through
    :meth:`set_gate_provider`. Neither extension changes hook plumbing.
    """

    def __init__(
        self, base_model: nn.Module, cfg, device: Optional[torch.device] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.cfg = cfg
        self.device = device or next(base_model.parameters()).device
        self.target_layers = list(cfg.fixed_layers)
        self._layers = self._backbone_layers()
        self._validate_target_layers()

        hidden_size = getattr(base_model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(base_model.config, "n_embd", None)
        if hidden_size is None:
            raise ValueError("backbone config does not expose hidden_size or n_embd")

        norms = load_layer_norms(cfg.layer_norms_calibration)
        self.steer_nets = nn.ModuleDict(
            {
                str(layer): SteerNet(
                    hidden_dim=hidden_size,
                    rank=cfg.lora_rank,
                    lora_alpha=cfg.lora_alpha,
                    dropout=cfg.lora_dropout,
                    kappa=cfg.steer_kappa,
                    amplitude_mode=getattr(cfg, "amplitude_mode", "relative"),
                    rho_max=getattr(cfg, "rho_max", 0.04),
                    layer_norms=norms,
                    layer_idx=layer,
                ).to(self.device)
                for layer in self.target_layers
            }
        )

        self._hook_handles = []
        self._hooks_disabled = False
        self._route_selector: Optional[RouteSelector] = None
        self._gate_provider: Optional[GateProvider] = None
        self._routing_override = ("only", set(self.target_layers))
        self._route_matrix: Optional[torch.BoolTensor] = None
        self._selection: Optional[torch.BoolTensor] = None
        self._attention_mask: Optional[torch.Tensor] = None
        self._pending_shield_mask: Optional[torch.Tensor] = None
        self._active_shield_mask: Optional[torch.Tensor] = None
        self._gain: float = 1.0
        self._control: Optional[str] = None
        self._control_seed: int = 0
        self._control_gen: Optional[torch.Generator] = None
        self._steer_scale: Optional[float] = None
        self._is_decode_step = False
        self._collect_routing_trace = False
        self.routing_step_log = []
        self.guard_state: Dict[str, object] = {}
        self._collect_injection_trace = False
        self.injection_trace: Dict[int, list] = {}

        self._register_hooks()
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

    def _backbone_layers(self):
        try:
            return self.base_model.model.layers
        except AttributeError as error:
            raise ValueError(
                "backbone must expose transformer blocks at model.layers"
            ) from error

    def _validate_target_layers(self) -> None:
        if not self.target_layers:
            raise ValueError("fixed route cannot be empty")
        invalid = [
            layer
            for layer in self.target_layers
            if layer < 0 or layer >= len(self._layers)
        ]
        if invalid:
            raise ValueError(f"fixed route contains invalid layer indices: {invalid}")

    def _register_hooks(self) -> None:
        reference = weakref.ref(self)
        for position, layer_index in enumerate(self.target_layers):
            steer_net = self.steer_nets[str(layer_index)]
            handle = self._layers[layer_index].register_forward_hook(
                self._make_layer_hook(reference, position, steer_net)
            )
            self._hook_handles.append(handle)
        self._hook_handles.append(
            self.base_model.register_forward_pre_hook(
                self._prepare_forward, with_kwargs=True
            )
        )

    @staticmethod
    def _make_layer_hook(reference, position: int, steer_net: SteerNet):
        def hook(_, __, output):
            wrapper = reference()
            if wrapper is None or wrapper._hooks_disabled:
                return output
            return wrapper._apply_steering(position, steer_net, output)

        return hook

    def _prepare_forward(self, _, args, kwargs) -> None:
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None and len(args) > 1:
            attention_mask = args[1]
        if input_ids is None:
            raise ValueError("Phi2WithRSC requires input_ids")

        self._attention_mask = attention_mask
        # Detect decode: single-token input = autoregressive decode step.
        # Do NOT require attention_mask — newer transformers (>=4.44) may omit
        # it during generate(), using cache_position instead, which caused
        # every decode step to be misdetected as prefill (resetting guard_state,
        # conv cache, and re-applying prompt shield → blocking few-shot decode).
        self._is_decode_step = bool(input_ids.shape[1] == 1)
        self._selection = self._select_route(input_ids.shape[0], input_ids.device)
        if not self._is_decode_step:
            self.guard_state = {"rel_max": 0.0, "nonfinite": False}
            self._active_shield_mask = self._pending_shield_mask
            self.reset_steernet_cache()
            self.injection_trace = {}
        else:
            self._active_shield_mask = None
        if self._collect_routing_trace:
            self.routing_step_log.append({"routes": self.current_routes()})

    def _select_route(self, batch_size: int, device: torch.device) -> torch.BoolTensor:
        if self._routing_override is not None:
            mode, value = self._routing_override
            if mode == "none":
                return torch.zeros(
                    batch_size, len(self.target_layers), dtype=torch.bool, device=device
                )
            if mode == "only":
                route = torch.tensor(
                    [layer in value for layer in self.target_layers],
                    dtype=torch.bool,
                    device=device,
                )
                return route.expand(batch_size, -1)
            if mode == "matrix":
                if (
                    self._route_matrix is None
                    or self._route_matrix.shape[0] != batch_size
                ):
                    raise ValueError("route matrix must have one row per batch item")
                return self._route_matrix.to(device)
            raise ValueError(f"unsupported route override: {mode}")
        if self._route_selector is not None:
            selection = self._route_selector(batch_size, device)
            if selection.shape != (batch_size, len(self.target_layers)):
                raise ValueError("route selector returned an invalid route matrix")
            return selection.to(device=device, dtype=torch.bool)
        return torch.ones(
            batch_size, len(self.target_layers), dtype=torch.bool, device=device
        )

    def _apply_steering(self, position: int, steer_net: SteerNet, output):
        is_tuple = isinstance(output, tuple)
        hidden, suffix = (
            (output[0], output[1:]) if is_tuple else (output, ())
        )
        was_2d = hidden.ndim == 2
        if was_2d:
            hidden = hidden.unsqueeze(0)
        selection = self._selection[:, position].view(-1, 1, 1).to(hidden.dtype)
        if not bool(selection.any()):
            return output

        hidden_float = hidden.float()
        valid_mask = self._valid_token_mask(hidden_float)
        valid_mask = self._apply_prompt_shield(valid_mask)
        routing_k = self._selection.sum(dim=1)
        amp_k = getattr(self.cfg, "steer_amplitude_k", None)
        active_k = (
            torch.full_like(routing_k, float(amp_k), dtype=hidden_float.dtype)
            if amp_k is not None
            else routing_k.to(hidden_float.dtype)
        )
        delta = steer_net(
            hidden_float,
            active_k=active_k,
            is_decode=self._is_decode_step,
            injection_mask=valid_mask,
        )
        gate = self._gate_multiplier(position, hidden_float, delta)
        injected = delta * selection * gate * self._effective_scale()
        injected = injected * valid_mask.unsqueeze(-1).to(injected.dtype)
        if self._control == "random":
            if self._control_gen is None:
                self._control_gen = torch.Generator(device=injected.device).manual_seed(self._control_seed)
            noise = torch.randn(
                injected.shape, generator=self._control_gen, device=injected.device, dtype=injected.dtype
            )
            energy = injected.norm(dim=-1, keepdim=True)
            injected = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-6) * energy
        self._record_guard(injected, hidden_float, valid_mask)
        if self._collect_injection_trace:
            relative = injected.norm(dim=-1) / hidden_float.norm(dim=-1).clamp_min(1e-6)
            self.injection_trace.setdefault(self.target_layers[position], []).append(relative.detach().cpu())
        result = (hidden_float + injected).to(hidden.dtype)
        if was_2d:
            result = result.squeeze(0)
        return (result, *suffix) if is_tuple else result

    def _valid_token_mask(self, hidden: torch.Tensor) -> torch.BoolTensor:
        if self._attention_mask is None:
            return torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        return self._attention_mask[:, -hidden.shape[1] :].to(
            device=hidden.device, dtype=torch.bool
        )

    def _apply_prompt_shield(self, valid_mask: torch.BoolTensor) -> torch.BoolTensor:
        if not self.cfg.prompt_shielding or self._active_shield_mask is None:
            return valid_mask
        shield = self._active_shield_mask[:, -valid_mask.shape[1] :].to(
            valid_mask.device, torch.bool
        )
        return valid_mask & ~shield

    def _gate_multiplier(self, position: int, hidden: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if self._gate_provider is None:
            return torch.ones_like(delta[..., :1])
        gate = self._gate_provider(position, hidden, delta)
        return gate.to(delta.device, delta.dtype)

    def _effective_scale(self) -> float:
        base = 1.0 if getattr(self.cfg, "amplitude_mode", "relative") == "relative" else (
            self.cfg.steer_scale if self._steer_scale is None else self._steer_scale
        )
        return float(base * getattr(self, "_gain", 1.0))

    def _record_guard(
        self, delta: torch.Tensor, hidden: torch.Tensor, valid_mask: torch.BoolTensor
    ) -> None:
        if not bool(valid_mask.any()):
            return
        ratios = delta.norm(dim=-1) / hidden.norm(dim=-1).clamp_min(1e-6)
        relative_max = float(ratios[valid_mask].detach().max())
        self.guard_state["rel_max"] = max(
            float(self.guard_state.get("rel_max", 0.0)), relative_max
        )
        self.guard_state["nonfinite"] = bool(
            self.guard_state.get("nonfinite", False)
        ) or not bool(torch.isfinite(delta).all())

    def set_intervention_gain(self, gain: float) -> None:
        self._gain = float(gain)

    def set_control(self, mode: Optional[str], seed: int = 0) -> None:
        self._control, self._control_seed, self._control_gen = mode, int(seed), None

    def set_route_selector(self, selector: Optional[RouteSelector]) -> None:
        self._route_selector = selector

    def set_gate_provider(self, provider: Optional[GateProvider]) -> None:
        self._gate_provider = provider

    def set_routing_override(self, mode: Optional[str] = None, layers=None) -> None:
        if mode is None:
            self._routing_override = None
            self._route_matrix = None
            return
        if mode == "matrix":
            matrix = torch.as_tensor(layers, dtype=torch.bool)
            if matrix.ndim != 2 or matrix.shape[1] != len(self.target_layers):
                raise ValueError("route matrix must have shape (batch, fixed_layers)")
            self._route_matrix = matrix
            self._routing_override = (mode, None)
            return
        if mode not in {"none", "only"}:
            raise ValueError(f"unsupported route override: {mode}")
        selected = set(layers or [])
        unknown = selected - set(self.target_layers)
        if unknown:
            raise ValueError(f"route contains non-candidate layers: {sorted(unknown)}")
        self._routing_override = (mode, selected)

    def subset_matrix(self, subsets: Sequence[Sequence[int]]) -> torch.BoolTensor:
        return torch.tensor(
            [[layer in subset for layer in self.target_layers] for subset in subsets],
            dtype=torch.bool,
        )

    def current_routes(self):
        if self._selection is None:
            return []
        return [
            tuple(
                layer
                for layer, enabled in zip(self.target_layers, row.tolist())
                if enabled
            )
            for row in self._selection
        ]

    def set_prompt_shield_mask(self, mask: Optional[torch.Tensor]) -> None:
        self._pending_shield_mask = mask

    def set_steer_scale(self, scale: Optional[float]) -> None:
        self._steer_scale = scale

    def set_routing_trace(self, enabled: bool) -> None:
        self._collect_routing_trace = bool(enabled)

    def set_injection_trace(self, enabled: bool) -> None:
        self._collect_injection_trace = bool(enabled)

    def set_telemetry(self, enabled: bool) -> None:
        for steer_net in self.steer_nets.values():
            steer_net.telemetry = bool(enabled)

    def reset_steernet_cache(self) -> None:
        for steer_net in self.steer_nets.values():
            steer_net.reset_cache()

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    @contextmanager
    def hooks_disabled(self):
        previous = self._hooks_disabled
        self._hooks_disabled = True
        try:
            yield
        finally:
            self._hooks_disabled = previous

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        self.reset_steernet_cache()
        return self.base_model.generate(*args, **kwargs)
