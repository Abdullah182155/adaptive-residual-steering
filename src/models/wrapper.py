"""Phi-2 wrapper with extensible fixed-route residual steering hooks."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import Callable, Dict, Optional, Sequence
import weakref

import torch
import torch.nn as nn

from src.steering.steernet import (
    FixedLowRankResidualSteerNet,
    PromptConditionedSteerNet,
    SteerNet,
    load_layer_norms,
)
from src.steering.route_bank import (
    normalize_route_catalog,
    route_key,
    validate_route_expert_states,
)


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

        norms = load_layer_norms(
            cfg.layer_norms_calibration,
            statistic=getattr(cfg, "steernet_norm_stat", "median_norm"),
        )
        if getattr(cfg, "steernet_arch", "v5") == "prompt_conditioned_residual":
            self.steer_nets = nn.ModuleDict(
                {
                    str(layer): PromptConditionedSteerNet(
                        hidden_dim=hidden_size,
                        rank=cfg.lora_rank,
                        rho_max=cfg.rho_max,
                        layer_norm_ref=norms.get(layer, 0.0),
                        layer_idx=layer,
                        max_strength=cfg.prompt_strength_max,
                    ).to(self.device)
                    for layer in self.target_layers
                }
            )
        elif getattr(cfg, "steernet_arch", "v5") == "v2":
            self.steer_nets = nn.ModuleDict(
                {
                    str(layer): FixedLowRankResidualSteerNet(
                        hidden_dim=hidden_size,
                        rank=cfg.lora_rank,
                        rho_max=cfg.rho_max,
                        layer_norm_ref=norms.get(layer, 0.0),
                        layer_idx=layer,
                    ).to(self.device)
                    for layer in self.target_layers
                }
            )
        else:
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
                        mag_head_init_bias=getattr(cfg, "mag_head_init_bias", -3.0),
                        init_gain=getattr(cfg, "steernet_init_gain", 1.0),
                        use_mag_head=getattr(cfg, "steernet_use_mag_head", True),
                        use_soft_cap=getattr(cfg, "steernet_use_soft_cap", True),
                        use_causal_conv=getattr(cfg, "steernet_use_conv", True),
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
        self.bank_experts = nn.ModuleDict()
        self._bank_route_catalog: tuple[tuple[int, ...], ...] = ((),)
        self._active_route_experts: Optional[list[str]] = None
        self._attention_mask: Optional[torch.Tensor] = None
        self._pending_shield_mask: Optional[torch.Tensor] = None
        self._active_shield_mask: Optional[torch.Tensor] = None
        self._pending_conditioning_mask: Optional[torch.Tensor] = None
        self._active_conditioning_mask: Optional[torch.Tensor] = None
        self._cached_prompt_strengths: Dict[int, torch.Tensor] = {}
        self._gain: float = 1.0
        self._route_gains: Optional[torch.Tensor] = None
        self._intervention_gains: Optional[torch.Tensor] = None
        self._prompt_strength_override: Optional[torch.Tensor] = None
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
        self._capture_deltas = False
        self._captured_deltas: Dict[int, torch.Tensor] = {}
        self._captured_steering_inputs: Optional[Dict[int, torch.Tensor]] = None

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
            default_conditioning_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones_like(input_ids, dtype=torch.bool)
            )
            self._active_conditioning_mask = (
                self._pending_conditioning_mask
                if self._pending_conditioning_mask is not None
                else default_conditioning_mask
            )
            self._cached_prompt_strengths.clear()
            self.reset_steernet_cache()
            self.injection_trace = {}
        else:
            self._active_shield_mask = None
            self._active_conditioning_mask = None
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
        if self._captured_steering_inputs is not None:
            self._captured_steering_inputs[self.target_layers[position]] = (
                hidden_float.detach()
            )
        valid_mask = self._valid_token_mask(hidden_float)
        valid_mask = self._apply_prompt_shield(valid_mask)
        routing_k = self._selection.sum(dim=1)
        amp_k = getattr(self.cfg, "steer_amplitude_k", None)
        active_k = (
            torch.full_like(routing_k, float(amp_k), dtype=hidden_float.dtype)
            if amp_k is not None
            else routing_k.to(hidden_float.dtype)
        )
        route_gain = self._route_gain_for(hidden_float)
        steer_kwargs = {
            "active_k": active_k,
            "is_decode": self._is_decode_step,
            "injection_mask": valid_mask,
        }
        if self._active_route_experts is not None:
            delta = self._apply_route_experts(
                position, hidden_float, active_k, valid_mask, route_gain
            )
        elif isinstance(steer_net, PromptConditionedSteerNet):
            if self._is_decode_step:
                prompt_strength = self._cached_prompt_strengths.get(
                    self.target_layers[position]
                )
                if prompt_strength is None:
                    raise RuntimeError(
                        "prompt strength was not cached during generation prefill"
                    )
                delta, _ = steer_net(
                    hidden_float,
                    strength_override=prompt_strength,
                    gain=route_gain,
                    **steer_kwargs,
                )
            else:
                conditioning_mask = self._active_conditioning_mask
                if conditioning_mask is None:
                    raise RuntimeError("prompt conditioning mask is unavailable")
                conditioning_mask = conditioning_mask[:, -hidden_float.shape[1] :].to(
                    device=hidden_float.device, dtype=torch.bool
                )
                delta, predicted_strength = steer_net(
                    hidden_float,
                    conditioning_mask=conditioning_mask,
                    strength_override=self._prompt_strength_override,
                    gain=route_gain,
                    **steer_kwargs,
                )
                if not self.training:
                    self._cached_prompt_strengths[self.target_layers[position]] = (
                        predicted_strength.detach()
                    )
        elif isinstance(steer_net, FixedLowRankResidualSteerNet):
            delta = steer_net(hidden_float, gain=route_gain, **steer_kwargs)
        else:
            delta = steer_net(hidden_float, **steer_kwargs)
        gate = self._gate_multiplier(position, hidden_float, delta)
        injected = delta * selection * gate * self._effective_scale()
        intensity = self._intensity_for(hidden_float)
        if intensity is not None:
            injected = injected * intensity.view(-1, 1, 1)
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
        if self._capture_deltas:
            self._captured_deltas[self.target_layers[position]] = injected
        result = (hidden_float + injected).to(hidden.dtype)
        if was_2d:
            result = result.squeeze(0)
        return (result, *suffix) if is_tuple else result

    def _apply_route_experts(
        self,
        position: int,
        hidden: torch.Tensor,
        active_k: torch.Tensor,
        valid_mask: torch.BoolTensor,
        route_gain: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply each example's independent route expert at this layer."""
        layer = self.target_layers[position]
        if self._active_route_experts is None:
            raise RuntimeError("route experts were requested without an active route")
        keys = self._active_route_experts
        batch_size = hidden.shape[0]
        if len(keys) != batch_size:
            if batch_size % len(keys) != 0:
                raise ValueError("route expert count must divide the active generation batch")
            repeat = batch_size // len(keys)
            keys = [key for key in keys for _ in range(repeat)]
        delta = torch.zeros_like(hidden)
        active = self._selection[:, position]
        groups: dict[str, list[int]] = {}
        for row, key in enumerate(keys):
            if bool(active[row]) and key != "k0":
                groups.setdefault(key, []).append(row)
        for key, row_ids in groups.items():
            if key not in self.bank_experts or str(layer) not in self.bank_experts[key]:
                raise ValueError(f"route expert {key!r} has no adapter for active layer {layer}")
            indices = torch.tensor(row_ids, device=hidden.device, dtype=torch.long)
            local_gain = route_gain
            if local_gain is not None and local_gain.ndim > 0:
                local_gain = local_gain.index_select(0, indices)
            local = self.bank_experts[key][str(layer)](
                hidden.index_select(0, indices),
                active_k=active_k.index_select(0, indices),
                is_decode=self._is_decode_step,
                injection_mask=valid_mask.index_select(0, indices),
                gain=local_gain,
            )
            delta.index_copy_(0, indices, local)
        return delta

    def add_route_expert(
        self,
        route: Sequence[int],
        initial_layer_states: Optional[dict[int | str, dict[str, torch.Tensor]]] = None,
    ) -> str:
        """Create an independent adapter set for a route, optionally from experts."""
        raw_route = tuple(int(layer) for layer in route)
        if len(set(raw_route)) != len(raw_route):
            raise ValueError("route layers must be unique")
        normalized = tuple(sorted(raw_route))
        if not normalized or not set(normalized).issubset(self.target_layers):
            raise ValueError("a trainable route expert must contain candidate layers")
        key = route_key(normalized)
        if key in self.bank_experts:
            return key
        modules = nn.ModuleDict()
        states = initial_layer_states or {}
        for layer in normalized:
            module = deepcopy(self.steer_nets[str(layer)])
            state = states.get(layer, states.get(str(layer)))
            if state is not None:
                module.load_state_dict(state, strict=True)
            modules[str(layer)] = module
        self.bank_experts[key] = modules
        return key

    def route_expert_state(self, route: Sequence[int]) -> dict[str, torch.Tensor]:
        """Return an independent route expert state on CPU."""
        key = route_key(route)
        if key not in self.bank_experts:
            raise KeyError(f"route expert {key!r} is not installed")
        return {
            name: value.detach().cpu().clone()
            for name, value in self.bank_experts[key].state_dict().items()
        }

    def install_route_experts(
        self,
        route_catalog: Sequence[Sequence[int]],
        expert_states: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        """Replace the installed route catalog with independently weighted experts."""
        normalized = normalize_route_catalog(route_catalog, self.target_layers)
        validate_route_expert_states(normalized, expert_states)
        installed = nn.ModuleDict()
        for route in normalized:
            if not route:
                continue
            key = route_key(route)
            if key not in expert_states:
                raise ValueError(f"route expert checkpoint is missing weights for {key}")
            modules = nn.ModuleDict()
            route_state = expert_states[key]
            for layer in route:
                module = deepcopy(self.steer_nets[str(layer)])
                prefix = f"{layer}."
                layer_state = {
                    name[len(prefix):]: value
                    for name, value in route_state.items()
                    if name.startswith(prefix)
                }
                if not layer_state:
                    raise ValueError(f"route expert {key} has no state for layer {layer}")
                module.load_state_dict(layer_state, strict=True)
                modules[str(layer)] = module
            installed[key] = modules
        self.bank_experts = installed
        self._bank_route_catalog = normalized
        self.set_route_experts(None)

    def set_route_experts(self, routes: Optional[Sequence[Sequence[int]]]) -> None:
        """Set one expert route per batch item; ``None`` selects shared adapters."""
        if routes is None:
            self._active_route_experts = None
            return
        if not routes:
            raise ValueError("route expert selection cannot be empty")
        keys = [route_key(route) for route in routes]
        unknown = [key for key in keys if key != "k0" and key not in self.bank_experts]
        if unknown:
            raise ValueError(f"requested route experts are not installed: {unknown}")
        self._active_route_experts = keys

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

    def _route_gain_for(self, hidden: torch.Tensor) -> Optional[torch.Tensor]:
        if self._route_gains is None:
            return None
        gains = self._route_gains.to(device=hidden.device, dtype=hidden.dtype)
        if gains.ndim == 0:
            return gains
        if gains.ndim != 1 or gains.shape[0] != hidden.shape[0]:
            raise ValueError("route gains must be a scalar or one value per example")
        return gains

    def _intensity_for(self, hidden: torch.Tensor) -> Optional[torch.Tensor]:
        """Return post-cap per-example intervention scales for the active batch."""
        if self._intervention_gains is None:
            return None
        gains = self._intervention_gains.to(device=hidden.device, dtype=hidden.dtype)
        if gains.ndim == 0:
            return gains.expand(hidden.shape[0])
        if gains.ndim != 1 or gains.shape[0] != hidden.shape[0]:
            raise ValueError("intervention gains must be scalar or one value per example")
        return gains

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
        self._route_gains = None
        self._intervention_gains = None

    def set_intervention_gains(self, gains: Optional[torch.Tensor]) -> None:
        """Set post-cap intensity, allowing each batch item its own nonnegative scale.

        Applying intensity after SteerNet's norm cap keeps its meaning stable:
        zero disables steering, one preserves the learned bounded residual, and
        values between zero and one attenuate that residual without changing its
        direction or the configured route.
        """
        if gains is None:
            self._intervention_gains = None
            return
        values = torch.as_tensor(gains).detach()
        if not torch.isfinite(values).all() or bool((values < 0).any()):
            raise ValueError("intervention gains must be finite and non-negative")
        self._intervention_gains = values

    def set_route_gains(self, gains: Optional[torch.Tensor]) -> None:
        """Set per-example V2 gains; the V2 module applies them before its norm cap."""
        if gains is None:
            self._route_gains = None
            return
        values = torch.as_tensor(gains).detach()
        if not torch.isfinite(values).all() or bool((values < 0).any()):
            raise ValueError("route gains must be finite and non-negative")
        self._route_gains = values

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

    def set_prompt_conditioning_mask(self, mask: Optional[torch.Tensor]) -> None:
        """Set the prompt-only token mask used by question-conditioned strength heads."""
        self._pending_conditioning_mask = mask

    def set_prompt_strength_override(self, strength: Optional[torch.Tensor | float]) -> None:
        """Override learned prompt strengths for controlled action evaluation."""
        if strength is None:
            self._prompt_strength_override = None
            return
        values = torch.as_tensor(strength).detach()
        maximum = float(getattr(self.cfg, "prompt_strength_max", 1.25))
        if (
            not torch.isfinite(values).all()
            or bool((values < 0).any())
            or bool((values > maximum).any())
        ):
            raise ValueError(f"prompt strength must be finite and between 0 and {maximum}")
        self._prompt_strength_override = values

    def get_prompt_strengths(self) -> Dict[int, torch.Tensor]:
        """Return the most recently cached per-layer strengths on CPU."""
        return {
            layer: value.detach().float().cpu().clone()
            for layer, value in self._cached_prompt_strengths.items()
        }

    def set_steer_scale(self, scale: Optional[float]) -> None:
        self._steer_scale = scale

    def set_routing_trace(self, enabled: bool) -> None:
        self._collect_routing_trace = bool(enabled)

    def set_injection_trace(self, enabled: bool) -> None:
        self._collect_injection_trace = bool(enabled)

    def set_telemetry(self, enabled: bool) -> None:
        modules = list(self.steer_nets.values()) + [
            expert for route in self.bank_experts.values() for expert in route.values()
        ]
        for steer_net in modules:
            if hasattr(steer_net, "telemetry"):
                steer_net.telemetry = bool(enabled)

    def reset_steernet_cache(self) -> None:
        modules = list(self.steer_nets.values()) + [
            expert for route in self.bank_experts.values() for expert in route.values()
        ]
        for steer_net in modules:
            reset = getattr(steer_net, "reset_cache", None)
            if reset is not None:
                reset()
        self._cached_prompt_strengths.clear()

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

    @contextmanager
    def capture_base_hidden(self):
        """Run a forward pass without steering, capturing hidden states at target layers."""
        captured: Dict[int, torch.Tensor] = {}
        handles = []
        for layer_idx in self.target_layers:
            def _make_hook(li):
                def _hook(module, _input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    captured[li] = h.detach().float()
                return _hook
            handles.append(self._layers[layer_idx].register_forward_hook(_make_hook(layer_idx)))
        prev = self._hooks_disabled
        self._hooks_disabled = True
        try:
            yield captured
        finally:
            for h in handles:
                h.remove()
            self._hooks_disabled = prev

    @contextmanager
    def capture_steering_inputs(self):
        """Capture the pre-adapter hidden state seen by each active steering head."""
        previous = self._captured_steering_inputs
        captured: Dict[int, torch.Tensor] = {}
        self._captured_steering_inputs = captured
        try:
            yield captured
        finally:
            self._captured_steering_inputs = previous

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        self.reset_steernet_cache()
        return self.base_model.generate(*args, **kwargs)
