import math
import warnings
import weakref
from typing import List, Optional, Tuple, Dict, Any, Set, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.configs.config import RSCConfig
from src.steering.steernet import RSCSteerNet
from src.gating.gate import RSCUsefulnessGate, masked_route_allocation, token_competitive_allocation
from src.routing.router import JointLayerRouter
from src.routing.rloo import sample_k, sample_subset_plackett_luce
from contextlib import contextmanager
from typing import Sequence as _Sequence
from src.utils.failure_policy import FatalContractError, get_manifest

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
            self.target_layers,
            self.hidden_dim,
            n_layers,
            cfg.max_active_layers,
            layer_dropout=getattr(cfg, "router_layer_dropout", 0.0),  # BUG-C2 fix: fallback 0.15→0.0 (matches config.py; avoids train/eval mismatch)
            difficulty_aware=getattr(cfg, "router_difficulty_aware", True),
            synergy_limit=getattr(cfg, "router_synergy_limit", 0.75),
            use_demo_features=getattr(cfg, "router_demo_features", False),
        ).to(self.device)

        self._hooks = []
        self._attn_mask_cache = None
        self._selected_counts = None
        self._desired_k = None
        self._precomputed_routes = None
        self._router_probabilities = None
        self._router_logits = None
        self._router_margins = None
        self._route_allocation = None
        self._token_competitive_alpha = None
        self._last_inferred_k = None
        self._router_context_mask_pending = None
        self._router_context_mask_active = None
        self._prompt_shield_mask_pending = None
        self._prompt_shield_mask_active = None
        self.prompt_shielding = getattr(cfg, "prompt_shielding", True)
        self._router_warmup = False
        self._warmup_cursor = 0
        self._warmup_width_visits = [0] * self.cfg.max_active_layers
        self._prompt_level_selected_count = [0] * len(self.target_layers)
        self._prompt_level_total = 0
        self._routing_override = None
        # Set after Phase 1B.  A router may only learn/deploy routes composed
        # of layers that survived the Phase-1 safety contract and received
        # co-adaptation updates.  ``None`` preserves the complete catalog for
        # Phase-1 audits and fixed experimental controls.
        self._eligible_routing_layers: Optional[Set[int]] = None
        # Calibrated Phase-1 scales are part of the routing contract. They are
        # separate from transient gate-freeze state so reloads remain faithful.
        self._certified_gate_scales: Dict[int, float] = {}
        if not getattr(cfg, "use_router", True) or getattr(cfg, "fixed_layers", None) is not None:
            fl = cfg.fixed_layers or list(self.target_layers)
            self.set_routing_override("only", fl)
        # Evaluation must be reproducible.  Sampling is useful for policy
        # training, but an evaluation route is a deterministic argmax policy.
        self._router_inference_mode = getattr(cfg, "router_eval_mode", "argmax")
        self._router_exploration = True
        self._joint_grad_mode = False  # When True (Phase 2C), task gradient flows through both alpha and delta
        self.deployment_disabled = False  # Set to True if finalize_deployment decides baseline only
        self.deployment_modes_by_shot = {}

        # Cached decode attributes
        self._cached_decode_routes = None
        self._cached_decode_k = None
        self._cached_decode_probs = None
        self._cached_decode_logits = None
        self._cached_decode_margins = None
        self._cached_decode_active_k = None
        self._cached_decode_allocation = None
        self._active_k_realized = None
        self._collect_routing_trace = False
        self.last_routing_trace = None
        self._hooks_disabled = False
        self._steer_scale = None
        self.routing_step_log = []
        self.guard_state = {}
        # Per-generation evidence used by the qualitative evaluator.  Values
        # are detached scalar alphas only; tracing never retains a model graph.
        self._collect_gate_trace = False
        self.last_gate_trace = None
        self._trace_cached_decode = False
        self._steering_telemetry = {}
        # Python-only diagnostic state: it must not become a checkpoint tensor
        # or retain a computation graph between micro-batches.
        self._route_tuple_counts: Dict[Tuple[int, ...], int] = {}
        self._active_pre_cap_records = []

        self.register_buffer(
            "routing_k_hist",
            torch.zeros(cfg.max_active_layers + 1, dtype=torch.long, device=self.device),
        )
        self.register_buffer("routing_forwards", torch.tensor(0, dtype=torch.long, device=self.device))

        # ---- M6b state ----
        self._catalog = None
        self._catalog_membership = None
        self._route_is_catalog = False
        self._last_catalog_idx = None
        self._router_policy = None          # None -> fall back to cfg.router_policy
        self._abstain_theta = None          # None -> fall back to cfg.abstain_theta
        self._shot_bucket = None
        self._gate_mode = "learned"
        self._manifest = None
        if not hasattr(self, "_joint_grad_mode"):
            self._joint_grad_mode = False
        if not hasattr(self, "_steer_scale"):
            self._steer_scale = None
        if not isinstance(getattr(self, "guard_state", None), dict):
            self.guard_state = {}
        self._guards_enabled = bool(getattr(cfg, "guards_enabled", True))
        self._guard_rel_norm_cap = float(getattr(cfg, "guard_rel_norm_cap", 0.02))
        self._guard_k_max = int(getattr(cfg, "guard_k_max", 3))
        self._guard_rel_max = None
        self._guard_nonfinite = None

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
                self.hidden_dim,
                self.cfg.lora_rank,
                self.cfg.lora_alpha,
                self.cfg.lora_dropout,
                magnitude_mode=getattr(self.cfg, "steer_magnitude_mode", "direct"),
                max_magnitude=getattr(self.cfg, "steer_max_magnitude", 2.0),
                max_relative_norm=getattr(self.cfg, "steer_max_relative_norm", 0.04),     # BUG-C3 fix: 0.15→0.04
                orthogonal_projection=getattr(self.cfg, "steer_orthogonal_projection", False),
                k_max=getattr(self.cfg, "max_active_layers", 4),
                telemetry_reservoir_size=getattr(self.cfg, "pre_cap_telemetry_reservoir_size", 8192),
            ).to(self.device)
            self.gates[str(idx)] = gate
            self.steer_nets[str(idx)] = steer
            layer_pos = self.target_layers.index(idx)

            def make_hook(layer_idx, layer_pos, g, s):
                def hook_fn(module, args, output):
                    wrapper = self_ref()
                    if wrapper is None:
                        return output
                    if wrapper._hooks_disabled:
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
                        if not wrapper._route_is_catalog:
                            if wrapper._routing_override is None:
                                wrapper.router.selected_forwards[layer_pos] += selected.long().sum()
                        delta = s(h_fp, active_k=wrapper._active_k_realized if getattr(wrapper.cfg, 'steernet_k_conditioning', True) else None)
                        if wrapper._token_competitive_alpha is not None:
                            # Advance the causal gate cache using the native
                            # layer state, then substitute the shared
                            # competitive allocation for its amplitude.
                            _ = g(h_fp, delta, global_context=wrapper.router._last_semantic_shared)
                            alpha = wrapper._token_competitive_alpha[:, :, layer_pos:layer_pos + 1].to(dtype=h_fp.dtype)
                            g.last_alpha = alpha.detach()
                            g._alpha_for_aux = alpha
                        else:
                            alpha = g(h_fp, delta, global_context=wrapper.router._last_semantic_shared)
                        # A route has one shared intervention budget.  Gates
                        # retain token-level strength, while selected layers
                        # receive a masked-softmax allocation instead of each
                        # injecting a full residual on the same tokens.
                        allocation = (
                            torch.ones_like(alpha)
                            if (wrapper._token_competitive_alpha is not None or wrapper._route_allocation is None)
                            else wrapper._route_allocation[:, layer_pos].to(alpha.dtype).view(-1, 1, 1)
                        )
                        alpha = alpha * allocation
                        g._alpha_for_aux = alpha
                        g.last_alpha = alpha.detach()
                        if wrapper._collect_gate_trace:
                            if wrapper.last_gate_trace is None:
                                wrapper.last_gate_trace = {int(layer): [] for layer in wrapper.target_layers}
                            # The prefill's final alpha controls the first
                            # generated token; every cached single-token call
                            # then aligns with the next generated token.
                            traced = alpha.detach().float().squeeze(-1)
                            # BUG-I4 fix: previous if/else were identical (both traced[:, -1]).
                            # Prefill (T>1): use mean across all valid positions for a representative summary.
                            # Cached decode (T=1): use the single token gate value directly.
                            if wrapper._trace_cached_decode:
                                token_alpha = traced[:, -1]      # autoregressive single-token step
                            else:
                                token_alpha = traced.mean(dim=1) # prefill: mean over all positions
                            wrapper.last_gate_trace[int(layer_idx)].append(float(token_alpha[0].cpu().item()))
                        route = selected.to(alpha.dtype).view(-1, 1, 1)
                        route_st = route
                        # Each individual SteerNet is bounded, but that alone
                        # does not bound their cumulative displacement across a
                        # route.  Give a route one shared budget: K=1 is
                        # untouched; K>1 is scaled by K**(-p), p=0.5 by
                        # default.  The same operation is used in training and
                        # generation, so a multi-layer route is never evaluated
                        # in a larger norm regime than it was trained in.
                        active_k = wrapper._active_k_realized.to(dtype=alpha.dtype).clamp(min=1)
                        budget_power = float(getattr(wrapper.cfg, "steer_route_budget_power", 0.0))
                        route_budget = active_k.pow(-budget_power).view(-1, 1, 1)
                        steer_mask = mask
                        if wrapper.prompt_shielding and wrapper._prompt_shield_mask_active is not None:
                            sh = wrapper._prompt_shield_mask_active
                            if sh.ndim == 2 and sh.shape[0] == h_fp.shape[0] and sh.shape[1] >= h_fp.shape[1]:
                                sh_aligned = sh[:, -h_fp.shape[1]:].to(device=h_fp.device, dtype=torch.bool)
                                steer_mask = steer_mask & (~sh_aligned)
                        # Gate regularisers and telemetry must operate on
                        # positions that receive an injected residual.  Using
                        # the padding-only mask included shielded few-shot
                        # demonstrations, where the gate cannot affect loss.
                        g._valid_mask = steer_mask
                        valid = steer_mask & selected.view(-1, 1)
                        s._last_valid_mask = valid
                        # The activity-floor loss needs physical intervention
                        # energy, not merely a large gate on a zero residual.
                        g._effective_energy_for_aux = (
                            delta.norm(dim=-1) * alpha.squeeze(-1) * route_budget.squeeze(-1)
                        )
                        if valid.any() and getattr(s, "last_pre_cap_q", None) is not None:
                            wrapper._active_pre_cap_records.append((s, s.last_pre_cap_q, valid))
                        if wrapper._routing_override is None:
                            if valid.any():
                                raw_ratio = (delta.norm(dim=-1) / h_fp.norm(dim=-1).clamp_min(1e-6))[valid]
                                effective = delta * alpha * route_budget
                                effective_ratio = (effective.norm(dim=-1) / h_fp.norm(dim=-1).clamp_min(1e-6))[valid]
                                metrics = wrapper._steering_telemetry.setdefault(
                                    int(layer_idx), {
                                        "count": 0, "raw_sum": 0.0, "effective_sum": 0.0,
                                        "raw_max": 0.0, "effective_max": 0.0,
                                        "energy_sum": 0.0, "energy_max": 0.0,
                                    }
                                )
                                metrics["count"] += int(valid.sum().item())
                                metrics["raw_sum"] += float(raw_ratio.sum().item())
                                metrics["effective_sum"] += float(effective_ratio.sum().item())
                                metrics["raw_max"] = max(metrics["raw_max"], float(raw_ratio.max().item()))
                                metrics["effective_max"] = max(metrics["effective_max"], float(effective_ratio.max().item()))
                                effective_energy = effective.norm(dim=-1)[valid]
                                metrics["energy_sum"] += float(effective_energy.sum().item())
                                metrics["energy_max"] = max(metrics["energy_max"], float(effective_energy.max().item()))
                        if wrapper._routing_override is None:
                            g.record_diagnostics(alpha, steer_mask)
                        allow_grad = wrapper._joint_grad_mode or getattr(wrapper.cfg, "gate_task_gradient", False)
                        scale = (
                            wrapper._steer_scale
                            if wrapper._steer_scale is not None
                            else getattr(wrapper.cfg, "steer_scale", 1.0)
                        )
                        alpha_eff = (alpha if allow_grad else alpha.detach()) * scale
                        delta_injected = (
                            route_st * alpha_eff * delta * route_budget
                            * steer_mask.to(alpha.dtype).unsqueeze(-1)
                        )
                        h_new = h_fp + delta_injected
                        record = getattr(wrapper, "_record_envelope", None)
                        if record is not None:
                            record(layer_pos, delta_injected.detach(), h_fp, steer_mask, h_new)
                        wrapper._update_guard(delta_injected, h_fp, steer_mask, selected, layer_idx=layer_idx)
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
            self._trace_cached_decode = is_cached_decode
            if self._collect_gate_trace and not is_cached_decode:
                self.last_gate_trace = {int(layer): [] for layer in self.target_layers}
            self._selected_counts = torch.zeros(batch, dtype=torch.long, device=next(self.gates.parameters()).device)
            self._active_pre_cap_records = []
            for sn in self.steer_nets.values():
                sn.last_pre_cap_q = None
                sn._last_valid_mask = None
            for gate in self.gates.values():
                gate.last_alpha = None
                gate._alpha_for_aux = None
                gate._valid_mask = None

            if is_cached_decode and self._cached_decode_routes is not None:
                self._precomputed_routes = self._cached_decode_routes
                self._desired_k = self._cached_decode_k
                self._router_probabilities = self._cached_decode_probs
                self._router_logits = self._cached_decode_logits
                self._router_margins = self._cached_decode_margins
                self._active_k_realized = self._cached_decode_active_k
                self._route_allocation = self._cached_decode_allocation
                self._token_competitive_alpha = None
                self._prompt_shield_mask_active = None
            else:
                self._desired_k = None
                self._precomputed_routes = None
                self._router_probabilities = None
                self._router_logits = None
                self._router_margins = None
                self._active_k_realized = None
                self._route_allocation = None
                self._token_competitive_alpha = None
                if not is_cached_decode:
                    self._cached_decode_routes = None
                    self._cached_decode_k = None
                    self._cached_decode_probs = None
                    self._cached_decode_logits = None
                    self._cached_decode_margins = None
                    self._cached_decode_active_k = None
                    self._cached_decode_allocation = None
                    self.guard_state.clear()
                    self._prompt_shield_mask_active = self._prompt_shield_mask_pending
                    for g in self.gates.values():
                        if hasattr(g, "reset_conv_cache"):
                            g.reset_conv_cache()

            self._router_context_mask_active = self._router_context_mask_pending
            self._router_context_mask_pending = None
            self._prompt_shield_mask_pending = None
            return args, kwargs

        self._mask_capture_hook = self.base_model.register_forward_pre_hook(
            _capture_mask_pre_hook, with_kwargs=True
        )
        self._hooks.append(self._mask_capture_hook)

    def set_prompt_shield_mask(self, mask: Optional[torch.Tensor] = None):
        """Set a boolean mask (B, T) where True indicates tokens that should be shielded
        (i.e. zero residual steering perturbation and excluded from router pooling)."""
        self._prompt_shield_mask_pending = mask

    def set_prompt_shielding(self, enabled: bool):
        """Enable or disable prompt shielding."""
        self.prompt_shielding = enabled

    def set_gate_freeze(self, freeze: bool, value: float | Dict[int, float] = 1.0):
        """Freeze gates globally or at calibrated per-layer intensities."""
        values = value if isinstance(value, dict) else None
        for layer, gate in self.gates.items():
            gate._freeze_gate = freeze
            gate._freeze_gate_value = float(
                values.get(int(layer), 1.0) if values is not None else value
            )

    def set_certified_gate_scales(self, scales: Optional[Dict[int, float]]) -> None:
        """Record validated Phase-1B layer scales for checkpoint portability."""
        if scales is None:
            self._certified_gate_scales = {}
            return
        normalized = {int(layer): float(scale) for layer, scale in scales.items()}
        unknown = set(normalized) - set(self.target_layers)
        if unknown:
            raise ValueError(f"certified gate scales contain non-candidates: {sorted(unknown)}")
        if any(not math.isfinite(scale) or scale < 0.0 for scale in normalized.values()):
            raise ValueError("certified gate scales must be finite and non-negative")
        self._certified_gate_scales = normalized

    def set_router_warmup(self, enabled: bool):
        self._router_warmup = enabled

    def set_router_inference_mode(self, mode: str):
        if mode not in {"sample", "argmax"}:
            raise ValueError('router inference mode must be "argmax" or "sample"')
        self._router_inference_mode = mode

    def set_router_exploration(self, enabled: bool):
        self._router_exploration = bool(enabled)

    def set_router_context_mask(self, context_mask=None):
        self._router_context_mask_pending = context_mask

    def set_eligible_routing_layers(self, layers: Optional[List[int]] = None):
        """Restrict learned routing to a certified Phase-1B layer pool.

        Explicit routing overrides remain available for controlled audits, but
        ordinary router inference and RLOO sampling are masked to this pool.
        This prevents a confirmation-failed or never-co-adapted SteerNet from
        silently re-entering downstream adaptive phases.
        """
        if layers is None:
            self._eligible_routing_layers = None
        else:
            selected = {int(layer) for layer in layers}
            unknown = selected - set(self.target_layers)
            if unknown:
                raise ValueError(f"eligible routing layers contain non-candidates: {sorted(unknown)}")
            if not selected:
                raise ValueError("eligible routing layers must not be empty")
            self._eligible_routing_layers = selected
        # Cached decode routes were constructed under the preceding contract.
        self._cached_decode_routes = None
        self._cached_decode_k = None
        self._cached_decode_probs = None
        self._cached_decode_logits = None
        self._cached_decode_margins = None
        self._cached_decode_active_k = None
        self._cached_decode_allocation = None

    def get_routing_manifest(self) -> Dict[str, Any]:
        """Return non-tensor routing decisions required for reproducible reload.

        ARS weights alone do not encode the certified Phase-1B pool or the
        per-shot deployment decision.  Keeping this state in the manifest
        prevents a newly constructed wrapper from silently exposing its full
        catalog after loading a portable component checkpoint.
        """
        return {
            "eligible_routing_layers": (
                sorted(int(layer) for layer in self._eligible_routing_layers)
                if self._eligible_routing_layers is not None else None
            ),
            "deployment_disabled": bool(self.deployment_disabled),
            "deployment_modes_by_shot": {
                str(shot): str(mode) for shot, mode in self.deployment_modes_by_shot.items()
            },
            "certified_gate_scales": {
                str(layer): float(scale)
                for layer, scale in getattr(self, "_certified_gate_scales", {}).items()
            },
        }

    def load_routing_manifest(self, manifest: Optional[Dict[str, Any]]) -> None:
        """Restore validated non-tensor routing state from a checkpoint sidecar."""
        if manifest is None:
            return
        if not isinstance(manifest, dict):
            raise TypeError("routing manifest must be a dictionary")
        eligible = manifest.get("eligible_routing_layers")
        if eligible is None:
            self.set_eligible_routing_layers(None)
        else:
            if not isinstance(eligible, (list, tuple)):
                raise TypeError("routing manifest eligible_routing_layers must be a list or null")
            self.set_eligible_routing_layers([int(layer) for layer in eligible])
        self.deployment_disabled = bool(manifest.get("deployment_disabled", False))
        modes = manifest.get("deployment_modes_by_shot", {})
        if not isinstance(modes, dict):
            raise TypeError("routing manifest deployment_modes_by_shot must be a dictionary")
        self.deployment_modes_by_shot = {str(shot): str(mode) for shot, mode in modes.items()}
        scales = manifest.get("certified_gate_scales", {})
        if not isinstance(scales, dict):
            raise TypeError("routing manifest certified_gate_scales must be a dictionary")
        self.set_certified_gate_scales({int(layer): float(scale) for layer, scale in scales.items()})

    def routing_eligibility_mask(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return the learned-routing eligibility mask in catalog order."""
        target_device = device or self.device
        if self._eligible_routing_layers is None:
            return torch.ones(len(self.target_layers), dtype=torch.bool, device=target_device)
        return torch.tensor(
            [layer in self._eligible_routing_layers for layer in self.target_layers],
            dtype=torch.bool,
            device=target_device,
        )

    def mask_routing_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask ineligible logits with a finite sentinel for stable policy math."""
        eligible = self.routing_eligibility_mask(logits.device)
        if not eligible.any():
            raise RuntimeError("learned routing has no eligible Phase-1B layers")
        # ``-inf`` reaches log-softmax/entropy/preference terms and can turn
        # an otherwise zero-probability product into NaN.  The legal action
        # space is bounded separately by K_max, so a finite sentinel is both
        # semantically equivalent for routing and numerically safe.
        return logits.masked_fill(~eligible.unsqueeze(0), -1.0e4)

    def mask_routing_k_logits(self, k_logits: torch.Tensor) -> torch.Tensor:
        """Disallow K values that cannot be realised by the eligible pool."""
        max_k = min(self.cfg.max_active_layers, int(self.routing_eligibility_mask(k_logits.device).sum().item()))
        # Modern routers expose K=0..Kmax.  Retain the legacy fallback for a
        # K=1..Kmax head so older checkpoints can still be inspected.
        if k_logits.shape[-1] > self.cfg.max_active_layers:
            legal = torch.arange(k_logits.shape[-1], device=k_logits.device) <= max_k
            if not bool(getattr(self.cfg, "router_k0_enabled", True)):
                legal[0] = False
        else:
            legal = torch.arange(1, k_logits.shape[-1] + 1, device=k_logits.device) <= max_k
        return k_logits.masked_fill(~legal.unsqueeze(0), -1.0e4)

    def set_routing_override(self, mode=None, layers=None):
        if mode is None:
            self._routing_override = None
            return
        if mode == "all":
            mode = "only"
            layers = list(self.target_layers)
        if mode not in {"only", "exclude", "none", "force", "matrix", "balanced"}:
            raise ValueError("routing override must be one of: only, exclude, none, force, matrix, balanced, all")
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

    def set_gate_trace(self, enabled: bool):
        """Collect detached per-token gate values for one generated sample."""
        self._collect_gate_trace = bool(enabled)
        if enabled:
            self.last_gate_trace = {int(layer): [] for layer in self.target_layers}
        else:
            self.last_gate_trace = None

    def _valid_mask_for(self, h: torch.Tensor) -> torch.Tensor:
        mask = self._attn_mask_cache
        if mask is None:
            return torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        if mask.ndim != 2 or mask.shape[0] != h.shape[0] or mask.shape[1] < h.shape[1]:
            raise RuntimeError(f"RSC attention-mask mismatch: mask={tuple(mask.shape)}, hidden={tuple(h.shape)}")
        return mask[:, -h.shape[1] :].to(device=h.device, dtype=torch.bool)

    def _router_mask_for(self, h: torch.Tensor, fallback_mask: torch.Tensor) -> torch.Tensor:
        mask = self._router_context_mask_active
        def mask_stats(value: Optional[torch.Tensor]) -> str:
            if value is None:
                return "mask=None"
            valid = value.detach().to(dtype=torch.bool)
            counts = valid.sum(dim=1).detach().cpu().tolist() if valid.ndim == 2 else []
            return f"shape={tuple(valid.shape)}, valid_tokens_per_row={counts}"
        if mask is None and self.prompt_shielding and self._prompt_shield_mask_active is not None:
            sh = self._prompt_shield_mask_active
            if sh.ndim == 2 and sh.shape[0] == h.shape[0] and sh.shape[1] >= h.shape[1]:
                sh_aligned = sh[:, -h.shape[1]:].to(device=h.device, dtype=torch.bool)
                unshielded = fallback_mask & (~sh_aligned)
                if unshielded.any():
                    mask = unshielded
                else:
                    warnings.warn(
                        "Prompt shielding left no router-context tokens; falling back to the full attention mask "
                        f"({mask_stats(sh_aligned)}; fallback {mask_stats(fallback_mask)}).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        if mask is None or mask.ndim != 2 or mask.shape[0] != h.shape[0] or mask.shape[1] < h.shape[1]:
            if self.prompt_shielding and self._prompt_shield_mask_active is not None:
                warnings.warn(
                    "Prompt shielding router context was invalid; falling back to the full attention mask "
                    f"({mask_stats(mask)}; fallback {mask_stats(fallback_mask)}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return fallback_mask
        aligned = mask[:, -h.shape[1] :].to(device=h.device, dtype=torch.bool)
        if not bool(aligned.any(dim=1).all()):
            warnings.warn(
                "Prompt shielding router context was empty for one or more rows; using the full attention mask for those rows "
                f"({mask_stats(aligned)}; fallback {mask_stats(fallback_mask)}).",
                RuntimeWarning,
                stacklevel=2,
            )
        return torch.where(aligned.any(dim=1, keepdim=True), aligned, fallback_mask)

    def _prepare_token_competitive_allocation(self, h: torch.Tensor, selected_matrix: torch.Tensor) -> None:
        """Prepare per-token selected-layer allocation without mutating gate caches."""
        self._token_competitive_alpha = None
        use_competitive = (
            bool(getattr(self.cfg, "token_competitive_allocation", False))
            and self._routing_override is None
            and not any(gate._freeze_gate or gate.gating_mode == "constant" for gate in self.gates.values())
        )
        if not use_competitive:
            return
        global_ctx = getattr(self.router, "_last_semantic_shared", None)
        logits = []
        for layer in self.target_layers:
            gate = self.gates[str(layer)]
            with torch.no_grad():
                preview_delta = self.steer_nets[str(layer)](h)
            logits.append(gate.gate_logits(h, preview_delta, global_ctx, update_cache=False).squeeze(-1))
        gate_logits = torch.stack(logits, dim=-1)
        beta = token_competitive_allocation(
            gate_logits, selected_matrix,
            temperature=float(getattr(self.cfg, "gate_allocation_temperature", 0.80)),
        )
        masked_logits = gate_logits.masked_fill(~selected_matrix.unsqueeze(1), float("-inf"))
        total_strength = torch.sigmoid(masked_logits.max(dim=-1, keepdim=True).values)
        total_strength = torch.where(
            selected_matrix.any(dim=-1, keepdim=True).unsqueeze(1), total_strength, torch.zeros_like(total_strength)
        )
        self._token_competitive_alpha = beta * total_strength

    def _finalize_selection(self, selected_matrix: torch.Tensor, margins: torch.Tensor) -> torch.Tensor:
        K = self.cfg.max_active_layers
        counts = selected_matrix.sum(dim=-1)
        over_budget = counts > K
        if over_budget.any():
            ranks = margins.argsort(dim=-1, descending=True).argsort(dim=-1)
            capped = ranks < K
            selected_matrix = torch.where(over_budget.unsqueeze(-1), selected_matrix & capped, selected_matrix)
        return selected_matrix

    def _prepare_routes(self, h: torch.Tensor, mask: torch.Tensor):
        if self._precomputed_routes is not None:
            # Diagnostics describe the learned router only.  Baseline and
            # fixed-route controls must not dilute their denominator.
            if self._routing_override is None:
                self.router.seen_forwards += h.shape[0]
            self._prepare_token_competitive_allocation(h, self._precomputed_routes)
            return
        if self._selected_counts is None or self._selected_counts.shape[0] != h.shape[0]:
            raise RuntimeError("RSC router state was not initialised by the model forward pre-hook")

        # ---- M6b: catalog policy branch ----
        self._route_is_catalog = False
        self._last_catalog_idx = None
        if self._catalog_policy_active():
            self._prepare_routes_catalog(h, mask)
            return

        router_mask = self._router_mask_for(h, mask)
        demo_mask = None
        if bool(getattr(self.cfg, "router_demo_features", False)):
            shield = self._prompt_shield_mask_active
            if shield is not None and shield.ndim == 2 and shield.shape[0] == h.shape[0] and shield.shape[1] >= h.shape[1]:
                demo_mask = mask & shield[:, -h.shape[1]:].to(device=h.device, dtype=torch.bool)
            else:
                demo_mask = torch.zeros_like(mask, dtype=torch.bool)
        u, k_logits = self.router(h, router_mask, demo_h=h if demo_mask is not None else None, demo_mask=demo_mask)
        learned_logits = self.mask_routing_logits(u)
        learned_k_logits = self.mask_routing_k_logits(k_logits)
        # A prescribed audit route may intentionally inspect any catalog
        # member.  Eligibility constrains learned routing, never an explicit
        # experimental control.
        trace_logits = learned_logits if self._routing_override is None else u
        self._router_logits = trace_logits
        self._router_margins = trace_logits
        self._router_probabilities = F.softmax(trace_logits, dim=-1)
        if self._routing_override is None:
            self.router.seen_forwards += h.shape[0]

        n_candidates = len(self.target_layers)
        eligible_mask = self.routing_eligibility_mask(h.device)
        eligible_layers = [layer for layer, allowed in zip(self.target_layers, eligible_mask.tolist()) if allowed]
        n_eligible = len(eligible_layers)
        K_max = min(self.cfg.max_active_layers, n_eligible)

        # An explicit route is an audit/benchmark contract.  It must win over
        # background warmup coverage so a requested baseline remains truly
        # unsteered and a requested single layer is actually the one measured.
        if self._router_warmup and self._routing_override is None:
            warmup_max_k = int(getattr(self.cfg, "router_warmup_max_k", 4))
            if getattr(self.cfg, "router_warmup_random_subsets", True):
                import random as _random
                k_max = min(K_max, warmup_max_k, n_eligible)
                width = _random.randint(1, max(1, k_max))
                forced_indices = set(_random.sample(eligible_layers, width))
            else:
                width = 1 + (self._warmup_cursor % max(1, min(K_max, warmup_max_k)))
                stride = max(1, n_eligible // width)
                pos = self._warmup_width_visits[width - 1]
                forced_indices = {eligible_layers[(pos + j * stride) % n_eligible] for j in range(width)}
                self._warmup_width_visits[width - 1] += 1
            selected_matrix = torch.tensor(
                [idx in forced_indices for idx in self.target_layers], device=h.device, dtype=torch.bool
            ).unsqueeze(0).expand(h.shape[0], -1)
            self._warmup_cursor += 1
            self._desired_k = selected_matrix.long().sum(dim=-1).clamp(min=1, max=K_max)
        elif self._routing_override is None:
            synergy = getattr(self.router, "_last_synergy_matrix", None)
            if self.training and self._router_exploration:
                with torch.no_grad():
                    k_real, _ = sample_k(learned_k_logits, k_max=K_max)
                    selected_matrix, _ = sample_subset_plackett_luce(learned_logits, k_real, synergy_matrix=synergy)
                self._desired_k = k_real
            else:
                if self._router_inference_mode == "sample":
                    with torch.no_grad():
                        k_real, _ = sample_k(learned_k_logits, k_max=K_max)
                        selected_matrix, _ = sample_subset_plackett_luce(learned_logits, k_real, synergy_matrix=synergy)
                    k_real = k_real.clamp(min=0, max=K_max)
                else:
                    # Deterministic evaluation: select the learned K (including
                    # K=0) and greedily construct the cooperative subset.
                    with torch.no_grad():
                        k_idx = learned_k_logits.float().argmax(dim=-1)
                        if learned_k_logits.shape[-1] > self.cfg.max_active_layers:
                            k_real = k_idx.clamp(min=0, max=K_max)
                        else:
                            k_real = (k_idx + 1).clamp(min=1, max=K_max)
                    selected_matrix = torch.zeros(h.shape[0], n_candidates, dtype=torch.bool, device=h.device)
                    for b in range(h.shape[0]):
                        kb = int(k_real[b].item())
                        cur_u = learned_logits[b].clone()
                        sel_idx = []
                        for _ in range(min(kb, n_candidates)):
                            for prev in sel_idx:
                                cur_u[prev] = float("-inf")
                            best = cur_u.argmax().item()
                            sel_idx.append(best)
                            if synergy is not None:
                                cur_u = cur_u + 0.5 * synergy[b, best]
                        selected_matrix[b, sel_idx] = True
                self._desired_k = k_real
            self.router.observe(u, selected_matrix, eligible_mask=eligible_mask)
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
            elif mode == "balanced":
                # Fixed SteerNet ablation: a legal, evenly-spaced K-route.
                # This is deliberately not "all" candidates, which is outside
                # the trained intervention regime and therefore not an ablation.
                fixed_k = int(getattr(self.cfg, "fixed_steer_eval_k", 2))
                fixed_k = max(1, min(fixed_k, K_max, n_candidates))
                requested_layers = getattr(self.cfg, "fixed_steer_eval_layers", None)
                if requested_layers:
                    positions = torch.tensor(
                        [self.target_layers.index(layer) for layer in requested_layers if layer in self.target_layers],
                        device=h.device, dtype=torch.long,
                    )
                    if positions.numel() == 0:
                        raise ValueError("fixed_steer_eval_layers contains no candidate layers")
                    positions = positions[:fixed_k]
                else:
                    positions = torch.linspace(0, n_candidates - 1, steps=fixed_k, device=h.device).round().long()
                selected_matrix = torch.zeros(h.shape[0], n_candidates, dtype=torch.bool, device=h.device)
                selected_matrix[:, positions] = True
            else:
                eligible = torch.tensor([idx not in layers for idx in self.target_layers], device=h.device)
                scores = u.masked_fill(~eligible.unsqueeze(0), float("-inf"))
                selected_matrix = self._finalize_selection((scores > float("-inf")) & eligible.unsqueeze(0), scores)
            # Overrides are measured controls: a "none" baseline requests and
            # realizes K=0.  Do not clamp its telemetry to one.
            if mode == "matrix":
                if selected_matrix.size(-1) != len(self.target_layers):
                    raise FatalContractError("E_SHAPE", f"matrix width {selected_matrix.size(-1)} != n_candidates {len(self.target_layers)}")
                if (selected_matrix.sum(-1) > self.cfg.max_active_layers).any():
                    get_manifest().warn("W_K_EXCEEDS_CAP", f"Some rows exceed max_active_layers={self.cfg.max_active_layers}")
                self._desired_k = selected_matrix.long().sum(dim=-1)  # F2: uncapped, min 0
            else:
                self._desired_k = selected_matrix.long().sum(dim=-1).clamp(min=0, max=K_max)

        self.router._last_prediction = self._desired_k
        self._last_inferred_k = self._desired_k
        self._precomputed_routes = selected_matrix
        self._active_k_realized = selected_matrix.sum(dim=-1)
        if self._routing_override is None and not self._router_warmup:
            for row in selected_matrix.detach().cpu().tolist():
                route = tuple(layer for layer, selected in zip(self.target_layers, row) if selected)
                self._route_tuple_counts[route] = self._route_tuple_counts.get(route, 0) + 1
        if self._routing_override is not None:
            # Forced Phase-1B/audit routes are an intervention contract, not
            # a chance for the immature router logits to attenuate one member
            # of a deliberately balanced pair.
            self._route_allocation = selected_matrix.to(dtype=self._router_logits.dtype)
        elif bool(getattr(self.cfg, "factorized_gate_allocation", True)):
            self._route_allocation = masked_route_allocation(
                self._router_logits, selected_matrix,
                temperature=float(getattr(self.cfg, "gate_allocation_temperature", 0.70)),
            )
        else:
            self._route_allocation = selected_matrix.to(dtype=self._router_logits.dtype)

        self._prepare_token_competitive_allocation(h, selected_matrix)

        if h.shape[1] > 1:
            self._cached_decode_routes = selected_matrix.detach()
            self._cached_decode_k = self._desired_k.detach()
            self._cached_decode_probs = self._router_probabilities.detach()
            self._cached_decode_logits = self._router_logits.detach()
            self._cached_decode_margins = self._router_margins.detach()
            self._cached_decode_active_k = self._active_k_realized.detach()
            self._cached_decode_allocation = self._route_allocation.detach()

    def _select_layer(self, layer_idx: int, layer_pos: int, h: torch.Tensor, mask: torch.Tensor):
        if layer_idx == self.target_layers[0]:
            self._reset_guard(h)
            self._prepare_routes(h, mask)
            if self._collect_routing_trace:
                self.routing_step_log.append({
                    "routes": self.current_route_tuples(),
                    "q_len": int(h.shape[1]),
                })
        if self._precomputed_routes is None or self._desired_k is None:
            raise RuntimeError("RSC routes were not prepared at the first candidate layer")
        selected = self._precomputed_routes[:, layer_pos]
        if self._route_is_catalog:
            probability = self._catalog_layer_probs()[:, layer_pos]
            margin = probability
            layer_logit = torch.log(probability.clamp(min=1e-8))
        else:
            probability = self._router_probabilities[:, layer_pos]
            margin = self._router_margins[:, layer_pos]
            layer_logit = self._router_logits[:, layer_pos]
        fallback = torch.zeros_like(selected)
        if self._collect_routing_trace:
            if self.last_routing_trace is None:
                self.last_routing_trace = {}
            self.last_routing_trace[int(layer_idx)] = {
                "probability": probability.detach().float().cpu().tolist(),
                "logit": layer_logit.detach().float().cpu().tolist(),
                "selection_rate_ema": float(self.router.selection_rate_ema[layer_pos].item()),
                "probability_ema": float(self.router.selection_rate_ema[layer_pos].item()),
                "logit_ema": float(self.router.logit_ema[layer_pos].item()),
                "margin_z": margin.detach().float().cpu().tolist(),
                "selected": selected.detach().cpu().tolist(),
                "fallback": fallback.detach().cpu().tolist(),
                "desired_k": self._desired_k.detach().cpu().tolist(),
                "realized_k": self._active_k_realized.detach().cpu().tolist() if self._active_k_realized is not None else [],
                "route_allocation": self._route_allocation[:, layer_pos].detach().float().cpu().tolist() if self._route_allocation is not None else [],
                "predicted_k": self.router._last_prediction.detach().cpu().tolist(),
            }
        self._selected_counts = self._selected_counts + selected.long()
        if layer_idx == self.target_layers[-1] and not self._router_warmup and self._routing_override is None:
            counts = self._selected_counts.clamp(max=self.cfg.max_active_layers)
            self.routing_k_hist.add_(torch.bincount(counts, minlength=self.cfg.max_active_layers + 1))
            self.routing_forwards.add_(counts.numel())
        return selected, probability

    # ---- M6b: question-span / shot-bucket plumbing ----
    def set_question_span_mask(self, mask=None):
        """Alias of set_router_context_mask (pending -> active at the next model forward)."""
        self.set_router_context_mask(mask)

    def set_shot_bucket(self, b: Union[int, Sequence[int], torch.Tensor, None]):
        if b is None:
            self._shot_bucket = None
        elif isinstance(b, int):
            self._shot_bucket = int(b)
        elif torch.is_tensor(b):
            t = b.detach().long()
            self._shot_bucket = int(t.item()) if t.dim() == 0 else t.reshape(-1).clone()
        else:
            self._shot_bucket = torch.as_tensor(list(b), dtype=torch.long)

    def _shot_bucket_for(self, h: torch.Tensor):
        sb = self._shot_bucket
        B = h.shape[0]
        if sb is None:
            return None
        if isinstance(sb, int):
            return torch.full((B,), sb, dtype=torch.long, device=h.device)
        t = sb.to(device=h.device, dtype=torch.long).reshape(-1)
        if t.numel() == 1 and B > 1:
            t = t.expand(B)
        return t

    # ---- M6b: policy / catalog / abstain setters ----
    def set_catalog(self, catalog):
        self._catalog = catalog
        self._catalog_membership = None
        if self._catalog is not None:
            n_subsets = len(self._catalog.subsets) if hasattr(self._catalog, "subsets") else len(self._catalog)
            if getattr(self.router, "catalog_head", None) is None or self.router.catalog_head.out_features != n_subsets:
                self.router.catalog_head = nn.Linear(
                    self.router.d_model, n_subsets
                ).to(self.device)
                nn.init.zeros_(self.router.catalog_head.weight)
                nn.init.zeros_(self.router.catalog_head.bias)

    def set_abstain_threshold(self, theta: float):
        self._abstain_theta = float(theta)

    def set_theta_abs(self, theta: float):
        self.set_abstain_threshold(theta)

    def set_k_scale_mode(self, mode):
        self._k_scale_mode = mode
        if hasattr(self, 'cfg') and self.cfg is not None:
            self.cfg.k_scale_mode = mode

    def reset_guard_state(self):
        self._guard_rel_max = None
        self._guard_nonfinite = None
        if not isinstance(getattr(self, 'guard_state', None), dict):
            self.guard_state = {}
        self.guard_state.clear()
        self.guard_state['enabled'] = getattr(self, '_guards_enabled', True)
        self.guard_state['cap'] = getattr(self, '_guard_rel_norm_cap', 0.02)
        self.guard_state['k_max'] = getattr(self, '_guard_k_max', 3)
        self.guard_state['finite_check'] = getattr(self, '_guard_finite_check', True)
        self.guard_state['revert_to_k0'] = getattr(self, '_guard_revert_to_k0', True)
        self.guard_state['rel_norms'] = {}

    def reset_routing_diagnostics(self):
        if hasattr(self, 'routing_k_hist'):
            self.routing_k_hist.zero_()
        if hasattr(self, 'routing_forwards'):
            self.routing_forwards.zero_()
        if hasattr(self, '_route_tuple_counts'):
            self._route_tuple_counts.clear()

    def set_router_policy(self, mode: str):
        if mode not in {"legacy", "catalog"}:
            raise ValueError('router policy must be "legacy" or "catalog"')
        self._router_policy = mode

    def _policy(self) -> str:
        if self._router_policy is not None:
            return self._router_policy
        return getattr(self.cfg, "router_policy", "legacy")

    def _catalog_policy_active(self) -> bool:
        return (
            self._routing_override is None
            and not self._router_warmup
            and self._policy() == "catalog"
        )

    def _prepare_routes_catalog(self, h: torch.Tensor, mask: torch.Tensor):
        """Catalog policy: one argmax class per batch row, optional abstention to the empty class."""
        if self._catalog is None:
            from src.routing.catalog import Catalog
            layers = getattr(self.cfg, "catalog_layers", None) or self.target_layers
            self._catalog = Catalog.build(layers, getattr(self.cfg, "catalog_max_k", 3))
            self._catalog_membership = None

        if getattr(self.router, "catalog_head", None) is None:
            self.router.catalog_head = nn.Linear(
                self.router.d_model, len(self._catalog.subsets)
            ).to(self.device)
            nn.init.zeros_(self.router.catalog_head.weight)
            nn.init.zeros_(self.router.catalog_head.bias)

        router_mask = self._router_mask_for(h, mask)
        with torch.no_grad():
            logits = self.router.catalog_logits(h, router_mask, shot_bucket=self._shot_bucket_for(h))
            p = F.softmax(logits.float(), dim=-1)
            idx = p.argmax(dim=-1)
            # Abstain rule:
            # 1. Primary confidence margin check: if top1_prob < theta, abstain to k=0 (idx=0)
            # 2. Legacy/empty-set fallback: if p[:, 0] >= theta
            theta = self._abstain_theta
            if theta is None:
                theta = getattr(self.cfg, "abstain_theta", 1.01)
            if theta <= 1.0:
                top1_prob = p.max(dim=-1).values
                abstain_cond = (top1_prob < theta) | (p[:, 0] >= theta)
                idx = torch.where((idx != 0) & abstain_cond, torch.zeros_like(idx), idx)

        selected_matrix = self._catalog.to_candidate_matrix(idx, self.target_layers).to(h.device).bool()
        self._desired_k = self._catalog.sizes().to(h.device)[idx]
        self._active_k_realized = self._desired_k.clamp(min=1)
        self.router.seen_forwards += h.shape[0]
        self.router.selected_forwards += selected_matrix.long().sum(dim=0).to(self.router.selected_forwards.device)

        self._router_logits = logits.detach()
        self._router_margins = logits.detach()
        self._router_probabilities = p
        self._last_catalog_idx = idx.detach()
        self._route_is_catalog = True
        self._precomputed_routes = selected_matrix
        self._precomputed_route_tuples = [
            tuple(sorted(self.target_layers[j] for j in range(len(self.target_layers)) if bool(selected_matrix[b, j])))
            for b in range(h.shape[0])
        ]
        self.router._last_prediction = self._desired_k
        self._last_inferred_k = self._desired_k

        if h.shape[1] > 1:
            self._cached_decode_routes = selected_matrix.detach()
            self._cached_decode_k = self._desired_k.detach()
            self._cached_decode_probs = self._router_probabilities.detach()
            self._cached_decode_logits = self._router_logits.detach()
            self._cached_decode_margins = self._router_margins.detach()
            self._cached_decode_active_k = self._active_k_realized.detach()

    def _catalog_layer_probs(self) -> torch.Tensor:
        """(B, n_candidates) marginal probability that each candidate layer is in the sampled class."""
        p = self._router_probabilities.float()
        if self._catalog_membership is None or self._catalog_membership.shape[0] != p.shape[1]:
            n = len(self._catalog.subsets)
            self._catalog_membership = self._catalog.to_candidate_matrix(
                torch.arange(n), self.target_layers
            ).float()
        return p @ self._catalog_membership.to(p.device)

    # ---- M6b: steer scale / gate mode ----
    def set_steer_scale(self, scale: Optional[float] = None):
        self._steer_scale = None if scale is None else float(scale)

    def set_manifest(self, manifest):
        self._manifest = manifest

    def set_gate_mode(self, mode: str, scale: Optional[float] = None):
        if mode not in {"learned", "const"}:
            raise ValueError('gate mode must be "learned" or "const"')
        self._gate_mode = mode
        if mode == "const":
            self.set_gate_freeze(True, 1.0)
            self.set_steer_scale(1.0 if scale is None else scale)
        else:
            self.set_gate_freeze(False)
            self.set_steer_scale(None)
        if self._manifest is not None:
            self._manifest.set("gate_mode", mode)

    @property
    def gate_mode(self) -> str:
        return self._gate_mode

    # ---- M6b: guards (record only, never raise) ----
    def set_guards(
        self,
        enabled: bool = True,
        rel_norm_cap: float = 0.02,
        k_max: int = 3,
        finite_check: bool = True,
        revert_to_k0: bool = True,
        **kwargs,
    ):
        self._guards_enabled = bool(enabled)
        self._guard_rel_norm_cap = float(rel_norm_cap)
        self._guard_k_max = int(k_max)
        self._guard_finite_check = bool(finite_check)
        self._guard_revert_to_k0 = bool(revert_to_k0)
        self.guard_state["enabled"] = self._guards_enabled
        self.guard_state["cap"] = self._guard_rel_norm_cap
        self.guard_state["k_max"] = self._guard_k_max
        if "rel_norms" not in self.guard_state:
            self.guard_state["rel_norms"] = {}
        self.guard_state["finite_check"] = self._guard_finite_check
        self.guard_state["revert_to_k0"] = self._guard_revert_to_k0
        if "rel_norms" not in self.guard_state:
            self.guard_state["rel_norms"] = {}

    def _reset_guard(self, h: torch.Tensor):
        B = h.shape[0]
        self._guard_rel_max = torch.zeros(B, device=h.device)
        self._guard_nonfinite = torch.zeros(B, dtype=torch.bool, device=h.device)
        self.guard_state["tripped"] = torch.zeros(B, dtype=torch.bool, device=h.device)
        self.guard_state["enabled"] = self._guards_enabled
        self.guard_state["cap"] = self._guard_rel_norm_cap
        self.guard_state["k_max"] = self._guard_k_max

    def _update_guard(self, delta_injected, h_fp, steer_mask, selected, layer_idx=None):
        """tripped = (rel_max > cap) | nonfinite | (K > k_max), per batch row. Never raises."""
        with torch.no_grad():
            d = delta_injected.detach().float()
            hh = h_fp.detach().float()
            if self._guard_rel_max is None or self._guard_rel_max.shape[0] != d.shape[0]:
                self._reset_guard(h_fp)
            act = selected.detach().bool().reshape(-1)
            valid = steer_mask.bool() & act.unsqueeze(1)
            finite = torch.isfinite(d).all(dim=-1) & torch.isfinite(hh).all(dim=-1)
            nonfinite = ((~finite) & valid).any(dim=1)
            rel = d.norm(dim=-1) / hh.norm(dim=-1).clamp(min=1e-6)
            rel = torch.nan_to_num(rel, nan=0.0).masked_fill(~valid, 0.0)
            self._guard_rel_max = torch.maximum(self._guard_rel_max, rel.max(dim=1).values)
            self._guard_nonfinite = self._guard_nonfinite | nonfinite
            routes = self._precomputed_routes
            if torch.is_tensor(routes):
                k = routes.long().sum(dim=-1).to(d.device)
            else:
                k = torch.tensor([len(r) for r in routes], device=d.device)
            tripped = (
                (self._guard_rel_max > self._guard_rel_norm_cap)
                | self._guard_nonfinite
                | (k > self._guard_k_max)
            )
            self.guard_state["tripped"] = tripped
            self.guard_state["rel_max_tracked"] = self._guard_rel_max.clone()
            self.guard_state["k"] = k
            if layer_idx is not None:
                rn = self.guard_state.setdefault("rel_norms", {})
                valid_rel = rel[valid]
                if valid_rel.numel() > 0:
                    prev = rn.get(int(layer_idx))
                    val_flat = valid_rel.detach().cpu()
                    rn[int(layer_idx)] = torch.cat([prev, val_flat]) if prev is not None else val_flat

    # ---- M6b: route prediction / canonical keys ----
    def _clear_decode_caches(self):
        self._cached_decode_routes = None
        self._cached_decode_k = None
        self._cached_decode_probs = None
        self._cached_decode_logits = None
        self._cached_decode_margins = None
        self._cached_decode_active_k = None
        self._cached_decode_allocation = None

    @torch.no_grad()
    def predict_routes(self, input_ids, attention_mask, question_span_mask=None, shot_bucket=None) -> List[dict]:
        """Single no-grad prefill; returns [{"route", "class_idx", "p_empty"}] per batch row.

        Diagnostic counters are restored and decode caches cleared, so a subsequent
        generate() recomputes routes from scratch.
        """
        saved = {
            "sel": self.router.selected_forwards.clone(),
            "seen": self.router.seen_forwards.clone(),
            "khist": self.routing_k_hist.clone(),
            "kfwd": self.routing_forwards.clone(),
        }
        prev_bucket = self._shot_bucket
        if shot_bucket is not None:
            self.set_shot_bucket(shot_bucket)
        self.set_question_span_mask(question_span_mask)
        try:
            with self.counterfactual_mode():
                self.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            routes = [tuple(int(x) for x in r) for r in self.current_route_tuples()]
            B = len(routes)
            if self._route_is_catalog and self._last_catalog_idx is not None:
                class_idx = [int(i) for i in self._last_catalog_idx.cpu().tolist()]
                p_empty = [float(x) for x in self._router_probabilities[:, 0].float().cpu().tolist()]
            else:
                lookup = {}
                if self._catalog is not None:
                    lookup = {tuple(sorted(s)): i for i, s in enumerate(self._catalog.subsets)}
                class_idx = [lookup.get(r, -1) for r in routes]
                p_empty = [1.0 if len(r) == 0 else 0.0 for r in routes]
            out = [
                {"route": routes[b], "class_idx": class_idx[b], "p_empty": p_empty[b]}
                for b in range(B)
            ]
        finally:
            self.router.selected_forwards.copy_(saved["sel"])
            self.router.seen_forwards.copy_(saved["seen"])
            self.routing_k_hist.copy_(saved["khist"])
            self.routing_forwards.copy_(saved["kfwd"])
            self._shot_bucket = prev_bucket
            self._clear_decode_caches()
        return out

    @staticmethod
    def route_key(row) -> Tuple[Tuple[int, ...], str, Optional[float]]:
        """Canonical (route_tuple, gate_mode, steer_scale_or_None). Empty route -> ((), "none", None).

        `row` is a predict_routes() dict (optionally with "gate_mode"/"steer_scale") or a bare route sequence.
        A learned-gate row with scale None or 1.0 canonicalises to scale None.
        """
        if isinstance(row, dict):
            route = row.get("route", ())
            gate_mode = row.get("gate_mode", "learned")
            scale = row.get("steer_scale", None)
        else:
            route, gate_mode, scale = row, "learned", None
        route = tuple(sorted(int(x) for x in route))
        if len(route) == 0:
            return ((), "none", None)
        if scale is not None:
            scale = float(scale)
            if gate_mode == "learned" and scale == 1.0:
                scale = None
        return (route, str(gate_mode), scale)

    def reset_routing_diagnostics(self):
        self.router.selected_forwards.zero_()
        self.router.seen_forwards.zero_()
        self.routing_k_hist.zero_()
        self.routing_forwards.zero_()
        self._prompt_level_selected_count = [0] * len(self.target_layers)
        self._prompt_level_total = 0
        self._steering_telemetry = {}
        self._route_tuple_counts = {}
        for gate in self.gates.values():
            gate.reset_diagnostics()

    def routing_diagnostics(self):
        diag = {}
        for pos, layer in enumerate(self.target_layers):
            seen = int(self.router.seen_forwards[pos].item())
            selected = int(self.router.selected_forwards[pos].item())
            empirical_rate = (selected / seen) if seen > 0 else float(self.router.selection_rate_ema[pos].item())
            diag[int(layer)] = {
                "selection_rate_ema": float(empirical_rate),
                "selection_rate_batch": float(empirical_rate),
                "logit_ema": float(self.router.logit_ema[pos].item()),
                "selected_forwards": selected,
                "activations_count": selected,
                "seen_forwards": seen,
            }
        return diag

    def routing_k_distribution(self):
        total = max(1, int(self.routing_forwards.item()))
        return {k: int(count.item()) / total for k, count in enumerate(self.routing_k_hist)}

    def route_tuple_entropy(self) -> float:
        """Return Shannon entropy in bits over observed complete route tuples."""
        total = sum(self._route_tuple_counts.values())
        if total <= 0:
            return 0.0
        return float(-sum(
            (count / total) * math.log2(count / total)
            for count in self._route_tuple_counts.values() if count > 0
        ))

    def effective_intervention_energy(self) -> Dict[int, Dict[str, float]]:
        """Summarize realised gate-scaled residual energy by layer.

        Calculation/prose labels are intentionally omitted here because the
        wrapper only receives attention/shield masks; training can enrich this
        telemetry later without retaining labels in model hooks.
        """
        total_energy = sum(float(row.get("energy_sum", 0.0)) for row in self._steering_telemetry.values())
        result: Dict[int, Dict[str, float]] = {}
        for layer in self.target_layers:
            row = self._steering_telemetry.get(int(layer), {})
            count = max(1, int(row.get("count", 0)))
            energy = float(row.get("energy_sum", 0.0))
            result[int(layer)] = {
                "mean_effective_norm": energy / count,
                "total_effective_energy": energy,
                "energy_fraction": energy / total_energy if total_energy > 0 else 0.0,
                "dominant_token_fraction": 0.0,
                "prose_energy": 0.0,
                "calculation_energy": 0.0,
            }
        return result

    def effective_k_diagnostics(self, min_energy_fraction: float = 0.05) -> Dict[str, Any]:
        """Identify selected layers with a material share of total intervention energy."""
        energy = self.effective_intervention_energy()
        threshold = max(0.0, float(min_energy_fraction))
        effective = [
            layer for layer, row in energy.items()
            if row["total_effective_energy"] > 0.0 and row["energy_fraction"] >= threshold
        ]
        return {
            "min_energy_fraction": threshold,
            "effective_layers": effective,
            "effective_layer_count": len(effective),
            "energy": energy,
        }

    def intervention_diagnostics(self):
        """Summarize batch-union K/layer usage and token-gate diversity."""
        k_dist = self.routing_k_distribution()
        all_layer_rates = [row["selection_rate_ema"] for row in self.routing_diagnostics().values()]
        eligible = self.routing_eligibility_mask(self.device).tolist()
        layer_rates = [rate for rate, allowed in zip(all_layer_rates, eligible) if allowed]
        layer_total = sum(layer_rates)
        layer_probs = [rate / layer_total for rate in layer_rates if rate > 0.0]
        k_probs = [p for p in k_dist.values() if p > 0.0]
        def normalized_entropy(probs, support):
            if support <= 1 or not probs:
                return 0.0
            entropy = -sum(p * float(torch.log(torch.tensor(p)).item()) for p in probs)
            return entropy / float(torch.log(torch.tensor(float(support))).item())
        gate_metrics = {int(layer): gate.get_saturation_metrics() for layer, gate in self.gates.items()}
        effective = self.effective_k_diagnostics(
            float(getattr(self.cfg, "effective_k_min_energy_fraction", 0.05))
        )
        routing = self.routing_diagnostics()
        active_union = [
            layer for layer, allowed in zip(self.target_layers, eligible)
            if allowed and routing[int(layer)]["selected_forwards"] > 0
        ]
        ordered = list(self.target_layers)
        def depth_band(layer: int) -> str:
            # Match the Phase-1B physical-depth contract.  A safe pool can
            # omit neighbours, but L15 must remain a middle layer rather than
            # becoming "late" merely because L13 was excluded.
            if ordered and min(ordered) <= 13 and max(ordered) >= 20:
                if layer <= 13:
                    return "early"
                if layer <= 18:
                    return "middle"
                return "late"
            position = ordered.index(layer)
            if position < max(1, len(ordered) // 3):
                return "early"
            if position < max(2, (2 * len(ordered)) // 3):
                return "middle"
            return "late"
        active_bands = sorted({depth_band(layer) for layer in active_union})
        return {
            "realized_k_distribution": k_dist,
            "realized_k_support": sum(p > 0.0 for p in k_dist.values()),
            "realized_k_entropy": round(normalized_entropy(k_probs, len(k_dist)), 4),
            "eligible_routing_layers": [
                layer for layer, allowed in zip(self.target_layers, eligible) if allowed
            ],
            "active_layer_union": active_union,
            "active_depth_bands": active_bands,
            "active_depth_band_count": len(active_bands),
            "layer_selection_entropy": round(normalized_entropy(layer_probs, len(layer_rates)), 4),
            "route_tuple_entropy": round(self.route_tuple_entropy(), 4),
            "route_tuple_counts": {"+".join(map(str, route)) or "K0": count for route, count in self._route_tuple_counts.items()},
            "active_layer_count": len(active_union),
            "effective_layer_count": effective["effective_layer_count"],
            "effective_layers": effective["effective_layers"],
            "effective_energy": effective["energy"],
            "gate": gate_metrics,
            "steering": self.steering_diagnostics(),
        }

    def steering_diagnostics(self) -> Dict[int, Dict[str, float]]:
        """Return observed residual displacement, not just configured bounds."""
        summary = {}
        for layer in self.target_layers:
            row = self._steering_telemetry.get(int(layer), {})
            count = max(1, int(row.get("count", 0)))
            summary[int(layer)] = {
                "intervened_tokens": int(row.get("count", 0)),
                "mean_raw_relative_norm": round(float(row.get("raw_sum", 0.0)) / count, 6),
                "mean_effective_relative_norm": round(float(row.get("effective_sum", 0.0)) / count, 6),
                "max_raw_relative_norm": round(float(row.get("raw_max", 0.0)), 6),
                "max_effective_relative_norm": round(float(row.get("effective_max", 0.0)), 6),
            }
        return summary

    def compute_active_steer_norm_loss(
        self,
        q_hi: float = 0.75,
        huber_delta: float = 0.10,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute Huber penalty on pre-squash norm ratios exceeding q_hi for active SteerNets (Engine v2.1)."""
        losses = []
        q_all = []
        records = getattr(self, "_active_pre_cap_records", [])
        if records:
            for sn, q, valid in records:
                if q is not None and valid is not None and valid.any():
                    q_valid = q.squeeze(-1)[valid]
                    excess = torch.relu(q_valid - float(q_hi))
                    if excess.numel() > 0:
                        loss_sn = F.huber_loss(excess, torch.zeros_like(excess), delta=float(huber_delta), reduction="mean")
                        losses.append(loss_sn)
                        q_all.append(q_valid.detach())
        else:
            for sn in self.steer_nets.values():
                q = getattr(sn, "last_pre_cap_q", None)
                valid = getattr(sn, "_last_valid_mask", None)
                if q is not None and valid is not None and valid.any():
                    q_valid = q.squeeze(-1)[valid]
                    excess = torch.relu(q_valid - float(q_hi))
                    if excess.numel() > 0:
                        loss_sn = F.huber_loss(excess, torch.zeros_like(excess), delta=float(huber_delta), reduction="mean")
                        losses.append(loss_sn)
                        q_all.append(q_valid.detach())

        # Clear records and stale references immediately to prevent graph leakage across micro-batches
        self._active_pre_cap_records = []
        for sn in self.steer_nets.values():
            sn.last_pre_cap_q = None
            sn._last_valid_mask = None

        if not losses:
            return torch.tensor(0.0, device=self.device), {"q_mean": 0.0, "q_p95": 0.0, "q_max": 0.0, "pct_above_1": 0.0}
        total_loss = torch.stack(losses).mean()
        q_cat = torch.cat(q_all)
        stats = {
            "q_mean": float(q_cat.mean().item()),
            "q_p95": float(torch.quantile(q_cat, 0.95).item()) if q_cat.numel() > 1 else float(q_cat.max().item()),
            "q_max": float(q_cat.max().item()),
            "pct_above_1": float((q_cat > 1.0).float().mean().item() * 100.0),
        }
        return total_loss, stats

    def pre_cap_norm_diagnostics(self) -> Dict[str, Any]:
        """Summarize global pre-cap norm ratios across all SteerNets."""
        ratios = {}
        for layer, sn in self.steer_nets.items():
            r = getattr(sn, "_last_pre_cap_ratio", None)
            ratios[int(layer)] = float(r.item()) if isinstance(r, torch.Tensor) else (float(r) if r is not None else 0.0)
        return ratios

    def get_layer_synergy_matrix(self) -> Optional[torch.Tensor]:
        return self.router.get_layer_synergy_matrix()

    def get_cooperation_analysis(self) -> dict:
        """Report cooperation only among the currently legal router pool."""
        eligible = [
            layer for layer, allowed in zip(
                self.target_layers, self.routing_eligibility_mask(self.device).tolist(),
            ) if allowed
        ]
        return self.router.get_cooperation_analysis(eligible_layers=eligible)


    def set_steer_scale(self, s):
        """Set constant steering bypass scale (float). None disables bypass (uses learned gate)."""
        self._steer_scale = s

    @contextmanager
    def hooks_disabled(self):
        """Reference 'unwrapped' forward; keeps pre-hook running for router features."""
        old = self._hooks_disabled
        self._hooks_disabled = True
        try:
            yield
        finally:
            self._hooks_disabled = old

    @contextmanager
    def counterfactual_mode(self):
        """eval(), no_grad context; saves/restores training flag, routing override,
        steer_scale, gate_freeze, router context mask. Used by table builder and RLOO."""
        was_training = self.training
        old_override = self._routing_override
        old_scale = self._steer_scale
        old_freeze = [(g, g._freeze_gate, g._freeze_gate_value) for g in self.gates.values()]
        old_ctx_pending = self._router_context_mask_pending
        self.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            self.train(was_training)
            self._routing_override = old_override
            self._steer_scale = old_scale
            for g, freeze_flag, freeze_val in old_freeze:
                g._freeze_gate = freeze_flag
                g._freeze_gate_value = freeze_val
            self._router_context_mask_pending = old_ctx_pending

    def subset_matrix(self, subsets):
        """Convert layer-id subsets to (B, n_candidates) bool matrix.
        Raises ValueError if any layer id is not a candidate."""
        nc = len(self.target_layers)
        mat = torch.zeros(len(subsets), nc, dtype=torch.bool)
        idx_map = {l: i for i, l in enumerate(self.target_layers)}
        for row, subset in enumerate(subsets):
            for lid in subset:
                if lid not in idx_map:
                    raise ValueError(
                        f"Layer {lid} is not a candidate layer. Valid: {self.target_layers}"
                    )
                mat[row, idx_map[lid]] = True
        return mat

    def current_route_tuples(self):
        """Layer ids per batch row from _precomputed_routes (valid at prefill AND cached decode)."""
        if self._precomputed_routes is None:
            return []
        return [
            tuple(self.target_layers[i] for i, v in enumerate(row) if v)
            for row in self._precomputed_routes.tolist()
        ]

    def _record_envelope(self, layer_pos, step, h_fp, mask, h_new):
        """Record per-item rel-norm and non-finite status into guard_state. Always runs (cheap)."""
        with torch.no_grad():
            norm_h = h_fp.norm(dim=-1).clamp(min=1e-8)          # (B, T)
            rel = (step.norm(dim=-1) / norm_h * mask).max(dim=-1).values  # (B,)
            nonfinite = (~torch.isfinite(h_new)).any(dim=-1).any(dim=-1)   # (B,)
            if "rel_max" not in self.guard_state:
                self.guard_state["rel_max"] = rel
                self.guard_state["nonfinite"] = nonfinite
            else:
                prev_rel = self.guard_state["rel_max"]
                if prev_rel.shape == rel.shape:
                    self.guard_state["rel_max"] = torch.maximum(prev_rel, rel)
                    self.guard_state["nonfinite"] = self.guard_state["nonfinite"] | nonfinite
    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)
