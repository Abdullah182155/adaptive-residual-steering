import weakref
from typing import List, Optional, Tuple, Dict, Any, Set
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.configs.config import RSCConfig
from src.steering.steernet import RSCSteerNet
from src.gating.gate import RSCUsefulnessGate, masked_route_allocation
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
            self.target_layers,
            self.hidden_dim,
            n_layers,
            cfg.max_active_layers,
            layer_dropout=getattr(cfg, "router_layer_dropout", 0.0),  # BUG-C2 fix: fallback 0.15→0.0 (matches config.py; avoids train/eval mismatch)
            difficulty_aware=getattr(cfg, "router_difficulty_aware", True),
            synergy_limit=getattr(cfg, "router_synergy_limit", 0.75),
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
        # Per-generation evidence used by the qualitative evaluator.  Values
        # are detached scalar alphas only; tracing never retains a model graph.
        self._collect_gate_trace = False
        self.last_gate_trace = None
        self._trace_cached_decode = False
        self._steering_telemetry = {}
        self._active_pre_cap_records = []

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
                        if wrapper._routing_override is None:
                            wrapper.router.selected_forwards[layer_pos] += selected.long().sum()
                        delta = s(h_fp, active_k=wrapper._active_k_realized)
                        alpha = g(h_fp, delta, global_context=wrapper.router._last_semantic_shared)
                        # A route has one shared intervention budget.  Gates
                        # retain token-level strength, while selected layers
                        # receive a masked-softmax allocation instead of each
                        # injecting a full residual on the same tokens.
                        allocation = wrapper._route_allocation[:, layer_pos].to(alpha.dtype).view(-1, 1, 1)
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
                        if valid.any() and getattr(s, "last_pre_cap_q", None) is not None:
                            wrapper._active_pre_cap_records.append((s, s.last_pre_cap_q, valid))
                        if wrapper._routing_override is None:
                            if valid.any():
                                raw_ratio = (delta.norm(dim=-1) / h_fp.norm(dim=-1).clamp_min(1e-6))[valid]
                                effective = delta * alpha * route_budget
                                effective_ratio = (effective.norm(dim=-1) / h_fp.norm(dim=-1).clamp_min(1e-6))[valid]
                                metrics = wrapper._steering_telemetry.setdefault(
                                    int(layer_idx), {"count": 0, "raw_sum": 0.0, "effective_sum": 0.0, "raw_max": 0.0, "effective_max": 0.0}
                                )
                                metrics["count"] += int(valid.sum().item())
                                metrics["raw_sum"] += float(raw_ratio.sum().item())
                                metrics["effective_sum"] += float(effective_ratio.sum().item())
                                metrics["raw_max"] = max(metrics["raw_max"], float(raw_ratio.max().item()))
                                metrics["effective_max"] = max(metrics["effective_max"], float(effective_ratio.max().item()))
                        if wrapper._routing_override is None:
                            g.record_diagnostics(alpha, steer_mask)
                        if wrapper._joint_grad_mode:
                            h_new = h_fp + route_st * alpha * delta * route_budget * steer_mask.to(alpha.dtype).unsqueeze(-1)
                        else:
                            h_new = h_fp + route_st * alpha.detach() * delta * route_budget * steer_mask.to(alpha.dtype).unsqueeze(-1)
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

            if is_cached_decode and self._cached_decode_routes is not None:
                self._precomputed_routes = self._cached_decode_routes
                self._desired_k = self._cached_decode_k
                self._router_probabilities = self._cached_decode_probs
                self._router_logits = self._cached_decode_logits
                self._router_margins = self._cached_decode_margins
                self._active_k_realized = self._cached_decode_active_k
                self._route_allocation = self._cached_decode_allocation
                self._prompt_shield_mask_active = None
            else:
                self._desired_k = None
                self._precomputed_routes = None
                self._router_probabilities = None
                self._router_logits = None
                self._router_margins = None
                self._active_k_realized = None
                self._route_allocation = None
                if not is_cached_decode:
                    self._cached_decode_routes = None
                    self._cached_decode_k = None
                    self._cached_decode_probs = None
                    self._cached_decode_logits = None
                    self._cached_decode_margins = None
                    self._cached_decode_active_k = None
                    self._cached_decode_allocation = None
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

    def set_gate_freeze(self, freeze: bool, value: float = 1.0):
        for gate in self.gates.values():
            gate._freeze_gate = freeze
            gate._freeze_gate_value = value

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
        if mask is None and self.prompt_shielding and self._prompt_shield_mask_active is not None:
            sh = self._prompt_shield_mask_active
            if sh.ndim == 2 and sh.shape[0] == h.shape[0] and sh.shape[1] >= h.shape[1]:
                sh_aligned = sh[:, -h.shape[1]:].to(device=h.device, dtype=torch.bool)
                unshielded = fallback_mask & (~sh_aligned)
                if unshielded.any():
                    mask = unshielded
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
        return selected_matrix

    def _prepare_routes(self, h: torch.Tensor, mask: torch.Tensor):
        if self._precomputed_routes is not None:
            # Diagnostics describe the learned router only.  Baseline and
            # fixed-route controls must not dilute their denominator.
            if self._routing_override is None:
                self.router.seen_forwards += h.shape[0]
            return
        if self._selected_counts is None or self._selected_counts.shape[0] != h.shape[0]:
            raise RuntimeError("RSC router state was not initialised by the model forward pre-hook")

        router_mask = self._router_mask_for(h, mask)
        u, k_logits = self.router(h, router_mask)
        self._router_logits = u
        self._router_margins = u
        self._router_probabilities = F.softmax(u, dim=-1)
        if self._routing_override is None:
            self.router.seen_forwards += h.shape[0]

        n_candidates = len(self.target_layers)
        K_max = self.cfg.max_active_layers

        # An explicit route is an audit/benchmark contract.  It must win over
        # background warmup coverage so a requested baseline remains truly
        # unsteered and a requested single layer is actually the one measured.
        if self._router_warmup and self._routing_override is None:
            warmup_max_k = int(getattr(self.cfg, "router_warmup_max_k", 4))
            if getattr(self.cfg, "router_warmup_random_subsets", True):
                import random as _random
                k_max = min(K_max, warmup_max_k, n_candidates)
                width = _random.randint(1, max(1, k_max))
                forced_indices = set(_random.sample(self.target_layers, width))
            else:
                width = 1 + (self._warmup_cursor % max(1, min(K_max, warmup_max_k)))
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
            synergy = getattr(self.router, "_last_synergy_matrix", None)
            if self.training and self._router_exploration:
                with torch.no_grad():
                    k_real, _ = sample_k(k_logits)
                    selected_matrix, _ = sample_subset_plackett_luce(u, k_real, synergy_matrix=synergy)
                self._desired_k = k_real
            else:
                if self._router_inference_mode == "sample":
                    with torch.no_grad():
                        k_real, _ = sample_k(k_logits, k_max=K_max)
                        selected_matrix, _ = sample_subset_plackett_luce(u, k_real, synergy_matrix=synergy)
                    k_real = k_real.clamp(min=0, max=K_max)
                else:
                    # Deterministic evaluation: select the learned K (including
                    # K=0) and greedily construct the cooperative subset.
                    with torch.no_grad():
                        k_idx = k_logits.float().argmax(dim=-1)
                        if k_logits.shape[-1] > K_max:
                            k_real = k_idx.clamp(min=0, max=K_max)
                        else:
                            k_real = (k_idx + 1).clamp(min=1, max=K_max)
                    selected_matrix = torch.zeros(h.shape[0], n_candidates, dtype=torch.bool, device=h.device)
                    for b in range(h.shape[0]):
                        kb = int(k_real[b].item())
                        cur_u = u[b].clone()
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
            self._desired_k = selected_matrix.long().sum(dim=-1).clamp(min=0, max=K_max)

        self.router._last_prediction = self._desired_k
        self._last_inferred_k = self._desired_k
        self._precomputed_routes = selected_matrix
        self._active_k_realized = selected_matrix.sum(dim=-1)
        if self._routing_override is not None:
            # Forced Phase-1B/audit routes are an intervention contract, not
            # a chance for the immature router logits to attenuate one member
            # of a deliberately balanced pair.
            self._route_allocation = selected_matrix.to(dtype=self._router_logits.dtype) / self._active_k_realized.clamp_min(1).to(dtype=self._router_logits.dtype).unsqueeze(-1)
        elif bool(getattr(self.cfg, "factorized_gate_allocation", True)):
            self._route_allocation = masked_route_allocation(
                self._router_logits, selected_matrix,
                temperature=float(getattr(self.cfg, "gate_allocation_temperature", 0.70)),
            )
        else:
            self._route_allocation = selected_matrix.to(dtype=self._router_logits.dtype)

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
                "realized_k": self._active_k_realized.detach().cpu().tolist(),
                "route_allocation": self._route_allocation[:, layer_pos].detach().float().cpu().tolist(),
                "predicted_k": self.router._last_prediction.detach().cpu().tolist(),
            }
        self._selected_counts = self._selected_counts + selected.long()
        if layer_idx == self.target_layers[-1] and not self._router_warmup and self._routing_override is None:
            counts = self._selected_counts.clamp(max=self.cfg.max_active_layers)
            self.routing_k_hist.add_(torch.bincount(counts, minlength=self.cfg.max_active_layers + 1))
            self.routing_forwards.add_(counts.numel())
        return selected, probability

    def reset_routing_diagnostics(self):
        self.router.selected_forwards.zero_()
        self.router.seen_forwards.zero_()
        self.routing_k_hist.zero_()
        self.routing_forwards.zero_()
        self._steering_telemetry = {}
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

    def intervention_diagnostics(self):
        """Summarize realized K, layer usage, and token-gate diversity."""
        k_dist = self.routing_k_distribution()
        layer_rates = [row["selection_rate_ema"] for row in self.routing_diagnostics().values()]
        layer_total = sum(layer_rates)
        layer_probs = [rate / layer_total for rate in layer_rates if rate > 0.0]
        k_probs = [p for p in k_dist.values() if p > 0.0]
        def normalized_entropy(probs, support):
            if support <= 1 or not probs:
                return 0.0
            entropy = -sum(p * float(torch.log(torch.tensor(p)).item()) for p in probs)
            return entropy / float(torch.log(torch.tensor(float(support))).item())
        gate_metrics = {int(layer): gate.get_saturation_metrics() for layer, gate in self.gates.items()}
        return {
            "realized_k_distribution": k_dist,
            "realized_k_support": sum(p > 0.0 for p in k_dist.values()),
            "realized_k_entropy": round(normalized_entropy(k_probs, len(k_dist)), 4),
            "layer_selection_entropy": round(normalized_entropy(layer_probs, len(self.target_layers)), 4),
            "active_layer_count": sum(rate > 0.02 for rate in layer_rates),
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
        return self.router.get_cooperation_analysis()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)
