"""Phase 2 Router training: per-question layer selection via RLOO policy gradient.

Architecture:
  - JointLayerRouter: two-pass transformer + cross-layer synergy matrix
  - RSCUsefulnessGate: per-token gate (optional)
  - RLOO: leave-one-out advantage estimator with Cooperative Plackett-Luce sampling
  - Probe hook: captures hidden states at a fixed "read layer" for router input

Usage:
    result = train_phase2_router(model, train_data, eval_data, cfg, device, tokenizer)
    save_router_weights(router, gate, cfg, cfg.router_save_path)
    router, gate = load_router_weights(model, cfg, cfg.router_save_path, device)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.configs.config import RSCConfig
from src.routing.router import JointLayerRouter
from src.routing.rloo import compute_router_rloo_loss, per_example_answer_nll


PHI2_HIDDEN_DIM = 2560
PHI2_N_LAYERS = 32


class ProbeCapture:
    """Registers a forward hook on a backbone layer to capture hidden states."""

    def __init__(self):
        self._h: Optional[torch.Tensor] = None
        self._mask: Optional[torch.Tensor] = None
        self._handle = None

    def register(self, layer: nn.Module) -> "ProbeCapture":
        self._handle = layer.register_forward_hook(self._hook)
        return self

    def _hook(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        self._h = h.detach().float()

    def set_mask(self, mask: Optional[torch.Tensor]) -> None:
        self._mask = mask

    @property
    def h(self) -> Optional[torch.Tensor]:
        return self._h

    @property
    def valid_mask(self) -> Optional[torch.Tensor]:
        return self._mask

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class GateAdaptor:
    """Wraps RSCUsefulnessGate to match wrapper's GateProvider(position, delta) signature.

    Captures hidden states per layer via pre-hooks so the gate can receive (h, delta).
    """

    def __init__(self, gate, router: JointLayerRouter):
        self.gate = gate
        self.router = router
        self._cached_h: Dict[int, torch.Tensor] = {}
        self._handles: List = []

    def register_hooks(self, model) -> None:
        for position, layer_idx in enumerate(model.target_layers):
            def _make_pre_hook(pos):
                def pre_hook(module, input):
                    h = input[0] if isinstance(input, (tuple, list)) else input
                    if isinstance(h, torch.Tensor):
                        self._cached_h[pos] = h.detach().float()
                return pre_hook
            handle = model._layers[layer_idx].register_forward_pre_hook(_make_pre_hook(position))
            self._handles.append(handle)

    def __call__(self, position: int, delta: torch.Tensor) -> torch.Tensor:
        h = self._cached_h.get(position)
        if h is None:
            return torch.ones_like(delta[..., :1])
        global_ctx = getattr(self.router, "_last_semantic_shared", None)
        alpha = self.gate(h.to(delta.device), delta, global_context=global_ctx)
        return alpha

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _router_diversity_check(router: JointLayerRouter, min_k: int) -> Tuple[bool, dict]:
    """Return (health_ok, stats) based on router's EMA tracking buffers."""
    stats = router.route_diversity_stats()
    health_ok = (
        stats["k_ge2_rate"] >= 0.85
        and stats["layer_entropy_bits"] >= 1.5
        and stats["mean_k"] >= float(min_k)
    )
    return health_ok, stats


def train_phase2_router(
    model,
    train_data,
    eval_data,
    cfg: RSCConfig,
    device: torch.device,
    tokenizer=None,
) -> Dict[str, Any]:
    """Train JointLayerRouter + optional RSCUsefulnessGate on top of frozen SteerNets.

    Returns a results dict with training history and final router state.
    """
    # ── Freeze Phase 1 SteerNets ──────────────────────────────────────────────
    for param in model.steer_nets.parameters():
        param.requires_grad = False

    # ── Build router ──────────────────────────────────────────────────────────
    router = JointLayerRouter(
        target_layers=cfg.fixed_layers,
        hidden_dim=PHI2_HIDDEN_DIM,
        n_layers=PHI2_N_LAYERS,
        max_active_layers=cfg.router_max_k,
        semantic_dim=cfg.router_semantic_dim,
        d_model=cfg.router_d_model,
        n_heads=cfg.router_n_heads,
    ).to(device)

    # ── Build gate (optional) ─────────────────────────────────────────────────
    gate = None
    gate_adaptor = None
    if cfg.use_gate:
        from src.gating.gate import RSCUsefulnessGate
        gate = RSCUsefulnessGate(PHI2_HIDDEN_DIM, cfg).to(device)
        gate_adaptor = GateAdaptor(gate, router)
        gate_adaptor.register_hooks(model)
        model.set_gate_provider(gate_adaptor)

    # ── Probe hook: captures hidden states from first candidate layer ─────────
    probe_layer_idx = cfg.fixed_layers[0]
    probe = ProbeCapture().register(model._layers[probe_layer_idx])

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    trainable_params = list(router.parameters())
    if gate is not None:
        trainable_params += list(gate.parameters())

    optimizer = torch.optim.Adam(trainable_params, lr=cfg.phase2_lr, betas=(0.9, 0.999))
    total_steps = max(1, len(train_data) * cfg.phase2_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=cfg.phase2_lr * 0.1
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    rows = train_data if isinstance(train_data, list) else [train_data[i] for i in range(len(train_data))]
    history: List[Dict[str, float]] = []
    best_val_nll = float("inf")
    best_router_state = None
    best_gate_state = None

    for epoch in range(cfg.phase2_epochs):
        router.train()
        if gate is not None:
            gate.train()

        warmup_k = cfg.router_min_k if epoch < cfg.router_warmup_epochs else None
        epoch_losses: List[float] = []
        epoch_diags: List[Dict[str, float]] = []

        perm = torch.randperm(len(rows)).tolist()
        progress = tqdm(perm, desc=f"router epoch {epoch+1}/{cfg.phase2_epochs}", leave=False)

        for idx in progress:
            row = rows[idx]
            ids = row["input_ids"].unsqueeze(0).to(device)
            mask = row["attention_mask"].unsqueeze(0).to(device)
            labels = row["labels"].unsqueeze(0).to(device) if "labels" in row else None
            if labels is None:
                continue

            # ── Probe forward: capture hidden states (routing disabled) ────
            model.set_routing_override("none")
            probe.set_mask(mask.bool())
            with torch.no_grad():
                model(input_ids=ids, attention_mask=mask)

            if probe.h is None:
                continue

            cached_h = probe.h  # (1, T, D) float32
            cached_valid = mask.bool()

            # ── Compute RLOO loss ──────────────────────────────────────────
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss, diag = compute_router_rloo_loss(
                    router=router,
                    model=model,
                    ids=ids,
                    mask=mask,
                    labels=labels,
                    cfg=cfg,
                    cached_h=cached_h,
                    cached_valid_mask=cached_valid,
                    warmup_k=warmup_k,
                )

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_losses.append(float(loss.detach()))
            epoch_diags.append(diag)
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

        # ── Epoch diagnostics ────────────────────────────────────────────────
        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        mean_k = sum(d["mean_k"] for d in epoch_diags) / max(len(epoch_diags), 1)
        diversity_ok, div_stats = _router_diversity_check(router, cfg.router_min_k)

        print(
            f"[router epoch {epoch+1}] loss={mean_loss:.4f}  mean_k={mean_k:.2f}  "
            f"k≥{cfg.router_min_k}_rate={div_stats['k_ge2_rate']:.2%}  "
            f"layer_entropy={div_stats['layer_entropy_bits']:.2f} bits  "
            f"health={'OK' if diversity_ok else 'COLLAPSED'}"
        )

        if not diversity_ok:
            print(
                f"  [warning] Router diversity collapsed! "
                f"Doubling beta_entropy_k: {cfg.router_beta_entropy_k} → {cfg.router_beta_entropy_k * 2}"
            )
            cfg.router_beta_entropy_k *= 2

        # ── Validation NLL under router routing ──────────────────────────────
        router.eval()
        if gate is not None:
            gate.eval()
        val_nll = _eval_router_nll(model, router, eval_data, cfg, device, probe)
        print(f"  val_nll={val_nll:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": mean_loss,
            "mean_k": mean_k,
            "diversity_ok": diversity_ok,
            "val_nll": val_nll,
            **div_stats,
        })

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_router_state = {k: v.cpu().clone() for k, v in router.state_dict().items()}
            best_gate_state = (
                {k: v.cpu().clone() for k, v in gate.state_dict().items()} if gate else None
            )

    # ── Restore best checkpoint ───────────────────────────────────────────────
    if best_router_state is not None:
        router.load_state_dict(best_router_state)
    if gate is not None and best_gate_state is not None:
        gate.load_state_dict(best_gate_state)

    # ── Install selector for inference ────────────────────────────────────────
    router.eval()
    model.set_routing_override(None)
    model.set_route_selector(router.make_selector(
        min_k=cfg.router_min_k, max_k=cfg.router_max_k
    ))

    # ── Cleanup probe ─────────────────────────────────────────────────────────
    probe.remove()

    return {
        "history": history,
        "best_val_nll": best_val_nll,
        "router": router,
        "gate": gate,
    }


@torch.no_grad()
def _eval_router_nll(
    model, router: JointLayerRouter, eval_data, cfg: RSCConfig,
    device: torch.device, probe: ProbeCapture,
) -> float:
    """Evaluate mean NLL using greedy router decisions on the validation set."""
    rows = eval_data if isinstance(eval_data, list) else [eval_data[i] for i in range(len(eval_data))]
    selector = router.make_selector(min_k=cfg.router_min_k, max_k=cfg.router_max_k)
    model.set_routing_override(None)
    model.set_route_selector(selector)

    nlls: List[float] = []
    for row in rows[:min(len(rows), 200)]:
        ids = row["input_ids"].unsqueeze(0).to(device)
        mask = row["attention_mask"].unsqueeze(0).to(device)
        labels = row["labels"].unsqueeze(0).to(device) if "labels" in row else None
        if labels is None:
            continue
        # Probe forward to populate router cache
        model.set_routing_override("none")
        probe.set_mask(mask.bool())
        model(input_ids=ids, attention_mask=mask)
        if probe.h is None:
            continue
        # Update router cache so selector uses fresh hidden states
        router(probe.h, mask.bool())
        # Evaluate with router selector
        model.set_routing_override(None)
        model.set_route_selector(selector)
        logits = model(input_ids=ids, attention_mask=mask).logits
        nll = per_example_answer_nll(logits, labels)
        nlls.append(float(nll.mean().detach()))

    # Restore fixed route for subsequent Phase 1 evaluations
    model.set_routing_override("only", cfg.fixed_layers)
    model.set_route_selector(None)
    return sum(nlls) / max(len(nlls), 1)


def save_router_weights(
    router: JointLayerRouter,
    gate,
    cfg: RSCConfig,
    path: str,
) -> None:
    """Save router and gate state dicts with config metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "router": router.state_dict(),
            "gate": gate.state_dict() if gate is not None else None,
            "fixed_layers": list(cfg.fixed_layers),
            "router_min_k": cfg.router_min_k,
            "router_max_k": cfg.router_max_k,
            "router_semantic_dim": cfg.router_semantic_dim,
            "router_d_model": cfg.router_d_model,
            "router_n_heads": cfg.router_n_heads,
            "use_gate": cfg.use_gate,
        },
        path,
    )
    print(f"[router] Saved checkpoint to {path}")


def load_router_weights(
    model,
    cfg: RSCConfig,
    path: str,
    device: torch.device,
) -> Tuple[JointLayerRouter, Optional[Any]]:
    """Load router (and optional gate) checkpoint, install on model for inference."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if ckpt["fixed_layers"] != list(cfg.fixed_layers):
        raise ValueError(
            f"Router checkpoint has fixed_layers={ckpt['fixed_layers']} "
            f"but config expects {list(cfg.fixed_layers)}"
        )

    router = JointLayerRouter(
        target_layers=cfg.fixed_layers,
        hidden_dim=PHI2_HIDDEN_DIM,
        n_layers=PHI2_N_LAYERS,
        max_active_layers=ckpt.get("router_max_k", cfg.router_max_k),
        semantic_dim=ckpt.get("router_semantic_dim", cfg.router_semantic_dim),
        d_model=ckpt.get("router_d_model", cfg.router_d_model),
        n_heads=ckpt.get("router_n_heads", cfg.router_n_heads),
    ).to(device)
    router.load_state_dict(ckpt["router"])
    router.eval()

    gate = None
    gate_adaptor = None
    if ckpt.get("use_gate") and ckpt["gate"] is not None:
        from src.gating.gate import RSCUsefulnessGate
        gate = RSCUsefulnessGate(PHI2_HIDDEN_DIM, cfg).to(device)
        gate.load_state_dict(ckpt["gate"])
        gate.eval()
        gate_adaptor = GateAdaptor(gate, router)
        gate_adaptor.register_hooks(model)
        model.set_gate_provider(gate_adaptor)

    # Probe hook: captures hidden states at probe layer during no-steer forward
    probe_layer_idx = cfg.fixed_layers[0]
    probe = ProbeCapture().register(model._layers[probe_layer_idx])

    min_k = ckpt.get("router_min_k", cfg.router_min_k)
    max_k = ckpt.get("router_max_k", cfg.router_max_k)
    N_layers = len(cfg.fixed_layers)

    @torch.no_grad()
    def router_prefill(ids: torch.Tensor, mask: torch.Tensor) -> None:
        """Run a probe forward to capture hidden states, compute route, fix via matrix override.

        Must be called BEFORE model.generate() for each question so the route is
        computed from the current prompt's hidden states and stays fixed throughout
        the entire generation (prefill + all decode steps).
        """
        # 1. Probe forward (no steering) → populates probe.h
        model.set_routing_override("none")
        probe.set_mask(mask.bool())
        model(input_ids=ids, attention_mask=mask)

        if probe.h is None:
            # Fallback: activate all layers (same as fixed)
            model.set_routing_override("only", cfg.fixed_layers)
            return

        # 2. Compute route from router
        h = probe.h.to(ids.device)
        valid = mask.bool()
        router(h, valid)
        sel = router.make_selector(min_k=min_k, max_k=max_k)
        route_matrix = sel(ids.shape[0], ids.device)  # (B, N_layers) bool

        # 3. Fix route via matrix override — stays constant for entire generate()
        model.set_routing_override("matrix", route_matrix)

    # Clear any stale override/selector — route will be set per-question via router_prefill
    model.set_route_selector(None)
    model.set_routing_override("only", cfg.fixed_layers)  # safe default until first prefill

    print(f"[router] Loaded checkpoint from {path}")
    return router, gate, router_prefill
