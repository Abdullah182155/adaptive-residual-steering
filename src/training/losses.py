import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
import re
import numpy as np
from src.data.dataset import PromptTemplateBank
from src.evaluation.verifier import extract_final_answer

class LRReducer:
    """Cosine LR schedule with linear warmup and plateau-triggered peak reduction.

    ISSUE-10 fix — two bugs in the original:
    1. Warmup was a no-op: `step = max(0, current_step - warmup_steps)` made
       `step == 0` throughout warmup, so `cosine_factor == 1.0` from step 1 (full LR).
    2. Plateau set `current_step = warmup_steps`, restarting the cosine from its peak —
       potentially *raising* LR after a plateau (e.g. 2.75e-4 → jump to 2.5e-4 peak).

    Fixed behaviour:
    - Linear ramp from 0 → base_lr over warmup_steps.
    - Monotone cosine decay from base_lr → min_lr over the remaining steps.
    - Plateau only shrinks the peak (base_lrs) and keeps the schedule position.
    """

    def __init__(self, optimizer, total_steps, factor=0.5, patience=3,
                 min_lr=1e-6, delta=1e-4, warmup_steps=0):
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.current_step = 0
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.delta = delta
        self.best_loss = float("inf")
        self.wait = 0
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.original_base_lrs = list(self.base_lrs)
        self.n_reductions = 0

    def _schedule_factor(self) -> float:
        if self.warmup_steps > 0 and self.current_step < self.warmup_steps:
            # Linear warmup: 1/warmup_steps → 1.0
            return (self.current_step + 1) / float(self.warmup_steps)
        remaining = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, (self.current_step - self.warmup_steps) / remaining)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self, val_loss):
        """Plateau: shrink the peak LR but keep the schedule position (never raises LR)."""
        if val_loss < self.best_loss - self.delta:
            self.best_loss, self.wait = val_loss, 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            self.n_reductions += 1
            # Reduce the peak; do NOT reset current_step (would restart cosine from peak)
            self.base_lrs = [max(lr * self.factor, self.min_lr) for lr in self.base_lrs]
            self.wait = 0
            print(
                f"  ⚠️ LR peak reduced (×{self.factor}^{self.n_reductions}): "
                f"{[f'{lr:.2e}' for lr in self.base_lrs]}"
            )

    def scheduler_step(self):
        f = self._schedule_factor()
        for i, pg in enumerate(self.optimizer.param_groups):
            pg["lr"] = max(self.min_lr, self.base_lrs[i] * f)
        self.current_step += 1

    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state required for an exact staged-phase resume."""
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "current_step": self.current_step,
            "factor": self.factor,
            "patience": self.patience,
            "min_lr": self.min_lr,
            "delta": self.delta,
            "best_loss": self.best_loss,
            "wait": self.wait,
            "base_lrs": list(self.base_lrs),
            "original_base_lrs": list(self.original_base_lrs),
            "n_reductions": self.n_reductions,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore a validated scheduler snapshot without resetting its progress."""
        required = {"current_step", "best_loss", "wait", "base_lrs", "n_reductions"}
        missing = sorted(required - set(state))
        if missing:
            raise ValueError(f"LRReducer resume state is missing fields: {missing}")
        for name in (
            "total_steps", "warmup_steps", "current_step", "factor", "patience",
            "min_lr", "delta", "best_loss", "wait", "n_reductions",
        ):
            if name in state:
                setattr(self, name, state[name])
        self.base_lrs = list(state["base_lrs"])
        self.original_base_lrs = list(state.get("original_base_lrs", self.base_lrs))
        if len(self.base_lrs) != len(self.optimizer.param_groups):
            raise ValueError("LRReducer resume state does not match optimizer parameter groups.")
        for index, group in enumerate(self.optimizer.param_groups):
            group["lr"] = max(self.min_lr, self.base_lrs[index] * self._schedule_factor())

class CheckpointManager:
    def __init__(self, checkpoint_dir, keep_last=3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last = keep_last
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_loss = float("inf")
        self.best_path = None
        self.recent = []

    def save(self, model, optimizer, epoch, val_loss, phase="p1"):
        state = {
            "steer_state": {
                k: v.cpu() for k, v in model.state_dict().items() if "steer_net" in k or "gate" in k or "router" in k
            },
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        }
        path = os.path.join(self.checkpoint_dir, f"{phase}_epoch{epoch}.pt")
        torch.save(state, path)
        self.recent.append(path)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_path = os.path.join(self.checkpoint_dir, f"{phase}_best.pt")
            torch.save(state, self.best_path)

        while len(self.recent) > self.keep_last:
            old = self.recent.pop(0)
            if old != self.best_path and os.path.exists(old):
                os.remove(old)

class GradientMonitor:
    def __init__(self, warn_threshold=2.0):
        self.warn_threshold = warn_threshold
        self.norms = []
        self.explosions = 0
        self.vanishes = 0

    def check(self, params, step):
        total_norm = 0.0
        has_nan = False
        for p in params:
            if p.grad is not None:
                g = p.grad.data
                if not torch.isfinite(g).all():
                    has_nan = True
                    continue
                total_norm += g.norm(2).item() ** 2
        total_norm = total_norm**0.5
        if has_nan:
            return
        self.norms.append(total_norm)
        if total_norm > self.warn_threshold:
            self.explosions += 1
            if self.explosions <= 3:
                print(f"    ⚠️ Grad explosion at step {step}: {total_norm:.4f}")
        elif total_norm < 1e-6:
            self.vanishes += 1

    def summary(self):
        if not self.norms:
            return
        print(f"  Gradient summary: mean={np.mean(self.norms):.4f}, explosions={self.explosions}")

def compute_l1_gate(model) -> torch.Tensor:
    """L1 sparsity on gate outputs."""
    device = next(iter(p for g in model.gates.values() for p in g.parameters()), torch.tensor(0.0)).device
    total = torch.tensor(0.0, device=device)
    count = 0
    for gate in model.gates.values():
        alpha = gate._alpha_for_aux
        if alpha is None:
            continue
        a = alpha.float().squeeze(-1)
        vm = gate._valid_mask
        if vm is not None and vm.shape == a.shape:
            denom = vm.sum().clamp(min=1)
            total = total + (a * vm.float()).sum() / denom
        else:
            total = total + a.mean()
        count += 1
    return total / max(1, count)

def compute_gate_budget_loss(model, target_sparsity: float = 0.20) -> torch.Tensor:
    """Penalize total route-level intervention mass, not each selected layer.

    Factorized routing makes ``_alpha_for_aux`` the effective per-layer gate
    mass.  Budgeting each layer independently accidentally allows a K=2 route
    to spend roughly twice the configured intervention budget.  Summing active
    gates tokenwise gives K=1 and K>1 routes one comparable budget while still
    letting the allocation mechanism time-share it across layers.
    """
    device = next(iter(p for g in model.gates.values() for p in g.parameters()), torch.tensor(0.0)).device
    rows = []
    masks = []
    for gate in model.gates.values():
        alpha = gate._alpha_for_aux
        if alpha is None:
            continue
        a = alpha.float().squeeze(-1)
        vm = gate._valid_mask
        if vm is not None and vm.shape == a.shape:
            valid = vm.bool()
        else:
            valid = torch.ones_like(a, dtype=torch.bool)
        if rows and a.shape != rows[0].shape:
            raise ValueError("active gate tensors must share batch/token dimensions")
        rows.append(a * valid.float())
        masks.append(valid)
    if not rows:
        return torch.zeros((), device=device)
    route_mass = torch.stack(rows, dim=0).sum(dim=0)
    route_valid = torch.stack(masks, dim=0).any(dim=0)
    mean_mass = route_mass[route_valid].mean() if route_valid.any() else torch.zeros((), device=device)
    return (mean_mass - float(target_sparsity)).pow(2)

def compute_gate_diversity_loss(model, floor: float = 0.05) -> torch.Tensor:
    """Penalize low per-sequence gate variance, excluding single max token."""
    device = next(iter(p for g in model.gates.values() for p in g.parameters()), torch.tensor(0.0)).device
    total = torch.tensor(0.0, device=device)
    count = 0
    for gate in model.gates.values():
        alpha = gate._alpha_for_aux
        if alpha is None:
            continue
        a = alpha.float().squeeze(-1)
        vm = gate._valid_mask
        for b in range(a.size(0)):
            a_b = a[b]
            valid_b = vm[b] if (vm is not None and vm.shape == a.shape) else torch.ones_like(a_b, dtype=torch.bool)
            if valid_b.sum() < 3:
                continue
            vals = a_b[valid_b]
            std_b = vals.std(unbiased=False)
            span_b = vals.max() - vals.min()
            # Penalize both low variance and collapsed dynamic range (< 0.15 span)
            total = total + F.relu(floor - std_b) ** 2 + 0.5 * F.relu(0.15 - span_b) ** 2
            count += 1
    return total / max(1, count)


def compute_gate_overlap_loss(model) -> torch.Tensor:
    """Penalize simultaneous effective gate mass on the same routed token.

    Only active gates expose ``_alpha_for_aux``.  This keeps the objective
    neutral for K=0/K=1 while making K>=2 routes learn complementary token
    roles instead of radial co-activation.
    """
    device = next(iter(p for g in model.gates.values() for p in g.parameters()), torch.tensor(0.0)).device
    rows = []
    for gate in model.gates.values():
        alpha = getattr(gate, "_alpha_for_aux", None)
        valid = getattr(gate, "_valid_mask", None)
        if alpha is None or valid is None or valid.shape != alpha.shape[:2]:
            continue
        rows.append(alpha.float().squeeze(-1) * valid.float())
    if len(rows) < 2:
        return torch.zeros((), device=device)
    stacked = torch.stack(rows, dim=0)
    # sum_{i<j} a_i*a_j, normalized by valid token/layer pairs.
    pair_mass = (stacked.sum(dim=0).square() - stacked.square().sum(dim=0)) * 0.5
    valid_pairs = (stacked.gt(0).sum(dim=0) >= 2)
    if not valid_pairs.any():
        return torch.zeros((), device=device)
    return pair_mass[valid_pairs].mean()


def compute_route_activity_floor_loss(model, min_fraction: float = 0.05) -> torch.Tensor:
    """Penalise selected layers that receive a negligible share of gate mass.

    This is a differentiable micro-batch surrogate for effective-K health: it
    does not force equal gates, but prevents a nominal K>=2 route from hiding
    a fully starved member behind a selected-layer telemetry count.
    """
    device = next(iter(p for g in model.gates.values() for p in g.parameters()), torch.tensor(0.0)).device
    masses = []
    for gate in model.gates.values():
        alpha = getattr(gate, "_alpha_for_aux", None)
        valid = getattr(gate, "_valid_mask", None)
        if alpha is None or valid is None or valid.shape != alpha.shape[:2]:
            continue
        masses.append((alpha.float().squeeze(-1) * valid.float()).sum())
    if len(masses) < 2:
        return torch.zeros((), device=device)
    mass = torch.stack(masses)
    total = mass.sum().clamp_min(1e-8)
    fractions = mass / total
    return F.relu(float(min_fraction) - fractions).square().mean()

def compute_antisat_gate(model, threshold: float = 0.92) -> torch.Tensor:
    """Penalise gate values above threshold to prevent saturation."""
    device = next(iter(p for g in model.gates.values() for p in g.parameters()), torch.tensor(0.0)).device
    total = torch.tensor(0.0, device=device)
    count = 0
    layer_keys = sorted(model.gates.keys(), key=lambda k: int(k))
    n_layers = max(1, len(layer_keys))
    for rank_idx, key in enumerate(layer_keys):
        gate = model.gates[key]
        alpha = gate._alpha_for_aux
        if alpha is None:
            continue
        a = alpha.float().squeeze(-1)
        layer_threshold = threshold - 0.30 * (1.0 - rank_idx / (n_layers - 1 + 1e-8))
        layer_threshold = max(0.50, layer_threshold)
        over = F.relu(a - layer_threshold) ** 2
        vm = gate._valid_mask
        if vm is not None and vm.shape == over.shape:
            denom = vm.sum().clamp(min=1)
            total = total + (over * vm.float()).sum() / denom
        else:
            total = total + over.mean()
        count += 1
    return total / max(1, count)
 
def compute_steer_norm_loss(
    model,
    q_hi: float = 0.75,
    huber_delta: float = 0.10,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Cap-aware Huber penalty on pre-squash SteerNet norm ratios exceeding q_hi (Engine v2.1)."""
    if hasattr(model, "compute_active_steer_norm_loss"):
        return model.compute_active_steer_norm_loss(q_hi=q_hi, huber_delta=huber_delta)
    steer_nets = getattr(model, "steer_nets", {})
    nets = list(steer_nets.values()) if hasattr(steer_nets, "values") else []
    if not nets:
        device = torch.device("cpu")
        return torch.tensor(0.0, device=device), {"q_mean": 0.0, "q_p95": 0.0, "q_max": 0.0, "pct_above_1": 0.0}
    device = next(iter(p for sn in nets for p in sn.parameters()), torch.tensor(0.0)).device
    losses = []
    q_all = []
    for sn in nets:
        q = getattr(sn, "last_pre_cap_q", None)
        valid = getattr(sn, "_last_valid_mask", None)
        if q is not None and valid is not None and valid.any():
            q_valid = q.squeeze(-1)[valid]
            excess = F.relu(q_valid - float(q_hi))
            if excess.numel() > 0:
                loss_sn = F.huber_loss(excess, torch.zeros_like(excess), delta=float(huber_delta), reduction="mean")
                losses.append(loss_sn)
                q_all.append(q_valid.detach())
    for sn in nets:
        sn.last_pre_cap_q = None
        sn._last_valid_mask = None
    if not losses:
        return torch.tensor(0.0, device=device), {"q_mean": 0.0, "q_p95": 0.0, "q_max": 0.0, "pct_above_1": 0.0}
    total_loss = torch.stack(losses).mean()
    q_cat = torch.cat(q_all)
    stats = {
        "q_mean": float(q_cat.mean().item()),
        "q_p95": float(torch.quantile(q_cat, 0.95).item()) if q_cat.numel() > 1 else float(q_cat.max().item()),
        "q_max": float(q_cat.max().item()),
        "pct_above_1": float((q_cat > 1.0).float().mean().item() * 100.0),
    }
    return total_loss, stats

def log_gpu_memory(label: str, cfg=None, reset_peak: bool = True):
    """Report CUDA allocation/cache peaks without turning telemetry into a fatal guard.

    PyTorch's ``reserved`` value includes reusable caching-allocator blocks.
    Capacity policy belongs to ``trainer.enforce_gpu_memory_budget``, where a
    cache flush can be attempted before a controlled checkpointed stop.
    """
    if cfg is not None and not getattr(cfg, "log_gpu_memory", True):
        return
    if not torch.cuda.is_available():
        return
    alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
    print(f"    [mem] {label}: peak_allocated={alloc_mb:,.0f}MB  peak_reserved={reserved_mb:,.0f}MB")
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    return {"peak_allocated_mb": float(alloc_mb), "peak_reserved_mb": float(reserved_mb)}

def _extract_final_number(text: str) -> Optional[float]:
    """Helper using verifier extract_final_answer with fallback."""
    num = extract_final_answer(text)
    if num is not None:
        return num
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            return None
    return None

def compute_pg_loss(
    model,
    tokenizer,
    input_ids,
    labels,
    device,
    cfg,
    baseline_ema,
    attention_masks=None,
    train_batch_ids=None,
):
    """Outcome-based policy gradient loss measuring answer correctness under full steering vs no steering."""
    N = min(cfg.pg_n_samples, input_ids.size(0))
    advantages = []

    model.eval()
    with torch.no_grad():
        for i in range(N):
            ids_i = input_ids[i : i + 1]
            lbl_i = labels[i : i + 1]
            pad_id = tokenizer.pad_token_id

            q_mask = (lbl_i[0] == -100) & (ids_i[0] != pad_id)
            if q_mask.sum() == 0:
                continue
            q_ids = ids_i[0][q_mask].unsqueeze(0)
            q_attn = torch.ones_like(q_ids)

            gold_mask = lbl_i[0] != -100
            if gold_mask.sum() == 0:
                continue
            gold_text = tokenizer.decode(ids_i[0][gold_mask], skip_special_tokens=True)
            gold_num = _extract_final_number(gold_text)
            if gold_num is None:
                continue

            # Gate ON (1.0)
            model.set_gate_freeze(True, value=1.0)
            gen_on = model.generate(
                input_ids=q_ids,
                attention_mask=q_attn,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_id,
                use_cache=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            text_on = tokenizer.decode(gen_on[0][q_ids.size(1) :], skip_special_tokens=True)
            pred_on = _extract_final_number(text_on)
            r_on = 1.0 if (pred_on is not None and abs(pred_on - gold_num) < 0.01) else 0.0

            # Gate OFF (0.0)
            model.set_gate_freeze(True, value=0.0)
            gen_off = model.generate(
                input_ids=q_ids,
                attention_mask=q_attn,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_id,
                use_cache=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            text_off = tokenizer.decode(gen_off[0][q_ids.size(1) :], skip_special_tokens=True)
            pred_off = _extract_final_number(text_off)
            r_off = 1.0 if (pred_off is not None and abs(pred_off - gold_num) < 0.01) else 0.0

            advantages.append(r_on - r_off)

    model.set_gate_freeze(False)
    model.train()
    log_gpu_memory("compute_pg_loss after generation rounds", cfg)

    if not advantages:
        return torch.tensor(0.0, device=device)

    mean_adv = float(np.mean(advantages))
    baseline_ema[0] = (
        cfg.pg_baseline_momentum * baseline_ema[0] + (1 - cfg.pg_baseline_momentum) * mean_adv
    )
    centred_adv = mean_adv - baseline_ema[0]

    if abs(centred_adv) < 1e-5:
        return torch.tensor(0.0, device=device)

    # Forward on same batch to get live alphas
    attn = (
        attention_masks
        if attention_masks is not None
        else (input_ids != tokenizer.pad_token_id).long()
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        _ = model(input_ids=input_ids, attention_mask=attn)

    pg_total = torch.tensor(0.0, device=device)
    n_terms = 0
    for gate in model.gates.values():
        alpha = gate._alpha_for_aux
        if alpha is None:
            continue
        a = alpha.float().clamp(1e-8, 1 - 1e-8).squeeze(-1)
        vm = gate._valid_mask
        # BUG-05 fix: Unified REINFORCE — loss = -(adv * log(a))
        # adv > 0 (steering helps) → minimizing pushes log(a) up → a → 1 ✓
        # adv < 0 (steering hurts) → minimizing pushes log(a) down → a → 0 ✓
        # Old branched code had sign error in else-branch (drove a→1 when steering hurts).
        if vm is not None and vm.shape == a.shape:
            answer_mask = labels[:, :a.size(1)].ne(-100) if labels is not None else None
            if answer_mask is not None:
                answer_mask = answer_mask[:a.size(0), :a.size(1)]
                vm = vm & answer_mask
            weight_sum = vm.float().sum().clamp(min=1)
            pg_layer = -(centred_adv * (torch.log(a) * vm.float()).sum() / weight_sum)
        else:
            pg_layer = -(centred_adv * torch.log(a).mean())
        pg_total = pg_total + pg_layer
        n_terms += 1

    return pg_total / max(1, n_terms)

@torch.no_grad()
def _span_logprob(model, input_ids, attention_mask, labels, gate_value):
    """Mean log-prob of the labelled span under a fixed gate value via teacher-forced forward pass."""
    model.set_gate_freeze(True, value=gate_value)
    with torch.amp.autocast("cuda", enabled=input_ids.is_cuda):
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, :-1, :]
    tgt = labels[:, 1:]
    valid = tgt != -100
    # ``log_softmax`` materialised a second full (B,T,V) float tensor.  Phi-2
    # has a 51k vocabulary, so that temporary alone is hundreds of MB for a
    # PG probe.  Cross entropy keeps only the per-token result we need.
    token_nll = F.cross_entropy(
        logits.flatten(0, 1).float(), tgt.reshape(-1), ignore_index=-100, reduction="none"
    ).view_as(tgt)
    denom = valid.float().sum(-1).clamp(min=1)
    return -(token_nll * valid.float()).sum(-1) / denom


@torch.no_grad()
def compute_pg_advantage_lite(
    model, tokenizer, input_ids, labels, device, cfg, baseline_ema, attention_masks=None,
    router_context_mask=None, shield_mask=None,
):
    """Estimate a detached Gate-ON vs Gate-OFF advantage before a task graph exists.

    The old implementation called this probe *after* the task backward graph
    had been created and then built another Phi-2 graph solely to recover gate
    activations.  On a T4 that overlaps two full activation graphs and OOMs.
    This function performs only two sequential no-grad forwards; the caller
    applies the resulting scalar to activations from its ordinary task forward.
    """
    attn = attention_masks if attention_masks is not None else (input_ids != tokenizer.pad_token_id).long()
    module_modes = {module: module.training for module in model.modules()}
    gate_state = [(gate._freeze_gate, gate._freeze_gate_value) for gate in model.gates.values()]
    try:
        model.eval()  # deterministic counterfactual reward; no dropout noise.
        # The wrapper consumes pending masks on every full prefill.  Reinstall
        # them for both arms so their reward difference uses the exact same
        # question-only routing and few-shot shielding contract.
        context = router_context_mask
        if context is None:
            context = attn.bool() & labels.eq(-100)
        model.set_router_context_mask(context)
        if hasattr(model, "set_prompt_shield_mask"):
            model.set_prompt_shield_mask(shield_mask)
        lp_on = _span_logprob(model, input_ids, attn, labels, 1.0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.set_router_context_mask(context)
        if hasattr(model, "set_prompt_shield_mask"):
            model.set_prompt_shield_mask(shield_mask)
        lp_off = _span_logprob(model, input_ids, attn, labels, 0.0)
        advantage = lp_on - lp_off
    finally:
        for gate, (frozen, value) in zip(model.gates.values(), gate_state):
            gate._freeze_gate = frozen
            gate._freeze_gate_value = value
        for module, was_training in module_modes.items():
            module.train(was_training)

    mean_adv = advantage.mean().item()
    std_adv = advantage.std().item() if advantage.numel() > 1 else 1.0
    baseline_ema[0] = (
        cfg.pg_baseline_momentum * baseline_ema[0] + (1 - cfg.pg_baseline_momentum) * mean_adv
    )
    return (mean_adv - baseline_ema[0]) / max(std_adv, 1e-6)


def compute_pg_gate_loss_from_live_alphas(model, centred_adv: float) -> torch.Tensor:
    """REINFORCE gate loss using alphas from the caller's existing task graph."""
    device = next(model.parameters()).device
    if abs(centred_adv) < 1e-5:
        return torch.tensor(0.0, device=device)
    pg_total = torch.tensor(0.0, device=device)
    n_terms = 0
    for gate in model.gates.values():
        alpha = gate._alpha_for_aux
        if alpha is None:
            continue
        a = alpha.float().clamp(1e-8, 1 - 1e-8).squeeze(-1)
        vm = gate._valid_mask
        if vm is not None and vm.shape == a.shape:
            weight_sum = vm.float().sum().clamp(min=1)
            pg_layer = -(centred_adv * (torch.log(a) * vm.float()).sum() / weight_sum)
        else:
            pg_layer = -(centred_adv * torch.log(a).mean())
        pg_total = pg_total + pg_layer
        n_terms += 1
    return pg_total / max(1, n_terms)

def compute_pg_loss_lite(
    model,
    tokenizer,
    input_ids,
    labels,
    device,
    cfg,
    baseline_ema,
    attention_masks=None,
    train_batch_ids=None,
):
    """Legacy standalone teacher-forced PG helper.

    Training uses ``compute_pg_advantage_lite`` before the task forward plus
    ``compute_pg_gate_loss_from_live_alphas`` after it, so it never overlaps
    two Phi-2 graphs.  This helper remains for external callers.
    """
    attn = (
        attention_masks
        if attention_masks is not None
        else (input_ids != tokenizer.pad_token_id).long()
    )

    centred_adv = compute_pg_advantage_lite(
        model, tokenizer, input_ids, labels, device, cfg, baseline_ema, attn
    )
    model.train()
    model.set_gate_freeze(False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    return compute_pg_gate_loss_from_live_alphas(model, centred_adv)

def run_pg_loss(
    model,
    tokenizer,
    input_ids,
    labels,
    device,
    cfg,
    baseline_ema,
    attention_masks=None,
    train_batch_ids=None,
):
    """Dispatches to generation-based or teacher-forced counterfactual PG loss."""
    if getattr(cfg, "pg_mode", "generation") == "teacher_forced":
        return compute_pg_loss_lite(
            model,
            tokenizer,
            input_ids,
            labels,
            device,
            cfg,
            baseline_ema,
            attention_masks,
            train_batch_ids,
        )
    return compute_pg_loss(
        model,
        tokenizer,
        input_ids,
        labels,
        device,
        cfg,
        baseline_ema,
        attention_masks,
        train_batch_ids,
    )
