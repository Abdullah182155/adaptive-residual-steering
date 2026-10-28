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
    """Penalise selected layers that receive negligible effective residual energy.

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
        energy = getattr(gate, "_effective_energy_for_aux", None)
        if energy is not None and energy.shape == alpha.shape[:2]:
            masses.append((energy.float() * valid.float()).sum())
        else:
            # Compatibility with minimal unit-test doubles.
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


# ==== M4 additions ====
from typing import Sequence


def compute_calc_token_weights(tokenizer, labels: torch.Tensor, ids: torch.Tensor, w: float = 1.75) -> torch.Tensor:
    """Return per-token weight tensor (same shape as labels).

    Weight = w for supervised tokens whose id is in contrastive._CONTRAST_MATH_IDS
             (digits, operators, =, $, %, .).
    Weight = 1.0 for all other non-masked tokens.
    Weight = 0.0 where labels == -100.

    If _CONTRAST_MATH_IDS is None or empty, falls back to uniform weight 1.0
    (non-masked tokens only). The id set is read at call time because it is
    populated lazily by the contrastive module.
    """
    from src.training import contrastive as _contrastive

    valid = labels.ne(-100)
    weights = valid.to(torch.float32)
    math_ids = getattr(_contrastive, "_CONTRAST_MATH_IDS", None)
    if math_ids:
        math_t = torch.tensor(sorted(math_ids), dtype=torch.long, device=labels.device)
        is_math = torch.isin(ids.to(labels.device), math_t) & valid
        weights = torch.where(is_math, torch.full_like(weights, float(w)), weights)
    return weights


def weighted_answer_nll(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute weighted cross-entropy loss.

    logits: (B, T, V)
    labels: (B, T)   -100 for masked positions
    weights: (B, T)  0 for masked positions, positive elsewhere

    Returns scalar mean loss (sum of weight*ce / sum of weights).
    Raises ValueError if all weights are zero.
    """
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    shift_w = weights[:, 1:].to(shift_logits.dtype) * shift_labels.ne(-100).to(shift_logits.dtype)
    denom = shift_w.sum()
    if float(denom.item()) <= 0.0:
        raise ValueError("weighted_answer_nll: all weights are zero")
    ce = F.cross_entropy(shift_logits.transpose(1, 2), shift_labels, ignore_index=-100, reduction="none")
    return (ce * shift_w).sum() / denom


def unlikelihood_on_divergence(logits_neg: torch.Tensor, neg_labels: torch.Tensor, div_token_mask: torch.BoolTensor) -> torch.Tensor:
    """Unlikelihood loss on divergent-line tokens: mean[-log(1 - p(tok))] over divergent tokens.

    logits_neg: (B, T, V)
    neg_labels:  (B, T) with -100 on prompt
    div_token_mask: (B, T) bool, subset of (neg_labels != -100) positions

    Returns scalar >= 0. Returns 0.0 if div_token_mask has no True entries.
    """
    shift_labels = neg_labels[:, 1:]
    mask = div_token_mask[:, 1:].to(torch.bool) & shift_labels.ne(-100)
    if not bool(mask.any()):
        return torch.tensor(0.0, device=logits_neg.device)
    sel_logits = logits_neg[:, :-1, :][mask].float()  # (N, V)
    tgt = shift_labels[mask]
    logp = F.log_softmax(sel_logits, dim=-1).gather(1, tgt.unsqueeze(-1)).squeeze(-1)
    p = logp.exp().clamp(max=1.0 - 1e-6)
    return (-torch.log1p(-p)).mean()


def collate_pad(rows: Sequence[dict], pad_id: int) -> dict:
    """Pad a list of row dicts to uniform length along dim=1.

    Each row must have 'input_ids' (LongTensor 1-D) and 'labels' (LongTensor 1-D).
    Optional keys 'neg_input_ids', 'neg_labels', 'div_token_mask' are padded too
    (only when every row carries a non-None value for that key).
    Returns a single batch dict with tensors of shape (B, T_max). Also returns
    'attention_mask' (and 'neg_attention_mask' when neg_input_ids is present).
    Missing optional keys are absent in the output.
    """
    if len(rows) == 0:
        raise ValueError("collate_pad: empty rows")

    def _pad(seqs, fill, dtype):
        t_max = max(int(s.numel()) for s in seqs)
        out = torch.full((len(seqs), t_max), fill, dtype=dtype)
        for i, s in enumerate(seqs):
            out[i, : s.numel()] = s.to(dtype)
        return out

    def _seqs(key, dtype):
        return [torch.as_tensor(r[key]).to(dtype).reshape(-1) for r in rows]

    def _has(key):
        return all((key in r) and (r[key] is not None) for r in rows)

    batch = {}
    id_seqs = _seqs("input_ids", torch.long)
    batch["input_ids"] = _pad(id_seqs, pad_id, torch.long)
    batch["labels"] = _pad(_seqs("labels", torch.long), -100, torch.long)
    batch["attention_mask"] = _pad([torch.ones(s.numel(), dtype=torch.long) for s in id_seqs], 0, torch.long)

    if _has("neg_input_ids"):
        neg_seqs = _seqs("neg_input_ids", torch.long)
        batch["neg_input_ids"] = _pad(neg_seqs, pad_id, torch.long)
        batch["neg_attention_mask"] = _pad([torch.ones(s.numel(), dtype=torch.long) for s in neg_seqs], 0, torch.long)
    if _has("neg_labels"):
        batch["neg_labels"] = _pad(_seqs("neg_labels", torch.long), -100, torch.long)
    if _has("div_token_mask"):
        batch["div_token_mask"] = _pad(_seqs("div_token_mask", torch.bool), False, torch.bool)
    return batch


# ===========================================================================
# M7: Dominance Hinge, Token Labels, Gate BCE, and Paired Gate Evaluation
# ===========================================================================

def dominance_hinge(nll_gated: torch.Tensor, nll_ungated: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Per-example dominance hinge loss: mean(relu(nll_gated - nll_ungated.detach() + margin)).

    Gradient flows ONLY through nll_gated.
    """
    diff = nll_gated - nll_ungated.detach() + float(margin)
    return F.relu(diff).mean()


class TokenLabelStore:
    """Keyed canonical_token_key or (split, qid, n_shot) -> (labels: torch.BoolTensor[T_ans], weights: torch.Tensor)."""

    def __init__(self, data: Optional[Dict[Any, Tuple[torch.Tensor, torch.Tensor]]] = None):
        from src.utils.keys import canonical_token_key
        self._data = {}
        if data is not None:
            for k, v in data.items():
                if isinstance(k, tuple):
                    ck = canonical_token_key(*k)
                elif isinstance(k, str) and ":" in k:
                    ck = k
                else:
                    ck = canonical_token_key("train", k, 0)
                self._data[ck] = v
                self._data[k] = v  # preserve raw key for backward compatibility

    def keys(self):
        return self._data.keys()

    def get(self, key: Any) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        from src.utils.keys import canonical_token_key
        if key in self._data:
            return self._data[key]
        if isinstance(key, tuple):
            ck = canonical_token_key(*key)
        elif isinstance(key, str) and ":" in key:
            ck = key
        else:
            ck = canonical_token_key("train", key, 0)
        return self._data.get(ck)

    def __contains__(self, key: Any) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        return len(self._data)


def counterfactual_token_labels(table, router_choice: np.ndarray, tau_u: float = 0.005) -> TokenLabelStore:
    """Build TokenLabelStore from counterfactual table and router argmax classes.

    u_t = token_nll[n, 0, t] - token_nll[n, c_n, t] where c_n = router_choice[n].
    label = 1[u_t > tau_u]. Rows with c_n == 0 have no label (empty / abstain class).
    Weights balance positive and negative tokens on valid positions.
    """
    from src.utils.keys import canonical_token_key

    store_dict = {}
    N = len(table.qid) if hasattr(table, "qid") else len(table)
    choices = np.asarray(router_choice)
    for i in range(N):
        c_n = int(choices[i])
        if c_n == 0:
            continue
        u_t = table.token_nll[i, 0, :] - table.token_nll[i, c_n, :]
        mask = table.token_mask[i]
        u_valid = u_t[mask]
        if len(u_valid) == 0:
            continue
        lbls_bool = torch.as_tensor(u_valid > float(tau_u), dtype=torch.bool)
        n_pos = int(lbls_bool.sum().item())
        n_neg = int((~lbls_bool).sum().item())
        n_tot = len(lbls_bool)
        if n_pos > 0 and n_neg > 0:
            w_pos = n_tot / (2.0 * n_pos)
            w_neg = n_tot / (2.0 * n_neg)
            weights = torch.where(
                lbls_bool,
                torch.full((n_tot,), w_pos, dtype=torch.float32),
                torch.full((n_tot,), w_neg, dtype=torch.float32),
            )
        else:
            weights = torch.ones(n_tot, dtype=torch.float32)
        key = canonical_token_key(table.split[i], table.qid[i], table.n_shot[i])
        store_dict[key] = (lbls_bool, weights)
    return TokenLabelStore(store_dict)


def gate_bce_tokens(
    model,
    rows: Sequence[dict],
    store: Optional[TokenLabelStore],
    device: torch.device,
) -> torch.Tensor:
    """Weighted BCE on gate._alpha_for_aux over selected layers on valid answer tokens."""
    if store is None or len(store) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    bce_terms = []
    weight_terms = []

    for b, r in enumerate(rows):
        key = (str(r.get("split", "train")), str(r.get("qid", "")), int(r.get("n_shot", r.get("shot", 0))))
        item = store.get(key)
        if item is None:
            continue
        target_labels, target_weights = item
        lbls = torch.as_tensor(r.get("labels", [])).reshape(-1)
        ans_positions = (lbls != -100).nonzero(as_tuple=True)[0]
        if len(ans_positions) == 0 or len(target_labels) == 0:
            continue
        t_m = min(len(ans_positions), len(target_labels))
        ans_idx = ans_positions[-t_m:].to(device)
        tgt = target_labels[-t_m:].to(device).float()
        w = target_weights[-t_m:].to(device).float()

        # Retrieve active layer selection for row b if available
        active_layers = None
        if hasattr(model, "current_route_tuples"):
            routes = model.current_route_tuples()
            if b < len(routes):
                active_layers = set(routes[b])

        for layer_idx, gate in model.gates.items():
            # If active route exists for this sample, only supervise active layer gates
            if active_layers is not None and len(active_layers) > 0:
                try:
                    if int(layer_idx) not in active_layers:
                        continue
                except (ValueError, TypeError):
                    pass
            alpha = getattr(gate, "_alpha_for_aux", None)
            if alpha is None or alpha.shape[0] <= b:
                continue
            # alpha: (B, T, 1)
            # FIX: Align causal next-token index - next-token prediction at position p
            # is driven by the hidden state and gate at position p-1
            ans_idx_shifted = torch.clamp(ans_idx - 1, min=0)
            pred = alpha[b, ans_idx_shifted, 0].float().clamp(1e-7, 1.0 - 1e-7)
            term = F.binary_cross_entropy(pred, tgt, weight=w, reduction="sum")
            bce_terms.append(term)
            weight_terms.append(w.sum())

    if not bce_terms or sum(weight_terms) <= 0.0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    total_bce = sum(bce_terms)
    total_w = sum(weight_terms)
    return total_bce / total_w.clamp(min=1e-6)


def gate_bce_tokens_with_live_fallback(
    model,
    rows: Sequence[dict],
    store: Optional["TokenLabelStore"],
    device: torch.device,
    loss_mat_g: Optional[torch.Tensor] = None,
    loss_mat_u: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    tau_u: float = 0.005,
) -> torch.Tensor:
    """Token BCE using CF store when available; live per-token NLL-diff labels as fallback.

    For rows with a CF store hit, uses the pre-computed counterfactual labels (higher
    quality, but sparse — typically only ~1-3% of training rows).  For all other rows,
    derives token-level binary labels on-the-fly from the current batch's gated vs
    ungated per-token NLL matrices:

        live_label[b, t] = 1  if  loss_mat_u[b, t] - loss_mat_g[b, t] > tau_u
                             0  otherwise

    i.e. a token is labelled "beneficial" when steering lowers its NLL by more than
    tau_u.  This provides a dense, always-available supervision signal that replaces
    the sparse CF store.

    Args:
        model: Phi2WithRSC with gate._alpha_for_aux populated from the gated forward pass.
        rows: list of row dicts (current batch).
        store: TokenLabelStore built from CF table (may be None or sparse).
        device: torch device.
        loss_mat_g: (B, T) per-token NLL from gated forward (detached, no grad).
        loss_mat_u: (B, T) per-token NLL from ungated forward (detached, no grad).
        valid_mask: (B, T) bool mask of non-padding answer tokens.
        tau_u: minimum NLL improvement to label a token as beneficial.
    """
    # Try CF store first
    cf_result = gate_bce_tokens(model, rows, store, device)
    if loss_mat_g is None or loss_mat_u is None or valid_mask is None:
        return cf_result

    # Build live labels for rows not covered by the CF store
    live_bce_terms = []
    live_weight_terms = []

    for b, r in enumerate(rows):
        # Skip if CF store already covered this row
        if store is not None and len(store) > 0:
            key = (str(r.get("split", "train")), str(r.get("qid", "")), int(r.get("n_shot", r.get("shot", 0))))
            if store.get(key) is not None:
                continue  # CF store hit: already included in cf_result

        # Derive live labels from the NLL difference at valid (answer) positions
        row_valid = valid_mask[b]  # (T,) bool
        if not row_valid.any():
            continue

        # NLL improvement per token: positive means gated is better.
        # loss_mat has shape (T-1,) because it's derived from shifted logits.
        # Clamp valid positions to the loss matrix length to avoid out-of-bounds.
        nll_diff_tok = (loss_mat_u[b] - loss_mat_g[b])  # (T-1,) — positive = gated is better
        L_loss = nll_diff_tok.shape[0]
        # Restrict to positions within the loss matrix bounds
        valid_positions = row_valid.nonzero(as_tuple=True)[0]
        valid_positions = valid_positions[valid_positions < L_loss]
        if len(valid_positions) == 0:
            continue
        diff_valid = nll_diff_tok[valid_positions]  # (V,)
        # PHYSICAL LOGIC: out_u is computed with gate_mode="const", scale=steer_scale (full steering alpha=1.0).
        # out_g is computed with gate_mode="learned" (attenuated/gated alpha in [0, 1]).
        # diff_valid = loss_mat_u - loss_mat_g.
        # If loss_u > loss_g (diff_valid > 0): constant full steering caused higher loss than gated steering.
        # This proves full steering is harmful here; the token needs less steering (target alpha -> 0.0).
        # If loss_u < loss_g (diff_valid < 0): constant full steering caused lower loss than gated steering.
        # This proves full steering is helpful here; the token needs more steering (target alpha -> 1.0).
        live_labels = (diff_valid < 0).float().to(device)
        n_pos = int((live_labels > 0.5).sum().item())
        n_neg = int((live_labels <= 0.5).sum().item())
        n_tot = len(live_labels)
        if n_tot == 0:
            continue
        if n_pos > 0 and n_neg > 0:
            w_pos = n_tot / (2.0 * n_pos)
            w_neg = n_tot / (2.0 * n_neg)
            live_weights = torch.where(live_labels > 0.5,
                torch.full((n_tot,), w_pos, dtype=torch.float32, device=device),
                torch.full((n_tot,), w_neg, dtype=torch.float32, device=device),
            )
        else:
            live_weights = torch.ones(n_tot, dtype=torch.float32, device=device)

        # Collect gate alpha values at valid positions across active layers
        for layer_idx, gate in model.gates.items():
            alpha = getattr(gate, "_alpha_for_aux", None)
            if alpha is None or alpha.shape[0] <= b:
                continue
            pred = alpha[b, valid_positions, 0].float().clamp(1e-7, 1.0 - 1e-7)
            if pred.shape[0] != n_tot:
                # Lengths may differ if seq is shorter; align to minimum
                m = min(pred.shape[0], n_tot)
                pred = pred[-m:]
                tgt = live_labels[-m:]
                w = live_weights[-m:]
            else:
                tgt = live_labels
                w = live_weights
            term = F.binary_cross_entropy(pred, tgt, weight=w, reduction="sum")
            live_bce_terms.append(term)
            live_weight_terms.append(w.sum())

    if not live_bce_terms:
        return cf_result

    live_total = sum(live_bce_terms) / sum(live_weight_terms).clamp(min=1e-6)
    # Combine: if cf_result is non-zero (some CF store hits), average both signals;
    # otherwise return only the live signal.
    cf_val = cf_result.detach().item() if torch.is_tensor(cf_result) else float(cf_result)
    if abs(cf_val) > 1e-9:
        return (cf_result + live_total) * 0.5
    return live_total


def gate_total_loss(
    nll_gated: torch.Tensor,
    nll_ungated: torch.Tensor,
    bce_tok: torch.Tensor,
    mean_alpha: torch.Tensor,
    cfg,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute combined V5 gate loss.

    Loss = NLL_gated + lambda_dom*L_dom + lambda_nll_diff*(NLL_gated - NLL_ungated).mean()
           + lambda_tok*BCE + lambda_l1*mean_alpha

    The lambda_nll_diff term directly minimises the mean per-example NLL gap
    (gated - ungated), providing an explicit reward signal when the gate suppresses
    steering on harmful tokens.  Without it, the dominance hinge only penalises
    positive gaps but never rewards negative ones, leaving the gate unable to learn
    selective suppression when most training examples are harmful.
    """
    margin = float(getattr(cfg, "gate_dom_margin", 0.0))
    l_dom = dominance_hinge(nll_gated, nll_ungated, margin=margin)
    lambda_dom = float(getattr(cfg, "lambda_dom", 1.0))
    lambda_tok = float(getattr(cfg, "lambda_tok", 0.3))
    lambda_l1 = float(getattr(cfg, "lambda_l1_gate_v5", 0.005))
    # Direct NLL-diff minimisation: drives gate to reduce nll_gated below nll_ungated.
    # Default 1.0 — same scale as lambda_dom, but this term is always active (no relu).
    lambda_nll_diff = float(getattr(cfg, "lambda_nll_diff", 1.0))

    bce_val = bce_tok if torch.is_tensor(bce_tok) else torch.tensor(float(bce_tok), device=nll_gated.device)
    alpha_val = mean_alpha if torch.is_tensor(mean_alpha) else torch.tensor(float(mean_alpha), device=nll_gated.device)

    nll_diff = (nll_gated - nll_ungated.detach()).mean()
    total = nll_gated.mean() + lambda_dom * l_dom + lambda_nll_diff * nll_diff + lambda_tok * bce_val + lambda_l1 * alpha_val
    metrics = {
        "loss_total": float(total.detach().item()),
        "nll_gated": float(nll_gated.mean().detach().item()),
        "nll_ungated": float(nll_ungated.mean().detach().item()),
        "loss_dom": float(l_dom.detach().item()),
        "loss_nll_diff": float(nll_diff.detach().item()),
        "loss_bce_tok": float(bce_val.detach().item()),
        "mean_alpha": float(alpha_val.detach().item()),
    }
    return total, metrics


def gate_label_auc(
    model,
    rows: Sequence[dict],
    store: TokenLabelStore,
    device: torch.device,
) -> float:
    """Advisory diagnostic: ROC AUC between gate alphas and counterfactual token labels."""
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for r in rows:
            key = (str(r.get("split", "train")), str(r.get("qid", "")), int(r.get("n_shot", r.get("shot", 0))))
            item = store.get(key)
            if item is None:
                continue
            target_labels, _ = item
            lbls = torch.as_tensor(r.get("labels", [])).reshape(-1)
            ans_positions = (lbls != -100).nonzero(as_tuple=True)[0]
            if len(ans_positions) == 0 or len(target_labels) == 0:
                continue
            t_m = min(len(ans_positions), len(target_labels))
            ans_idx = ans_positions[-t_m:].to(device)
            tgt = target_labels[-t_m:].cpu().numpy()

            layer_preds = []
            for gate in model.gates.values():
                alpha = getattr(gate, "last_alpha", None)
                if alpha is not None:
                    layer_preds.append(alpha[0, ans_idx, 0].float().cpu().numpy())
            if layer_preds:
                mean_p = np.mean(layer_preds, axis=0)
                all_preds.extend(mean_p.tolist())
                all_tgts.extend(tgt.tolist())

    if len(all_tgts) < 2:
        return 0.5
    y = np.asarray(all_tgts, dtype=bool)
    p = np.asarray(all_preds, dtype=np.float32)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    pos_p = p[y]
    neg_p = p[~y]
    # Fast Mann-Whitney U calculation
    u = sum((pos_p[:, None] > neg_p[None, :]).sum() + 0.5 * (pos_p[:, None] == neg_p[None, :]).sum())
    return float(u / (n_pos * n_neg))


def paired_gate_vs_ungated(
    model,
    rows: Sequence[dict],
    cfg,
    device: torch.device,
    pad_id: int = 0,
) -> Dict[str, Any]:
    """Compute per-example NLL diffs between gated and ungated (full pass-through) steering."""
    was_training = model.training
    model.eval()
    nll_gated_list = []
    nll_ungated_list = []

    with torch.no_grad():
        for r in rows:
            batch = collate_pad([r], pad_id)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            # Gated forward
            model.set_gate_mode("learned")
            model.set_routing_override(None)
            if hasattr(model, "set_router_context_mask"):
                model.set_router_context_mask(mask.bool() & (lbls == -100))
            out_g = model(input_ids=ids, attention_mask=mask)
            valid = lbls[:, 1:].ne(-100)
            if not valid.any():
                continue
            nll_g = F.cross_entropy(
                out_g.logits[:, :-1, :].transpose(1, 2),
                lbls[:, 1:],
                ignore_index=-100,
                reduction="sum",
            ) / valid.sum().clamp(min=1)

            # Ungated counterfactual forward with the same route
            routes = model.current_route_tuples()
            with model.counterfactual_mode():
                model.set_gate_mode("const", scale=getattr(cfg, "steer_scale", 1.0))
                model.set_routing_override("matrix", model.subset_matrix(routes).to(device))
                out_u = model(input_ids=ids, attention_mask=mask)
                nll_u = F.cross_entropy(
                    out_u.logits[:, :-1, :].transpose(1, 2),
                    lbls[:, 1:],
                    ignore_index=-100,
                    reduction="sum",
                ) / valid.sum().clamp(min=1)
                model.set_routing_override(None)
                model.set_gate_mode("learned")

            nll_gated_list.append(float(nll_g.item()))
            nll_ungated_list.append(float(nll_u.item()))

    if was_training:
        model.train()

    g_arr = np.asarray(nll_gated_list, dtype=np.float32)
    u_arr = np.asarray(nll_ungated_list, dtype=np.float32)
    diffs = g_arr - u_arr
    mean_diff = float(diffs.mean()) if len(diffs) > 0 else 0.0

    # 95% bootstrap CI
    ci_low, ci_high = mean_diff, mean_diff
    if len(diffs) > 1:
        rng = np.random.default_rng(42)
        boot_means = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(1000)]
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))

    return {
        "nll_gated": [float(x) for x in g_arr],
        "nll_ungated": [float(x) for x in u_arr],
        "diffs": [float(x) for x in diffs],
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "gate_adopted_nll": bool(mean_diff < 0.0),
    }
