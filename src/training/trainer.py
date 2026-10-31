import gc
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, is_dataclass
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

from src.configs.config import RSCConfig
from src.evaluation.verifier import epoch_end_accuracy
from src.evaluation.multi_dataset import build_benchmark_prompt, evaluate_sample_answer
from src.evaluation.verifier import generate_response
from src.routing.rloo import compute_router_rloo_loss, per_example_answer_nll
from src.training.losses import (
    LRReducer,
    CheckpointManager,
    GradientMonitor,
    compute_l1_gate,
    compute_gate_budget_loss,
    compute_antisat_gate,
    compute_gate_diversity_loss,
    compute_gate_overlap_loss,
    compute_route_activity_floor_loss,
    compute_pg_advantage_lite,
    compute_pg_gate_loss_from_live_alphas,
    log_gpu_memory,
)
from src.training.contrastive import compute_contrastive_gate_loss
from src.experiments.checkpointing import (
    ARS_PREFIXES,
    ars_state_dict,
    load_ars_checkpoint_strict,
    load_checkpoint_manifest,
    save_ars_checkpoint,
)
from src.steering.steernet import post_cap_ratio_from_pre_cap_q
from src.utils.failure_policy import (
    FatalContractError,
    RunManifest,
    get_manifest,
    set_manifest,
    to_jsonable,
)


# --------------------------------------------------------------------------- #
# M0 helpers
# --------------------------------------------------------------------------- #
def _assert_finite_params(model, stage_name: str = ""):
    """Fatal (E_NONFINITE_PARAMS) if any SteerNet / Gate / Router parameter is NaN or Inf."""
    manifest = get_manifest()
    for group_name in ("steer_nets", "gates", "router"):
        module = getattr(model, group_name, None)
        if module is None:
            continue
        if isinstance(module, (torch.nn.ModuleDict, dict)):
            items = module.items()
        elif isinstance(module, torch.nn.Module):
            items = [("", module)]
        else:
            continue
        for sub_name, sub_mod in items:
            if not isinstance(sub_mod, torch.nn.Module):
                continue
            for pname, p in sub_mod.named_parameters():
                with torch.no_grad():
                    bad = bool(torch.isnan(p).any() or torch.isinf(p).any())
                if bad:
                    manifest.fatal(
                        "E_NONFINITE_PARAMS",
                        f"Non-finite params detected in {group_name}.{sub_name}.{pname} during stage {stage_name}",
                        stage=stage_name,
                    )


class NonfiniteLossTracker:
    """Counts consecutive non-finite losses; raises E_NONFINITE_LOSS_STREAK at the limit."""

    def __init__(self, limit: int = 20, stage_name: str = ""):
        self.limit = int(limit)
        self.stage_name = stage_name
        self.count = 0

    @classmethod
    def from_cfg(cls, cfg, stage_name: str = "") -> "NonfiniteLossTracker":
        return cls(getattr(cfg, "max_consecutive_nonfinite_loss", 20), stage_name)

    def observe(self, is_finite: bool) -> int:
        if is_finite:
            self.count = 0
            return 0
        self.count += 1
        if self.count >= self.limit:
            get_manifest().fatal(
                "E_NONFINITE_LOSS_STREAK",
                f"Exceeded {self.limit} consecutive non-finite loss steps",
                stage=self.stage_name,
            )
        return self.count


def _phase1_runtime_config_sha256(cfg: RSCConfig) -> str:
    """Fingerprint Phase-1-relevant runtime choices, excluding resume plumbing."""
    raw = asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
    ignored = {
        "resume_phase1", "resume_phase2", "phase1_report_save_path", "checkpoint_dir",
        "lora_save_path", "phase2_save_path", "router_bootstrap_save_path",
        "steer_finetune_save_path", "gate_only_save_path", "phase2_training_state_path",
    }
    selected = {key: value for key, value in raw.items() if key not in ignored}
    payload = json.dumps(selected, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _phase1_report_provenance(checkpoint_path: str, cfg: Optional[RSCConfig] = None) -> Dict[str, str]:
    """Bind a cached acceptance report to the exact Phase-1 checkpoint.

    A report selected under different weights, source revision, or architecture
    must never be trusted during resume.  The checkpoint manifest already
    records an exact ARS state hash and configuration, so reuse it instead of
    inventing a second, weaker identity scheme.
    """
    manifest = load_checkpoint_manifest(checkpoint_path)
    if not manifest or not manifest.get("state_sha256"):
        raise RuntimeError(
            f"Phase-1 resume requires a manifest with state_sha256: {checkpoint_path}"
        )
    config_json = json.dumps(manifest.get("config", {}), sort_keys=True, default=str)
    provenance = {
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_state_sha256": str(manifest["state_sha256"]),
        "checkpoint_git_commit": str(manifest.get("git_commit", "unknown")),
        "checkpoint_config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
    }
    if cfg is not None:
        provenance["runtime_config_sha256"] = _phase1_runtime_config_sha256(cfg)
    return provenance


def _cache_phase1_report(
    report: Dict[str, Any], report_path: str, checkpoint_path: str, cfg: Optional[RSCConfig] = None,
) -> Dict[str, Any]:
    """Atomically persist a Phase-1 report with checkpoint provenance."""
    payload = dict(report)
    payload["cache_provenance"] = _phase1_report_provenance(checkpoint_path, cfg)
    directory = os.path.dirname(os.path.abspath(report_path))
    os.makedirs(directory, exist_ok=True)
    temporary_path = report_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    os.replace(temporary_path, report_path)
    return payload


def _phase1_report_matches_checkpoint(
    report: Dict[str, Any], checkpoint_path: str, cfg: Optional[RSCConfig] = None,
) -> bool:
    """Return whether a cached report was generated from the loaded weights."""
    cached = report.get("cache_provenance")
    if not isinstance(cached, dict):
        return False
    expected = _phase1_report_provenance(checkpoint_path, cfg)
    return all(cached.get(key) == value for key, value in expected.items())


def _set_batch_context(model, mask: torch.Tensor, labels: torch.Tensor, shield_mask: Optional[torch.Tensor] = None):
    """Install exact few-shot shielding and route only from the live question prompt."""
    prompt = mask.bool() & labels.eq(-100)
    if shield_mask is not None:
        shield = shield_mask.bool() & mask.bool()
        if hasattr(model, "set_prompt_shield_mask"):
            model.set_prompt_shield_mask(shield)
        prompt = prompt & ~shield
    elif hasattr(model, "set_prompt_shield_mask"):
        model.set_prompt_shield_mask(None)
    model.set_router_context_mask(prompt)


def _small_run_waiver(n_validation_examples: int, cfg: RSCConfig) -> bool:
    """Whether diagnostics are informative-only because the holdout is tiny.

    This never changes losses, routing, or checkpoints.  It only prevents a
    smoke test with an intrinsically unreliable holdout from being mislabeled
    as a failed scientific experiment.
    """
    minimum = int(getattr(cfg, "min_reliable_validation_examples", 16))
    return bool(getattr(cfg, "allow_inconclusive_small_runs", True)) and n_validation_examples < minimum


def _optimizer_has_gradients(parameters) -> bool:
    """Whether this accumulation window actually called backward on an optimizer parameter."""
    return any(parameter.grad is not None for parameter in parameters)


def _cuda_memory_snapshot() -> Dict[str, float]:
    """Return allocator and device-memory measurements in MiB.

    ``reserved`` is intentionally exposed as a cache-pressure signal; it is
    not equivalent to live tensor allocation and must not by itself kill a
    long-running Kaggle experiment.
    """
    if not torch.cuda.is_available():
        return {"available": False}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    mib = float(1024 ** 2)
    return {
        "available": True,
        "allocated_mb": float(torch.cuda.memory_allocated() / mib),
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / mib),
        "reserved_mb": float(torch.cuda.memory_reserved() / mib),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved() / mib),
        "free_mb": float(free_bytes / mib),
        "total_mb": float(total_bytes / mib),
    }


def enforce_gpu_memory_budget(label: str, cfg: RSCConfig) -> Dict[str, Any]:
    """Reclaim cached CUDA blocks before requesting a controlled staged stop.

    The active-allocation limit is the primary safety contract.  A high
    reservation triggers ``gc.collect`` plus ``empty_cache`` first because
    PyTorch deliberately retains reusable allocator blocks.  Callers save a
    resumable state checkpoint when the returned action is ``"emergency"``;
    this helper never raises a raw CUDA-memory RuntimeError.
    """
    before = _cuda_memory_snapshot()
    if not before.get("available", False):
        return {"action": "disabled", "before": before, "after": before}

    active_limit = max(0, int(getattr(cfg, "cuda_active_memory_limit_mb", 0)))
    reserved_limit = max(0, int(getattr(cfg, "cuda_reserved_memory_limit_mb", 0)))
    min_free = max(0, int(getattr(cfg, "cuda_memory_min_free_mb", 0)))
    cache_pressure = (
        (reserved_limit > 0 and max(before["reserved_mb"], before["peak_reserved_mb"]) > reserved_limit)
        or (min_free > 0 and before["free_mb"] < min_free)
    )

    after = before
    reclaimed = False
    if cache_pressure:
        gc.collect()
        torch.cuda.empty_cache()
        after = _cuda_memory_snapshot()
        reclaimed = True
        print(
            f"    [mem] {label}: cache pressure reclaimed "
            f"reserved={before['reserved_mb']:.0f}->{after['reserved_mb']:.0f}MB "
            f"free={before['free_mb']:.0f}->{after['free_mb']:.0f}MB"
        )

    active_peak = max(after["allocated_mb"], after["peak_allocated_mb"])
    insufficient_free = min_free > 0 and after["free_mb"] < min_free
    if (active_limit > 0 and active_peak > active_limit) or insufficient_free:
        reason = (
            f"active peak {active_peak:.0f}MB exceeds {active_limit}MB"
            if active_limit > 0 and active_peak > active_limit
            else f"free device memory {after['free_mb']:.0f}MB is below {min_free}MB"
        )
        print(f"    ⚠️ [mem] {label}: controlled emergency requested ({reason}).")
        torch.cuda.reset_peak_memory_stats()
        return {"action": "emergency", "reason": reason, "before": before, "after": after}
    if reclaimed:
        torch.cuda.reset_peak_memory_stats()
        return {"action": "reclaimed", "before": before, "after": after}
    torch.cuda.reset_peak_memory_stats()
    return {"action": "ok", "before": before, "after": after}


def _to_cpu_tree(value: Any) -> Any:
    """Copy tensors recursively to CPU for a portable training-state file."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    return value


def _phase2_runtime_state(model) -> Dict[str, Any]:
    """Capture non-tensor routing state required to continue Phase 2 exactly."""
    eligible = None
    if hasattr(model, "routing_eligibility_mask") and hasattr(model, "target_layers"):
        device = model.device if hasattr(model, "device") else next(model.parameters()).device
        mask = model.routing_eligibility_mask(device).detach().cpu().tolist()
        eligible = [int(layer) for layer, allowed in zip(model.target_layers, mask) if allowed]
    certified_scales = None
    if hasattr(model, "_certified_gate_scales") and model._certified_gate_scales:
        certified_scales = {int(layer): float(scale) for layer, scale in model._certified_gate_scales.items()}
    return {
        "eligible_layers": eligible,
        "certified_gate_scales": certified_scales,
        "router_warmup": bool(getattr(model, "_router_warmup", False)),
        "router_exploration": bool(getattr(model, "_router_exploration", False)),
        "router_inference_mode": getattr(model, "_router_inference_mode", "argmax"),
    }


def _restore_phase2_runtime_state(model, state: Dict[str, Any]) -> None:
    """Restore routing controls after a validated Phase-2 state load."""
    eligible = state.get("eligible_layers")
    if eligible is not None and hasattr(model, "set_eligible_routing_layers"):
        model.set_eligible_routing_layers([int(layer) for layer in eligible])
    scales = state.get("certified_gate_scales")
    if scales is not None and hasattr(model, "set_certified_gate_scales"):
        model.set_certified_gate_scales({int(layer): float(scale) for layer, scale in scales.items()})
    if hasattr(model, "set_router_warmup"):
        model.set_router_warmup(bool(state.get("router_warmup", False)))
    if hasattr(model, "set_router_exploration"):
        model.set_router_exploration(bool(state.get("router_exploration", False)))
    if hasattr(model, "set_router_inference_mode"):
        model.set_router_inference_mode(state.get("router_inference_mode", "argmax"))


def _capture_rng_state() -> Dict[str, Any]:
    """Capture all stochastic sources used by staged training."""
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore a checkpointed stochastic state before resuming a stage."""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_phase2_training_state(
    path: str,
    *,
    model,
    optimizer: torch.optim.Optimizer,
    scaler,
    lr_reducer: LRReducer,
    stage_name: str,
    epochs: int,
    epoch: int,
    next_batch_start: int,
    indices: List[int],
    global_step: int,
    best_acc: float,
    best_val: float,
    patience_ctr: int,
    baseline_ema: List[float],
    effective_rloo_batch_size: int,
    train_steer: bool,
    train_gate: bool,
    train_router: bool,
    save_path: str,
) -> Dict[str, Any]:
    """Atomically persist complete Phase-2 state for exact restart after pressure.

    This intentionally supplements, rather than replaces, the portable
    tensor-only ARS checkpoint consumed by evaluation notebooks.
    """
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    payload = {
        "format": "ars-phase2-training-state-v1",
        "ars_state": ars_state_dict(model),
        "optimizer": _to_cpu_tree(optimizer.state_dict()),
        "scaler": _to_cpu_tree(scaler.state_dict()),
        "lr_reducer": lr_reducer.state_dict(),
        "rng": _to_cpu_tree(_capture_rng_state()),
        "runtime": _phase2_runtime_state(model),
        "progress": {
            "stage_name": stage_name,
            "epochs": int(epochs),
            "epoch": int(epoch),
            "next_batch_start": int(next_batch_start),
            "indices": list(indices),
            "global_step": int(global_step),
            "best_acc": float(best_acc),
            "best_val": float(best_val),
            "patience_ctr": int(patience_ctr),
            "baseline_ema": list(baseline_ema),
            "effective_rloo_batch_size": int(effective_rloo_batch_size),
            "train_steer": bool(train_steer),
            "train_gate": bool(train_gate),
            "train_router": bool(train_router),
            "save_path": str(save_path),
        },
    }
    temporary = destination + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return {"path": destination, "format": payload["format"], "global_step": global_step}


def load_phase2_training_state(path: str, model, device: torch.device) -> Dict[str, Any]:
    """Load and validate a complete Phase-2 state before any resumed update."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CFG.resume_phase2 is True but state was not found at {path}.")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != "ars-phase2-training-state-v1":
        raise RuntimeError(f"Invalid Phase-2 training state: {path}")
    required = {"ars_state", "optimizer", "scaler", "lr_reducer", "rng", "runtime", "progress"}
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"Phase-2 training state is incomplete: missing {missing}")
    saved = payload["ars_state"]
    expected = {name: value for name, value in model.state_dict().items() if name.startswith(ARS_PREFIXES)}
    unexpected = sorted(set(saved) - set(expected))
    mismatched = sorted(name for name in set(saved) & set(expected) if tuple(saved[name].shape) != tuple(expected[name].shape))
    if unexpected or mismatched or set(saved) != set(expected):
        raise RuntimeError(
            "Phase-2 training state is incompatible with this ARS wrapper: "
            f"unexpected={unexpected}, mismatched={mismatched}, missing={sorted(set(expected) - set(saved))}"
        )
    result = model.load_state_dict(saved, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Phase-2 state produced unexpected model keys: {result.unexpected_keys}")
    _restore_phase2_runtime_state(model, payload["runtime"])
    return payload


def _answer_aware_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    cfg: RSCConfig,
    calc_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Differentiable completion loss targeting calculation/outcome tokens and final answer tail (Engine v2)."""
    return per_example_answer_nll(
        logits,
        labels,
        answer_tail_tokens=getattr(cfg, "answer_tail_tokens", 0),
        answer_tail_weight=getattr(cfg, "answer_tail_weight", 1.0),
        calc_mask=calc_mask,
        calc_target_mode=getattr(cfg, "steer_loss_target", "calculation"),
        calc_weight=getattr(cfg, "calc_token_weight", 3.0),
    ).mean()


def split_steer_router(
    rows: list,
    router_fraction: float = 0.2,
    seed: int = 42,
) -> tuple:
    """Split training rows into steer-set and router-set (ISSUE-4 data split).

    SteerNets and Phase 2A use steer_data (80%).
    Router/Gate (Phase 1.5, 2B/C/D) use router_data (20%).
    This prevents the router from seeing the same examples used to train the
    SteerNets, reducing data leakage in the routing objective.

    Args:
        rows: Full list of training feature dicts.
        router_fraction: Fraction of rows reserved for router/gate training.
        seed: Random seed for reproducibility.

    Returns:
        (steer_data, router_data) tuple of lists.
    """
    import random as _random
    rng = _random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    split = int(len(rows) * (1 - router_fraction))
    steer_idx = indices[:split]
    router_idx = indices[split:]
    return [rows[i] for i in steer_idx], [rows[i] for i in router_idx]


def _trim_batch(
    ids: torch.Tensor,
    mask: torch.Tensor,
    lbls: torch.Tensor,
    shield: torch.Tensor = None,
    *extras: Optional[torch.Tensor],
) -> tuple:
    """Trim trailing padding from a right-padded micro-batch (P6 speed fix).

    GSM8K sequences are ~150-300 tokens but are padded to max_tok_len=512.
    Trimming to the longest real sequence in the batch cuts forward/backward
    compute by ~2x without changing gradient direction. Supports optional extra
    sequence tensors (e.g. calc_mask).

    Args:
        ids:    input_ids   [B, T]
        mask:   attn_mask   [B, T]
        lbls:   labels      [B, T]
        shield: prompt_shield [B, T] or None
        *extras: optional tensors with shape [B, T]

    Returns:
        (ids, mask, lbls, shield, *trimmed_extras) trimmed to length of longest non-padding token.
    """
    keep = int(mask.sum(dim=1).max().item())
    ids = ids[:, :keep]
    mask = mask[:, :keep]
    lbls = lbls[:, :keep]
    if shield is not None:
        shield = shield[:, :keep]
    if extras:
        trimmed_extras = [ex[:, :keep] if ex is not None else None for ex in extras]
        return (ids, mask, lbls, shield, *trimmed_extras)
    return ids, mask, lbls, shield


def gate_is_degenerate(intervention_diag: dict, std_tol: float = 1e-2, sat_pct: float = 95.0) -> bool:
    """Detect if learned token gates have saturated to a constant (alpha ≈ 1.0) everywhere (R3 fix)."""
    if not intervention_diag or "gate" not in intervention_diag:
        return False
    rows = [g for g in intervention_diag["gate"].values() if g.get("intervened_tokens", 0) > 0]
    return bool(rows) and all(
        g.get("token_std", 0.0) < std_tol and g.get("high_saturation_pct", 0.0) > sat_pct
        for g in rows
    )


def collect_pre_cap_ratio(model: torch.nn.Module, reset: bool = True) -> dict:
    """Collect SteerNet pre-cap norm saturation statistics across candidate layers (R2 / R4)."""
    from src.steering.steernet import pre_cap_report
    report = pre_cap_report(model, reset=reset)
    if report:
        return report
    # Fallback to last-forward mean if no accumulation was active
    ratios = {}
    steer_nets = getattr(model, "steer_nets", {})
    if isinstance(steer_nets, torch.nn.ModuleDict):
        for k, sn in steer_nets.items():
            ratio = getattr(sn, "_last_pre_cap_ratio", None)
            if ratio is not None:
                val = float(ratio.item()) if hasattr(ratio, "item") else float(ratio)
                ratios[int(k) if str(k).isdigit() else k] = {"mean": round(val, 4)}
    return ratios


def save_steer_weights(
    model, path: str, cfg: Optional[RSCConfig] = None, phase: str = "ars", prefixes=None,
    metadata: Optional[Dict[str, Any]] = None, extra_metadata: Optional[Dict[str, Any]] = None,
):
    """Save ARS tensors with an auditable phase/config/hash sidecar."""
    extra = metadata or extra_metadata
    model_cfg = getattr(model, "cfg", None) or cfg
    report = save_ars_checkpoint(model, path, phase=phase, cfg=cfg or model_cfg, prefixes=prefixes, extra_metadata=extra)
    print(f"✅ Saved {phase} ARS weights to {path} ({report['tensor_count']} tensors; manifest={report['manifest_path']})")


def load_steer_weights(model, path: str, device: Optional[torch.device] = None):
    """Load RSC-specific weights into model and attach provenance metadata."""
    if device is None:
        device = next(model.parameters()).device
    report = load_ars_checkpoint_strict(model, path, device=device)
    phase = (report.get("manifest") or {}).get("phase", "legacy-unmanifested")
    print(f"✅ Loaded strict ARS checkpoint from {path} (phase={phase}, tensors={report['saved_ars_tensors']})")

    # Compute checkpoint SHA-256 for CF table shard cache invalidation
    sha256 = None
    if os.path.exists(path):
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha256 = h.hexdigest()

    is_steer_ckpt = phase in ("ars", "s1_steernet", "legacy-unmanifested") or "steernet" in path.lower()
    if is_steer_ckpt or getattr(model, "_audit_ckpt_meta", None) is None:
        model._audit_ckpt_meta = {
            "sha256": sha256,
            "path": path,
            "phase": phase,
            "tensors": report.get("saved_ars_tensors", 0),
        }


def evaluate_loss(
    model, data, cfg: RSCConfig, device: torch.device, progress_desc: Optional[str] = None,
    answer_aware: bool = False,
) -> float:
    """Compute standard or final-answer-aware completion loss with visible progress."""
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_weight = 0

    starts = range(0, len(data), cfg.batch_size)
    # Acceptance runs are deliberately visible: deleting completed bars makes
    # the several-minute isolated-layer audit appear to have frozen.
    iterator = tqdm(starts, desc=progress_desc, unit="batch", leave=True) if progress_desc else starts
    for start in iterator:
        batch_idx = list(range(start, min(start + cfg.batch_size, len(data))))
        ids = torch.stack([data[i]["input_ids"] for i in batch_idx]).to(device)
        mask = torch.stack([data[i]["attention_mask"] for i in batch_idx]).to(device)
        lbls = torch.stack([data[i]["labels"] for i in batch_idx]).to(device)
        shield = None
        if "prompt_shield_mask" in data[batch_idx[0]]:
            shield = torch.stack([data[i]["prompt_shield_mask"] for i in batch_idx]).to(device)
        calc_mask = None
        if "calc_mask" in data[batch_idx[0]]:
            calc_mask = torch.stack([data[i]["calc_mask"] for i in batch_idx]).to(device)

        ids, mask, lbls, shield, calc_mask = _trim_batch(ids, mask, lbls, shield, calc_mask)

        if hasattr(model, "_attn_mask_cache"):
            model._attn_mask_cache = mask
        _set_batch_context(model, mask, lbls, shield)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(input_ids=ids, attention_mask=mask, labels=lbls, use_cache=False)

        n_tokens = (lbls != -100).sum().item()
        if n_tokens > 0:
            if answer_aware:
                batch_nll = _answer_aware_loss(out.logits, lbls, cfg, calc_mask=calc_mask).item()
                batch_weight = ids.shape[0]
            else:
                batch_nll = out.loss.item()
                batch_weight = n_tokens
            total_nll += batch_nll * batch_weight
            total_weight += batch_weight
            if progress_desc:
                iterator.set_postfix(nll=f"{total_nll / total_weight:.4f}")

    model.train(was_training)
    return total_nll / max(1, total_weight)


def _phase1_zero_shot_generation_probe(
    model,
    tokenizer,
    data,
    cfg: RSCConfig,
    device: torch.device,
    n_samples: int,
    layer: Optional[int] = None,
    gate_value: float = 1.0,
    progress_desc: Optional[str] = None,
) -> Dict[str, Any]:
    """Measure a Phase-1 route on held-out benchmark-format generation.

    Teacher-forced NLL is useful for optimisation but cannot certify an
    autoregressive intervention.  This deliberately reconstructs the exact
    zero-shot ``Question/Answer`` prompt used by GSM8K evaluation from the
    held-out rows and scores free generation with the production parser.
    It is a validation probe, never a test-set selector.
    """
    requested_rows = list(data)[: min(max(0, int(n_samples)), len(data))]
    rows = [row for row in requested_rows if row.get("question") and row.get("final_num") not in (None, "")]
    if len(rows) != len(requested_rows):
        print(
            "  ⚠️ Phase-1 held-out generation probe skipped malformed rows without raw "
            "question/final answer metadata; this run cannot establish capacity from tokenized prompts."
        )

    was_training = model.training
    old_override = model._routing_override
    old_mode = getattr(model, "_router_inference_mode", "argmax")
    correct = 0
    answer_present = 0
    details = []
    try:
        model.eval()
        model.set_router_inference_mode("argmax")
        if layer is None:
            model.set_routing_override("none")
            model.set_gate_freeze(False)
        else:
            model.set_routing_override("only", [int(layer)])
            model.set_gate_freeze(True, float(gate_value))

        token_lengths = []
        iterator = tqdm(rows, desc=progress_desc, unit="sample", leave=True) if progress_desc else rows
        for row in iterator:
            prompt = build_benchmark_prompt("gsm8k", row["question"], n_shot=0)
            response = generate_response(
                model, tokenizer, prompt, device,
                max_new_tokens=int(getattr(cfg, "phase1_generation_max_new_tokens", 256)),
            )
            token_lengths.append(len(tokenizer.encode(response, add_special_tokens=False)))
            score = evaluate_sample_answer("gsm8k", response, row["final_num"])
            correct += int(score["correct"])
            answer_present += int(score["has_answer"])
            details.append({
                "question": row["question"],
                "gold": row["final_num"],
                "response": response,
                "extracted": score["extracted"],
                "correct": bool(score["correct"]),
            })
            if progress_desc:
                iterator.set_postfix(acc=f"{correct / max(1, len(details)):.1%}", valid=len(details))
    finally:
        model.set_gate_freeze(True, 1.0)
        model.set_router_inference_mode(old_mode)
        if old_override is None:
            model.set_routing_override(None)
        else:
            model.set_routing_override(old_override[0], old_override[1])
        if was_training:
            model.train()

    total = len(details)
    import numpy as _np
    median_toks = float(_np.median(token_lengths)) if token_lengths else 0.0
    return {
        "accuracy": correct / max(1, total),
        "answer_present": answer_present / max(1, total),
        "median_tokens": median_toks,
        "n_samples": total,
        "details": details,
    }


def _phase1_probe_with_norm(
    model, tokenizer, rows, cfg: RSCConfig, device: torch.device, n_samples: int,
    layer: Optional[int], gate_value: float, progress_desc: str,
) -> Dict[str, Any]:
    """Generate a Phase-1 route and attach reset-scoped raw norm statistics."""
    collect_pre_cap_ratio(model, reset=True)
    probe = _phase1_zero_shot_generation_probe(
        model, tokenizer, rows, cfg, device, n_samples, layer=layer,
        gate_value=gate_value, progress_desc=progress_desc,
    )
    norms = collect_pre_cap_ratio(model, reset=True)
    probe["pre_cap_norm"] = norms.get(int(layer), {}) if layer is not None else {}
    return probe


def _phase1_norm_metrics(result: Dict[str, Any], gate_scale: float) -> Tuple[float, float, float]:
    """Return raw, post-cap, and gate-scaled post-cap p95 ratios.

    Raw ``q`` describes how often the trainable residual enters the soft
    saturation envelope.  It is useful telemetry, but safety is determined by
    the bounded output which is actually injected into the backbone.
    """
    norm = result.get("pre_cap_norm", {})
    raw_q_p95 = float(norm.get("p95", norm.get("mean", 0.0)))
    post_cap_q_p95 = float(norm.get("post_cap_p95", post_cap_ratio_from_pre_cap_q(raw_q_p95)))
    return raw_q_p95, post_cap_q_p95, float(gate_scale) * post_cap_q_p95


def _phase1_pair_is_safe(
    result: Dict[str, Any], base_median_tokens: float, cfg: RSCConfig, gate_scale: float = 1.0,
) -> Tuple[bool, float, float]:
    """Return whether a generation pair preserves CoT and realized displacement safety.

    The returned q remains the raw pre-cap p95 for backwards-compatible
    diagnostics.  The acceptance decision intentionally uses post-cap q.
    """
    completion_ratio = float(result.get("median_tokens", 0.0)) / max(1.0, base_median_tokens)
    q_p95, _post_cap_q_p95, effective_q_p95 = _phase1_norm_metrics(result, gate_scale)
    safe = (
        completion_ratio >= float(getattr(cfg, "phase1_min_completion_len_ratio", 0.85))
        and effective_q_p95 <= float(getattr(cfg, "phase1_max_effective_generation_p95_norm_ratio", 1.0))
    )
    return safe, completion_ratio, q_p95


def _phase1_calibration_rank_key(item: Tuple[Dict[str, Any], Dict[str, Any]]) -> Tuple[float, ...]:
    """Rank safe Phase-1 pairs with every selection field made explicit."""
    row, pair = item
    return (
        float(pair["accuracy"]), float(pair["answer_present"]), -float(row["val_nll"]),
        -float(pair.get("effective_post_cap_q_p95", pair["pre_cap_q_p95"])), -float(pair["gate_scale"]),
    )


def _phase1_confirmation_verdict(
    *,
    confirmation_gain: float,
    calibration_gain: float,
    combined_gain: float,
    confirmation_n: int,
    safe: bool,
    nll_safe: bool,
    cfg: RSCConfig,
) -> Tuple[bool, bool, List[str]]:
    """Apply strict Phase-1 gates and record compact-screen noise tolerance.

    A confirmation result remains the primary promotion criterion.  The
    Compact-screen noise is recorded for diagnosis, but it cannot promote a
    fixed control.  Promotion requires a strictly positive independently
    measured confirmation gain; calibration wins never substitute for it.
    """
    minimum_gain = float(getattr(
        cfg, "phase1_min_validation_accuracy_gain", cfg.phase1_min_train_accuracy_gain
    ))
    # Accuracy is calculated as floating-point ratios; retain the intended
    # exact binomial boundary (for example 12/20 - 13/20 == -1/20).
    epsilon = 1e-12
    combined_nonregressing = combined_gain >= -epsilon
    direct_accuracy_pass = confirmation_gain > max(0.0, minimum_gain) + epsilon and (
        not bool(getattr(cfg, "phase1_require_nonnegative_combined_gain", True))
        or combined_nonregressing
    )
    one_question = 1.0 / max(1, int(confirmation_n))
    allow_nonregressing = bool(getattr(cfg, "phase1_allow_nonregressing_confirmation", False))
    if allow_nonregressing:
        accuracy_pass = (
            (confirmation_gain >= -epsilon and combined_nonregressing)
            or (
                confirmation_n <= 32
                and confirmation_gain >= -2 * one_question - epsilon
                and calibration_gain > epsilon
                and combined_nonregressing
            )
            or direct_accuracy_pass
        )
        noise_tolerated = (
            confirmation_n <= 32
            and confirmation_gain >= -2 * one_question - epsilon
            and calibration_gain > epsilon
            and combined_nonregressing
        )
    else:
        accuracy_pass = direct_accuracy_pass
        noise_tolerated = bool(
            getattr(cfg, "phase1_allow_confirmation_noise_tolerance", True)
            and confirmation_n <= 32
            and confirmation_gain >= -one_question - epsilon
            and calibration_gain >= 0.10 - epsilon
            and combined_nonregressing
        )
    reasons = []
    if not safe:
        reasons.append("confirmation_safety")
    if not nll_safe:
        reasons.append("nll_reduction")
    if not accuracy_pass:
        reasons.append("confirmation_accuracy")
    return safe and nll_safe and accuracy_pass, noise_tolerated, reasons


def phase1_steernet_acceptance(model, tokenizer, train_data, eval_data, cfg: RSCConfig, device: torch.device) -> Dict[str, Any]:
    """Certify one independently useful, generation-safe Phase-1 intervention.

    The compact calibration partition jointly searches every ``(layer, alpha)``
    pair.  A disjoint confirmation partition then decides promotion.  This
    avoids both teacher-forced selection bias and the former alpha=1-only
    screen that discarded safe low-intensity residuals before measuring them.
    """
    train_n = min(int(cfg.phase1_acceptance_examples), len(train_data))
    val_n = min(int(cfg.phase1_acceptance_examples), len(eval_data))
    small_run_waiver = _small_run_waiver(val_n, cfg)
    train_probe = list(train_data)[:train_n]
    val_probe = list(eval_data)[:val_n]
    screen_n = min(max(0, int(getattr(cfg, "phase1_generation_screen_examples", 0))), val_n)
    confirmation_n = min(
        max(0, int(getattr(cfg, "phase1_generation_confirmation_examples", 16))),
        max(0, val_n - screen_n),
    )
    calibration_rows = val_probe[:screen_n]
    confirmation_rows = val_probe[screen_n:screen_n + confirmation_n]
    old_override = model._routing_override
    old_layers = getattr(cfg, "fixed_steer_eval_layers", None)
    old_gate_scale = float(getattr(cfg, "fixed_steer_eval_gate_scale", 1.0))
    started_at = time.monotonic()
    try:
        print("\n" + "=" * 80)
        print("PHASE 1 ACCEPTANCE AUDIT: Generation-Safe SteerNet Capacity")
        print(
            f"         NLL: {train_n} train / {val_n} validation; joint (layer, alpha) "
            f"calibration: {screen_n}; disjoint confirmation: {confirmation_n}."
        )
        print("         Router and learned gate are disabled during this audit.")
        if small_run_waiver:
            print("         ⚠️ INCONCLUSIVE DEBUG RUN: validation set is below the reliability floor.")
        print("=" * 80)
        if screen_n <= 0 or confirmation_n <= 0:
            raise RuntimeError(
                "Phase-1 acceptance requires non-empty disjoint calibration and confirmation generation partitions."
            )

        model.set_routing_override("none")
        model.set_gate_freeze(False)
        print("  [1/4] Baseline answer-weighted teacher-forced NLL...")
        base_train_nll = evaluate_loss(model, train_probe, cfg, device, "Acceptance baseline train NLL", answer_aware=True)
        base_val_nll = evaluate_loss(model, val_probe, cfg, device, "Acceptance baseline val NLL", answer_aware=True)

        candidates = []
        print(f"  [2/4] Single-layer NLL sweep across {len(model.target_layers)} SteerNets...")
        for layer in model.target_layers:
            model.set_routing_override("only", [layer])
            model.set_gate_freeze(True, 1.0)
            row = {
                "layer": int(layer),
                "train_nll": evaluate_loss(model, train_probe, cfg, device, f"Acceptance L{layer} train NLL", answer_aware=True),
                "val_nll": evaluate_loss(model, val_probe, cfg, device, f"Acceptance L{layer} val NLL", answer_aware=True),
                "scale_screen": [],
            }
            candidates.append(row)
            print(f"        L{layer}: train_nll={row['train_nll']:.4f} val_nll={row['val_nll']:.4f}")

        scale_grid = sorted({min(1.0, max(0.0, float(scale))) for scale in getattr(cfg, "phase1_gate_scale_grid", (1.0,))})
        if not scale_grid:
            raise RuntimeError("phase1_gate_scale_grid must contain at least one scale.")
        initial_n = min(
            screen_n,
            max(1, int(getattr(cfg, "phase1_successive_halving_examples", screen_n))),
        )
        initial_rows = calibration_rows[:initial_n]
        print(
            f"  [3/4] Successive-halving calibration: {len(candidates)} layers × {scale_grid} "
            f"({initial_n} initial each; promote one safe route per depth band to {screen_n})."
        )
        base_initial = _phase1_probe_with_norm(
            model, tokenizer, initial_rows, cfg, device, initial_n, None, 1.0,
            "Acceptance baseline initial calibration",
        )
        initial_median_tokens = max(1.0, float(base_initial.get("median_tokens", 0.0)))
        initial_pairs = []
        for row in candidates:
            row["initial_scale_screen"] = []
            for scale in scale_grid:
                screen = _phase1_probe_with_norm(
                    model, tokenizer, initial_rows, cfg, device, initial_n, row["layer"], scale,
                    f"Acceptance initial L{row['layer']} alpha={scale:.2f}",
                )
                safe, completion_ratio, q_p95 = _phase1_pair_is_safe(
                    screen, initial_median_tokens, cfg, gate_scale=scale,
                )
                _raw_q, post_cap_q_p95, effective_q_p95 = _phase1_norm_metrics(screen, scale)
                pair = {
                    "layer": row["layer"], "gate_scale": scale,
                    "accuracy": screen["accuracy"], "answer_present": screen["answer_present"],
                    "median_tokens": screen.get("median_tokens", 0.0),
                    "completion_len_ratio": completion_ratio, "pre_cap_q_p95": q_p95,
                    "post_cap_q_p95": post_cap_q_p95,
                    "effective_post_cap_q_p95": effective_q_p95,
                    "effective_pre_cap_q_p95": scale * q_p95,
                    "pre_cap_norm": screen.get("pre_cap_norm", {}), "safe": safe,
                }
                row["initial_scale_screen"].append(pair)
                if safe:
                    initial_pairs.append((row, pair))

        # A one-per-band finalist set controls multiplicity and preserves the
        # early/middle/late curriculum hypothesis.  The full calibration split
        # remains unseen until after this deterministic initial selection.
        initial_ranked = sorted(initial_pairs, key=_phase1_calibration_rank_key, reverse=True)
        finalists = []
        selected_bands = []
        candidates_per_band = max(1, int(getattr(cfg, "phase1b_candidates_per_band", 2)))
        for band in ("early", "middle", "late"):
            seen_band_layers = set()
            for item in initial_ranked:
                layer_id = int(item[0]["layer"])
                if _phase1_depth_band(layer_id, model.target_layers) == band and layer_id not in seen_band_layers:
                    finalists.append(item)
                    seen_band_layers.add(layer_id)
                    if band not in selected_bands:
                        selected_bands.append(band)
                    if len(seen_band_layers) >= candidates_per_band:
                        break
        if not finalists:
            rejected_layers = [row["layer"] for row in candidates]
            warning = "  ⚠️ Phase-1 acceptance FAILED: no initial (layer, alpha) pair met CoT and realized-displacement safety."
            print(warning)
            return {
                "candidates": candidates, "passed": False, "status": "FAIL_NO_SAFE_PAIR",
                "truncation_rejected_layers": rejected_layers, "confirmation_trials": [],
            }

        if initial_n == screen_n:
            base_screen = base_initial
        else:
            base_screen = _phase1_probe_with_norm(
                model, tokenizer, calibration_rows, cfg, device, screen_n, None, 1.0,
                "Acceptance baseline calibration",
            )
        base_screen_acc = base_screen["accuracy"]
        base_median_tokens = max(1.0, float(base_screen.get("median_tokens", 0.0)))
        print(
            f"        promoted bands={selected_bands}; baseline calibration: "
            f"acc={base_screen_acc:.1%} median_tokens={base_median_tokens:.0f}"
        )
        eligible_pairs = []
        for row, initial_pair in finalists:
            scale = float(initial_pair["gate_scale"])
            if initial_n == screen_n:
                pair = dict(initial_pair)
                safe = bool(pair["safe"])
                completion_ratio = float(pair["completion_len_ratio"])
                q_p95 = float(pair["pre_cap_q_p95"])
                post_cap_q_p95 = float(pair["post_cap_q_p95"])
                effective_q_p95 = float(pair["effective_post_cap_q_p95"])
            else:
                screen = _phase1_probe_with_norm(
                    model, tokenizer, calibration_rows, cfg, device, screen_n, row["layer"], scale,
                    f"Acceptance finalist L{row['layer']} alpha={scale:.2f}",
                )
                safe, completion_ratio, q_p95 = _phase1_pair_is_safe(
                    screen, base_median_tokens, cfg, gate_scale=scale,
                )
                _raw_q, post_cap_q_p95, effective_q_p95 = _phase1_norm_metrics(screen, scale)
                pair = {
                    "layer": row["layer"], "gate_scale": scale,
                    "accuracy": screen["accuracy"], "answer_present": screen["answer_present"],
                    "median_tokens": screen.get("median_tokens", 0.0),
                    "completion_len_ratio": completion_ratio, "pre_cap_q_p95": q_p95,
                    "post_cap_q_p95": post_cap_q_p95,
                    "effective_post_cap_q_p95": effective_q_p95,
                    "effective_pre_cap_q_p95": scale * q_p95,
                    "pre_cap_norm": screen.get("pre_cap_norm", {}), "safe": safe,
                }
            row["scale_screen"].append(pair)
            if safe:
                eligible_pairs.append((row, pair))
            status = "" if safe else " ⚠️ REJECTED"
            print(
                f"        finalist L{row['layer']} alpha={scale:.2f}: acc={pair['accuracy']:.1%} "
                f"tokens={pair['median_tokens']:.0f} (ratio {completion_ratio:.2f}) "
                f"raw_q95={q_p95:.2f} postcap_q95={post_cap_q_p95:.2f} "
                f"effective_postcap_q95={effective_q_p95:.2f}{status}"
            )

        # A non-finalist was not disproven: it simply lost the inexpensive
        # first-round screen to a stronger route in its depth band.  Report
        # only routes that actually failed a safety measurement as rejected.
        rejected_layers = [
            row["layer"] for row in candidates
            if row.get("initial_scale_screen")
            and not any(pair["safe"] for pair in row["initial_scale_screen"])
        ]
        if not eligible_pairs:
            warning = "  ⚠️ Phase-1 acceptance FAILED: no (layer, alpha) pair met CoT and norm safety constraints."
            print(warning)
            # ``train_rsc`` owns the fail-closed transition so it can persist
            # this diagnostic report before it stops downstream phases.
            return {
                "candidates": candidates,
                "passed": False,
                "status": "FAIL_NO_SAFE_PAIR",
                "truncation_rejected_layers": rejected_layers,
                "confirmation_trials": [],
            }

        # Accuracy drives calibration selection; answer presence, NLL,
        # realized effective post-cap displacement and lower intensity are
        # deterministic secondary tie-breakers.
        ranked_pairs = sorted(eligible_pairs, key=_phase1_calibration_rank_key, reverse=True)
        calibration_ranking = []
        print("        eligible calibration ranking (acc, answer-present, val-NLL, CoT, q95, alpha):")
        for rank, (row, pair) in enumerate(ranked_pairs, start=1):
            ranked = {
                "rank": rank, "layer": row["layer"], "gate_scale": pair["gate_scale"],
                "accuracy": pair["accuracy"], "answer_present": pair["answer_present"],
                "val_nll": row["val_nll"], "completion_len_ratio": pair["completion_len_ratio"],
                "pre_cap_q_p95": pair["pre_cap_q_p95"],
                "post_cap_q_p95": pair.get("post_cap_q_p95"),
                "effective_post_cap_q_p95": pair.get("effective_post_cap_q_p95"),
                "safe": pair.get("safe", True),
                "selection_key": list(_phase1_calibration_rank_key((row, pair))),
            }
            calibration_ranking.append(ranked)
            print(
                f"          #{rank}: L{row['layer']} alpha={pair['gate_scale']:.2f} "
                f"acc={pair['accuracy']:.1%} present={pair['answer_present']:.1%} "
                f"val_nll={row['val_nll']:.4f} CoT={pair['completion_len_ratio']:.2f} "
                f"raw_q95={pair['pre_cap_q_p95']:.2f} "
                f"effective_postcap_q95={pair.get('effective_post_cap_q_p95', 0.0):.2f}"
            )
        selection_reason = (
            "accuracy > answer_present > lower_val_nll > "
            "lower_effective_postcap_q95 > lower_alpha"
        )
        max_confirmation_candidates = max(1, int(getattr(cfg, "phase1_max_confirmation_candidates", 3)))
        confirmation_candidates = ranked_pairs[:max_confirmation_candidates]
        print(
            f"  [4/4] Disjoint confirmation: up to {len(confirmation_candidates)} ranked candidates "
            f"({confirmation_n} each)..."
        )
        base_confirmation = _phase1_probe_with_norm(
            model, tokenizer, confirmation_rows, cfg, device, confirmation_n, None, 1.0,
            "Acceptance baseline confirmation",
        )
        base_val_acc = base_confirmation["accuracy"]
        confirmation_trials = []
        accepted = None
        for confirmation_rank, (candidate, calibration) in enumerate(confirmation_candidates, start=1):
            print(
                f"        confirmation candidate #{confirmation_rank}: L{candidate['layer']} "
                f"alpha={calibration['gate_scale']:.2f}..."
            )
            candidate_confirmation = _phase1_probe_with_norm(
                model, tokenizer, confirmation_rows, cfg, device, confirmation_n,
                candidate["layer"], calibration["gate_scale"],
                f"Acceptance L{candidate['layer']} alpha={calibration['gate_scale']:.2f} confirmation",
            )
            confirm_safe, confirm_ratio, confirm_q_p95 = _phase1_pair_is_safe(
                candidate_confirmation, max(1.0, float(base_confirmation.get("median_tokens", 0.0))), cfg,
                gate_scale=calibration["gate_scale"],
            )
            _confirm_raw_q, confirm_post_cap_q_p95, confirm_effective_q_p95 = _phase1_norm_metrics(
                candidate_confirmation, calibration["gate_scale"],
            )
            confirmation_gain = float(candidate_confirmation["accuracy"]) - float(base_val_acc)
            calibration_gain = float(calibration["accuracy"]) - float(base_screen_acc)
            combined_gain = (
                (float(calibration["accuracy"]) * screen_n + float(candidate_confirmation["accuracy"]) * confirmation_n)
                - (float(base_screen_acc) * screen_n + float(base_val_acc) * confirmation_n)
            ) / max(1, screen_n + confirmation_n)
            nll_reduction = (base_train_nll - candidate["train_nll"]) / max(base_train_nll, 1e-8)
            val_nll_reduction = (base_val_nll - candidate["val_nll"]) / max(base_val_nll, 1e-8)
            nll_safe = (
                nll_reduction >= float(cfg.phase1_min_train_nll_reduction)
                and val_nll_reduction >= float(cfg.phase1_min_val_nll_reduction)
            )
            passed, noise_tolerated, failure_reasons = _phase1_confirmation_verdict(
                confirmation_gain=confirmation_gain,
                calibration_gain=calibration_gain,
                combined_gain=combined_gain,
                confirmation_n=confirmation_n,
                safe=confirm_safe,
                nll_safe=nll_safe,
                cfg=cfg,
            )
            trial = {
                "rank": confirmation_rank,
                "layer": candidate["layer"],
                "gate_scale": calibration["gate_scale"],
                "calibration": calibration,
                "candidate": candidate_confirmation,
                "confirmation_gain": confirmation_gain,
                "calibration_gain": calibration_gain,
                "combined_accuracy_gain": combined_gain,
                "completion_len_ratio": confirm_ratio,
                "pre_cap_q_p95": confirm_q_p95,
                "post_cap_q_p95": confirm_post_cap_q_p95,
                "effective_post_cap_q_p95": confirm_effective_q_p95,
                "effective_pre_cap_q_p95": float(calibration["gate_scale"]) * confirm_q_p95,
                "safe": confirm_safe,
                "train_relative_nll_reduction": nll_reduction,
                "val_relative_nll_reduction": val_nll_reduction,
                "nll_safe": nll_safe,
                "noise_tolerance_used": noise_tolerated,
                "passed": passed,
                "failure_reasons": failure_reasons,
            }
            confirmation_trials.append(trial)
            verdict = "PASS" if passed else f"FAIL ({', '.join(failure_reasons)})"
            tolerance = " [one-question tolerance]" if noise_tolerated else ""
            print(
                f"          {verdict}{tolerance}: confirmation Δ={confirmation_gain * 100:+.1f}pp, "
                f"calibration Δ={calibration_gain * 100:+.1f}pp, combined Δ={combined_gain * 100:+.1f}pp, "
                f"CoT={confirm_ratio:.2f}, raw_q95={confirm_q_p95:.2f}, "
                f"effective_postcap_q95={confirm_effective_q_p95:.2f}"
            )
            if passed:
                accepted = (candidate, calibration, candidate_confirmation, trial)
                break

        # Preserve legacy report keys even on a failed audit: they describe
        # the first ranked confirmation attempt, while confirmation_trials
        # provides the complete fallback record.
        selected_candidate, selected_calibration, selected_confirmation, selected_trial = (
            accepted if accepted is not None else (
                confirmation_candidates[0][0], confirmation_candidates[0][1],
                confirmation_trials[0]["candidate"], confirmation_trials[0],
            )
        )
        if accepted is not None:
            cfg.fixed_steer_eval_layers = [selected_candidate["layer"]]
            cfg.fixed_steer_eval_gate_scale = float(selected_calibration["gate_scale"])
        report = {
            "baseline_train_nll": base_train_nll, "baseline_val_nll": base_val_nll,
            "baseline_screen_accuracy": base_screen_acc, "baseline_validation_accuracy": base_val_acc,
            "candidates": candidates, "selected_layer": selected_candidate["layer"],
            "selected_validation_accuracy": selected_confirmation["accuracy"],
            "accuracy_comparison": "disjoint_generation_confirmation",
            "accuracy_comparison_baseline": base_val_acc,
            "fixed_steer_gate_scale": float(selected_calibration["gate_scale"]),
            "gate_scale_candidates": selected_candidate["scale_screen"],
            "selected_calibration": selected_calibration,
            "calibration_ranking": calibration_ranking,
            "successive_halving": {
                "initial_examples": initial_n,
                "full_calibration_examples": screen_n,
                "promoted_depth_bands": selected_bands,
                "finalist_count": len(finalists),
            },
            "selection_reason": selection_reason,
            "confirmation": {
                "baseline": base_confirmation, "candidate": selected_confirmation,
                "completion_len_ratio": selected_trial["completion_len_ratio"],
                "pre_cap_q_p95": selected_trial["pre_cap_q_p95"],
                "safe": selected_trial["safe"],
            },
            "confirmation_trials": confirmation_trials,
            "train_relative_nll_reduction": selected_trial["train_relative_nll_reduction"],
            "val_relative_nll_reduction": selected_trial["val_relative_nll_reduction"],
            "validation_accuracy_gain": selected_trial["confirmation_gain"], "validation_examples": val_n,
            "calibration_examples": screen_n, "confirmation_examples": confirmation_n,
            "reliability_waived": small_run_waiver, "truncation_rejected_layers": rejected_layers,
            "passed": accepted is not None,
        }
        print(
            "  Phase-1 acceptance: "
            f"L{selected_candidate['layer']} alpha={selected_calibration['gate_scale']:.2f} | train/val NLL Δ="
            f"{selected_trial['train_relative_nll_reduction'] * 100:+.1f}%/"
            f"{selected_trial['val_relative_nll_reduction'] * 100:+.1f}% | "
            f"confirmation Δ={selected_trial['confirmation_gain'] * 100:+.1f}pp | "
            f"CoT={selected_trial['completion_len_ratio']:.2f} raw_q95={selected_trial['pre_cap_q_p95']:.2f} "
            f"effective_postcap_q95={selected_trial['effective_post_cap_q_p95']:.2f}"
        )
        print(f"  Acceptance audit finished in {(time.monotonic() - started_at) / 60:.1f} min.")
        if not report["passed"] and small_run_waiver:
            report["status"] = "INCONCLUSIVE"
        elif not report["passed"]:
            report["status"] = "FAIL"
            warning = "  ⚠️ Phase-1 SteerNet acceptance FAILED; failing closed before routing."
            print(warning)
        else:
            report["status"] = "PASS"
        return report
    finally:
        model.set_gate_freeze(True, 1.0)
        if old_override is None:
            model.set_routing_override(None)
        else:
            model.set_routing_override(old_override[0], old_override[1])
        if not locals().get("accepted", None):
            cfg.fixed_steer_eval_layers = old_layers
            cfg.fixed_steer_eval_gate_scale = old_gate_scale


def audit_ars_intervention_health(model, eval_data, cfg: RSCConfig, device: torch.device) -> Dict[str, Any]:
    """Fail closed for research-sized evaluations with routing/gate collapse.

    This is evaluated with the same deterministic argmax route and prompt
    shielding used by generation.  It is deliberately a health gate, not a
    benchmark: test labels never choose a route or tune a threshold.  Tiny
    smoke runs report the same diagnostics as INCONCLUSIVE instead of making
    claims from an unreliable holdout.
    """
    previous_mode = getattr(model, "_router_inference_mode", "argmax")
    previous_override = model._routing_override
    model.reset_routing_diagnostics()
    model.eval()
    model.set_router_inference_mode("argmax")
    model.set_routing_override(None)
    model.set_gate_freeze(False)
    n = min(int(cfg.intervention_audit_examples), len(eval_data))
    small_run_waiver = _small_run_waiver(n, cfg)
    try:
        with torch.no_grad():
            for i in range(n):
                row = eval_data[i]
                ids = row["input_ids"].unsqueeze(0).to(device)
                mask = row["attention_mask"].unsqueeze(0).to(device)
                labels = row["labels"].unsqueeze(0).to(device)
                shield = row.get("prompt_shield_mask")
                shield = shield.unsqueeze(0).to(device) if shield is not None else None
                model._attn_mask_cache = mask
                _set_batch_context(model, mask, labels, shield)
                _ = model(input_ids=ids, attention_mask=mask, use_cache=False)
        report = model.intervention_diagnostics()
        failures = []
        k_dist = report["realized_k_distribution"]
        if cfg.router_require_eval_k0 and k_dist.get(0, 0.0) <= 0.0:
            failures.append("K=0 was never selected")
        k_ge2_rate = sum(float(rate) for k, rate in k_dist.items() if int(k) >= 2)
        report["realized_k_ge2_rate"] = k_ge2_rate
        if bool(getattr(cfg, "router_require_eval_k_ge2", True)) and k_ge2_rate < float(getattr(cfg, "router_min_eval_k_ge2_rate", 0.02)):
            failures.append(
                f"K>=2 rate={k_ge2_rate:.3f} < {float(getattr(cfg, 'router_min_eval_k_ge2_rate', 0.02)):.3f}"
            )
        if report["realized_k_support"] < int(cfg.router_min_eval_k_support):
            failures.append(f"K support={report['realized_k_support']} < {cfg.router_min_eval_k_support}")
        if report["realized_k_entropy"] < float(cfg.router_min_eval_k_entropy):
            failures.append(f"K entropy={report['realized_k_entropy']:.3f} < {cfg.router_min_eval_k_entropy}")
        # Active means the union of deterministic selections across the whole
        # preflight batch, not an impossible requirement that every prompt
        # activate every certified layer.
        if report["active_layer_count"] < int(cfg.router_min_eval_active_layers):
            failures.append(f"active layers={report['active_layer_count']} < {cfg.router_min_eval_active_layers}")
        min_bands = int(getattr(cfg, "router_min_eval_active_depth_bands", 2))
        if report.get("active_depth_band_count", 0) < min_bands:
            failures.append(
                f"active depth bands={report.get('active_depth_band_count', 0)} < {min_bands}"
            )
        if report["layer_selection_entropy"] < float(cfg.router_min_eval_layer_entropy):
            failures.append(f"layer entropy={report['layer_selection_entropy']:.3f} < {cfg.router_min_eval_layer_entropy}")
        min_route_entropy = float(getattr(cfg, "router_min_eval_route_tuple_entropy", 0.0))
        if report.get("route_tuple_entropy", 0.0) < min_route_entropy:
            failures.append(
                f"route-tuple entropy={report.get('route_tuple_entropy', 0.0):.3f} < {min_route_entropy:.3f}"
            )
        if report.get("effective_layer_count", 0) < int(cfg.router_min_eval_active_layers):
            failures.append(
                f"effective layers={report.get('effective_layer_count', 0)} < {cfg.router_min_eval_active_layers}"
            )
        active_gates = [row for row in report["gate"].values() if row["intervened_tokens"] > 0]
        if not active_gates:
            failures.append("no gate received intervened tokens")
        else:
            mean_seq_std = sum(row["mean_sequence_std"] for row in active_gates) / len(active_gates)
            max_high_sat = max(row["high_saturation_pct"] for row in active_gates)
            report["mean_gate_sequence_std"] = round(mean_seq_std, 4)
            report["max_gate_high_saturation_pct"] = round(max_high_sat, 2)
            if mean_seq_std < float(cfg.gate_min_eval_sequence_std):
                failures.append(f"gate sequence std={mean_seq_std:.4f} < {cfg.gate_min_eval_sequence_std}")
            if max_high_sat > float(cfg.gate_max_eval_high_saturation_pct):
                failures.append(f"gate high saturation={max_high_sat:.1f}% > {cfg.gate_max_eval_high_saturation_pct}%")
        report["passed"] = not failures
        report["failures"] = failures
        report["validation_examples"] = n
        report["reliability_waived"] = small_run_waiver
        report["status"] = "PASS" if not failures else ("INCONCLUSIVE" if small_run_waiver else "FAIL_CONTINUED")
        print(f"  ARS intervention audit: {report}")
        if failures and small_run_waiver:
            print(
                "  ⚠️ ARS health is INCONCLUSIVE, not PASS: this tiny validation holdout cannot establish "
                "K/layer/gate diversity. Continuing only as a pipeline smoke test."
            )
        elif failures:
            warning = (
                "  ⚠️ ARS intervention health FAILED. Final checkpoint is retained for diagnosis only: "
                + "; ".join(failures)
            )
            print(warning)
            if not getattr(cfg, "continue_on_diagnostic_failure", True):
                raise RuntimeError(warning)
        return report
    finally:
        model.set_router_inference_mode(previous_mode)
        if previous_override is None:
            model.set_routing_override(None)
        else:
            model.set_routing_override(previous_override[0], previous_override[1])


def audit_router_policy_health(model, eval_data, cfg: RSCConfig, device: torch.device) -> Dict[str, Any]:
    """Verify deterministic K/layer diversity before expensive gate training.

    Bootstrap sampling can look diverse even if every deployment argmax is the
    same K and triplet.  This preflight runs the deployment policy itself, so
    a collapsed router is rejected before Phase 2B consumes hours of GPU time.
    """
    previous_mode = getattr(model, "_router_inference_mode", "argmax")
    previous_override = model._routing_override
    model.reset_routing_diagnostics()
    model.eval()
    model.set_router_inference_mode("argmax")
    model.set_routing_override(None)
    n = min(int(getattr(cfg, "router_preflight_examples", 128)), len(eval_data))
    waiver = _small_run_waiver(n, cfg)
    try:
        with torch.no_grad():
            for i in range(n):
                row = eval_data[i]
                ids = row["input_ids"].unsqueeze(0).to(device)
                mask = row["attention_mask"].unsqueeze(0).to(device)
                labels = row["labels"].unsqueeze(0).to(device)
                shield = row.get("prompt_shield_mask")
                shield = shield.unsqueeze(0).to(device) if shield is not None else None
                model._attn_mask_cache = mask
                _set_batch_context(model, mask, labels, shield)
                _ = model(input_ids=ids, attention_mask=mask, use_cache=False)
        report = model.intervention_diagnostics()
        failures = []
        k_dist = report["realized_k_distribution"]
        if cfg.router_require_eval_k0 and k_dist.get(0, 0.0) <= 0.0:
            failures.append("K=0 was never selected")
        k_ge2_rate = sum(float(rate) for k, rate in k_dist.items() if int(k) >= 2)
        report["realized_k_ge2_rate"] = k_ge2_rate
        if bool(getattr(cfg, "router_require_eval_k_ge2", True)) and k_ge2_rate < float(getattr(cfg, "router_min_eval_k_ge2_rate", 0.02)):
            failures.append(
                f"K>=2 rate={k_ge2_rate:.3f} < {float(getattr(cfg, 'router_min_eval_k_ge2_rate', 0.02)):.3f}"
            )
        if report["realized_k_support"] < int(cfg.router_min_eval_k_support):
            failures.append(f"K support={report['realized_k_support']} < {cfg.router_min_eval_k_support}")
        if report["realized_k_entropy"] < float(cfg.router_min_eval_k_entropy):
            failures.append(f"K entropy={report['realized_k_entropy']:.3f} < {cfg.router_min_eval_k_entropy}")
        if report["active_layer_count"] < int(cfg.router_min_eval_active_layers):
            failures.append(f"active layers={report['active_layer_count']} < {cfg.router_min_eval_active_layers}")
        # This is intentionally a batch-union property.  A useful policy may
        # select an early+late route on only some prompts; requiring every
        # prompt to use every layer would reject that valid specialization.
        min_bands = int(getattr(cfg, "router_min_eval_active_depth_bands", 2))
        if report.get("active_depth_band_count", 0) < min_bands:
            failures.append(
                f"active depth bands={report.get('active_depth_band_count', 0)} < {min_bands}"
            )
        if report["layer_selection_entropy"] < float(cfg.router_min_eval_layer_entropy):
            failures.append(f"layer entropy={report['layer_selection_entropy']:.3f} < {cfg.router_min_eval_layer_entropy}")
        report.update({
            "validation_examples": n,
            "reliability_waived": waiver,
            "passed": not failures,
            "failures": failures,
            "status": "PASS" if not failures else ("INCONCLUSIVE" if waiver else "FAIL"),
        })
        print(f"  Router deployment preflight: {report}")
        if failures and not waiver:
            warning = (
                "  ⚠️ Router deployment preflight FAILED. Failing closed before Phase 2: "
                + "; ".join(failures)
            )
            print(warning)
            # The top-level curriculum owns termination/rollback.  Raising
            # here used to bypass report persistence and restore logic in
            # ``train_rsc``, leaving an interrupted Kaggle session with no
            # clean diagnostic artifact.
        return report
    finally:
        model.set_router_inference_mode(previous_mode)
        if previous_override is None:
            model.set_routing_override(None)
        else:
            model.set_routing_override(previous_override[0], previous_override[1])


def _evaluate_and_maybe_checkpoint(
    model, tokenizer, eval_data, cfg: RSCConfig, device: torch.device, save_path: str,
    best_acc: float, best_val: float, patience_ctr: int, train_router: bool, label: str,
):
    """Evaluates val CE loss + accuracy and saves checkpoint if improved."""
    if train_router:
        model.reset_routing_diagnostics()
    val_loss = evaluate_loss(model, eval_data, cfg, device, answer_aware=True)
    acc = epoch_end_accuracy(model, tokenizer, eval_data, device, n_samples=cfg.accuracy_eval_n, seed=cfg.seed)
    # A generation sample is a noisy checkpoint signal.  It may break ties,
    # but it must never replace a checkpoint whose teacher-forced validation
    # loss is materially better (the old rule did exactly that in Phase 2B).
    val_improved = val_loss < best_val - getattr(cfg, "early_stop_delta", 1e-4)
    acc_improved = acc > best_acc + 0.005
    val_tied = abs(val_loss - best_val) <= getattr(cfg, "early_stop_delta", 1e-4)
    improved = val_improved or (getattr(cfg, "use_accuracy_checkpoint", True) and val_tied and acc_improved)
    note = ""
    if train_router:
        diag = model.routing_diagnostics()
        n_candidates = max(1, len(diag))
        dead = sum(1 for s in diag.values() if s["selected_forwards"] / max(1, s["seen_forwards"]) < 0.02)
        veto_frac = getattr(cfg, "router_dead_layer_veto_frac", 1.0)
        dead_ratio = dead / n_candidates
        within = not (dead_ratio >= veto_frac)  # 1.0 => warn when 100% of layers are dead
        manifest = get_manifest()
        manifest.diagnostic(
            "W_DEAD_LAYERS",
            ok=within,
            message=f"Dead layer ratio {dead}/{n_candidates} exceeds veto threshold",
            stage="checkpoint",
            dead=dead,
            n_candidates=n_candidates,
            ratio=dead_ratio,
            veto_frac=veto_frac,
            label=label,
            accuracy_improved=improved,
        )
        if not within:
            if getattr(cfg, "failure_policy", "warn") == "strict" and improved:
                improved = False
                note = f"  (accuracy improved but router shows {dead}/{n_candidates} dead layers -- not saved [strict])"
            else:
                note = f"  (W_DEAD_LAYERS: {dead}/{n_candidates} dead layers -- recorded, checkpoint not vetoed)"
        else:
            k_dist = model.routing_k_distribution()
            k_str = ", ".join(f"K={k}:{v:.2f}" for k, v in k_dist.items() if v > 0.01)
            intervention = model.intervention_diagnostics() if hasattr(model, "intervention_diagnostics") else {}
            note = (
                f"  (active layers: {n_candidates - dead}/{n_candidates} | {k_str}"
                f" | K-H={intervention.get('realized_k_entropy', 0.0):.2f}"
                f" | L-H={intervention.get('layer_selection_entropy', 0.0):.2f})"
            )
    print(f"    [checkpoint check] {label}: val={val_loss:.4f}  acc={acc:.3f}  {'[SAVED]' if improved else ''}{note}")
    if hasattr(model, "intervention_diagnostics"):
        gate_rows = model.intervention_diagnostics().get("gate", {})
        active_gates = [row for row in gate_rows.values() if row.get("intervened_tokens", 0) > 0]
        if active_gates:
            mean_gate_std = sum(row["mean_sequence_std"] for row in active_gates) / len(active_gates)
            max_high_sat = max(row["high_saturation_pct"] for row in active_gates)
            print(
                f"    [gate health] selected_layers={len(active_gates)} "
                f"mean_token_std={mean_gate_std:.4f} max_high_sat={max_high_sat:.1f}%"
            )
    if train_router and hasattr(model, "router_k_cap_gap_summary"):
        print(model.router_k_cap_gap_summary())
    best_state = None
    if improved:
        best_acc = max(best_acc, acc)
        best_val, patience_ctr = val_loss, 0
        best_state = {
            k: v.clone() for k, v in model.state_dict().items()
            if "steer_net" in k or "gate" in k or "router" in k
        }
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        save_ars_checkpoint(model, save_path, phase=label, cfg=cfg)
    else:
        patience_ctr += 1
    return best_acc, best_val, best_state, patience_ctr, improved, val_loss, acc


def _phase1_epoch_generation_screen(model, tokenizer, eval_data, cfg: RSCConfig, device: torch.device) -> Optional[Dict[str, Any]]:
    """Run the bounded per-epoch generation safety screen used for checkpoint selection."""
    if tokenizer is None:
        return None
    n_samples = min(len(eval_data), max(0, int(getattr(cfg, "phase1_epoch_generation_screen_examples", 0))))
    if n_samples <= 0:
        return None
    rows = list(eval_data)[:n_samples]
    scales = sorted({
        min(1.0, max(0.0, float(scale)))
        for scale in getattr(cfg, "phase1_epoch_gate_scale_grid", (0.2, 0.4))
    })
    if not scales:
        return None
    old_warmup = getattr(model, "_router_warmup", False)
    model.set_router_warmup(False)
    try:
        baseline = _phase1_probe_with_norm(
            model, tokenizer, rows, cfg, device, n_samples, None, 1.0,
            "Phase1 epoch baseline generation screen",
        )
        base_median_tokens = max(1.0, float(baseline.get("median_tokens", 0.0)))
        safe_pairs = []
        for layer in model.target_layers:
            for scale in scales:
                probe = _phase1_probe_with_norm(
                    model, tokenizer, rows, cfg, device, n_samples, layer, scale,
                    f"Phase1 epoch L{layer} alpha={scale:.2f}",
                )
                safe, completion_ratio, q_p95 = _phase1_pair_is_safe(
                    probe, base_median_tokens, cfg, gate_scale=scale,
                )
                _raw_q_p95, post_cap_q_p95, effective_q_p95 = _phase1_norm_metrics(probe, scale)
                if safe:
                    safe_pairs.append({
                        "layer": int(layer), "gate_scale": scale, "accuracy": probe["accuracy"],
                        "answer_present": probe["answer_present"], "completion_len_ratio": completion_ratio,
                        "pre_cap_q_p95": q_p95, "post_cap_q_p95": post_cap_q_p95,
                        "effective_post_cap_q_p95": effective_q_p95,
                        "effective_pre_cap_q_p95": scale * q_p95,
                    })
        # A checkpoint cannot win merely by avoiding CoT/norm failures: its
        # best safe route must also not underperform the matched baseline.
        eligible = [row for row in safe_pairs if row["accuracy"] >= baseline["accuracy"]]
        if not eligible:
            return {
                "baseline": baseline, "safe_pairs": safe_pairs, "eligible": [],
                "best": None, "status": "NO_NONREGRESSING_SAFE_PAIR",
            }
        best = max(
            eligible,
            key=lambda row: (row["accuracy"], row["answer_present"], -row["effective_post_cap_q_p95"], -row["gate_scale"]),
        )
        return {"baseline": baseline, "safe_pairs": safe_pairs, "eligible": eligible, "best": best, "status": "SAFE_NONREGRESSING"}
    finally:
        model.set_router_warmup(old_warmup)


def train_phase1_steernet(model, train_data, eval_data, cfg: RSCConfig, device: torch.device, tokenizer=None):
    """Phase 1: Pure SteerNet training with gates pinned at 1.0 and router in coverage warmup."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log_gpu_memory("train_phase1 start", cfg)

    print("\n" + "=" * 80)
    print("PHASE 1: Pure SteerNet Training (Hidden States)")
    print("         Gates FIXED at 1.0; deterministic K=1 rotation")
    # Use v5-specific learning rate if v5 architecture is active
    phase1_lr = getattr(cfg, "phase1_lr_v5", cfg.phase1_lr) if getattr(cfg, "steernet_arch", "v4") == "v5" else cfg.phase1_lr
    print(f"         lr={phase1_lr}, epochs={cfg.phase1_epochs}")
    print("=" * 80)

    model.set_gate_freeze(True, 1.0)
    model.set_router_warmup(True)
    model.set_router_exploration(False)
    for p in model.router.parameters():
        p.requires_grad = False

    steer_params = []
    for sn in model.steer_nets.values():
        for p in sn.parameters():
            p.requires_grad = True
            steer_params.append(p)
    print(f"  SteerNet params: {sum(p.numel() for p in steer_params):,}")

    optimizer_p1 = torch.optim.AdamW(
        steer_params,
        lr=phase1_lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-6,
    )

    steps_per_epoch = len(train_data) // (cfg.effective_batch_size)
    total_p1 = max(1, steps_per_epoch * cfg.phase1_epochs)
    warmup_p1 = int(total_p1 * cfg.phase1_warmup_ratio)

    lr_reducer_p1 = LRReducer(
        optimizer_p1,
        total_p1,
        factor=cfg.lr_reduce_factor,
        patience=cfg.lr_reduce_patience,
        min_lr=cfg.lr_reduce_min_lr,
        delta=cfg.early_stop_delta,
        warmup_steps=warmup_p1,
    )
    lr_reducer_p1.base_lrs = [cfg.phase1_lr]
    optimizer_p1.param_groups[0]["lr"] = cfg.phase1_lr
    ckpt_p1 = CheckpointManager(cfg.checkpoint_dir)
    grad_mon_p1 = GradientMonitor(cfg.grad_warn_threshold)

    best_val_p1 = float("inf")
    best_nll_state_p1 = None
    best_generation_state_p1 = None
    best_generation_rank = None
    best_generation_val = float("inf")
    p1_patience_ctr = 0
    logs = {"p1_train": [], "p1_val": [], "p1_generation_screen": []}
    scaler_p1 = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(max(0, cfg.phase1_epochs)):
        model.train()
        # The Phi-2 backbone is frozen.  Its dropout must not make Phase 1
        # optimise a different function from eval()/generate(), where the
        # dropout is disabled.  Phase 2 already enforced this invariant; its
        # absence here was a direct train/deployment distribution mismatch.
        model.base_model.eval()
        # The phase-specific modules use their normal training mode, but the
        # router only supplies deterministic warmup coverage and is frozen.
        model.router.eval()
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        epoch_loss, n_optimizer_steps = 0.0, 0
        accum_count_p1 = 0
        optimizer_p1.zero_grad(set_to_none=True)

        pbar = tqdm(
            range(0, len(indices), cfg.batch_size), desc=f"Phase1 {epoch+1}/{cfg.phase1_epochs}"
        )
        for start in pbar:
            batch_idx = indices[start : start + cfg.batch_size]
            ids = torch.stack([train_data[i]["input_ids"] for i in batch_idx]).to(device)
            mask = torch.stack([train_data[i]["attention_mask"] for i in batch_idx]).to(device)
            lbls = torch.stack([train_data[i]["labels"] for i in batch_idx]).to(device)
            shield = None
            if "prompt_shield_mask" in train_data[batch_idx[0]]:
                shield = torch.stack([train_data[i]["prompt_shield_mask"] for i in batch_idx]).to(device)
            calc_mask = None
            if "calc_mask" in train_data[batch_idx[0]]:
                calc_mask = torch.stack([train_data[i]["calc_mask"] for i in batch_idx]).to(device)
            ids, mask, lbls, shield, calc_mask = _trim_batch(ids, mask, lbls, shield, calc_mask)
            model._attn_mask_cache = mask
            _set_batch_context(model, mask, lbls, shield)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(input_ids=ids, attention_mask=mask, use_cache=False)
                loss_task = _answer_aware_loss(out.logits, lbls, cfg, calc_mask=calc_mask)
                loss_norm, norm_stats = model.compute_active_steer_norm_loss(
                    q_hi=getattr(cfg, "steer_norm_q_hi", 0.75),
                    huber_delta=getattr(cfg, "steer_norm_huber_delta", 0.10),
                )
                # Use v5-specific regularization if v5 architecture is active
                if getattr(cfg, "steernet_arch", "v4") == "v5":
                    lambda_norm = float(getattr(cfg, "lambda_steer_norm_v5", 0.30))
                else:
                    lambda_norm = float(getattr(cfg, "lambda_steer_norm", 0.60))
                loss = (loss_task + lambda_norm * loss_norm) / cfg.grad_accum

            if not torch.isfinite(loss):
                continue

            scaler_p1.scale(loss).backward()
            accum_count_p1 += 1

            is_last_p1 = start + cfg.batch_size >= len(indices)
            if accum_count_p1 == cfg.grad_accum or is_last_p1:
                scaler_p1.unscale_(optimizer_p1)
                # Phase 4: Log gradient norms for v5 diagnostics
                total_grad_norm = 0.0
                layer_grad_norms = {}
                for p in steer_params:
                    if p.grad is not None:
                        grad_norm = p.grad.norm().item()
                        total_grad_norm += grad_norm * grad_norm
                total_grad_norm = total_grad_norm ** 0.5
                if n_optimizer_steps % 10 == 0:  # Log every 10 steps
                    print(f"    Gradient norm: {total_grad_norm:.6f}")
                for p in steer_params:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        p.grad.zero_()
                torch.nn.utils.clip_grad_norm_(steer_params, cfg.clip_grad)
                if cfg.use_gradient_monitor:
                    grad_mon_p1.check(steer_params, n_optimizer_steps)
                scaler_p1.step(optimizer_p1)
                scaler_p1.update()
                lr_reducer_p1.scheduler_step()
                optimizer_p1.zero_grad(set_to_none=True)
                accum_count_p1 = 0
                n_optimizer_steps += 1

            epoch_loss += loss.item() * cfg.grad_accum
            pbar.set_postfix(
                loss=f"{loss.item() * cfg.grad_accum:.4f}",
                q_p95=f"{norm_stats['q_p95']:.2f}",
                q_max=f"{norm_stats['q_max']:.2f}",
            )

        avg_loss = epoch_loss / max(1, n_optimizer_steps * cfg.grad_accum)
        val_loss = evaluate_loss(model, eval_data, cfg, device, answer_aware=True)
        logs["p1_train"].append(avg_loss)
        logs["p1_val"].append(val_loss)

        improved = "✅" if val_loss < best_val_p1 - cfg.early_stop_delta else "  "
        if val_loss < best_val_p1 - cfg.early_stop_delta:
            best_val_p1 = val_loss
            p1_patience_ctr = 0
            best_nll_state_p1 = {
                k: v.clone()
                for k, v in model.state_dict().items()
                if "steer_net" in k or "router" in k
            }
        else:
            p1_patience_ctr += 1

        lr_reducer_p1.step(val_loss)
        lr_now = optimizer_p1.param_groups[0]["lr"]
        print(
            f"  Phase1 Epoch {epoch+1}: train={avg_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}  {improved}"
        )

        epoch_screen = _phase1_epoch_generation_screen(model, tokenizer, eval_data, cfg, device)
        if epoch_screen is None:
            logs["p1_generation_screen"].append({"epoch": epoch + 1, "status": "UNAVAILABLE"})
            if tokenizer is None:
                print("  ⚠️ Phase1 epoch generation screen skipped: tokenizer was not provided.")
        elif epoch_screen["best"] is None:
            status = epoch_screen.get("status", "NO_SAFE_PAIR")
            logs["p1_generation_screen"].append({"epoch": epoch + 1, "status": status})
            if status == "NO_NONREGRESSING_SAFE_PAIR":
                print(
                    "  ⚠️ Phase1 epoch generation screen found safe routes, but none matched the baseline "
                    "accuracy; retaining NLL checkpoint eligibility only."
                )
            else:
                print("  ⚠️ Phase1 epoch generation screen found no CoT- and norm-safe pair.")
        else:
            choice = epoch_screen["best"]
            baseline_acc = epoch_screen["baseline"]["accuracy"]
            rank = (
                choice["accuracy"] - baseline_acc, choice["accuracy"], choice["answer_present"],
                -choice["pre_cap_q_p95"], -choice["gate_scale"], -val_loss,
            )
            logs["p1_generation_screen"].append({
                "epoch": epoch + 1, "status": "SAFE", "baseline_accuracy": baseline_acc, **choice,
            })
            print(
                f"  Phase1 Epoch {epoch+1} generation screen: L{choice['layer']} alpha={choice['gate_scale']:.2f} "
                f"acc={choice['accuracy']:.1%} vs base={baseline_acc:.1%} "
                f"CoT={choice['completion_len_ratio']:.2f} q95={choice['pre_cap_q_p95']:.2f}"
            )
            if best_generation_rank is None or rank > best_generation_rank:
                best_generation_rank = rank
                best_generation_val = val_loss
                best_generation_state_p1 = {
                    k: v.clone()
                    for k, v in model.state_dict().items()
                    if "steer_net" in k or "router" in k
                }

        if (epoch + 1) % cfg.checkpoint_every_n == 0:
            ckpt_p1.save(model, optimizer_p1, epoch, val_loss, "p1")

        if p1_patience_ctr >= cfg.phase1_patience:
            print(f"  Phase1 early stopping (no improvement for {cfg.phase1_patience} epochs)")
            break

    best_state_p1 = best_generation_state_p1 or best_nll_state_p1
    if best_generation_state_p1:
        best_val_p1 = best_generation_val
        print("  ✅ Restored generation-screen-selected Phase 1 checkpoint.")
    else:
        print("  ✅ No epoch met the matched generation baseline; restored the NLL-best Phase 1 checkpoint.")
    if best_state_p1:
        model.load_state_dict(best_state_p1, strict=False)
    save_steer_weights(
        model, cfg.lora_save_path, cfg=cfg, phase="phase1-steernet", prefixes=("steer_nets.",)
    )
    grad_mon_p1.summary()
    print(f"\n✅ Phase 1 complete — best_val = {best_val_p1:.4f}")

    if hasattr(model, "pre_cap_norm_diagnostics"):
        ratios = model.pre_cap_norm_diagnostics()
        print(f"  Phase 1 SteerNet last-batch mean pre-cap norm ratios: {ratios}")
        max_ratio = max(ratios.values()) if ratios else 0.0
        thresh = float(getattr(cfg, "phase1_max_eval_p95_norm_ratio", 1.0))
        if max_ratio > thresh:
            print(f"  ℹ️ SteerNet last-batch ratio notice: max ratio = {max_ratio:.2f} > {thresh:.2f} (acceptance audit evaluates exact p95)")
        else:
            print(f"  ✅ SteerNet norm control healthy: max ratio = {max_ratio:.2f} <= {thresh:.2f}")

    model.set_router_warmup(False)
    model.reset_routing_diagnostics()
    log_gpu_memory("after Phase 1", cfg)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_state_p1, best_val_p1, logs


def _phase1_depth_band(layer: int, candidate_layers: List[int]) -> str:
    """Map catalog depth to a stable early/middle/late co-adaptation band."""
    # Use the *catalogue's physical depth*, not a position within the safe
    # subset.  Otherwise omitting L13 from a safe pool re-labels L15 as a
    # late layer and makes a nominally cross-band L15+L22 pair look adjacent.
    # The fallback keeps this helper useful for non-standard, very small
    # catalogues used by CPU tests.
    ordered = sorted(int(value) for value in candidate_layers)
    # Phi-2's canonical ARS catalogue is deliberately non-uniform across
    # representational roles: L11--13 are early, L15--18 are middle, and
    # L20--24 are late.  Preserve those role boundaries even when a safe
    # subset omits a neighbouring layer.
    if ordered and min(ordered) <= 13 and max(ordered) >= 20:
        if int(layer) <= 13:
            return "early"
        if int(layer) <= 18:
            return "middle"
        return "late"
    if len(ordered) >= 3:
        position = ordered.index(int(layer))
        if position < max(1, len(ordered) // 3):
            return "early"
        if position < max(2, (2 * len(ordered)) // 3):
            return "middle"
        return "late"
    if int(layer) == ordered[0]:
        return "early"
    if len(ordered) == 2 and int(layer) == ordered[1]:
        return "middle"
    return "late"


def _is_long_range_coadaptation_pair(
    first: int, second: int, candidate_layers: List[int],
) -> bool:
    """Return whether a pair joins a late layer to an earlier reasoning band."""
    bands = {
        _phase1_depth_band(int(first), candidate_layers),
        _phase1_depth_band(int(second), candidate_layers),
    }
    # L11+L24 and L15+L24 are the intended curriculum examples.  L11+L15
    # remains a legitimate ablation, but it is not a long-range pair and must
    # not consume the scarce K=2 co-adaptation budget by default.
    return "late" in bands and len(bands) == 2


def select_phase1b_candidate_layers(
    model, acceptance_report: Dict[str, Any], cfg: RSCConfig,
) -> Tuple[List[int], Dict[int, float], Dict[str, Any]]:
    """Build a safe, depth-diverse Phase-1B pool from calibration evidence.

    Confirmation failures are absolute exclusions.  Among remaining safe
    calibration routes, take the highest-ranked distinct layer from each depth
    band before filling spare capacity by rank.  This creates opportunities
    for long-range early/late cooperation without allowing unsafe adapters
    back into the learned router.
    """
    ranking = acceptance_report.get("calibration_ranking", [])
    failed_confirmation_layers = {
        int(layer)
        for trial in acceptance_report.get("confirmation_trials", [])
        for layer in [trial.get("layer")]
        if layer is not None and (
            not trial.get("safe", True)
            or not trial.get("nll_safe", True)
            or (not trial.get("passed", True) and trial.get("confirmation_gain", 0.0) < -0.10)
        )
    }
    candidate_rows: List[Dict[str, Any]] = []
    seen = set()
    for row in ranking:
        layer = int(row.get("layer", -1))
        if (
            row.get("safe", False)
            and layer in model.target_layers
            and layer not in failed_confirmation_layers
            and layer not in seen
        ):
            candidate_rows.append(dict(row))
            seen.add(layer)
    # A legacy/malformed ranking should not discard independently recorded
    # safe screens, but these fallback rows rank after the explicit ranking.
    for candidate in acceptance_report.get("candidates", []):
        layer = int(candidate.get("layer", -1))
        if layer in seen or layer in failed_confirmation_layers or layer not in model.target_layers:
            continue
        safe_scales = [entry for entry in candidate.get("scale_screen", []) if entry.get("safe", False)]
        if safe_scales:
            best = min(safe_scales, key=lambda entry: float(entry.get("gate_scale", 1.0)))
            candidate_rows.append({"layer": layer, "gate_scale": best.get("gate_scale", 1.0), "safe": True})
            seen.add(layer)

    limit = int(getattr(cfg, "phase1b_candidate_layers", 6))
    per_band = max(1, int(getattr(cfg, "phase1b_candidates_per_band", 2)))
    chosen: List[int] = []
    scales: Dict[int, float] = {}
    covered_bands = []
    for band in ("early", "middle", "late"):
        band_rows = [
            item for item in candidate_rows
            if _phase1_depth_band(int(item["layer"]), model.target_layers) == band
        ][:per_band]
        for row in band_rows:
            if len(chosen) >= limit:
                break
            layer = int(row["layer"])
            chosen.append(layer)
            scales[layer] = float(row.get("gate_scale", 1.0))
        if band_rows:
            covered_bands.append(band)
    for row in candidate_rows:
        if len(chosen) >= limit:
            break
        layer = int(row["layer"])
        if layer not in chosen:
            chosen.append(layer)
            scales[layer] = float(row.get("gate_scale", 1.0))
    return chosen, scales, {
        "failed_confirmation_layers": sorted(failed_confirmation_layers),
        "depth_bands": covered_bands,
        "ranked_safe_layers": [int(item["layer"]) for item in candidate_rows],
    }


def train_phase1b_coadaptation(
    model, train_data, acceptance_report: Dict[str, Any], cfg: RSCConfig, device: torch.device,
) -> Dict[str, Any]:
    """Teach a balanced safe K=1/K=2 pool before policy optimisation.

    Phase 1 proves independent capacity.  This stage deliberately does *not*
    ask the immature router to choose training pairs: it round-robins safe
    candidate singletons and pairs with frozen gates, one trimmed micro-batch
    at a time.  That is the memory-safe path to complementary residuals.
    """
    epochs = int(getattr(cfg, "phase1b_coadapt_epochs", 0))
    if epochs <= 0:
        return {"status": "SKIPPED", "candidate_layers": [], "steps": 0}
    layers, layer_scales, pool_report = select_phase1b_candidate_layers(model, acceptance_report, cfg)
    if pool_report["failed_confirmation_layers"]:
        print(
            "  [Phase 1B] Excluding confirmation-failed layers: "
            f"{pool_report['failed_confirmation_layers']}"
        )
    min_layers = int(getattr(cfg, "phase1b_min_candidate_layers", 3))
    min_bands = int(getattr(cfg, "phase1b_min_depth_bands", 3))
    if len(layers) < min_layers or len(pool_report["depth_bands"]) < min_bands:
        print(
            "  ⚠️ Phase 1B: insufficient safe depth-diverse pool; "
            f"layers={layers}, bands={pool_report['depth_bands']}"
        )
        return {
            "status": "SKIPPED_INSUFFICIENT_DIVERSE_LAYERS", "candidate_layers": layers,
            "candidate_gate_scales": layer_scales, "steps": 0, **pool_report,
        }
    # Long-range pairs are the cooperative hypothesis: early/middle trajectory
    # steering plus late answer extraction.  Adjacent routes remain available
    # only if no cross-band choice exists.
    pairs = [
        (layers[i], layers[j]) for i in range(len(layers)) for j in range(i + 1, len(layers))
        if _is_long_range_coadaptation_pair(layers[i], layers[j], model.target_layers)
    ]
    if not pairs:
        pairs = [(layers[i], layers[j]) for i in range(len(layers)) for j in range(i + 1, len(layers))]
    # Repeat the smaller route family and interleave it with the other so
    # every epoch has exactly equal K=1 and K=2 exposure independent of the
    # number of certified layers.  The old concatenation was 40/60 for four
    # layers and 67/33 for two layers, despite claiming to be balanced.
    singletons = [(layer,) for layer in layers]
    schedule = []
    for index in range(max(len(singletons), len(pairs))):
        schedule.append(singletons[index % len(singletons)])
        schedule.append(pairs[index % len(pairs)])
    params = [p for sn in model.steer_nets.values() for p in sn.parameters()]
    for p in params:
        p.requires_grad = True
    for p in model.router.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(params, lr=float(getattr(cfg, "phase1b_lr", 5e-5)), weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    steps, norm_total = 0, 0.0
    model.base_model.eval()
    if hasattr(model, "set_certified_gate_scales"):
        model.set_certified_gate_scales(layer_scales)
    model.set_gate_freeze(True, layer_scales)
    model.set_router_warmup(False)
    print(
        f"\nPHASE 1B: Balanced K=1/K=2 Co-adaptation | layers={layers} "
        f"pairs={len(pairs)} scales={layer_scales}"
    )
    try:
        for epoch in range(epochs):
            order = list(range(len(train_data)))
            random.Random(cfg.seed + epoch).shuffle(order)
            for step, row_idx in enumerate(order):
                row = train_data[row_idx]
                ids = row["input_ids"].unsqueeze(0).to(device)
                mask = row["attention_mask"].unsqueeze(0).to(device)
                labels = row["labels"].unsqueeze(0).to(device)
                shield = row.get("prompt_shield_mask")
                shield = shield.unsqueeze(0).to(device) if shield is not None else None
                calc = row.get("calc_mask")
                calc = calc.unsqueeze(0).to(device) if calc is not None else None
                ids, mask, labels, shield, calc = _trim_batch(ids, mask, labels, shield, calc)
                chosen = schedule[step % len(schedule)]
                route = torch.zeros(1, len(model.target_layers), dtype=torch.bool, device=device)
                for layer in chosen:
                    route[0, model.target_layers.index(layer)] = True
                optimizer.zero_grad(set_to_none=True)
                model.set_routing_override("matrix", route)
                _set_batch_context(model, mask, labels, shield)
                try:
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        out = model(input_ids=ids, attention_mask=mask, use_cache=False)
                        task = _answer_aware_loss(out.logits, labels, cfg, calc_mask=calc)
                        norm, _ = model.compute_active_steer_norm_loss(
                            q_hi=cfg.steer_norm_q_hi, huber_delta=cfg.steer_norm_huber_delta,
                        )
                        loss = float(getattr(cfg, "phase1b_joint_nll_weight", 1.0)) * task + cfg.lambda_steer_norm * norm
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    steps += 1
                    norm_total += float(norm.detach().item())
                finally:
                    model.set_routing_override(None)
                    del out
                if steps % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    enforce_gpu_memory_budget(f"Phase 1B step {steps}", cfg)
    finally:
        model.set_routing_override(None)
        model.set_gate_freeze(True, 1.0)
        for p in params:
            p.requires_grad = False
    return {
        "status": "COMPLETE", "candidate_layers": layers, "pair_count": len(pairs),
        "candidate_gate_scales": layer_scales,
        "schedule_k1_steps": len(schedule) // 2,
        "schedule_k2_steps": len(schedule) // 2,
        "steps": steps, "mean_norm_loss": norm_total / max(1, steps), **pool_report,
    }


def train_router_bootstrap(model, train_data, eval_data, cfg: RSCConfig, device: torch.device):
    """Phase 1.5: Router Bootstrap with SteerNet compatibility correction (RC-5).

    ROOT CAUSE of router K=1 collapse:
      Phase 1 trains each SteerNet independently to minimise NLL for K=1.
      When Phase 1.5 probes K=2 counterfactuals, independently-trained vectors
      always INTERFERE (their residual deltas partially cancel), so K=2 NLL is
      genuinely worse than K=1 NLL.  The router correctly learns K=1 is best.
      Preflight fails.  Phase 2 is skipped.  Vicious cycle.

    RC-5 fix — two mechanisms:
      1. Compatibility LR: SteerNets are kept TRAINABLE during bootstrap with a
         tiny lr = bootstrap_steer_compat_lr_scale * router_bootstrap_lr (default
         5%).  Small enough that Phase-1 corrections are not destroyed; large
         enough to nudge vectors toward complementarity over many steps.
      2. Joint-NLL correction: after each RLOO step, we do a DIFFERENTIABLE
         forward with the top-synergy K=2 pair and backprop only through
         SteerNet params.  This gives SteerNets a direct training signal to
         produce complementary (not interfering) deltas for K>1 routes.
         RLOO itself runs under torch.no_grad() so it cannot teach SteerNets
         this — the correction step is the only path for that gradient.

    SteerNets are snapshotted before bootstrap and restored if preflight later
    fails, so Phase-1 work is never permanently destroyed.
    """
    # --- Snapshot Phase-1 SteerNet state for safe restore on preflight failure ---
    steer_snapshot = {
        k: v.clone() for k, v in model.state_dict().items() if "steer_net" in k
    }

    # --- RC-5: keep SteerNets TRAINABLE with tiny compatibility LR ---
    compat_lr_scale = float(getattr(cfg, "bootstrap_steer_compat_lr_scale", 0.05))
    compat_lr = compat_lr_scale * cfg.router_bootstrap_lr
    steer_params = []
    for sn in model.steer_nets.values():
        for p in sn.parameters():
            p.requires_grad = True
            steer_params.append(p)

    lambda_joint = float(getattr(cfg, "bootstrap_joint_nll_weight", 0.3))

    model.set_gate_freeze(True, 1.0)
    model.set_router_warmup(False)
    model.set_routing_override(None)

    router_params = list(model.router.parameters())
    for p in router_params:
        p.requires_grad = True

    # Two param groups: router at full LR, SteerNets at compat_lr
    optimizer = torch.optim.AdamW(
        [
            {"params": router_params, "lr": cfg.router_bootstrap_lr},
            {"params": steer_params, "lr": compat_lr},
        ],
        weight_decay=cfg.weight_decay,
    )
    model.reset_routing_diagnostics()

    pool_size = min(cfg.router_bootstrap_n_examples, len(train_data))
    rng = random.Random(cfg.seed)
    subset_idx = rng.sample(range(len(train_data)), pool_size)
    # best_state now covers router AND steer_nets so the best joint checkpoint is restored
    best_state, best_loss, patience_ctr = None, float("inf"), 0

    print("\n" + "=" * 80)
    print("PHASE 1.5: Router Bootstrap + SteerNet Compatibility Correction (RC-5)")
    print(f"           epochs={cfg.router_bootstrap_epochs}, router_lr={cfg.router_bootstrap_lr:.2e}")
    print(f"           steer_compat_lr={compat_lr:.2e}  joint_nll_weight={lambda_joint}")
    print("=" * 80)

    eligible_layers = [
        layer for layer, allowed in zip(
            model.target_layers,
            model.routing_eligibility_mask(device).tolist()
            if hasattr(model, "routing_eligibility_mask") else [True] * len(model.target_layers),
        ) if allowed
    ]
    compatibility_required = lambda_joint > 0.0 and bool(steer_params) and len(eligible_layers) >= 2
    compatibility_telemetry = {
        "required": compatibility_required,
        "attempts": 0,
        "successful_steps": 0,
        "nonzero_gradient_steps": 0,
        "attempted_rloo_steps": 0,
        "finite_rloo_steps": 0,
        "nonfinite_loss_steps": 0,
        "nonfinite_gradient_steps": 0,
        "status": "RUNNING",
    }
    # Keep a live dictionary on the model so a raised correction failure still
    # leaves forensic evidence for the caller or an interactive notebook.
    model._last_router_bootstrap_compatibility = compatibility_telemetry
    for epoch in range(cfg.router_bootstrap_epochs):
        random.shuffle(subset_idx)
        model.train()
        running, n_steps = 0.0, 0
        running_joint_nll, n_joint_steps = 0.0, 0
        joint_attempts_epoch, joint_grad_steps_epoch = 0, 0
        counterfactual_k_actions = None
        bootstrap_invalid = False
        rloo_batch_size = max(2, int(getattr(cfg, "router_rloo_batch_size", 8)))
        pbar = tqdm(
            range(0, len(subset_idx), rloo_batch_size),
            desc=f"RouterBootstrap {epoch + 1}/{cfg.router_bootstrap_epochs}",
        )
        for start in pbar:
            batch_indices = subset_idx[start : start + rloo_batch_size]
            ids = torch.stack([train_data[i]["input_ids"] for i in batch_indices]).to(device)
            mask = torch.stack([train_data[i]["attention_mask"] for i in batch_indices]).to(device)
            lbls = torch.stack([train_data[i]["labels"] for i in batch_indices]).to(device)
            shield = None
            if "prompt_shield_mask" in train_data[batch_indices[0]]:
                shield = torch.stack([train_data[i]["prompt_shield_mask"] for i in batch_indices]).to(device)
            calc_mask = None
            if "calc_mask" in train_data[batch_indices[0]]:
                calc_mask = torch.stack([train_data[i]["calc_mask"] for i in batch_indices]).to(device)
            ids, mask, lbls, shield, calc_mask = _trim_batch(ids, mask, lbls, shield, calc_mask)
            model._attn_mask_cache = mask

            _set_batch_context(model, mask, lbls, shield)
            prompt_mask = mask.bool() & lbls.eq(-100) & ~(
                shield.bool() if shield is not None else torch.zeros_like(mask, dtype=torch.bool)
            )

            # --- Step 1: RLOO router loss (router params + SteerNet compat gradient) ---
            compatibility_telemetry["attempted_rloo_steps"] += 1
            warmup_epochs = max(1, int(getattr(cfg, "router_tariff_warmup_epochs", 1)))
            epoch_progress = float(epoch) + float(start) / max(1, len(subset_idx))
            tariff_scale = min(1.0, epoch_progress / float(warmup_epochs))
            temperature_final = float(getattr(cfg, "router_k_preference_temperature_final", cfg.router_k_preference_temperature))
            temperature_start = float(getattr(cfg, "router_k_preference_temperature", 0.10))
            anneal_epochs = int(getattr(cfg, "router_k_preference_anneal_epochs", 0))
            anneal_progress = min(1.0, epoch_progress / max(1, anneal_epochs)) if anneal_epochs > 0 else 0.0
            preference_temperature = temperature_start + (temperature_final - temperature_start) * anneal_progress
            try:
                loss, rloo_diag = compute_router_rloo_loss(
                    model, ids, mask, lbls, cfg, prompt_mask=prompt_mask,
                    shield_mask=shield, calc_mask=calc_mask, tariff_scale=tariff_scale,
                    k_preference_temperature=preference_temperature,
                )
            except FloatingPointError as exc:
                compatibility_telemetry["nonfinite_loss_steps"] += 1
                compatibility_telemetry["status"] = f"INVALID_NUMERICS: {exc}"
                bootstrap_invalid = True
                print(f"  ⚠️ Router bootstrap stopped: {compatibility_telemetry['status']}")
                break
            if not torch.isfinite(loss):
                compatibility_telemetry["nonfinite_loss_steps"] += 1
                compatibility_telemetry["status"] = "INVALID_NUMERICS: non-finite RLOO loss"
                bootstrap_invalid = True
                print(f"  ⚠️ Router bootstrap stopped: {compatibility_telemetry['status']}")
                break

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in router_params
            ):
                compatibility_telemetry["nonfinite_gradient_steps"] += 1
                compatibility_telemetry["status"] = "INVALID_NUMERICS: non-finite router gradient"
                bootstrap_invalid = True
                print(f"  ⚠️ Router bootstrap stopped: {compatibility_telemetry['status']}")
                break
            torch.nn.utils.clip_grad_norm_(router_params, cfg.clip_grad)
            # Note: steer_params grad clipping happens AFTER joint-NLL correction below

            # --- Step 2: Differentiable K=2 joint-NLL correction for SteerNets ---
            # RLOO runs under no_grad, so SteerNets only receive gradient from this step.
            # We pick the top-synergy pair so the correction reinforces actual cooperation.
            # OOM FIX: run on a single example (ids[:1]) instead of the full batch.
            # Full-batch logits.float() = [B,T,V] fp32 → ~800 MB at B=8; single = ~100 MB.
            if lambda_joint > 0.0 and steer_params and len(eligible_layers) >= 2:
                joint_attempts_epoch += 1
                compatibility_telemetry["attempts"] += 1
                coop = model.get_cooperation_analysis() if hasattr(model, "get_cooperation_analysis") else {}
                top_pairs = coop.get("top_synergy_pairs", [])
                # Prefer a distant cross-band pair: the curriculum hypothesis is
                # early/middle trajectory steering plus late answer extraction,
                # not redundant adjacent residuals.
                cross_band_pairs = [
                    pair for pair in top_pairs
                    if len(pair) >= 3
                    and int(pair[0]) in eligible_layers and int(pair[1]) in eligible_layers
                    and _is_long_range_coadaptation_pair(
                        int(pair[0]), int(pair[1]), model.target_layers,
                    )
                ]
                # Use best positive cross-band synergy; fall back to the
                # maximally distant eligible pair before any adjacent pair.
                if (
                    cross_band_pairs and cross_band_pairs[0][2] > 0
                ):
                    layer_i, layer_j = int(cross_band_pairs[0][0]), int(cross_band_pairs[0][1])
                else:
                    layer_i, layer_j = min(eligible_layers), max(eligible_layers)

                n_cands = len(model.target_layers)
                # Single example: avoids holding B×T×V graph in memory alongside RLOO activations
                s_ids  = ids[:1]
                s_mask = mask[:1]
                s_lbls = lbls[:1]
                s_shield = shield[:1] if shield is not None else None
                s_calc = calc_mask[:1] if calc_mask is not None else None
                k2_mask = torch.zeros(1, n_cands, dtype=torch.bool, device=device)
                if layer_i in model.target_layers:
                    k2_mask[:, model.target_layers.index(layer_i)] = True
                if layer_j in model.target_layers:
                    k2_mask[:, model.target_layers.index(layer_j)] = True

                model.train()
                model.set_gate_freeze(True, 1.0)
                model.set_routing_override("matrix", k2_mask)
                _set_batch_context(model, s_mask, s_lbls, s_shield)

                # Free fragmented cache before the differentiable forward
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                try:
                    with torch.amp.autocast("cuda", enabled=s_ids.is_cuda):
                        joint_out = model(input_ids=s_ids, attention_mask=s_mask, use_cache=False)
                        joint_nll = per_example_answer_nll(
                            joint_out.logits,
                            s_lbls,
                            answer_tail_tokens=int(getattr(cfg, "answer_tail_tokens", 0)),
                            answer_tail_weight=float(getattr(cfg, "answer_tail_weight", 1.0)),
                            calc_mask=s_calc,
                            calc_target_mode=getattr(cfg, "steer_loss_target", "calculation"),
                            calc_weight=getattr(cfg, "calc_token_weight", 3.0),
                        ).mean()
                    if not torch.isfinite(joint_nll):
                        raise FloatingPointError("non-finite K=2 compatibility joint-NLL")
                    (lambda_joint * joint_nll).backward()
                    has_steer_gradient = any(
                        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.detach().abs().sum().item() > 0.0
                        for p in steer_params
                    )
                    if not has_steer_gradient:
                        raise RuntimeError("K=2 compatibility correction produced no SteerNet gradient")
                    running_joint_nll += joint_nll.item()
                    n_joint_steps += 1
                    joint_grad_steps_epoch += 1
                    compatibility_telemetry["successful_steps"] += 1
                    compatibility_telemetry["nonzero_gradient_steps"] += 1
                finally:
                    model.set_routing_override(None)

            # Clip steer params AFTER both backward passes accumulate gradients
            if steer_params:
                torch.nn.utils.clip_grad_norm_(steer_params, cfg.clip_grad)
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in router_params):
                compatibility_telemetry["nonfinite_gradient_steps"] += 1
                compatibility_telemetry["status"] = "INVALID_NUMERICS: non-finite router parameter"
                bootstrap_invalid = True
                print(f"  ⚠️ Router bootstrap stopped: {compatibility_telemetry['status']}")
                break

            running += loss.item()
            n_steps += 1
            compatibility_telemetry["finite_rloo_steps"] += 1
            postfix_dict = {"loss": f"{loss.item():.4f}"}
            if rloo_diag:
                postfix_dict["mean_Kcf"] = f"{rloo_diag.get('mean_k_sampled', 0.0):.2f}"
                if "win_rate_vs_base" in rloo_diag:
                    postfix_dict["win_base"] = f"{rloo_diag['win_rate_vs_base'] * 100:.0f}%"
                counterfactual_k_actions = rloo_diag.get("counterfactual_k_actions", counterfactual_k_actions)
            pbar.set_postfix(**postfix_dict)

        avg_loss = running / n_steps if n_steps else float("nan")
        avg_joint = running_joint_nll / max(1, n_joint_steps) if n_joint_steps > 0 else float("nan")
        k_cf_str = (
            " | K counterfactuals={0," + ",".join(str(k) for k in counterfactual_k_actions) + "}"
            if counterfactual_k_actions else ""
        )
        diag = model.routing_diagnostics()
        active_cnt = sum(1 for s in diag.values() if s["selected_forwards"] > 0)
        coop = model.get_cooperation_analysis() if hasattr(model, "get_cooperation_analysis") else {}
        top_syn_str = ", ".join(f"L{a}+L{b}:{v:+.2f}" for a, b, v in coop.get("top_synergy_pairs", [])[:2])
        win_str = f" | win_base={rloo_diag.get('win_rate_vs_base', 0.0) * 100:.1f}%" if rloo_diag else ""
        k_pref_str = (
            f" | Kpref={rloo_diag.get('k_preference_loss', 0.0):.3f} "
            f"p(K0)={rloo_diag.get('k0_probability', 0.0):.2f} "
            f"Ktariff={rloo_diag.get('mean_tariff', 0.0):.3f}"
            f"@{rloo_diag.get('tariff_scale', 0.0):.2f}"
            if rloo_diag else ""
        )
        k_argmax_str = (
            " | argmax-K=" + ",".join(
                f"{k}:{v:.2f}" for k, v in rloo_diag.get("argmax_k_distribution", {}).items()
            ) if rloo_diag else ""
        )
        utility_str = ""
        if rloo_diag:
            utilities = rloo_diag.get("counterfactual_utility_by_k", {})
            pieces = []
            for k in (0, 1, 2):
                row = utilities.get(k, utilities.get(str(k)))
                if row:
                    pieces.append(
                        f"K{k}:raw={row['raw_reward']:+.3f}/net={row['cost_adjusted_reward']:+.3f}"
                    )
            if pieces:
                utility_str = " | utility[" + ", ".join(pieces) + "]"
        syn_str = f" | top synergy: [{top_syn_str}]" if top_syn_str else ""
        joint_str = (
            f" | joint_nll={avg_joint:.4f} "
            f"(K2={n_joint_steps}/{joint_attempts_epoch}, grad={joint_grad_steps_epoch})"
            if joint_attempts_epoch > 0 else ""
        )
        print(
            f"  RouterBootstrap Epoch {epoch + 1}: loss={avg_loss:.4f}"
            f"{win_str}{k_pref_str}{k_argmax_str}"
            f" | active layers={active_cnt}/{len(diag)}"
            f"{syn_str}{joint_str}{utility_str}{k_cf_str}"
        )
        if bootstrap_invalid:
            break
        if avg_loss < best_loss - 1e-4:
            best_loss, patience_ctr = avg_loss, 0
            # Save both router AND steer_nets so best joint checkpoint can be restored
            best_state = {
                k: v.clone() for k, v in model.state_dict().items()
                if "router" in k or "steer_net" in k
            }
        else:
            patience_ctr += 1
        if patience_ctr >= cfg.router_bootstrap_patience:
            break

    if compatibility_telemetry["status"] == "RUNNING":
        compatibility_telemetry["status"] = "COMPLETE"
    if compatibility_telemetry["status"] != "COMPLETE":
        if steer_snapshot:
            model.load_state_dict(steer_snapshot, strict=False)
        for sn in model.steer_nets.values():
            for p in sn.parameters():
                p.requires_grad = False
        model.set_routing_override(None)
        print("  ⚠️ Router bootstrap is invalid; restored certified Phase-1 SteerNet state for fixed fallback.")
        return None, float("inf"), steer_snapshot

    if compatibility_required and (
        compatibility_telemetry["attempts"] == 0
        or compatibility_telemetry["successful_steps"] == 0
        or compatibility_telemetry["nonzero_gradient_steps"] == 0
    ):
        compatibility_telemetry["status"] = "INVALID_NO_K2_COMPATIBILITY_GRADIENT"
        if steer_snapshot:
            model.load_state_dict(steer_snapshot, strict=False)
        print(f"  ⚠️ Router bootstrap invalid: {compatibility_telemetry}")
        return None, float("inf"), steer_snapshot

    if best_state:
        model.load_state_dict(best_state, strict=False)

    # Re-freeze SteerNets after bootstrap; router stays trained
    for sn in model.steer_nets.values():
        for p in sn.parameters():
            p.requires_grad = False

    model.set_routing_override(None)
    save_path = cfg.router_bootstrap_save_path
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    # BUG-C5 fix: was raw torch.save (no manifest/hash/config provenance).
    # Now uses save_ars_checkpoint for consistency with all other phases.
    save_ars_checkpoint(model, save_path, phase='phase1.5-router-bootstrap', cfg=cfg, prefixes=('router.',))
    print(f"✅ Router Bootstrap complete — best_probe_loss = {best_loss:.4f}")
    # steer_snapshot is returned so the caller can restore Phase-1 SteerNet state
    # if the router preflight subsequently fails (prevents corrupting Phase-1 work).
    return best_state, best_loss, steer_snapshot


def run_staged_phase(
    model, tokenizer, train_data, eval_data, cfg: RSCConfig, device: torch.device,
    stage_name: str, epochs: int, param_groups: List[Dict[str, Any]], save_path: str,
    train_steer: bool, train_gate: bool, train_router: bool,
    global_step: int = 0, baseline_ema: Optional[List[float]] = None, patience: int = 3,
    init_best_acc: float = 0.0, init_best_val: float = float("inf"),
    resume_state: Optional[Dict[str, Any]] = None,
) -> Tuple[int, float, float]:
    """Run a Phase-2 stage, optionally resuming its optimizer-exact state.

    Memory pressure never raises a raw allocator exception from this loop.  It
    first reclaims cache, reduces the next RLOO probe if needed, and writes a
    complete training-state checkpoint before returning control to the caller
    on genuine active-memory exhaustion.
    """
    if baseline_ema is None:
        baseline_ema = [0.0]
    print("\n" + "=" * 80)
    print(f"{stage_name}")
    print(
        f"         epochs={epochs}  train_steer={train_steer}  "
        f"train_gate={train_gate}  train_router={train_router}"
    )
    for g in param_groups:
        print(f"         group size={sum(p.numel() for p in g['params']):,}  lr={g['lr']:.2e}")
    print("=" * 80)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.98), eps=1e-8)
    all_params = [p for g in param_groups for p in g["params"]]
    steps_per_epoch = len(train_data) // cfg.effective_batch_size
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * cfg.phase2_warmup_ratio)
    lr_reducer = LRReducer(
        optimizer, total_steps, factor=cfg.lr_reduce_factor, patience=cfg.lr_reduce_patience,
        min_lr=cfg.lr_reduce_min_lr, warmup_steps=warmup_steps,
    )
    lr_reducer.base_lrs = [g["lr"] for g in param_groups]
    for i, g in enumerate(optimizer.param_groups):
        g["lr"] = lr_reducer.base_lrs[i]
    grad_mon = GradientMonitor(cfg.grad_warn_threshold)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_acc, best_val, best_state, patience_ctr = init_best_acc, init_best_val, None, 0
    start_epoch = 0
    resume_batch_start = 0
    resume_indices = None
    effective_rloo_batch_size = max(2, int(getattr(cfg, "router_rloo_batch_size", 8)))
    if resume_state is not None:
        progress = resume_state.get("progress", {})
        if progress.get("stage_name") != stage_name:
            raise RuntimeError(
                f"Phase-2 state stage mismatch: expected {stage_name!r}, "
                f"got {progress.get('stage_name')!r}."
            )
        expected_flags = (bool(train_steer), bool(train_gate), bool(train_router))
        saved_flags = tuple(bool(progress.get(key, False)) for key in ("train_steer", "train_gate", "train_router"))
        if saved_flags != expected_flags:
            raise RuntimeError("Phase-2 state trainable-component contract does not match the requested stage.")
        optimizer.load_state_dict(resume_state["optimizer"])
        scaler.load_state_dict(resume_state["scaler"])
        lr_reducer.load_state_dict(resume_state["lr_reducer"])
        _restore_rng_state(resume_state["rng"])
        start_epoch = int(progress["epoch"])
        resume_batch_start = int(progress["next_batch_start"])
        resume_indices = list(progress.get("indices", []))
        global_step = int(progress["global_step"])
        best_acc = float(progress["best_acc"])
        best_val = float(progress["best_val"])
        patience_ctr = int(progress["patience_ctr"])
        baseline_ema[:] = list(progress.get("baseline_ema", baseline_ema))
        effective_rloo_batch_size = max(2, int(progress.get("effective_rloo_batch_size", effective_rloo_batch_size)))
        print(
            f"  ↪️  Resuming {stage_name} at epoch {start_epoch + 1}, batch offset {resume_batch_start}, "
            f"global_step={global_step}, RLOO batch={effective_rloo_batch_size}."
        )

    # A previous controlled memory pause may leave this diagnostic on a
    # long-lived notebook wrapper.  It describes the old invocation only;
    # clearing it here prevents a successful resumed stage from being
    # misreported as paused.
    if hasattr(model, "_phase2_memory_emergency"):
        delattr(model, "_phase2_memory_emergency")

    stop_for_memory = False
    for epoch in range(start_epoch, epochs):
        model.train()
        # The frozen backbone must remain in inference mode.  Leaving it in
        # ``train`` here makes a frozen model's dropout part of the target the
        # small ARS modules are trying to optimise, while deployment uses
        # ``eval``.  Frozen route/steer modules also need their deployment
        # behaviour when training only the gate.
        model.base_model.eval()
        if not train_router:
            model.router.eval()
        if not train_steer:
            for steer_net in model.steer_nets.values():
                steer_net.eval()
        model.set_gate_freeze(not train_gate, 1.0)
        # Gate-only phases need the real task loss.  The gate still observes
        # detached h/delta, so this does not unfreeze or backpropagate into the
        # SteerNet; it only lets task error calibrate alpha.
        model._joint_grad_mode = bool(train_gate)
        if epoch == start_epoch and resume_indices:
            indices = resume_indices
            batch_start_offset = resume_batch_start
        else:
            indices = list(range(len(train_data)))
            random.shuffle(indices)
            batch_start_offset = 0
        epoch_task, n_opt_steps, accum_count = 0.0, 0, 0
        no_gradient_windows = 0

        epoch_gate_l1, epoch_gate_div, epoch_gate_antisat, epoch_gate_contrast, n_gate_aux_steps = 0.0, 0.0, 0.0, 0.0, 0
        epoch_policy_loss, epoch_mean_k, n_router_rloo_steps = 0.0, 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(
            range(batch_start_offset, len(indices), cfg.batch_size),
            desc=f"{stage_name.split(':')[0]} {epoch + 1}/{epochs}",
        )

        for start in pbar:
            batch_idx = indices[start : start + cfg.batch_size]
            ids = torch.stack([train_data[i]["input_ids"] for i in batch_idx]).to(device)
            mask = torch.stack([train_data[i]["attention_mask"] for i in batch_idx]).to(device)
            lbls = torch.stack([train_data[i]["labels"] for i in batch_idx]).to(device)
            shield = None
            if "prompt_shield_mask" in train_data[batch_idx[0]]:
                shield = torch.stack([train_data[i]["prompt_shield_mask"] for i in batch_idx]).to(device)
            calc_mask = None
            if "calc_mask" in train_data[batch_idx[0]]:
                calc_mask = torch.stack([train_data[i]["calc_mask"] for i in batch_idx]).to(device)
            ids, mask, lbls, shield, calc_mask = _trim_batch(ids, mask, lbls, shield, calc_mask)
            model._attn_mask_cache = mask
            _set_batch_context(model, mask, lbls, shield)

            # Compute the detached ON/OFF reward before creating this step's
            # task graph.  Reusing the live gate alphas below avoids the old
            # extra graph on a 4-example PG batch, the direct cause of T4 OOM.
            pg_advantage = None
            if (
                train_gate
                and not train_steer
                and float(getattr(cfg, "lambda_pg", 0.0)) > 0.0
                and accum_count == 0
                and global_step % cfg.pg_every_n_steps == 0
            ):
                pg_advantage = compute_pg_advantage_lite(
                    model, tokenizer, ids, lbls, device, cfg, baseline_ema,
                    attention_masks=mask,
                    router_context_mask=(
                        mask.bool() & lbls.eq(-100) & ~(
                            shield.bool() if shield is not None else torch.zeros_like(mask, dtype=torch.bool)
                        )
                    ),
                    shield_mask=shield,
                )
                _set_batch_context(model, mask, lbls, shield)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(input_ids=ids, attention_mask=mask, use_cache=False)
                task_loss = _answer_aware_loss(out.logits, lbls, cfg, calc_mask=calc_mask)

            micro_loss = (task_loss / cfg.grad_accum) if (torch.isfinite(task_loss) and task_loss.requires_grad) else None

            if train_gate:
                l1_loss = compute_l1_gate(model)
                budget_loss = compute_gate_budget_loss(model, target_sparsity=getattr(cfg, "gate_target_sparsity", 0.20))
                asat_loss = compute_antisat_gate(model, threshold=0.85)
                contrast_loss = compute_contrastive_gate_loss(model, ids, lbls, tokenizer, device)
                diversity_loss = compute_gate_diversity_loss(model, floor=cfg.gate_diversity_floor)
                overlap_loss = compute_gate_overlap_loss(model)
                activity_floor_loss = compute_route_activity_floor_loss(
                    model, min_fraction=float(getattr(cfg, "route_activity_min_fraction", 0.05))
                )
                epoch_gate_l1 += l1_loss.item()
                epoch_gate_div += diversity_loss.item()
                epoch_gate_antisat += asat_loss.item()
                epoch_gate_contrast += contrast_loss.item()
                n_gate_aux_steps += 1
                aux_loss = (
                    cfg.lambda_l1_gate * l1_loss
                    + getattr(cfg, "lambda_gate_budget", 0.50) * budget_loss
                    + cfg.lambda_gate_antisat * asat_loss
                    + cfg.lambda_gate_usefulness * contrast_loss
                    + cfg.lambda_gate_diversity * diversity_loss
                    + float(getattr(cfg, "lambda_gate_overlap", 0.0)) * overlap_loss
                    + float(getattr(cfg, "lambda_route_activity_floor", 0.0)) * activity_floor_loss
                ) / cfg.grad_accum
                if torch.isfinite(aux_loss) and aux_loss.requires_grad:
                    micro_loss = aux_loss if micro_loss is None else (micro_loss + aux_loss)

            if pg_advantage is not None:
                pg_loss = compute_pg_gate_loss_from_live_alphas(model, pg_advantage) / cfg.grad_accum
                if torch.isfinite(pg_loss) and pg_loss.requires_grad:
                    # PG and task/aux losses share this forward's gate graph;
                    # combine them before one backward rather than retaining
                    # the whole Phi-2 graph for a second backward call.
                    micro_loss = (cfg.lambda_pg * pg_loss) if micro_loss is None else (micro_loss + cfg.lambda_pg * pg_loss)

            if micro_loss is not None:
                scaler.scale(micro_loss).backward()
            del out


            # BUG-08 fix: trigger RLOO only at the last micro-batch of the accumulation window
            # (i.e. just before the optimizer.step()). Previously triggered at accum_count==0
            # (first micro-batch), meaning router got grad_accum × more updates than task loss.
            is_last_accum = (accum_count == cfg.grad_accum - 1) or (start + cfg.batch_size >= len(indices))
            if train_router and is_last_accum and global_step % cfg.router_probe_every_n_steps == 0:
                # A B=1 RLOO probe cannot distinguish input-specific routing
                # from a globally collapsed argmax.  Draw a separate vectorised
                # probe minibatch so the mutual-information diversity term and
                # counterfactual rewards see heterogeneous prompts.
                probe_idx = random.sample(
                    range(len(train_data)),
                    min(effective_rloo_batch_size, len(train_data)),
                )
                probe_ids = torch.stack([train_data[i]["input_ids"] for i in probe_idx]).to(device)
                probe_mask = torch.stack([train_data[i]["attention_mask"] for i in probe_idx]).to(device)
                probe_lbls = torch.stack([train_data[i]["labels"] for i in probe_idx]).to(device)
                probe_shield = None
                if "prompt_shield_mask" in train_data[probe_idx[0]]:
                    probe_shield = torch.stack([train_data[i]["prompt_shield_mask"] for i in probe_idx]).to(device)
                probe_calc = None
                if "calc_mask" in train_data[probe_idx[0]]:
                    probe_calc = torch.stack([train_data[i]["calc_mask"] for i in probe_idx]).to(device)
                probe_ids, probe_mask, probe_lbls, probe_shield, probe_calc = _trim_batch(
                    probe_ids, probe_mask, probe_lbls, probe_shield, probe_calc
                )
                prompt_mask = probe_mask.bool() & probe_lbls.eq(-100) & ~(
                    probe_shield.bool() if probe_shield is not None else torch.zeros_like(probe_mask, dtype=torch.bool)
                )
                router_loss, rloo_diag = compute_router_rloo_loss(
                    model, probe_ids, probe_mask, probe_lbls, cfg,
                    prompt_mask=prompt_mask, shield_mask=probe_shield,
                    calc_mask=probe_calc,
                )
                if torch.isfinite(router_loss) and router_loss.requires_grad:
                    scaler.scale(router_loss).backward()
                if rloo_diag:
                    epoch_policy_loss += rloo_diag["policy_loss"]
                    epoch_mean_k += rloo_diag["mean_k_sampled"]
                    n_router_rloo_steps += 1
                del router_loss, rloo_diag, probe_ids, probe_mask, probe_lbls, probe_shield, probe_calc, prompt_mask
            accum_count += 1
            is_last = start + cfg.batch_size >= len(indices)
            if accum_count == cfg.grad_accum or is_last:
                if not _optimizer_has_gradients(all_params):
                    # Legal for K=0: with a frozen backbone and no selected
                    # layer, this window contains no trainable graph.  AMP
                    # cannot unscale before scale/backward, so record the
                    # condition and continue instead of crashing the run.
                    no_gradient_windows += 1
                    if no_gradient_windows == 1:
                        print(
                            f"  ⚠️ {stage_name.split(':')[0]} has a no-gradient window "
                            "(usually K=0/no selected layer); optimizer step skipped."
                        )
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                else:
                    scaler.unscale_(optimizer)
                    for p in all_params:
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            p.grad.zero_()
                    torch.nn.utils.clip_grad_norm_(all_params, cfg.clip_grad)
                    if cfg.use_gradient_monitor:
                        grad_mon.check(all_params, global_step)
                    scaler.step(optimizer)
                    scaler.update()
                    lr_reducer.scheduler_step()
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    global_step += 1
                    n_opt_steps += 1

                    watchdog_every = max(1, int(getattr(cfg, "cuda_memory_watchdog_every_n_steps", 50)))
                    state_every = max(1, int(getattr(cfg, "phase2_state_checkpoint_every_n_steps", watchdog_every)))
                    should_checkpoint_state = global_step % state_every == 0
                    memory_status = None
                    if global_step % watchdog_every == 0:
                        log_gpu_memory(f"run_staged_phase step {global_step}", cfg, reset_peak=False)
                        memory_status = enforce_gpu_memory_budget(f"{stage_name} step {global_step}", cfg)
                        if memory_status["action"] in {"reclaimed", "emergency"}:
                            reduced = max(2, int(getattr(cfg, "cuda_memory_reduced_rloo_batch_size", 4)))
                            if effective_rloo_batch_size > reduced:
                                effective_rloo_batch_size = reduced
                                print(
                                    f"    [mem] Reducing subsequent/resumed RLOO probe batch to {reduced}."
                                )
                        should_checkpoint_state = should_checkpoint_state or memory_status["action"] == "emergency"

                    next_batch_start = start + cfg.batch_size
                    if should_checkpoint_state:
                        save_phase2_training_state(
                            cfg.phase2_training_state_path,
                            model=model, optimizer=optimizer, scaler=scaler, lr_reducer=lr_reducer,
                            stage_name=stage_name, epochs=epochs, epoch=epoch, next_batch_start=next_batch_start,
                            indices=indices, global_step=global_step, best_acc=best_acc, best_val=best_val,
                            patience_ctr=patience_ctr, baseline_ema=baseline_ema,
                            effective_rloo_batch_size=effective_rloo_batch_size,
                            train_steer=train_steer, train_gate=train_gate, train_router=train_router,
                            save_path=save_path,
                        )
                    if memory_status is not None and memory_status["action"] == "emergency":
                        model._phase2_memory_emergency = memory_status
                        stop_for_memory = True
                        print("  ⚠️ Phase 2 paused safely after a resumable state checkpoint; no allocator crash was raised.")
                        break

                    if (
                        train_router
                        and cfg.router_checkpoint_every_n_steps > 0
                        and global_step % cfg.router_checkpoint_every_n_steps == 0
                    ):
                        best_acc, best_val, sub_state, patience_ctr, _, _, _ = _evaluate_and_maybe_checkpoint(
                            model, tokenizer, eval_data, cfg, device, save_path,
                            best_acc, best_val, patience_ctr, train_router,
                            label=f"step {global_step} (mid-epoch {epoch + 1})",
                        )
                        if sub_state is not None:
                            best_state = sub_state
                        if patience_ctr >= patience:
                            print(f"  {stage_name.split(':')[0]} early stopping (mid-epoch).")
                            break

            epoch_task += task_loss.item()
            postfix_kwargs = {"task": f"{task_loss.item():.4f}"}
            if pg_advantage is not None:
                postfix_kwargs["pg_adv"] = f"{pg_advantage:+.3f}"
            if train_router and n_router_rloo_steps > 0:
                postfix_kwargs["mean_k"] = f"{epoch_mean_k / n_router_rloo_steps:.2f}"
            pbar.set_postfix(**postfix_kwargs)

        if stop_for_memory:
            break
        if patience_ctr >= patience:
            break
        avg_task = epoch_task / max(1, n_opt_steps * cfg.grad_accum)
        best_acc, best_val, epoch_state, patience_ctr, improved, cur_val_loss, cur_acc = _evaluate_and_maybe_checkpoint(
            model, tokenizer, eval_data, cfg, device, save_path,
            best_acc, best_val, patience_ctr, train_router,
            label=f"epoch {epoch + 1} boundary",
        )
        if epoch_state is not None:
            best_state = epoch_state
        save_phase2_training_state(
            cfg.phase2_training_state_path,
            model=model, optimizer=optimizer, scaler=scaler, lr_reducer=lr_reducer,
            stage_name=stage_name, epochs=epochs, epoch=epoch + 1, next_batch_start=0, indices=[],
            global_step=global_step, best_acc=best_acc, best_val=best_val,
            patience_ctr=patience_ctr, baseline_ema=baseline_ema,
            effective_rloo_batch_size=effective_rloo_batch_size,
            train_steer=train_steer, train_gate=train_gate, train_router=train_router,
            save_path=save_path,
        )
        skip_note = f" | skipped_no_grad_windows={no_gradient_windows}" if no_gradient_windows else ""
        print(f"  {stage_name.split(':')[0]} Epoch {epoch + 1}: task={avg_task:.4f}  val={cur_val_loss:.4f}  acc={cur_acc:.3f}  {'✅' if improved else ''}{skip_note}")

    return global_step, best_val, best_acc


def train_streamlined_ars(model, tokenizer, train_data, eval_data, cfg: RSCConfig, device: torch.device):
    """Streamlined 2-Stage ARS Training Pipeline (Priority 5 / EXP-06).
    
    Stage 1: Representation Warmup (SteerNet on target layers, Gate = 1.0)
    Stage 2: Joint Gated Optimization (SteerNet + Token Gate simultaneously with decoupled LRs)
    """
    logs = {}
    print("\n" + "=" * 80)
    print("🚀 RUNNING STREAMLINED 2-STAGE ARS CURRICULUM")
    print("=" * 80)

    # Stage 1: Warmup
    orig_p1_epochs = cfg.phase1_epochs
    cfg.phase1_epochs = getattr(cfg, "streamlined_warmup_epochs", 3)
    best_state_p1, best_val_p1, p1_logs = train_phase1_steernet(
        model, train_data, eval_data, cfg, device, tokenizer=None
    )
    cfg.phase1_epochs = orig_p1_epochs
    logs.update(p1_logs)

    # Stage 2: Joint Gated Optimization
    model.set_gate_freeze(False)
    steer_params = []
    gate_params = []
    for sn in model.steer_nets.values():
        for p in sn.parameters():
            p.requires_grad = True
            steer_params.append(p)
    for g in model.gates.values():
        for p in g.parameters():
            p.requires_grad = True
            gate_params.append(p)

    train_router = getattr(cfg, "use_router", True) and getattr(cfg, "fixed_layers", None) is None
    for p in model.router.parameters():
        p.requires_grad = train_router

    steer_lr = getattr(cfg, "streamlined_steer_lr", 1e-4)
    gate_lr = getattr(cfg, "streamlined_gate_lr", 1e-3)
    param_groups = [
        {"params": steer_params, "lr": steer_lr},
        {"params": gate_params, "lr": gate_lr},
    ]
    if train_router:
        param_groups.append({
            "params": list(model.router.parameters()),
            "lr": cfg.router_bootstrap_lr * cfg.joint_finetune_router_lr_scale,
        })

    global_step, best_val_s2, best_acc_s2 = run_staged_phase(
        model, tokenizer, train_data, eval_data, cfg, device,
        stage_name="STREAMLINED STAGE 2: Joint Gated Optimization",
        epochs=getattr(cfg, "streamlined_joint_epochs", 3),
        param_groups=param_groups,
        save_path=cfg.phase2_save_path,
        train_steer=True, train_gate=True, train_router=train_router,
        global_step=0, baseline_ema=[0.0], patience=max(1, cfg.phase2_patience),
    )
    logs["streamlined_stage2_val"] = best_val_s2
    logs["streamlined_stage2_acc"] = best_acc_s2

    if os.path.exists(cfg.phase2_save_path):
        load_ars_checkpoint_strict(model, cfg.phase2_save_path, device=device)
        print(f"  ✅ Loaded final streamlined checkpoint from {cfg.phase2_save_path}")

    print("\n" + "=" * 80)
    print("🎉 STREAMLINED ARS TRAINING COMPLETE!")
    print(f"   Stage 1 best validation loss: {best_val_p1:.4f}")
    print(f"   Stage 2 best validation loss: {best_val_s2:.4f} (acc={best_acc_s2:.3f})")
    print(f"   Final saved checkpoint      : {cfg.phase2_save_path}")
    print("=" * 80)
    return logs, best_val_p1, best_val_s2


def _purge_ckpt(path: str) -> None:
    """ISSUE-15: delete a checkpoint and its manifest before a new run starts.

    Prevents phases that save nothing (all-NaN val, early exception, best_val
    never improving on init_best_val) from silently loading a previous run's file.
    """
    for p in (path, path + ".manifest.json"):
        if os.path.exists(p):
            os.remove(p)


def _fallback_to_certified_fixed_steernet(
    model, cfg: RSCConfig, logs: Dict[str, Any], steer_snapshot: Dict[str, torch.Tensor], reason: str,
) -> None:
    """Restore the certified Phase-1 route when adaptive routing is rejected.

    A failed router is not evidence that the independently confirmed SteerNet
    is ineffective.  Preserve that control for the notebook's full benchmark
    instead of terminating a costly Kaggle run without its fixed-route result.
    """
    if steer_snapshot:
        model.load_state_dict(steer_snapshot, strict=False)
    fixed_layers = list(getattr(cfg, "fixed_steer_eval_layers", None) or [])
    if not fixed_layers:
        raise RuntimeError(
            "Cannot activate certified fixed-SteerNet fallback without certified fixed SteerNet layers."
        )
    model.set_eligible_routing_layers(None)
    model.set_router_warmup(False)
    model.set_routing_override("only", fixed_layers)
    scales = getattr(model, "_certified_gate_scales", None)
    if scales:
        model.set_gate_freeze(True, scales)
    else:
        model.set_gate_freeze(True, float(getattr(cfg, "fixed_steer_eval_gate_scale", 1.0)))
    save_ars_checkpoint(model, cfg.phase2_save_path, phase="certified-fixed-steernet-fallback", cfg=cfg)
    logs["adaptive_phases_status"] = "FALLBACK_TO_CERTIFIED_FIXED_STEERNET"
    logs["fixed_steernet_fallback"] = {
        "layers": fixed_layers,
        "gate_scales": scales or float(getattr(cfg, "fixed_steer_eval_gate_scale", 1.0)),
        "reason": reason,
    }
    print(
        "  ↩️  FALLBACK_TO_CERTIFIED_FIXED_STEERNET: "
        f"L{fixed_layers[0]} alpha={cfg.fixed_steer_eval_gate_scale:.2f}; {reason}"
    )


def _evaluate_phase2_deployment(
    model, tokenizer, confirm_rows, cfg: RSCConfig, device: torch.device, min_deployment_questions: int,
) -> Dict[str, Any]:
    """Run the final paired deployment decision after normal or resumed Phase 2.

    Keeping this in a callable helper lets an interrupted Phase-2 run finish
    the same held-out confirmation protocol after restart, instead of leaving
    a model checkpoint without a deployment decision.
    """
    from src.evaluation.deployment import finalize_deployment_by_shot, paired_delta_ci

    def evaluate_flags(override: Optional[str], n_shot: int) -> List[int]:
        model.eval()
        if override == "fixed":
            fixed_layers = list(getattr(cfg, "fixed_steer_eval_layers", None) or [])
            if not fixed_layers:
                raise RuntimeError("Adaptive-vs-fixed deployment gate requires certified fixed SteerNet layers")
            model.set_routing_override("only", fixed_layers)
            scales = getattr(model, "_certified_gate_scales", None)
            if scales:
                model.set_gate_freeze(True, scales)
            else:
                model.set_gate_freeze(True, float(getattr(cfg, "fixed_steer_eval_gate_scale", 1.0)))
        else:
            model.set_gate_freeze(False)
            model.set_routing_override(override)
        flags = []
        for row in confirm_rows:
            question = row.get("question") if isinstance(row, dict) else None
            if not question and hasattr(row, "__getitem__"):
                try:
                    question = row["question"]
                except (KeyError, IndexError, TypeError):
                    question = None
            if not question:
                continue
            prompt = build_benchmark_prompt("gsm8k", question, n_shot=n_shot)
            response = generate_response(model, tokenizer, prompt, device, max_new_tokens=384)
            final_num = row.get("final_num") if isinstance(row, dict) else None
            if final_num is None and hasattr(row, "__getitem__"):
                try:
                    final_num = row["final_num"]
                except (KeyError, IndexError, TypeError):
                    final_num = None
            flags.append(int(evaluate_sample_answer("gsm8k", response, final_num)["correct"]))
        return flags

    if not confirm_rows:
        return {
            "deployment_mode": "baseline",
            "deployment_modes_by_shot": {"0": "baseline", "2": "baseline"},
            "deployment_flags": {"baseline": [], "ars": [], "baseline_clusters": [], "ars_clusters": []},
        }

    deployment_shots = tuple(getattr(cfg, "deployment_eval_shots", None) or (0, 2))
    print(f"\n  [deployment] Evaluating {len(confirm_rows)} reserved confirmation rows across shots={deployment_shots}...")
    base_by_shot, ars_by_shot, fixed_by_shot = [], [], []
    for n_shot in deployment_shots:
        baseline = evaluate_flags("none", n_shot)
        ars = evaluate_flags(None, n_shot)
        fixed = evaluate_flags("fixed", n_shot) if bool(getattr(cfg, "require_adaptive_beats_fixed", False)) else None
        if len(baseline) != len(confirm_rows) or len(ars) != len(confirm_rows) or len(baseline) != len(ars):
            raise RuntimeError(
                "Deployment confirmation did not produce one paired outcome per reserved question: "
                f"shot={n_shot}, baseline={len(baseline)}, ars={len(ars)}, expected={len(confirm_rows)}"
            )
        base_by_shot.append(baseline)
        ars_by_shot.append(ars)
        if fixed is not None:
            if len(fixed) != len(confirm_rows):
                raise RuntimeError("Fixed SteerNet deployment confirmation did not produce one outcome per reserved question")
            fixed_by_shot.append(fixed)
        delta, lower, upper = paired_delta_ci(baseline, ars)
        print(
            f"    {n_shot}-shot: Base={sum(baseline)}/{len(baseline)} ({sum(baseline)/len(baseline):.1%}) | "
            f"ARS={sum(ars)}/{len(ars)} ({sum(ars)/len(ars):.1%}) | "
            f"Δ={delta*100:+.2f}pp, 95% CI=[{lower*100:+.2f}, {upper*100:+.2f}]pp (n={len(baseline)})"
        )

    outcomes = {shot: (base_by_shot[index], ars_by_shot[index]) for index, shot in enumerate(deployment_shots)}
    fixed_outcomes = {shot: fixed_by_shot[index] for index, shot in enumerate(deployment_shots)} if fixed_by_shot else None
    modes_by_shot = finalize_deployment_by_shot(
        outcomes, cfg.phase2_save_path, min_n=min_deployment_questions,
        fixed_outcomes_by_shot=fixed_outcomes,
        require_adaptive_beats_fixed=bool(getattr(cfg, "require_adaptive_beats_fixed", False)),
    )
    base_clusters = [list(values) for values in zip(*base_by_shot)]
    ars_clusters = [list(values) for values in zip(*ars_by_shot)]
    delta, lower, upper = paired_delta_ci(base_clusters, ars_clusters)
    print(
        f"  [deployment] Multi-shot ({'+'.join(str(shot) for shot in deployment_shots)} clustered by question) "
        f"Δ = {delta*100:+.2f}pp, 95% CI = [{lower*100:+.2f}, {upper*100:+.2f}]pp "
        f"(questions={len(base_clusters)}, views={len(base_by_shot)})"
    )
    mode = (
        "ars" if all(value == "ars" for value in modes_by_shot.values())
        else "fixed_steernet" if modes_by_shot and all(value == "fixed_steernet" for value in modes_by_shot.values())
        else "baseline"
    )
    if hasattr(model, "deployment_disabled"):
        model.deployment_disabled = mode == "baseline"
        model.deployment_modes_by_shot = modes_by_shot
    print(f"  [deployment] Final deployment mode decided: {mode.upper()} (deployment_disabled={getattr(model, 'deployment_disabled', False)})")
    return {
        "deployment_mode": mode,
        "deployment_modes_by_shot": modes_by_shot,
        "deployment_flags": {
            "baseline": [flag for flags in base_by_shot for flag in flags],
            "ars": [flag for flags in ars_by_shot for flag in flags],
            "baseline_clusters": base_clusters,
            "ars_clusters": ars_clusters,
        },
    }


def _resume_phase2_from_training_state(
    model, tokenizer, router_data, eval_sel, confirm_rows, cfg: RSCConfig, device: torch.device,
) -> Tuple[Dict[str, Any], float, float]:
    """Resume the interrupted Phase-2 stage without replaying Phases 1--1.5.

    The saved stage contract determines trainable components and the exact
    intra-epoch cursor.  It is deliberately fail-closed: a state from another
    stage or model topology cannot silently degrade into a model-only warm
    start.
    """
    state = load_phase2_training_state(cfg.phase2_training_state_path, model, device)
    progress = state["progress"]
    stage_name = str(progress["stage_name"])
    train_steer = bool(progress["train_steer"])
    train_gate = bool(progress["train_gate"])
    train_router = bool(progress["train_router"])
    epochs = int(progress.get("epochs", 0))
    if epochs <= 0:
        raise RuntimeError("Phase-2 resume state has no valid total epoch count.")
    if not (train_steer or train_gate or train_router):
        raise RuntimeError("Phase-2 resume state has no trainable ARS component.")

    steer_params, gate_params, router_params = [], [], list(model.router.parameters())
    for steer_net in model.steer_nets.values():
        for parameter in steer_net.parameters():
            parameter.requires_grad = train_steer
            steer_params.append(parameter)
    for key in sorted(model.gates.keys(), key=lambda item: int(item)):
        for parameter in model.gates[key].parameters():
            parameter.requires_grad = train_gate
            gate_params.append(parameter)
    for parameter in router_params:
        parameter.requires_grad = train_router

    # Preserve the parameter-group ordering used by every Phase-2 stage.  The
    # optimizer payload validates group cardinality when loaded in
    # ``run_staged_phase``; a mismatch is an intentional hard failure.
    groups = []
    if train_steer:
        groups.append({"params": steer_params, "lr": cfg.steer_finetune_lr})
    if train_gate:
        groups.append({"params": gate_params, "lr": cfg.gate_only_lr})
    if train_router:
        groups.append({"params": router_params, "lr": cfg.router_bootstrap_lr * cfg.joint_finetune_router_lr_scale})
    expected_groups = len(state["optimizer"].get("param_groups", []))
    if len(groups) != expected_groups:
        raise RuntimeError(
            f"Phase-2 resume optimizer-group mismatch: state={expected_groups}, reconstructed={len(groups)}."
        )

    print(f"\n  [Phase 2 Resume] Loading exact staged state from {cfg.phase2_training_state_path}...")
    global_step, best_val, best_acc = run_staged_phase(
        model, tokenizer, router_data, eval_sel, cfg, device,
        stage_name=stage_name, epochs=epochs, param_groups=groups,
        save_path=str(progress["save_path"]),
        train_steer=train_steer, train_gate=train_gate, train_router=train_router,
        global_step=int(progress["global_step"]),
        baseline_ema=list(progress.get("baseline_ema", [0.0])),
        patience=cfg.phase2_patience,
        init_best_acc=float(progress["best_acc"]), init_best_val=float(progress["best_val"]),
        resume_state=state,
    )
    if getattr(model, "_phase2_memory_emergency", None):
        return {
            "adaptive_phases_status": "PAUSED_MEMORY_PRESSURE",
            "phase2_resume_state": cfg.phase2_training_state_path,
            "phase2_memory": model._phase2_memory_emergency,
            "global_step": global_step,
        }, best_val, best_val

    best_path = str(progress["save_path"])
    if not os.path.exists(best_path):
        raise RuntimeError("Resumed Phase 2 finished without a best model checkpoint.")
    load_ars_checkpoint_strict(model, best_path, device=device)
    save_ars_checkpoint(model, cfg.phase2_save_path, phase="phase2-resumed-complete", cfg=cfg)
    model.set_router_exploration(False)
    logs = {
        "adaptive_phases_status": "RESUMED_PHASE2_COMPLETE",
        "phase2_resume_state": cfg.phase2_training_state_path,
        "p2b_val": best_val,
        "p2b_acc": best_acc,
        "ars_intervention_audit": audit_ars_intervention_health(model, eval_sel, cfg, device),
    }
    logs.update(
        _evaluate_phase2_deployment(
            model, tokenizer, confirm_rows, cfg, device,
            int(getattr(cfg, "deployment_min_examples", 120)),
        )
    )
    return logs, best_val, best_val


def train_rsc(model, tokenizer, train_data, eval_data, cfg: RSCConfig, device: torch.device):
    """Complete multi-phase staged training pipeline:
    Phase 1: Pure SteerNet (Warmup Rotation, Gates=1.0)
    Phase 1.5: Router Bootstrap (RLOO Counterfactual Loss)
    Phase 2A: SteerNet Fine-Tune (Exhaustive Rotation)
    Phase 2B: Gate-Only Training (Dynamic Routing, Aux & PG Losses)
    Phase 2C: Light Joint Fine-Tuning (SteerNet + Gate + Router)
    Phase 2D: Gate Re-Tune (Against Final Router)
    """
    if getattr(cfg, "curriculum_mode", "full") == "streamlined":
        return train_streamlined_ars(model, tokenizer, train_data, eval_data, cfg, device)

    logs = {}
    manifest = get_manifest()
    if getattr(manifest, "_auto", False):
        manifest.configure(
            path=getattr(cfg, "manifest_path", "./run_manifest.json"),
            policy=getattr(cfg, "failure_policy", "warn"),
        )
    manifest.set("config", cfg.__dict__ if hasattr(cfg, "__dict__") else vars(cfg))
    manifest.set("steer_scale", getattr(cfg, "steer_scale", 0.30))

    resume_phase2 = bool(getattr(cfg, "resume_phase2", False))
    if resume_phase2 and not os.path.exists(cfg.phase2_training_state_path):
        raise FileNotFoundError(
            "CFG.resume_phase2 is True but training state was not found at "
            f"{cfg.phase2_training_state_path}. Refusing silent full retraining."
        )
    # A notebook may invoke ``train_rsc`` more than once on the same wrapper.
    # Phase 1 must begin against the full catalog; its newly certified Phase
    # 1B pool is installed later before router bootstrap.
    if hasattr(model, "set_eligible_routing_layers"):
        model.set_eligible_routing_layers(None)

    # ISSUE-15: purge stale checkpoint files from previous runs before writing new ones
    if not resume_phase2:
        for _ckpt_path in (
            getattr(cfg, "gate_only_save_path", None),
            getattr(cfg, "steer_finetune_save_path", None),
            getattr(cfg, "phase2_save_path", None),
        ):
            if _ckpt_path:
                _purge_ckpt(_ckpt_path)

    # ISSUE-4: Split training rows into steer-set (80%) and router-set (20%)
    steer_data, router_data = split_steer_router(list(train_data), router_fraction=0.2, seed=cfg.seed)

    # R6: Reserve held-out confirmation rows (up to deployment_examples or half of eval_data)
    # exclusively for the final deployment decision so selection and audits are strictly uncontaminated.
    # A deployment decision is invalid without the configured number of
    # independent questions, even though each question is viewed at two shots.
    n_dep = min(int(getattr(cfg, "deployment_examples", 300)), len(eval_data) // 2) if len(eval_data) >= 4 else 0
    min_deployment_questions = int(getattr(cfg, "deployment_min_examples", 120))
    if n_dep < min_deployment_questions:
        if getattr(cfg, "failure_policy", "warn") == "strict":
            raise RuntimeError(
                "Insufficient held-out evaluation rows for a deployment decision: "
                f"need {min_deployment_questions} independent confirmation questions plus an equal-sized "
                f"selection split, got eval_data={len(eval_data)} and confirmation_rows={n_dep}."
            )
        manifest.warn(
            "W_EVAL_ROWS",
            f"eval_data ({len(eval_data)}) too small for full deployment split",
            stage="setup",
            eval_data_len=len(eval_data),
            min_deployment_questions=min_deployment_questions,
        )
        eval_sel = list(eval_data)
        confirm_rows = list(eval_data)
    else:
        eval_sel = [eval_data[i] for i in range(len(eval_data) - n_dep)]
        confirm_rows = [eval_data[i] for i in range(len(eval_data) - n_dep, len(eval_data))]
    print(f"  [eval split] eval_data split: {len(eval_sel)} for selection/audits, {len(confirm_rows)} reserved for deployment confirmation.")

    if resume_phase2:
        return _resume_phase2_from_training_state(
            model, tokenizer, router_data, eval_sel, confirm_rows, cfg, device,
        )

    # PHASE 1: Pure SteerNet (uses steer_data, eval_sel)
    with manifest.stage("S1_steernet"):
        resume_phase1 = bool(getattr(cfg, "resume_phase1", False))
        phase1_report_path = getattr(cfg, "phase1_report_save_path", os.path.join(cfg.checkpoint_dir, "phase1_report.json"))
        loaded_existing_phase1 = False

        if resume_phase1 and not os.path.exists(cfg.lora_save_path):
            raise FileNotFoundError(
                "CFG.resume_phase1 is True but checkpoint was not found at "
                f"{cfg.lora_save_path}. Refusing silent full retraining."
            )

        if resume_phase1:
            print(f"\n  [Phase 1 Resume] Loading existing Phase 1 SteerNet weights from {cfg.lora_save_path}...")
            load_ars_checkpoint_strict(model, cfg.lora_save_path, device=device)
            best_val_p1 = 0.0
            p1_logs = {"p1_train": [], "p1_val": []}
            if os.path.exists(phase1_report_path):
                try:
                    with open(phase1_report_path, "r", encoding="utf-8") as f:
                        cached_report = json.load(f)
                    if not _phase1_report_matches_checkpoint(cached_report, cfg.lora_save_path, cfg):
                        raise RuntimeError("cache provenance does not match the loaded Phase-1 checkpoint")
                    logs["phase1_acceptance"] = cached_report
                    print(f"  [Phase 1 Resume] Loaded provenance-verified Phase 1 acceptance report from {phase1_report_path}.")
                    loaded_existing_phase1 = True
                except Exception as e:
                    print(f"  ⚠️ [Phase 1 Resume] Ignoring stale/corrupt report from {phase1_report_path}: {e}")
            if not loaded_existing_phase1:
                print("  [Phase 1 Resume] Re-evaluating acceptance audit on loaded weights...")
                if tokenizer is not None and len(eval_sel) > 0 and len(steer_data) > 0:
                    logs["phase1_acceptance"] = phase1_steernet_acceptance(
                        model, tokenizer, steer_data, eval_sel, cfg, device
                    )
                else:
                    logs["phase1_acceptance"] = {"passed": True, "reliability_waived": True, "status": "PASS"}
                try:
                    logs["phase1_acceptance"] = _cache_phase1_report(
                        logs["phase1_acceptance"], phase1_report_path, cfg.lora_save_path, cfg
                    )
                    print(f"  [Phase 1 Resume] Cached re-evaluated acceptance report at {phase1_report_path}.")
                except Exception as e:
                    manifest.warn("W_REPORT_CACHE", f"Could not cache re-evaluated Phase 1 acceptance report: {e}")
                loaded_existing_phase1 = True
            logs.update(p1_logs)

        if not loaded_existing_phase1:
            if tokenizer is not None:
                best_state_p1, best_val_p1, p1_logs = train_phase1_steernet(
                    model, steer_data, eval_sel, cfg, device, tokenizer=tokenizer
                )
            else:
                best_state_p1, best_val_p1, p1_logs = train_phase1_steernet(
                    model, steer_data, eval_sel, cfg, device
                )
            logs.update(p1_logs)
            if tokenizer is not None and len(eval_sel) > 0 and len(steer_data) > 0:
                logs["phase1_acceptance"] = phase1_steernet_acceptance(
                    model, tokenizer, steer_data, eval_sel, cfg, device
                )
                try:
                    logs["phase1_acceptance"] = _cache_phase1_report(
                        logs["phase1_acceptance"], phase1_report_path, cfg.lora_save_path, cfg
                    )
                except Exception as e:
                    manifest.warn("W_REPORT_CACHE", f"Could not save Phase 1 acceptance cache: {e}")
            else:
                logs["phase1_acceptance"] = {
                    "passed": True,
                    "reliability_waived": True,
                    "status": "PASS",
                    "selected_layer": (model.target_layers[0] if getattr(model, "target_layers", None) else 0),
                    "fixed_steer_gate_scale": getattr(cfg, "steer_scale", 0.30),
                }
        manifest.record_metrics("S1_steernet", best_val=best_val_p1)
        _assert_finite_params(model, "S1_steernet")

    # A provenance-verified resume restores adapter tensors, but the fixed
    # route is experiment metadata rather than a model parameter.  Rehydrate
    # it from the acceptance report so a later router rejection can always
    # fall back to the same independently confirmed `(layer, alpha)` control.
    resumed_report = logs["phase1_acceptance"]
    if resumed_report.get("passed", False):
        selected_layer = resumed_report.get("selected_layer")
        selected_scale = resumed_report.get("fixed_steer_gate_scale")
        if selected_layer is not None and selected_scale is not None:
            cfg.fixed_steer_eval_layers = [int(selected_layer)]
            cfg.fixed_steer_eval_gate_scale = float(selected_scale)
    p1_pre_cap = collect_pre_cap_ratio(model)
    if p1_pre_cap:
        print(f"  [telemetry] Phase 1 SteerNet pre-cap norm ratios (delta_norm / cap): {p1_pre_cap}")
        logs["phase1_pre_cap_ratios"] = p1_pre_cap

    # A router cannot repair a SteerNet that has already failed held-out free
    # generation.  Continuing Phase 2 in that state merely turns an NLL-only
    # artefact into a misleading "ARS" checkpoint, while consuming hours of
    # GPU time.  Preserve Phase 1 for diagnosis, save a strict checkpoint, and
    # return a no-intervention deployment state instead.
    phase1_report = logs["phase1_acceptance"]
    p1_passed = bool(phase1_report.get("passed", False) or phase1_report.get("reliability_waived", False))
    manifest.diagnostic(
        "W_PHASE1_CAPACITY",
        ok=p1_passed,
        message="Phase 1 held-out generation capacity gate failed",
        stage="phase1_acceptance",
        failures=phase1_report.get("failures", []),
    )
    if (
        not p1_passed
        and getattr(cfg, "failure_policy", "warn") == "strict"
        and bool(getattr(cfg, "skip_adaptive_phases_when_phase1_fails", False))
    ):
        print("\n" + "=" * 80)
        print("PHASE 1 CAPACITY GATE: ADAPTIVE PHASES SKIPPED")
        print(
            "         Held-out zero-shot generation did not establish an independently useful SteerNet."
        )
        print(
            "         Router/Gate optimisation is skipped; their NLL-only rewards would be scientifically invalid."
        )
        print("         Phase-1 checkpoint and acceptance evidence are retained for diagnosis.")
        print("=" * 80)
        model.set_routing_override("none")
        model.set_gate_freeze(False)
        save_ars_checkpoint(model, cfg.phase2_save_path, phase="phase1-failed-baseline-deployment", cfg=cfg)
        logs["adaptive_phases_status"] = "SKIPPED_UNPROVEN_STEERNET"
        logs["router_bootstrap_loss"] = None
        logs["router_bootstrap_preflight"] = None
        logs["ars_intervention_audit"] = {
            "status": "NOT_RUN_UNPROVEN_STEERNET",
            "passed": False,
            "failures": ["Phase-1 held-out generation capacity gate failed"],
        }
        print("\n🎉 TRAINING FINISHED WITH DIAGNOSTIC-ONLY PHASE-1 ARTIFACTS")
        print(f"   Phase 1 best validation loss : {best_val_p1:.4f}")
        print(f"   Final saved checkpoint        : {cfg.phase2_save_path}")
        return logs, best_val_p1, best_val_p1

    # Engine v3: establish pair compatibility before a learned router is
    # allowed to score combinatorial routes.  This is deliberately after the
    # strict single-layer acceptance gate, so weak adapters cannot hide in a
    # pair during co-adaptation.
    phase1_safe_snapshot = {
        name: value.detach().clone()
        for name, value in model.state_dict().items() if "steer_nets." in name
    }
    if getattr(cfg, "phase1b_coadapt_epochs", 0) > 0:
        logs["phase1b_coadaptation"] = train_phase1b_coadaptation(
            model, router_data, phase1_report, cfg, device,
        )
        phase1b_layers = logs["phase1b_coadaptation"].get("candidate_layers", [])
        min_pool = int(getattr(cfg, "phase1b_min_candidate_layers", 3))
        if len(phase1b_layers) < min_pool or logs["phase1b_coadaptation"].get("status") != "COMPLETE":
            manifest.diagnostic(
                "W_DEAD_LAYERS",
                ok=False,
                message="Phase 1B did not establish the required safe, depth-diverse co-adaptation pool.",
                stage="phase1b_coadaptation",
            )
            if getattr(cfg, "failure_policy", "warn") == "strict":
                _fallback_to_certified_fixed_steernet(
                    model, cfg, logs, phase1_safe_snapshot,
                    "Phase 1B did not establish the required safe, depth-diverse co-adaptation pool.",
                )
                logs["router_bootstrap_loss"] = None
                logs["router_bootstrap_preflight"] = None
                logs["ars_intervention_audit"] = {
                    "status": "NOT_RUN_INSUFFICIENT_DIVERSE_POOL", "passed": False,
                    "failures": ["Phase 1B safe co-adaptation pool was insufficient"],
                }
                return logs, best_val_p1, best_val_p1
        else:
            if hasattr(model, "set_eligible_routing_layers"):
                model.set_eligible_routing_layers(phase1b_layers)
                logs["eligible_routing_layers"] = list(phase1b_layers)
                print(f"  [routing contract] learned router restricted to Phase-1B layers: {phase1b_layers}")
            cfg.fixed_steer_eval_layers = list(phase1b_layers)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    enforce_gpu_memory_budget("after Phase 1B", cfg)

    # PHASE 1.5: Router Bootstrap (uses router_data, eval_sel; RC-5: returns steer_snapshot for safe restore)
    with manifest.stage("S2_synergy"):
        boot_res = train_router_bootstrap(model, router_data, eval_sel, cfg, device)
        if isinstance(boot_res, tuple) and len(boot_res) == 3:
            _, best_p15, steer_snapshot_p15 = boot_res
        elif isinstance(boot_res, tuple) and len(boot_res) == 2:
            _, best_p15 = boot_res
            steer_snapshot_p15 = None
        else:
            best_p15 = 0.0
            steer_snapshot_p15 = None
        logs["router_bootstrap_loss"] = best_p15
        logs["router_bootstrap_compatibility"] = dict(
            getattr(model, "_last_router_bootstrap_compatibility", {})
        )
        manifest.record_metrics("S2_synergy", best_probe_loss=best_p15)
        _assert_finite_params(model, "S2_synergy")

    try:
        logs["router_bootstrap_preflight"] = audit_router_policy_health(model, eval_sel, cfg, device)
    except TypeError:
        logs["router_bootstrap_preflight"] = audit_router_policy_health(model, cfg)
    router_report = logs["router_bootstrap_preflight"]
    passed = bool(router_report.get("passed", False) or router_report.get("reliability_waived", False))
    manifest.diagnostic(
        "W_ROUTER_COLLAPSE",
        ok=passed,
        message="Router preflight audit failed or collapsed",
        stage="phase2_router_preflight",
        failures=router_report.get("failures", []),
    )
    if (
        not passed
        and getattr(cfg, "failure_policy", "warn") == "strict"
        and bool(getattr(cfg, "skip_adaptive_phases_when_router_fails", False))
    ):
        print("\n" + "=" * 80)
        print("ROUTER DEPLOYMENT GATE: ADAPTIVE PHASES SKIPPED")
        print("         Deployment argmax collapsed; Gate/Phase 2 cannot make this a valid ARS policy.")
        print("         Retaining Phase-1 and Router diagnostics; no Full-ARS checkpoint claim is produced.")
        # Preserve the independently accepted Phase-1 route for the full
        # benchmark; adaptive failure cannot erase fixed-SteerNet evidence.
        _fallback_to_certified_fixed_steernet(
            model, cfg, logs, phase1_safe_snapshot,
            "Router bootstrap or deterministic multi-layer preflight was rejected.",
        )
        print("=" * 80)
        logs["ars_intervention_audit"] = {
            "status": "NOT_RUN_COLLAPSED_ROUTER",
            "passed": False,
            "failures": list(router_report.get("failures", [])),
        }
        # The Phase-1 artifact is intentionally retained.  Do not overwrite
        # the deployment checkpoint with an invalid gate/router composition.
        return logs, best_val_p1, best_val_p1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # SteerNet parameter collection and freezing
    steer_params_ft = []
    for sn in model.steer_nets.values():
        for p in sn.parameters():
            p.requires_grad = False
            steer_params_ft.append(p)

    # PHASE 2A: SteerNet Fine-tune
    with manifest.stage("S1_steernet_finetune"):
        for p in steer_params_ft:
            p.requires_grad = True
        for p in model.router.parameters():
            p.requires_grad = False
        model.set_gate_freeze(True, 1.0)
        model.set_router_warmup(True)

        global_step, best_p2a_val, best_p2a_acc = run_staged_phase(
            model, tokenizer, steer_data, eval_sel, cfg, device,
            stage_name="PHASE 2A: SteerNet Fine-tune (exhaustive rotation, gate=1.0)",
            epochs=getattr(cfg, "steer_finetune_epochs", 0),
            param_groups=[{"params": steer_params_ft, "lr": getattr(cfg, "steer_finetune_lr", 1e-4)}],
            save_path=cfg.steer_finetune_save_path,
            train_steer=True, train_gate=False, train_router=False,
            global_step=0, baseline_ema=[0.0], patience=cfg.phase2_patience,
        )
        logs["p2a_val"] = best_p2a_val
        manifest.record_metrics("S1_steernet_finetune", best_val=best_p2a_val, best_acc=best_p2a_acc)
        for p in steer_params_ft:
            p.requires_grad = False
        model.reset_routing_diagnostics()
        _assert_finite_params(model, "S1_steernet_finetune")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # PHASE 2B: Gate-only Training (uses router_data, eval_sel)
    with manifest.stage("S3_router"):
        for p in steer_params_ft:
            p.requires_grad = False
        model.set_gate_freeze(False)
        model.set_router_warmup(False)
        # Gate-only learning must see the same deterministic route that will be
        # deployed and validated.  Sampling routes here trained the gate against a
        # different intervention distribution from argmax evaluation.
        model.set_router_exploration(False)
        unified_adaptive = bool(getattr(cfg, "unified_adaptive_phase", False))
        for p in model.router.parameters():
            p.requires_grad = unified_adaptive
        gate_params_ft = []
        for key in sorted(model.gates.keys(), key=lambda k: int(k)):
            for p in model.gates[key].parameters():
                p.requires_grad = True
                gate_params_ft.append(p)

        gate_stage_groups = [{"params": gate_params_ft, "lr": cfg.gate_only_lr}]
        if unified_adaptive:
            gate_stage_groups.append({
                "params": list(model.router.parameters()),
                "lr": cfg.router_bootstrap_lr * cfg.joint_finetune_router_lr_scale,
            })
        global_step, best_p2b_val, best_p2b_acc = run_staged_phase(
            model, tokenizer, router_data, eval_sel, cfg, device,
            stage_name=("PHASE 2: Unified Gate+Router Adaptation" if unified_adaptive else "PHASE 2B: Gate-only Training (SteerNet + Router frozen)"),
            epochs=(int(getattr(cfg, "unified_adaptive_epochs", 2)) if unified_adaptive else cfg.gate_only_epochs),
            param_groups=gate_stage_groups,
            save_path=cfg.gate_only_save_path,
            train_steer=False, train_gate=True, train_router=unified_adaptive,
            global_step=global_step, baseline_ema=[0.0], patience=cfg.phase2_patience,
        )
        logs["p2b_val"] = best_p2b_val
        manifest.record_metrics("S3_router", best_val=best_p2b_val, best_acc=best_p2b_acc)

        if os.path.exists(cfg.gate_only_save_path):
            load_ars_checkpoint_strict(model, cfg.gate_only_save_path, device=device)
            save_ars_checkpoint(model, cfg.phase2_save_path, phase="phase2b-global-baseline", cfg=cfg)
        model.reset_routing_diagnostics()
        _assert_finite_params(model, "S3_router")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if unified_adaptive:
        # The unified stage already trained the gate and router together under
        # the deployment route.  Do not create zero-epoch legacy stages or
        # overwrite their truthful metrics with infinity placeholders.
        best_val_p2, best_acc_p2 = best_p2b_val, best_p2b_acc
        best_p2d_val, best_p2d_acc = best_p2b_val, best_p2b_acc
        logs["p2_val"] = [best_val_p2]
        logs["p2d_val"] = best_p2d_val
    else:
        # PHASE 2C: Coordinated Joint Fine-tune (uses router_data, eval_sel; SteerNet strictly FROZEN)
        with manifest.stage("S4_token_gate"):
            for p in steer_params_ft:
                p.requires_grad = False
            for p in gate_params_ft:
                p.requires_grad = True
            for p in model.router.parameters():
                p.requires_grad = True
            model.set_router_exploration(True)
            model._joint_grad_mode = True
            joint_groups = [
                {"params": gate_params_ft, "lr": cfg.gate_only_lr * cfg.joint_finetune_lr_scale},
                {"params": list(model.router.parameters()), "lr": cfg.router_bootstrap_lr * cfg.joint_finetune_router_lr_scale},
            ]
            global_step, best_val_p2, best_acc_p2 = run_staged_phase(
                model, tokenizer, router_data, eval_sel, cfg, device,
                stage_name="PHASE 2C: Coordinated Gate+Router Fine-tune (SteerNet Frozen)",
                epochs=cfg.joint_finetune_epochs, param_groups=joint_groups,
                save_path=cfg.phase2_save_path, train_steer=False, train_gate=True, train_router=True,
                global_step=global_step, baseline_ema=[0.0], patience=max(1, cfg.phase2_patience),
                init_best_acc=best_p2b_acc, init_best_val=best_p2b_val,
            )
            model._joint_grad_mode = False
            logs["p2_val"] = [best_val_p2]
            manifest.record_metrics("S4_token_gate", best_val=best_val_p2, best_acc=best_acc_p2)
            _assert_finite_params(model, "S4_token_gate")

        # PHASE 2D: Gate Re-tune (uses router_data, eval_sel)
        with manifest.stage("S3_gate_retune"):
            if os.path.exists(cfg.phase2_save_path):
                load_ars_checkpoint_strict(model, cfg.phase2_save_path, device=device)
            for p in model.router.parameters():
                p.requires_grad = False
            for p in steer_params_ft:
                p.requires_grad = False
            model.set_gate_freeze(False)
            model.set_router_exploration(False)
            global_step, best_p2d_val, best_p2d_acc = run_staged_phase(
                model, tokenizer, router_data, eval_sel, cfg, device,
                stage_name="PHASE 2D: Gate Re-tune (against final router)",
                epochs=cfg.gate_retune_epochs, param_groups=[{"params": gate_params_ft, "lr": cfg.gate_retune_lr}],
                save_path=cfg.phase2_save_path, train_steer=False, train_gate=True, train_router=False,
                global_step=global_step, baseline_ema=[0.0], patience=cfg.phase2_patience,
                init_best_acc=best_acc_p2, init_best_val=best_val_p2,
            )
            logs["p2d_val"] = best_p2d_val
            manifest.record_metrics("S3_gate_retune", best_val=best_p2d_val, best_acc=best_p2d_acc)
            _assert_finite_params(model, "S3_gate_retune")
    model.set_router_exploration(False)
    model.set_router_exploration(False)

    if os.path.exists(cfg.phase2_save_path):
        load_ars_checkpoint_strict(model, cfg.phase2_save_path, device=device)
        print(f"  ✅ Loaded final joint+gate-retune checkpoint from {cfg.phase2_save_path}")
    logs["ars_intervention_audit"] = audit_ars_intervention_health(model, eval_sel, cfg, device)

    p2_pre_cap = collect_pre_cap_ratio(model)
    if p2_pre_cap:
        print(f"  [telemetry] Final SteerNet pre-cap norm ratios (delta_norm / cap): {p2_pre_cap}")
        logs["final_pre_cap_ratios"] = p2_pre_cap

    if gate_is_degenerate(logs["ars_intervention_audit"]):
        print("  ⚠️ Gate is constant (alpha≈1 everywhere): 'Full ARS' is identical to Router+SteerNet; do not report a gate effect.")
        logs["gate_degenerate"] = True
    else:
        logs["gate_degenerate"] = False

    # Engine v3: decide each prompt format independently.  Correlated views
    # remain visible as pooled diagnostics, but a dormant 2-shot policy cannot
    # erase an independently validated 0-shot deployment decision.
    from src.evaluation.deployment import finalize_deployment_by_shot, paired_delta_ci
    from src.evaluation.multi_dataset import build_benchmark_prompt, evaluate_sample_answer
    from src.evaluation.verifier import generate_response

    def _eval_deployment_flags(target_model, target_tokenizer, rows, target_device, override="none", n_shot=0):
        target_model.eval()
        if override == "fixed":
            fixed_layers = list(getattr(cfg, "fixed_steer_eval_layers", None) or [])
            if not fixed_layers:
                raise RuntimeError("Adaptive-vs-fixed deployment gate requires certified fixed SteerNet layers")
            target_model.set_routing_override("only", fixed_layers)
            scales = getattr(target_model, "_certified_gate_scales", None)
            if scales:
                target_model.set_gate_freeze(True, scales)
            else:
                target_model.set_gate_freeze(True, float(getattr(cfg, "fixed_steer_eval_gate_scale", 1.0)))
        else:
            target_model.set_gate_freeze(False)
            target_model.set_routing_override(override)
        flags = []
        for r in rows:
            q = None
            if isinstance(r, dict):
                q = r.get("question")
            elif hasattr(r, "__getitem__"):
                try:
                    q = r["question"]
                except (KeyError, IndexError, TypeError):
                    pass
            if not q:
                continue
            prompt_text = build_benchmark_prompt("gsm8k", q, n_shot=n_shot)
            resp = generate_response(target_model, target_tokenizer, prompt_text, target_device, max_new_tokens=384)
            final_num = None
            if isinstance(r, dict):
                final_num = r.get("final_num")
            elif hasattr(r, "__getitem__"):
                try:
                    final_num = r["final_num"]
                except (KeyError, IndexError, TypeError):
                    pass
            res = evaluate_sample_answer("gsm8k", resp, final_num)
            flags.append(int(res["correct"]))
        return flags

    if len(confirm_rows) > 0:
        deployment_shots = tuple(getattr(cfg, "deployment_eval_shots", None) or (0, 2))
        print(f"\n  [deployment] Evaluating {len(confirm_rows)} reserved confirmation rows across shots={deployment_shots}...")
        base_flags = []
        ars_flags = []
        fixed_flags_by_shot = []
        base_flags_by_shot = []
        ars_flags_by_shot = []
        base_clusters, ars_clusters = [], []
        for n_shot in deployment_shots:
            bf = _eval_deployment_flags(model, tokenizer, confirm_rows, device, override="none", n_shot=n_shot)
            af = _eval_deployment_flags(model, tokenizer, confirm_rows, device, override=None, n_shot=n_shot)
            ff = _eval_deployment_flags(model, tokenizer, confirm_rows, device, override="fixed", n_shot=n_shot) if bool(getattr(cfg, "require_adaptive_beats_fixed", False)) else None
            if len(bf) != len(confirm_rows) or len(af) != len(confirm_rows) or len(bf) != len(af):
                raise RuntimeError(
                    "Deployment confirmation did not produce one paired outcome per reserved question: "
                    f"shot={n_shot}, baseline={len(bf)}, ars={len(af)}, expected={len(confirm_rows)}"
                )
            base_flags_by_shot.append(bf)
            ars_flags_by_shot.append(af)
            if ff is not None:
                if len(ff) != len(confirm_rows):
                    raise RuntimeError("Fixed SteerNet deployment confirmation did not produce one outcome per reserved question")
                fixed_flags_by_shot.append(ff)
            base_flags.extend(bf)
            ars_flags.extend(af)
            if len(bf) > 0 and len(af) == len(bf):
                d_s, lo_s, hi_s = paired_delta_ci(bf, af)
                print(
                    f"    {n_shot}-shot: Base={sum(bf)}/{len(bf)} ({sum(bf)/len(bf):.1%}) | "
                    f"ARS={sum(af)}/{len(af)} ({sum(af)/len(af):.1%}) | "
                    f"Δ={d_s*100:+.2f}pp, 95% CI=[{lo_s*100:+.2f}, {hi_s*100:+.2f}]pp (n={len(bf)})"
                )

        outcomes_by_shot = {shot: (base_flags_by_shot[i], ars_flags_by_shot[i]) for i, shot in enumerate(deployment_shots)}
        dep_modes_by_shot = finalize_deployment_by_shot(
            outcomes_by_shot, cfg.phase2_save_path, min_n=min_deployment_questions,
            fixed_outcomes_by_shot=(
                {shot: fixed_flags_by_shot[i] for i, shot in enumerate(deployment_shots)}
                if fixed_flags_by_shot else None
            ),
            require_adaptive_beats_fixed=bool(getattr(cfg, "require_adaptive_beats_fixed", False)),
        )
        if len(base_flags_by_shot) == len(deployment_shots) and len(ars_flags_by_shot) == len(deployment_shots):
            base_clusters = [list(values) for values in zip(*base_flags_by_shot)]
            ars_clusters = [list(values) for values in zip(*ars_flags_by_shot)]
            d, lo, hi = paired_delta_ci(base_clusters, ars_clusters)
            print(
                f"  [deployment] Multi-shot ({'+'.join(str(shot) for shot in deployment_shots)} clustered by question) "
                f"Δ = {d*100:+.2f}pp, 95% CI = [{lo*100:+.2f}, {hi*100:+.2f}]pp "
                f"(questions={len(base_clusters)}, views={len(base_flags_by_shot)})"
            )
            dep_mode = (
                "ars" if all(mode == "ars" for mode in dep_modes_by_shot.values())
                else "fixed_steernet" if dep_modes_by_shot and all(mode == "fixed_steernet" for mode in dep_modes_by_shot.values())
                else "baseline"
            )
        else:
            print("  [deployment] No valid evaluated confirmation items; defaulting to BASELINE deployment.")
            dep_mode = "baseline"
            dep_modes_by_shot = {str(shot): "baseline" for shot in deployment_shots}
    else:
        print("\n  [deployment] No held-out confirmation rows available; defaulting to BASELINE deployment.")
        dep_mode = "baseline"
        fallback_shots = tuple(getattr(cfg, "deployment_eval_shots", None) or (0, 2))
        dep_modes_by_shot = {str(shot): "baseline" for shot in fallback_shots}
        base_flags, ars_flags = [], []
        base_clusters, ars_clusters = [], []

    logs["deployment_mode"] = dep_mode
    logs["deployment_modes_by_shot"] = dep_modes_by_shot
    logs["deployment_flags"] = {
        "baseline": base_flags,
        "ars": ars_flags,
        "baseline_clusters": base_clusters if len(confirm_rows) > 0 else [],
        "ars_clusters": ars_clusters if len(confirm_rows) > 0 else [],
    }
    if hasattr(model, "deployment_disabled"):
        model.deployment_disabled = (dep_mode == "baseline")
        model.deployment_modes_by_shot = dep_modes_by_shot
    print(f"  [deployment] Final deployment mode decided: {dep_mode.upper()} (deployment_disabled={getattr(model, 'deployment_disabled', False)})")

    print("\n" + "=" * 80)
    print("🎉 FULL ARS TRAINING COMPLETE!")
    print(f"   Phase 1 best validation loss           : {best_val_p1:.4f}")
    print(f"   Router bootstrap best probe loss       : {best_p15:.4f}")
    print(f"   Phase 2A (SteerNet ft) best val        : {best_p2a_val:.4f}")
    print(f"   Phase 2B (Gate-only) best val          : {best_p2b_val:.4f}")
    print(f"   Phase 2C (joint ft) best val           : {best_val_p2:.4f}")
    print(f"   Phase 2D (gate re-tune) best val       : {best_p2d_val:.4f}")
    print(f"   Deployment mode decision               : {dep_mode.upper()}")
    print(f"   Final saved checkpoint                 : {cfg.phase2_save_path}")
    print("=" * 80)

    logs["manifest_path"] = getattr(cfg, "manifest_path", "./run_manifest.json")
    manifest.flush()

    return logs, best_val_p1, best_val_p2


# --------------------------------------------------------------------------- #
# M4: S1' SteerNet subset trainer additions
# --------------------------------------------------------------------------- #
import copy
import math
from contextlib import nullcontext

import numpy as np

from src.routing.catalog import Catalog
from src.routing.rloo import per_example_answer_nll
from src.training.losses import (
    compute_calc_token_weights,
    weighted_answer_nll,
    unlikelihood_on_divergence,
    collate_pad,
)


def sample_training_subset(catalog: "Catalog", rng: random.Random, k_choices: Tuple[int, ...] = (1, 2, 3)) -> Tuple[int, ...]:
    """Draw a random non-empty subset. K ~ Uniform(k_choices), subset ~ Uniform(C(5,K))."""
    valid_k = [k for k in k_choices if k > 0]
    if not valid_k:
        valid_k = [1, 2, 3]
    k = rng.choice(valid_k)
    candidates = [s for s in catalog.subsets if len(s) == k]
    if not candidates:
        raise ValueError(f"No catalog subsets of size {k}")
    return rng.choice(candidates)


def _autocast(device: torch.device):
    return torch.amp.autocast("cuda", enabled=device.type == "cuda")


def _set_ctx(model, mask: torch.Tensor, lbls: torch.Tensor) -> None:
    if hasattr(model, "_attn_mask_cache"):
        model._attn_mask_cache = mask
    if hasattr(model, "set_router_context_mask"):
        model.set_router_context_mask(mask.bool() & (lbls == -100))


def _restore_override(model, old) -> None:
    if old is None:
        model.set_routing_override(None)
    else:
        model.set_routing_override(old[0], old[1])


def _catalog_from_cfg(cfg, model) -> "Catalog":
    layers = getattr(cfg, "catalog_layers", None)
    if layers is None:
        layers = model.target_layers
    return Catalog.build(tuple(layers), getattr(cfg, "catalog_max_k", 3))


def _manifest():
    from src.utils.failure_policy import get_manifest
    return get_manifest()


def _ensure_contrast_sets(tokenizer, device: torch.device) -> bool:
    from src.training import contrastive as _c
    if getattr(_c, "_CONTRAST_MATH_IDS", None):
        return True
    builder = getattr(_c, "build_contrast_sets", None)
    try:
        if builder is not None:
            builder(tokenizer, device)
        else:
            _c.build_contrast_id_cache(tokenizer)
    except Exception as exc:  # noqa: BLE001
        _manifest().warn("S1_CALC_WEIGHTS_UNAVAILABLE", f"contrast sets not built ({exc}); using uniform token weights")
        return False
    ok = bool(getattr(_c, "_CONTRAST_MATH_IDS", None))
    if not ok:
        _manifest().warn("S1_CALC_WEIGHTS_EMPTY", "math token id set is empty; using uniform token weights")
    return ok


def evaluate_subset_nll(model, rows: List[dict], catalog: "Catalog", cfg, device: torch.device) -> np.ndarray:
    """Evaluate NLL for all 25 non-empty catalog subsets on the given rows.

    Returns (len(rows), C) float32 array where [:, 0] is NLL under empty subset
    and [:, 1:] are NLL for non-empty subsets.
    Uses model.counterfactual_mode() and model.set_routing_override("matrix", ...).
    """
    n_cols = len(catalog)
    out = np.zeros((len(rows), n_cols), dtype=np.float32)
    if not rows:
        return out
    eval_bs = max(1, int(getattr(cfg, "s1_eval_batch_size", 4)))
    old_override = getattr(model, "_routing_override", None)
    try:
        with model.counterfactual_mode(), torch.no_grad():
            for start in range(0, len(rows), eval_bs):
                chunk = rows[start : start + eval_bs]
                batch = collate_pad(chunk, 0)
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lbls = batch["labels"].to(device)
                bsz = ids.size(0)
                for c, subset in enumerate(catalog.subsets):
                    matrix = model.subset_matrix([subset] * bsz).to(device)
                    model.set_routing_override("matrix", matrix)
                    _set_ctx(model, mask, lbls)
                    with _autocast(device):
                        logits = model(input_ids=ids, attention_mask=mask).logits
                    nll = per_example_answer_nll(logits, lbls)
                    out[start : start + bsz, c] = nll.float().cpu().numpy()
    finally:
        _restore_override(model, old_override)
    return out


def _k_uniform_gain(nll: np.ndarray, catalog: "Catalog", k_choices: Tuple[int, ...]):
    """Mean NLL gain (empty - subset) with K uniform over k_choices, plus per-subset mean gains."""
    n_sub = nll.shape[1] - 1
    if nll.shape[0] == 0 or n_sub <= 0:
        return 0.0, np.zeros(max(0, n_sub), dtype=np.float64)
    gains = nll[:, :1] - nll[:, 1:]
    per_subset = gains.mean(axis=0).astype(np.float64)
    sizes = catalog.sizes().cpu().numpy()[1:]
    vals = [per_subset[sizes == k].mean() for k in k_choices if (sizes == k).any()]
    score = float(np.mean(vals)) if vals else float(per_subset.mean())
    return score, per_subset


def _draw_rows(rng: random.Random, pool: List[dict], n: int) -> List[dict]:
    if len(pool) >= n:
        return rng.sample(pool, n)
    return [rng.choice(pool) for _ in range(n)]


def _rows_to_questions_golds(tokenizer, rows: List[dict], n_max: int):
    """Decode prompt text and numeric gold from rows (prompt = labels == -100, non-pad)."""
    from src.evaluation.answers import extract_numeric_answer

    pad_id = tokenizer.pad_token_id
    questions, golds = [], []
    for r in rows:
        if len(questions) >= n_max:
            break
        ids = torch.as_tensor(r["input_ids"]).reshape(-1)
        lbl = torch.as_tensor(r["labels"]).reshape(-1)
        prompt_mask = lbl.eq(-100)
        if pad_id is not None:
            prompt_mask = prompt_mask & ids.ne(pad_id)
        ans_mask = lbl.ne(-100)
        if not bool(prompt_mask.any()) or not bool(ans_mask.any()):
            continue
        gold = extract_numeric_answer(tokenizer.decode(ids[ans_mask], skip_special_tokens=True))
        if gold is None:
            continue
        questions.append(tokenizer.decode(ids[prompt_mask], skip_special_tokens=True))
        golds.append(float(gold))
    return questions, golds


def _s1_fit(
    model, tokenizer, gold_rows: List[dict], onpolicy_rows: List[dict], sel_rows: List[dict],
    cfg, device: torch.device, catalog: "Catalog", epochs: int, lr: float, rng: random.Random,
    ckpt_mgr=None, label: str = "S1'",
) -> Dict[str, Any]:
    """Shared S1' training loop (subset-level layer-dropout). Returns logs + best state."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for p in model.parameters():
        p.requires_grad = False
    steer_params = []
    for sn in model.steer_nets.values():
        for p in sn.parameters():
            p.requires_grad = True
            steer_params.append(p)
    model.set_steer_scale(getattr(cfg, "steer_scale", 1.0))
    if hasattr(model, "set_gate_mode"):
        model.set_gate_mode("const", scale=getattr(cfg, "steer_scale", 1.0))
    if hasattr(model, "set_router_warmup"):
        model.set_router_warmup(False)

    n_total = len(gold_rows) + len(onpolicy_rows)
    if n_total == 0:
        raise ValueError("train_s1: no training rows")

    optimizer = torch.optim.AdamW(
        steer_params, lr=lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98), eps=1e-6
    )
    bs = max(1, cfg.batch_size)
    micro_per_epoch = max(1, math.ceil(n_total / bs))
    steps_per_epoch = max(1, math.ceil(micro_per_epoch / cfg.grad_accum))
    total_steps = max(1, steps_per_epoch * max(1, epochs))
    warmup = int(total_steps * cfg.phase1_warmup_ratio)
    lr_reducer = LRReducer(
        optimizer, total_steps, factor=cfg.lr_reduce_factor, patience=cfg.lr_reduce_patience,
        min_lr=cfg.lr_reduce_min_lr, delta=cfg.early_stop_delta, warmup_steps=warmup,
    )
    lr_reducer.base_lrs = [lr]
    optimizer.param_groups[0]["lr"] = lr
    grad_mon = GradientMonitor(cfg.grad_warn_threshold)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    train_loss_history: List[float] = []
    val_gain_history: List[float] = []
    best_gain = -float("inf")
    best_state = None
    best_subset = None
    best_subset_gain = None
    patience_ctr = 0
    micro_counter = 0
    frac = float(cfg.s1_onpolicy_fraction)

    for epoch in range(max(0, epochs)):
        model.train()
        if hasattr(model, 'base_model') and model.base_model is not None:
            model.base_model.eval()
        optimizer.zero_grad(set_to_none=True)
        accum, epoch_loss, n_loss = 0, 0.0, 0
        pbar = tqdm(range(micro_per_epoch), desc=f"{label} {epoch + 1}/{epochs}", leave=False)
        for m in pbar:
            use_onp = bool(onpolicy_rows) and (not gold_rows or rng.random() < frac)
            pool = onpolicy_rows if use_onp else gold_rows
            batch_rows = _draw_rows(rng, pool, bs)
            batch = collate_pad(batch_rows, pad_id)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)
            bsz = ids.size(0)

            subset = sample_training_subset(catalog, rng, cfg.s1_k_choices)
            model.set_routing_override("matrix", model.subset_matrix([subset] * bsz).to(device))
            _set_ctx(model, mask, lbls)
            with _autocast(device):
                out = model(input_ids=ids, attention_mask=mask)
            weights = compute_calc_token_weights(tokenizer, lbls, ids, cfg.s1_calc_token_weight)
            try:
                task_loss = weighted_answer_nll(out.logits, lbls, weights)
            except ValueError:
                continue
            total = task_loss

            micro_counter += 1
            if cfg.s1_lambda_unl > 0 and micro_counter % max(1, cfg.s1_unl_every_n_steps) == 0:
                paired = [
                    r for r in batch_rows
                    if r.get("neg_input_ids") is not None
                    and r.get("neg_labels") is not None
                    and r.get("div_token_mask") is not None
                ]
                if paired:
                    pb = collate_pad(paired, pad_id)
                    n_ids = pb["neg_input_ids"].to(device)
                    n_mask = pb["neg_attention_mask"].to(device)
                    n_lbl = pb["neg_labels"].to(device)
                    n_div = pb["div_token_mask"].to(device)
                    model.set_routing_override("matrix", model.subset_matrix([subset] * n_ids.size(0)).to(device))
                    _set_ctx(model, n_mask, n_lbl)
                    with _autocast(device):
                        out_neg = model(input_ids=n_ids, attention_mask=n_mask)
                    ul = unlikelihood_on_divergence(out_neg.logits, n_lbl, n_div)
                    if ul.requires_grad:
                        total = total + cfg.s1_lambda_unl * ul

            loss = total / cfg.grad_accum
            if not torch.isfinite(loss) or not loss.requires_grad:
                continue
            scaler.scale(loss).backward()
            accum += 1
            epoch_loss += float(task_loss.detach().item())
            n_loss += 1

            is_last = m == micro_per_epoch - 1
            if accum == cfg.grad_accum or is_last:
                scaler.unscale_(optimizer)
                for p in steer_params:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        p.grad.zero_()
                torch.nn.utils.clip_grad_norm_(steer_params, cfg.clip_grad)
                if cfg.use_gradient_monitor:
                    grad_mon.check(steer_params, micro_counter)
                scaler.step(optimizer)
                scaler.update()
                lr_reducer.scheduler_step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
            pbar.set_postfix(loss=f"{float(task_loss.detach().item()):.4f}")

        model.set_routing_override(None)
        avg_loss = epoch_loss / max(1, n_loss)
        train_loss_history.append(avg_loss)

        nll = evaluate_subset_nll(model, sel_rows, catalog, cfg, device)
        gain, per_subset = _k_uniform_gain(nll, catalog, cfg.s1_k_choices)
        val_gain_history.append(gain)
        lr_reducer.step(-gain)

        improved = (not sel_rows) or gain > best_gain + cfg.early_stop_delta
        if improved:
            best_gain = gain
            patience_ctr = 0
            best_state = {k: v.detach().clone() for k, v in model.steer_nets.state_dict().items()}
            if per_subset.size:
                idx = int(np.argmax(per_subset))
                best_subset = catalog.subsets[idx + 1]
                best_subset_gain = float(per_subset[idx])
        else:
            patience_ctr += 1
        print(
            f"  {label} Epoch {epoch + 1}: train={avg_loss:.4f}  val_gain(K-uniform)={gain:+.4f}  "
            f"best_subset={best_subset}  {'✅' if improved else ''}"
        )
        if ckpt_mgr is not None and (epoch + 1) % cfg.checkpoint_every_n == 0:
            ckpt_mgr.save(model, optimizer, epoch, -gain, "s1")
        if patience_ctr >= cfg.phase1_patience:
            print(f"  {label} early stopping (no improvement for {cfg.phase1_patience} epochs)")
            break

    grad_mon.summary()
    return {
        "train_loss_history": train_loss_history,
        "val_nll_gain_history": val_gain_history,
        "best_state": best_state,
        "best_gain": best_gain,
        "best_subset": best_subset,
        "best_subset_gain": best_subset_gain,
    }


def train_s1_pref(
    model,
    tokenizer,
    pairs: List,
    cfg,
    device: torch.device,
) -> Dict[str, Any]:
    """Train S1' SteerNets with preference (DPO) loss from mined pairs.
    
    Phase 1.4: Preference training with Step-DPO.
    
    Args:
        model: RSC model
        tokenizer: Tokenizer
        pairs: List of PreferencePair objects
        cfg: RSCConfig
        device: torch device
    
    Returns:
        Training logs
    """
    print("\n" + "=" * 80)
    print("S1': SteerNet preference trainer (DPO)")
    print(f"     pairs={len(pairs)}  lr={cfg.s1_lr}  epochs={cfg.s1_epochs}")
    print("=" * 80)
    
    # Freeze router and gate
    model.set_steer_scale(cfg.steer_scale)
    
    # Training setup
    optimizer = torch.optim.AdamW(
        [p for name, p in model.named_parameters() if "steer_nets" in name],
        lr=cfg.s1_lr,
    )
    
    beta = getattr(cfg, "dpo_beta", 0.1)
    kl_weight = getattr(cfg, "dpo_kl_weight", 0.01)
    
    model.train()
    loss_history = []
    
    grad_accum = max(1, getattr(cfg, "grad_accum", 4))
    nll_reg = getattr(cfg, "pref_nll_reg", 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def _get_chain_tokens(prompt: str, completion: str):
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        sep = "\n" if not prompt.endswith("\n") else ""
        full_text = prompt + sep + completion
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        p_len = len(p_ids)
        if len(full_ids) <= p_len:
            full_ids = p_ids + tokenizer.encode(completion, add_special_tokens=False)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:, :p_len] = -100
        return input_ids, attn_mask, labels

    def _eval_logprob(inp_ids, mask, lbls, is_steered=True):
        if not is_steered:
            with torch.no_grad():
                model.set_routing_override("none", ())
                out = model(input_ids=inp_ids, attention_mask=mask)
                model.set_routing_override(None)
        else:
            out = model(input_ids=inp_ids, attention_mask=mask)
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = lbls[:, 1:].contiguous()
        valid = (shift_labels != -100)
        n_tokens = valid.sum().clamp(min=1).float()
        token_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        return -token_loss / n_tokens, token_loss / n_tokens

    for epoch in range(cfg.s1_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        n_pairs = max(1, len(pairs))
        pbar = tqdm(pairs, desc=f"S1' Pref Epoch {epoch+1}/{cfg.s1_epochs}", leave=False)

        for pair in pbar:
            prompt = pair.y_plus.prompt
            text_pos = pair.y_plus.text
            text_neg = pair.y_minus.text

            inp_pos, mask_pos, lbl_pos = _get_chain_tokens(prompt, text_pos)
            inp_neg, mask_neg, lbl_neg = _get_chain_tokens(prompt, text_neg)

            # Compute reference logprobs (frozen unsteered base model)
            with torch.no_grad():
                ref_lp_pos, _ = _eval_logprob(inp_pos, mask_pos, lbl_pos, is_steered=False)
                ref_lp_neg, _ = _eval_logprob(inp_neg, mask_neg, lbl_neg, is_steered=False)

            # Compute steered logprobs under active SteerNet
            steer_lp_pos, nll_pos = _eval_logprob(inp_pos, mask_pos, lbl_pos, is_steered=True)
            steer_lp_neg, _ = _eval_logprob(inp_neg, mask_neg, lbl_neg, is_steered=True)

            delta_pos = steer_lp_pos - ref_lp_pos
            delta_neg = steer_lp_neg - ref_lp_neg
            margin = delta_pos - delta_neg

            # DPO preference loss: -log(sigmoid(beta * margin))
            dpo_loss = -torch.nn.functional.logsigmoid(beta * margin)
            total_loss = (dpo_loss + nll_reg * nll_pos) / grad_accum

            if not torch.isfinite(total_loss) or not total_loss.requires_grad:
                continue

            scaler.scale(total_loss).backward()
            accum += 1
            epoch_loss += float((dpo_loss + nll_reg * nll_pos).detach().item())

            if accum == grad_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                pbar.set_postfix(margin=f"{float(margin.item()):.3f}", loss=f"{float(dpo_loss.item()):.3f}")

        if accum > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = epoch_loss / n_pairs
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{cfg.s1_epochs}: loss={avg_loss:.4f}")

    return {
        "loss_history": loss_history,
        "n_pairs": len(pairs),
        "beta": beta,
    }


def train_s1_steernet(
    model,
    tokenizer,
    gold_rows: List[dict],
    onpolicy_rows: List[dict],
    sel_rows: List[dict],
    cfg,
    device: torch.device,
) -> Dict[str, Any]:
    """Train S1' SteerNets on on-policy + gold data with random subset-level layer-dropout.

    Router and gate are frozen (set_steer_scale(cfg.steer_scale)); only steer_nets train.
    Per epoch: evaluate_subset_nll on sel_rows. After the final epoch: paired greedy
    check (best subset vs baseline) on up to s1_final_gen_n rows, recorded as a manifest
    advisory. Saves weights with metadata.

    Returns logs dict with keys: train_loss_history, val_nll_gain_history,
    best_subset, final_gen_check.
    """
    print("\n" + "=" * 80)
    print("S1': SteerNet subset trainer (on-policy + gold, subset-level layer-dropout)")
    print(f"     epochs={cfg.s1_epochs}  lr={cfg.s1_lr}  onpolicy_fraction={cfg.s1_onpolicy_fraction}")
    print("=" * 80)

    catalog = _catalog_from_cfg(cfg, model)
    rng = random.Random(cfg.seed)
    _ensure_contrast_sets(tokenizer, device)

    fit = _s1_fit(
        model, tokenizer, gold_rows, onpolicy_rows, sel_rows, cfg, device, catalog,
        epochs=cfg.s1_epochs, lr=cfg.s1_lr, rng=rng,
        ckpt_mgr=CheckpointManager(cfg.checkpoint_dir),
    )
    if fit["best_state"] is not None:
        model.steer_nets.load_state_dict(fit["best_state"])
    best_subset = fit["best_subset"]

    final_gen = None
    n_gen = int(getattr(cfg, "s1_final_gen_n", 0))
    if n_gen > 0 and best_subset is not None and sel_rows:
        from src.evaluation.paired_generation import paired_greedy_eval

        questions, golds = _rows_to_questions_golds(tokenizer, sel_rows, n_gen)
        if questions:
            subset_layers = list(best_subset)
            res = paired_greedy_eval(
                model, tokenizer, questions, golds,
                lambda m: m.set_routing_override("none", None),
                lambda m: m.set_routing_override("only", subset_layers),
                device, batch_size=8, max_new_tokens=512,
            )
            final_gen = {
                "n": len(questions),
                "subset": subset_layers,
                "acc_baseline": res["acc_a"],
                "acc_best_subset": res["acc_b"],
                "b": res["b"],
                "c": res["c"],
                "delta": res["delta"],
            }
            _manifest().advisory("s1_final_gen_check", final_gen, "discordant counts (b,c)")
            print(
                f"  Final gen check: baseline={res['acc_a']:.3f}  best_subset={res['acc_b']:.3f}  "
                f"b={res['b']} c={res['c']}"
            )

    metadata = {
        "steer_scale": getattr(cfg, "steer_scale", 1.0),
        "k_scale_mode": getattr(getattr(model, "cfg", cfg), "k_scale_mode", getattr(cfg, "k_scale_mode", "sqrt")),
        "steernet_k_conditioning": getattr(
            getattr(model, "cfg", cfg), "steernet_k_conditioning", getattr(cfg, "steernet_k_conditioning", True)
        ),
        "best_subset": list(best_subset) if best_subset is not None else None,
        "catalog_layers": list(catalog.layers),
    }
    save_steer_weights(model, cfg.s1_save_path, metadata=metadata)
    model.set_routing_override(None)

    return {
        "train_loss_history": fit["train_loss_history"],
        "val_nll_gain_history": fit["val_nll_gain_history"],
        "best_subset": best_subset,
        "final_gen_check": final_gen,
    }


def _get_k_options(model, cfg):
    mcfg = getattr(model, "cfg", cfg)
    return getattr(mcfg, "k_scale_mode", "sqrt"), getattr(mcfg, "steernet_k_conditioning", True)


def _set_k_options(model, cfg, mode: str, cond: bool) -> None:
    targets = [cfg]
    mcfg = getattr(model, "cfg", None)
    if mcfg is not None and mcfg is not cfg:
        targets.append(mcfg)
    for t in targets:
        t.k_scale_mode = mode
        t.steernet_k_conditioning = cond
    if hasattr(model, "set_k_scale_mode"):
        model.set_k_scale_mode(mode)
    if hasattr(model, "set_k_conditioning"):
        model.set_k_conditioning(cond)


def run_k_scaling_ablation(
    model,
    tokenizer,
    rows: List[dict],
    sel_rows: List[dict],
    cfg,
    device: torch.device,
) -> Dict[str, Any]:
    """Run short ablation over {sqrt, none} x {k_cond on, off}.

    For each of 4 combinations: short train (s1_ablation_epochs, s1_ablation_n_train rows),
    score by mean val NLL gain with K uniform. Record all 4 to manifest as advisory.
    The best-scoring combination gives the recommended k_scaling and k_cond.

    Returns dict with keys: results (list of 4 dicts), winner_k_scaling, winner_k_cond.
    """
    catalog = _catalog_from_cfg(cfg, model)
    _ensure_contrast_sets(tokenizer, device)

    subsample_rng = random.Random(cfg.seed)
    n_train = int(cfg.s1_ablation_n_train)
    train_rows = list(rows)
    if len(train_rows) > n_train:
        train_rows = subsample_rng.sample(train_rows, n_train)
    onp = [r for r in train_rows if r.get("is_onpolicy")]
    gold = [r for r in train_rows if not r.get("is_onpolicy")]

    init_steer = copy.deepcopy(model.steer_nets.state_dict())
    init_mode, init_cond = _get_k_options(model, cfg)

    results: List[Dict[str, Any]] = []
    try:
        for mode in ("sqrt", "none"):
            for cond in (True, False):
                model.steer_nets.load_state_dict(init_steer)
                _set_k_options(model, cfg, mode, cond)
                fit = _s1_fit(
                    model, tokenizer, gold, onp, sel_rows, cfg, device, catalog,
                    epochs=cfg.s1_ablation_epochs, lr=cfg.s1_lr, rng=random.Random(cfg.seed),
                    ckpt_mgr=None, label=f"S1'-abl[{mode},k_cond={cond}]",
                )
                score = float(fit["best_gain"]) if np.isfinite(fit["best_gain"]) else 0.0
                res = {
                    "k_scaling": mode,
                    "k_cond": cond,
                    "score": score,
                    "final_train_loss": fit["train_loss_history"][-1] if fit["train_loss_history"] else None,
                }
                results.append(res)
                _manifest().advisory(
                    f"s1_k_ablation_{mode}_kcond_{'on' if cond else 'off'}",
                    res,
                    "mean val NLL gain with K uniform (higher is better)",
                )
    finally:
        model.steer_nets.load_state_dict(init_steer)
        _set_k_options(model, cfg, init_mode, init_cond)
        model.set_routing_override(None)

    best = max(results, key=lambda r: r["score"])
    winner_k_scaling, winner_k_cond = best["k_scaling"], best["k_cond"]
    _manifest().advisory(
        "s1_k_scaling_winner",
        {"k_scaling": winner_k_scaling, "k_cond": winner_k_cond, "score": best["score"]},
        "recommended k_scale_mode and steernet_k_conditioning from the 4-way ablation",
    )
    return {
        "results": results,
        "winner_k_scaling": winner_k_scaling,
        "winner_k_cond": winner_k_cond,
    }


# ===========================================================================
# M7: S3 Token Gate Trainer
# ===========================================================================

def train_s3_gate(
    model,
    tokenizer,
    train_rows: List[dict],
    val_rows: List[dict],
    cfg,
    device: torch.device,
    table=None,
) -> Dict[str, Any]:
    """Train token gates using direct task-loss, dominance hinge, and counterfactual BCE.

    SteerNets and router are strictly frozen; only gate parameters train.
    """
    import torch.nn.functional as F
    from src.training.losses import (
        TokenLabelStore,
        counterfactual_token_labels,
        gate_bce_tokens,
        gate_bce_tokens_with_live_fallback,
        gate_total_loss,
        compute_l1_gate,
        paired_gate_vs_ungated,
    )

    print("\n" + "=" * 80)
    print("S3: Direct Task-Loss Token Gate Trainer (Dominance Hinge + Counterfactual BCE)")
    print(f"     epochs={cfg.gate_epochs}  lr={cfg.gate_lr}  dom_margin={cfg.gate_dom_margin}")
    print("=" * 80)

    # 1. Freeze everything except gates
    for p in model.parameters():
        p.requires_grad = False

    gate_params = []
    for g in model.gates.values():
        for p in g.parameters():
            p.requires_grad = True
            gate_params.append(p)

    model.cfg.gate_task_gradient = True
    model.set_router_policy("catalog")
    if hasattr(model, "set_router_warmup"):
        model.set_router_warmup(False)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # 2. Build counterfactual token label store if table is provided
    store = None
    if table is not None:
        try:
            with torch.no_grad():
                pooled_t = torch.as_tensor(table.pooled, dtype=torch.float32, device=device)
                scalars_t = torch.as_tensor(table.scalars, dtype=torch.float32, device=device)
                shot_buckets_t = None
                if hasattr(table, "n_shot") and hasattr(model.router, "shot_to_bucket"):
                    shot_buckets_t = model.router.shot_to_bucket(table.n_shot).to(device)
                cat_logits = model.router.catalog_logits_from_features(pooled_t, scalars_t, shot_bucket=shot_buckets_t)
                choices = cat_logits.argmax(dim=-1).cpu().numpy()
            tau_u = getattr(cfg, "gate_tau_u", 0.005)
            store = counterfactual_token_labels(table, choices, tau_u=tau_u)
            print(f"  [S3] Loaded {len(store)} labeled counterfactual examples from table.")
        except Exception as e:
            print(f"  [S3] Warning: could not build counterfactual token labels: {e}")

    optimizer = torch.optim.AdamW(
        gate_params,
        lr=cfg.gate_lr,
        weight_decay=getattr(cfg, "weight_decay", 1e-2),
        betas=(0.9, 0.98),
        eps=1e-6,
    )

    bs = max(1, cfg.batch_size)
    micro_per_epoch = max(1, math.ceil(len(train_rows) / bs))
    rng = random.Random(cfg.seed)

    train_loss_history = []
    best_mean_diff = float("inf")
    best_gate_state = {k: v.cpu().clone() for k, v in model.gates.state_dict().items()}
    best_paired_eval = None

    for epoch in range(max(1, cfg.gate_epochs)):
        model.train()
        if hasattr(model, 'base_model') and model.base_model is not None:
            model.base_model.eval()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = []

        shuffled_indices = list(range(len(train_rows)))
        rng.shuffle(shuffled_indices)

        pbar = tqdm(
            range(micro_per_epoch),
            desc=f"S3 Epoch {epoch + 1}/{cfg.gate_epochs}",
            unit="batch",
            leave=True,
        )
        for m in pbar:
            batch_idx = shuffled_indices[m * bs : (m + 1) * bs]
            if not batch_idx:
                continue
            batch_rows = [train_rows[i] for i in batch_idx]
            if epoch == 0 and m == 0 and store is not None:
                from src.utils.keys import canonical_token_key
                matched_cnt = 0
                sample_keys = []
                for r in batch_rows:
                    # Phase 3 fix: Ensure consistent key generation
                    n_shot = r.get("n_shot", r.get("shot", 0))
                    key = canonical_token_key(r.get("split", "train"), r.get("qid", ""), n_shot)
                    if key in store:
                        matched_cnt += 1
                    if len(sample_keys) < 3:
                        sample_keys.append(key)
                print(f"\n  [S3 Diagnostics] First batch token label matches: {matched_cnt}/{len(batch_rows)} rows (store size={len(store)})")
                if matched_cnt == 0:
                    print(f"  [S3 Diagnostics] Sample row keys: {sample_keys}")
                    print(f"  [S3 Diagnostics] Sample store keys: {list(store.keys())[:3]}")
            batch = collate_pad(batch_rows, pad_id)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            # (i) Gated forward with grad
            model.set_gate_mode("learned")
            model.set_routing_override(None)
            _set_ctx(model, mask, lbls)
            out_g = model(input_ids=ids, attention_mask=mask)

            shift_logits_g = out_g.logits[:, :-1, :].float()
            shift_lbls = lbls[:, 1:]
            valid = shift_lbls.ne(-100)
            if not valid.any():
                continue

            loss_mat_g = F.cross_entropy(
                shift_logits_g.transpose(1, 2), shift_lbls, ignore_index=-100, reduction="none"
            )
            nll_gated = loss_mat_g.sum(dim=1) / valid.sum(dim=1).clamp(min=1)

            # (ii) Compute L1 immediately while _alpha_for_aux holds active autograd graph
            mean_alpha = compute_l1_gate(model)

            # (iii) Ungated forward under counterfactual_mode with the same route
            routes = model.current_route_tuples()
            with model.counterfactual_mode():
                model.set_gate_mode("const", scale=getattr(cfg, "steer_scale", 1.0))
                model.set_routing_override("matrix", model.subset_matrix(routes).to(device))
                out_u = model(input_ids=ids, attention_mask=mask)
                shift_logits_u = out_u.logits[:, :-1, :].float()
                loss_mat_u = F.cross_entropy(
                    shift_logits_u.transpose(1, 2), shift_lbls, ignore_index=-100, reduction="none"
                )
                nll_ungated = loss_mat_u.sum(dim=1) / valid.sum(dim=1).clamp(min=1)
                model.set_routing_override(None)
                model.set_gate_mode("learned")

            # (ii-b) Token BCE: prefer CF store entries when available; fall back to live
            # per-token NLL-diff labels derived from the current batch.
            # Live labels: token is "beneficial" (label=1) when ungated_nll > gated_nll,
            # i.e. the gate improves this token.  This dense signal replaces the sparse
            # CF store which covers only ~1-3% of training rows due to index mismatch.
            tau_u = float(getattr(cfg, "gate_tau_u", 0.005))
            bce_tok = gate_bce_tokens_with_live_fallback(
                model, batch_rows, store, device,
                loss_mat_g=loss_mat_g.detach(), loss_mat_u=loss_mat_u.detach(),
                valid_mask=valid, tau_u=tau_u,
            )

            # (iv) Combined loss
            total_loss, metrics = gate_total_loss(nll_gated, nll_ungated, bce_tok, mean_alpha, cfg)

            if torch.isfinite(total_loss):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(gate_params, getattr(cfg, "clip_grad", 1.0))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                epoch_losses.append(metrics["loss_total"])
                pbar.set_postfix({
                    "loss": f"{metrics['loss_total']:.4f}",
                    "dom": f"{metrics.get('loss_dom', 0.0):.4f}",
                    "a": f"{metrics.get('mean_alpha', 0.0):.3f}",
                })

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        train_loss_history.append(avg_loss)

        # Validation at epoch end
        val_sample = val_rows[: getattr(cfg, "gate_sel_n", 120)]
        paired_eval = paired_gate_vs_ungated(model, val_sample, cfg, device, pad_id=pad_id)
        print(
            f"  [S3 Epoch {epoch + 1}/{cfg.gate_epochs}] "
            f"loss={avg_loss:.4f}  "
            f"mean_diff(gated - ungated)={paired_eval['mean_diff']:.4f} "
            f"CI=[{paired_eval['ci_low']:.4f}, {paired_eval['ci_high']:.4f}] "
            f"adopted={paired_eval['gate_adopted_nll']}"
        )

        if paired_eval["mean_diff"] < best_mean_diff:
            best_mean_diff = paired_eval["mean_diff"]
            best_gate_state = {k: v.cpu().clone() for k, v in model.gates.state_dict().items()}
            best_paired_eval = paired_eval

    # Restore best checkpoint
    model.gates.load_state_dict(best_gate_state)

    # Save gate weights
    save_path = getattr(cfg, "gate_v5_save_path", "./checkpoints/gate_v5.pt")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(best_gate_state, save_path)
    print(f"  [S3] Saved gate checkpoint to {save_path}")

    # Emit manifest advisory / warning
    manifest = _manifest()
    gate_adopted = best_paired_eval["gate_adopted_nll"] if best_paired_eval else False
    if not gate_adopted:
        manifest.warn(
            "W_GATE_NOT_ADOPTED",
            f"Gate did not achieve negative mean NLL diff vs ungated reference (best diff: {best_mean_diff:.4f})",
        )
    manifest.advisory(
        "s3_gate_paired_eval",
        best_paired_eval,
        "Paired gated vs ungated NLL evaluation on selection split",
    )

    return {
        "train_loss_history": train_loss_history,
        "best_mean_diff": best_mean_diff,
        "paired_eval": best_paired_eval,
        "gate_adopted_nll": gate_adopted,
        "save_path": save_path,
    }
