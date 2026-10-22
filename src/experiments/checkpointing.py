"""Strict, provenance-preserving serialization for ARS component checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch


ARS_PREFIXES = ("steer_nets.", "gates.", "router.")

# These settings determine tensor topology or inference semantics.  An
# evaluator must reconstruct them from the checkpoint provenance *before*
# wrapping Phi-2; guessing rank alone can load a different router/gate model
# and silently invalidate a component comparison.
CHECKPOINT_CONFIG_FIELDS = frozenset({
    "model_name", "max_active_layers", "candidate_layers", "fixed_steer_eval_k",
    "fixed_steer_eval_layers", "fixed_steer_eval_gate_scale", "use_router",
    "lora_rank", "lora_alpha", "lora_dropout", "steer_magnitude_mode",
    "steer_max_magnitude", "steer_max_relative_norm", "steer_orthogonal_projection",
    "steer_route_budget_power", "gate_dim", "gate_dropout", "gate_init_alpha",
    "gating_mode", "gate_constant_value", "router_difficulty_aware",
    "router_synergy_limit", "router_eval_mode", "prompt_shielding",
    "effective_k_min_energy_fraction", "router_min_eval_route_tuple_entropy",
    "deployment_eval_shots",
    "router_k0_enabled", "router_demo_features", "token_competitive_allocation",
    "gate_allocation_temperature", "gate_allocation_temperature_final",
})


def is_ars_key(name: str) -> bool:
    return name.startswith(ARS_PREFIXES)


def ars_state_dict(model: torch.nn.Module, prefixes: Optional[Iterable[str]] = None) -> Dict[str, torch.Tensor]:
    """Return CPU ARS tensors, optionally restricted to declared components."""
    allowed = tuple(prefixes) if prefixes is not None else ARS_PREFIXES
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith(allowed)
    }
    if not state:
        raise ValueError("No ARS tensors matched the requested checkpoint components.")
    return state


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash names, shapes, dtypes, and exact tensor bytes deterministically."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        byte_view = value.reshape(1).view(torch.uint8) if value.ndim == 0 else value.view(torch.uint8)
        digest.update(byte_view.numpy().tobytes())
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except OSError:
        pass
    return "unknown"


def _json_config(cfg: Any) -> Dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    return {}


def load_checkpoint_manifest(path: str | Path) -> Optional[Dict[str, Any]]:
    """Read an ARS provenance sidecar without touching model tensors."""
    checkpoint = Path(path)
    manifest_path = checkpoint.with_suffix(checkpoint.suffix + ".manifest.json")
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint manifest must be a JSON object: {manifest_path}")
    return payload


def apply_checkpoint_config(cfg: Any, path: str | Path) -> Dict[str, Any]:
    """Restore checkpoint-defining settings onto ``cfg`` before wrapper creation.

    Only architecture/inference fields are restored.  Runtime choices such as
    data size, device, optimizer, and output paths deliberately remain under
    the caller's control.  The returned mapping is suitable for audit logs.
    """
    manifest = load_checkpoint_manifest(path)
    if manifest is None:
        return {}
    saved = manifest.get("config", {})
    if not isinstance(saved, dict):
        raise TypeError("Checkpoint manifest config must be a JSON object.")
    applied = {}
    for key in sorted(CHECKPOINT_CONFIG_FIELDS):
        if key in saved and hasattr(cfg, key):
            value = saved[key]
            # JSON represents tuple-valued configuration as lists.  The
            # selected fields currently need no tuple coercion, but preserve
            # regular Python scalar/list values exactly for dataclass config.
            setattr(cfg, key, value)
            applied[key] = value
    return applied


def save_ars_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    phase: str,
    cfg: Any,
    prefixes: Optional[Iterable[str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Save raw tensors plus a mandatory, human-readable provenance sidecar.

    The tensor file deliberately stays a normal state dict for compatibility
    with existing Kaggle notebooks.  The adjacent ``.manifest.json`` makes its
    architecture, phase, and exact bytes auditable before evaluation.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = ars_state_dict(model, prefixes=prefixes)
    torch.save(state, destination)
    deployment_mode = (
        extra_metadata.get("deployment_mode")
        if extra_metadata and "deployment_mode" in extra_metadata
        else ("baseline" if getattr(model, "deployment_disabled", False) else "ars")
    )
    manifest = {
        "format": "ars-checkpoint-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "deployment_mode": deployment_mode,
        "git_commit": _git_commit(),
        "state_sha256": state_sha256(state),
        "tensor_count": len(state),
        "components": sorted({name.split(".", 1)[0] for name in state}),
        "tensor_shapes": {name: list(value.shape) for name, value in sorted(state.items())},
        "config": _json_config(cfg),
    }
    if hasattr(model, "deployment_modes_by_shot"):
        manifest["deployment_modes_by_shot"] = dict(getattr(model, "deployment_modes_by_shot"))
    if hasattr(model, "get_routing_manifest"):
        manifest["routing_manifest"] = model.get_routing_manifest()
    if extra_metadata:
        manifest["extra"] = extra_metadata
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {"path": str(destination), "manifest_path": str(manifest_path), **manifest}


def load_ars_checkpoint_strict(
    model: torch.nn.Module, path: str | Path, device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load every saved ARS tensor exactly, rejecting shape/key drift."""
    map_location = device if device is not None else torch.device("cpu")
    payload = torch.load(path, map_location=map_location)
    state = payload.get("steer_state", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError("ARS checkpoint must contain a tensor state dictionary.")
    saved = {name: value for name, value in state.items() if is_ars_key(name)}
    expected = {name: value for name, value in model.state_dict().items() if is_ars_key(name)}
    unexpected = sorted(set(saved) - set(expected))
    mismatched = {
        name: {"checkpoint": list(saved[name].shape), "model": list(expected[name].shape)}
        for name in sorted(set(saved) & set(expected))
        if tuple(saved[name].shape) != tuple(expected[name].shape)
    }
    if not saved or unexpected or mismatched:
        raise RuntimeError(
            "ARS checkpoint is incompatible: "
            + json.dumps({"unexpected_ars_keys": unexpected, "shape_mismatches": mismatched}, indent=2)
        )
    result = model.load_state_dict(saved, strict=False)
    failed = sorted(set(saved) & set(result.missing_keys))
    if failed or result.unexpected_keys:
        raise RuntimeError(f"PyTorch did not apply all saved ARS tensors: missing={failed}, unexpected={result.unexpected_keys}")
    manifest = load_checkpoint_manifest(path)
    if manifest is not None:
        if manifest.get("state_sha256") != state_sha256(saved):
            raise RuntimeError(f"ARS checkpoint hash disagrees with its manifest: {Path(path).with_suffix(Path(path).suffix + '.manifest.json')}")
        if hasattr(model, "deployment_disabled"):
            model.deployment_disabled = (manifest.get("deployment_mode") == "baseline")
        if hasattr(model, "deployment_modes_by_shot") and manifest.get("deployment_modes_by_shot"):
            model.deployment_modes_by_shot = dict(manifest["deployment_modes_by_shot"])
        if hasattr(model, "load_routing_manifest"):
            model.load_routing_manifest(manifest.get("routing_manifest"))
    return {"saved_ars_tensors": len(saved), "manifest": manifest, "non_ars_missing": len(result.missing_keys)}
