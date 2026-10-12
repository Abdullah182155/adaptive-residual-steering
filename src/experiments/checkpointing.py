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
    manifest = {
        "format": "ars-checkpoint-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "git_commit": _git_commit(),
        "state_sha256": state_sha256(state),
        "tensor_count": len(state),
        "components": sorted({name.split(".", 1)[0] for name in state}),
        "tensor_shapes": {name: list(value.shape) for name, value in sorted(state.items())},
        "config": _json_config(cfg),
    }
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
    manifest_path = Path(path).with_suffix(Path(path).suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if manifest is not None and manifest.get("state_sha256") != state_sha256(saved):
        raise RuntimeError(f"ARS checkpoint hash disagrees with its manifest: {manifest_path}")
    return {"saved_ars_tensors": len(saved), "manifest": manifest, "non_ars_missing": len(result.missing_keys)}
