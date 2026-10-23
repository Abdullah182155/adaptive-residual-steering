"""Failure policy + run manifest (Task M0).

Policy:
  * "warn"   (default): diagnostics NEVER raise; they are recorded in the manifest.
  * "strict" (CI only): a failed diagnostic is escalated to FatalContractError.
Contract violations (non-finite params, non-finite loss streaks, ...) call
``RunManifest.fatal`` directly and raise under either policy.
"""
import json
import math
import os
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, NoReturn, Optional

import numpy as np
import torch

SCHEMA_VERSION = 1


class FatalContractError(RuntimeError):
    """Raised when a hard contract is violated. ``.code`` holds the error code."""

    def __init__(self, message: str, code: str = ""):
        super().__init__(message)
        self.code = code


def to_jsonable(x: Any) -> Any:
    """Recursively convert tensors / numpy / dataclasses / paths / sets to JSON-safe values."""
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        return to_jsonable(t.item() if t.numel() == 1 and t.ndim == 0 else t.tolist())
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, np.generic):
        return to_jsonable(x.item())
    if is_dataclass(x) and not isinstance(x, type):
        return to_jsonable(asdict(x))
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (set, frozenset)):
        items = [to_jsonable(v) for v in x]
        try:
            return sorted(items)
        except TypeError:
            return items
    return str(x)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunManifest:
    def __init__(self, path: str = "./run_manifest.json", policy: str = "warn", run_id: Optional[str] = None):
        if policy not in {"warn", "strict"}:
            raise ValueError('failure policy must be "warn" or "strict"')
        self.path = path
        self.policy = policy
        self._lock = threading.RLock()
        self._active_stage: Optional[str] = None
        self._auto = False  # True only for the lazily created default singleton
        self.data: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id or uuid.uuid4().hex[:12],
            "failure_policy": policy,
            "config": {},
            "steer_scale": 0.30,
            "gate_mode": "learned",
            "warnings": [],
            "advisories": {},
            "stages": {},
        }
        self.flush()

    # ------------------------------------------------------------------ setup
    def configure(self, path: Optional[str] = None, policy: Optional[str] = None) -> None:
        with self._lock:
            if path is not None:
                self.path = path
            if policy is not None:
                if policy not in {"warn", "strict"}:
                    raise ValueError('failure policy must be "warn" or "strict"')
                self.policy = policy
                self.data["failure_policy"] = policy
            self._auto = False
            self.flush()

    def set(self, key: str, value: Any) -> None:
        """Set a (possibly dotted) key, e.g. set("config.steer_scale", 0.30)."""
        with self._lock:
            parts = key.split(".")
            node = self.data
            for p in parts[:-1]:
                nxt = node.get(p)
                if not isinstance(nxt, dict):
                    nxt = {}
                    node[p] = nxt
                node = nxt
            node[parts[-1]] = to_jsonable(value)
            self.flush()

    # --------------------------------------------------------------- recording
    def warn(self, code: str, message: str, stage: str = "", **data) -> None:
        with self._lock:
            self.data["warnings"].append(
                {
                    "ts": _now_iso(),
                    "stage": stage or (self._active_stage or ""),
                    "code": code,
                    "message": message,
                    "data": to_jsonable(data),
                }
            )
            self.flush()

    def fatal(self, code: str, message: str, stage: str = "", **data) -> NoReturn:
        with self._lock:
            self.warn(code, message, stage, **data)
            stages = self.data["stages"]
            target = stage if stage in stages else self._active_stage
            if target and target in stages:
                stages[target]["status"] = "failed"
                stages[target]["t_end"] = time.time()
            self.flush()
        raise FatalContractError(f"[{code}] {message}", code=code)

    def diagnostic(self, code: str, ok: bool, message: str, stage: str = "", **data) -> bool:
        if ok:
            return True
        if self.policy == "strict":
            self.fatal(code, message, stage, **data)
        self.warn(code, message, stage, **data)
        return False

    def advisory(self, key: str, value: Any, interpretation: str = "") -> None:
        with self._lock:
            self.data["advisories"][key] = {
                "value": to_jsonable(value),
                "interpretation": interpretation,
                "ts": _now_iso(),
            }
            self.flush()

    def record_metrics(self, stage: str, **metrics) -> None:
        with self._lock:
            st = self.data["stages"].setdefault(
                stage, {"status": "running", "t_start": time.time(), "t_end": None, "metrics": {}}
            )
            st.setdefault("metrics", {}).update(to_jsonable(metrics))
            self.flush()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        with self._lock:
            prev = self._active_stage
            self._active_stage = name
            self.data["stages"][name] = {
                "status": "running",
                "t_start": time.time(),
                "t_end": None,
                "metrics": {},
            }
            self.flush()
        try:
            yield
        except BaseException:
            with self._lock:
                st = self.data["stages"][name]
                st["status"] = "failed"
                st["t_end"] = time.time()
                self._active_stage = prev
                self.flush()
            raise
        else:
            with self._lock:
                st = self.data["stages"][name]
                st["status"] = "completed"
                st["t_end"] = time.time()
                self._active_stage = prev
                self.flush()

    # ------------------------------------------------------------------- I/O
    def flush(self) -> None:
        """Atomic write: temp file in the same directory + os.replace."""
        with self._lock:
            directory = os.path.dirname(os.path.abspath(self.path)) or "."
            tmp_path = None
            try:
                os.makedirs(directory, exist_ok=True)
                fd, tmp_path = tempfile.mkstemp(prefix=".run_manifest_", suffix=".tmp", dir=directory)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(to_jsonable(self.data), f, indent=2)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass
                os.replace(tmp_path, self.path)
                tmp_path = None
            except OSError as exc:  # never let manifest I/O kill training; in-memory copy stays valid
                print(f"  ⚠️ RunManifest flush failed ({exc}); continuing with in-memory manifest")
            finally:
                if tmp_path is not None and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass


_MANIFEST: Optional[RunManifest] = None
_MANIFEST_LOCK = threading.Lock()


def get_manifest() -> RunManifest:
    global _MANIFEST
    with _MANIFEST_LOCK:
        if _MANIFEST is None:
            _MANIFEST = RunManifest()
            _MANIFEST._auto = True
        return _MANIFEST


def set_manifest(m: RunManifest) -> None:
    global _MANIFEST
    with _MANIFEST_LOCK:
        _MANIFEST = m
