"""Training package public API, loaded lazily to keep helpers decoupled."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "evaluate_loss",
    "load_steer_weights",
    "save_steer_weights",
    "train_phase1_steernet",
]


def __getattr__(name: str) -> Any:
    """Load legacy trainer exports on demand without importing them for helpers."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    trainer = import_module(".trainer", __name__)
    value = getattr(trainer, name)
    globals()[name] = value
    return value
