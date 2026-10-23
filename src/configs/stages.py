from enum import Enum


class Stage(str, Enum):
    S0 = "S0"      # Counterfactual Signal Audit (pre-training)
    S1 = "S1"      # SteerNet capacity verification (formerly Phase 1 / 2A)
    S0B = "S0b"    # Post-SteerNet signal audit
    S2 = "S2"      # Synergy exploration (formerly Phase 1.5)
    S3 = "S3"      # Router distillation / training (formerly Phase 2B / 2D)
    S4 = "S4"      # Token-gate dominance training (formerly Phase 2C)
    S5 = "S5"      # 5-condition frozen benchmark


LEGACY_TO_CANONICAL = {
    "phase1": Stage.S1.value,
    "phase1.5": Stage.S2.value,
    "2a": Stage.S1.value,
    "2b": Stage.S3.value,
    "2c": Stage.S4.value,
    "2d": Stage.S3.value,
}


def canonical_stage(name: str) -> str:
    """Map a legacy phase label (case-insensitive) to its canonical stage value."""
    key = str(name).strip().lower()
    return LEGACY_TO_CANONICAL.get(key, name)
