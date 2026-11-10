"""S5 deployment-envelope calibration for the fixed ARS route."""
import json
from pathlib import Path
from typing import Dict, Iterable
import torch


@torch.no_grad()
def measure_envelope(
    model, rows: Iterable[dict], cfg, device: torch.device
) -> Dict[str, object]:
    """Measure maximum residual ratio under the certified three-layer route."""
    model.eval()
    model.set_routing_override("only", cfg.fixed_layers)
    maximum, nonfinite_count, sample_count = 0.0, 0, 0
    row_items = (
        [rows[i] for i in range(len(rows))]
        if (not isinstance(rows, (list, tuple)) and hasattr(rows, "select"))
        else rows
    )
    for row in row_items:
        if hasattr(model, "set_prompt_shield_mask"):
            shield = row.get("prompt_shield_mask")
            model.set_prompt_shield_mask(
                None if shield is None else shield.unsqueeze(0).to(device)
            )
        model(
            input_ids=row["input_ids"].unsqueeze(0).to(device),
            attention_mask=row["attention_mask"].unsqueeze(0).to(device),
        )
        maximum = max(maximum, float(model.guard_state.get("rel_max", 0.0)))
        nonfinite_count += int(bool(model.guard_state.get("nonfinite", False)))
        sample_count += 1
    return {
        "route": list(cfg.fixed_layers),
        "samples": sample_count,
        "relative_norm_max": maximum,
        "relative_norm_cap": maximum,
        "nonfinite_count": nonfinite_count,
    }


def write_contract(path: str, *, cfg, envelope: Dict[str, object]) -> Dict[str, object]:
    """Write the deployment contract consumed by benchmark execution."""
    contract = {
        "version": 3,
        "model": cfg.model_name,
        "fixed_layers": list(cfg.fixed_layers),
        "steer_scale": cfg.steer_scale,
        "steer_kappa": cfg.steer_kappa,
        "k_scale_mode": cfg.k_scale_mode,
        "shield_policy": getattr(cfg, "shield_policy", "context"),
        "guard_revert_to_k0": cfg.guard_revert_to_k0,
        "envelope": envelope,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return contract


def load_contract(path: str) -> Dict[str, object]:
    """Read a deployment contract from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_contract(model, contract: Dict[str, object]) -> None:
    """Configure the wrapper for fixed-route deployment from a contract."""
    model.set_routing_override("only", contract["fixed_layers"])
    model.set_steer_scale(float(contract["steer_scale"]))
    # Restore guard setting from contract to ensure benchmark/training consistency
    if "guard_revert_to_k0" in contract and hasattr(model, "cfg"):
        model.cfg.guard_revert_to_k0 = bool(contract["guard_revert_to_k0"])
