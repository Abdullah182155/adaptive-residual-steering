"""S5 calibration for the fixed Phase 1 route."""
import json
from pathlib import Path
import torch

def measure_envelope(model, rows, cfg, device):
    """Measure the deployed fixed route, never a router surrogate."""
    model.set_routing_override("only", cfg.fixed_layers); model.eval(); maximum = 0.0
    with torch.no_grad():
        for row in rows:
            model(input_ids=row["input_ids"].unsqueeze(0).to(device), attention_mask=row["attention_mask"].unsqueeze(0).to(device))
            maximum = max(maximum, float(model.guard_state.get("rel_max", 0.0)))
    return {"route": list(cfg.fixed_layers), "relative_norm_max": maximum, "relative_norm_cap": cfg.steer_kappa}

def write_contract(path, *, cfg, envelope):
    contract = {"version": 1, "model": cfg.model_name, "fixed_layers": list(cfg.fixed_layers), "shot_scales": {"0": 1.0, "2": 0.50, "4": 0.35}, "envelope": envelope}
    Path(path).write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8"); return contract

def load_contract(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def apply_contract(model, contract):
    model.set_routing_override("only", contract["fixed_layers"])
    model.set_shot_aware_scale({int(k): float(v) for k, v in contract["shot_scales"].items()})
