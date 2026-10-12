import importlib.util
from pathlib import Path

import torch


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_checkpoint_steering.py"
_SPEC = importlib.util.spec_from_file_location("audit_checkpoint_steering", _SCRIPT)
audit = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(audit)


def test_tensor_inventory_records_each_ars_tensor_norm():
    state = {
        "gates.9.fc2.bias": torch.tensor([1.0]),
        "steer_nets.9.scale": torch.tensor(2.0),
        "router.k_head.bias": torch.tensor([0.0, 1.0]),
        "base_model.ignore": torch.tensor([9.0]),
    }
    inventory = audit.tensor_inventory(state)
    assert inventory["tensor_count"] == 3
    assert inventory["total_numel"] == 4
    assert {item["name"] for item in inventory["tensors"]} == {
        "gates.9.fc2.bias",
        "steer_nets.9.scale",
        "router.k_head.bias",
    }
    assert all(item["finite"] for item in inventory["tensors"])


def test_verify_ars_load_rejects_missing_tensor():
    model = audit.Phi2WithRSC(audit.SyntheticCausalLM(hidden_size=64, num_hidden_layers=10), audit.RSCConfig())
    state = {name: value.clone() for name, value in model.state_dict().items() if audit.is_ars_key(name)}
    state.pop(next(iter(state)))
    try:
        audit.verify_ars_load(model, state)
    except RuntimeError as exc:
        assert "missing_ars_keys" in str(exc)
    else:
        raise AssertionError("A checkpoint missing an ARS tensor must be rejected.")
