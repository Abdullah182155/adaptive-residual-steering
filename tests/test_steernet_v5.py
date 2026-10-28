"""
Tests for RSCSteerNetV5.

Phase 1.1: Test rank 64, per-layer cap, and position shielding.
"""

import pytest
import torch
from src.steering.steernet_v5 import RSCSteerNetV5


def test_steernet_v5_forward_shape():
    """Test RSCSteerNetV5 forward pass shape."""
    model = RSCSteerNetV5(
        hidden_dim=2560,  # Phi-2 hidden dim
        rank=64,
        layer_idx=13,
    )
    
    x = torch.randn(2, 10, 2560)
    delta = model(x)
    
    assert delta.shape == x.shape, f"Expected shape {x.shape}, got {delta.shape}"


def test_steernet_v5_position_shielding():
    """Test position-0 shielding."""
    model = RSCSteerNetV5(
        hidden_dim=2560,
        rank=64,
        layer_idx=13,
    )
    
    x = torch.randn(1, 5, 2560)
    position_ids = torch.arange(5).unsqueeze(0)  # [0, 1, 2, 3, 4]
    
    delta = model(x, position_ids=position_ids)
    
    # First position should have zero delta
    assert torch.allclose(delta[0, 0, :], torch.zeros(2560), atol=1e-6)
    
    # Other positions should have non-zero delta (not guaranteed due to randomness, but check shape)
    assert delta[0, 1:, :].shape == (4, 2560)


def test_steernet_v5_identity_init():
    """Test that zero-init up gives near-zero delta at init."""
    model = RSCSteerNetV5(
        hidden_dim=2560,
        rank=64,
        layer_idx=13,
    )
    
    x = torch.randn(1, 5, 2560)
    delta = model(x)
    
    # Delta should be very small at init (zero init up + small mag_head)
    assert torch.norm(delta).item() < 1.0


def test_steernet_v5_cap_envelope():
    """Test that delta is capped by C_l."""
    model = RSCSteerNetV5(
        hidden_dim=2560,
        rank=64,
        kappa=0.04,
        layer_norms={13: 10.0},  # N_l = 10
        layer_idx=13,
    )
    
    # Expected cap: C_l = 0.04 * 10 = 0.4
    expected_cap = 0.4
    
    x = torch.randn(1, 10, 2560) * 100  # Large input
    delta = model(x)
    
    # Delta norm should be bounded by cap * h_norm
    h_norm = torch.norm(x, dim=-1, keepdim=True)
    delta_norm = torch.norm(delta, dim=-1, keepdim=True)
    
    # Check soft cap (allow some numerical tolerance)
    ratio = delta_norm / (expected_cap * h_norm + 1e-7)
    assert ratio.max().item() < 1.1, f"Delta exceeds cap: max ratio {ratio.max().item()}"


def test_steernet_v5_telemetry():
    """Test telemetry collection."""
    model = RSCSteerNetV5(
        hidden_dim=2560,
        rank=64,
        layer_idx=13,
    )
    
    model.eval()
    x = torch.randn(1, 10, 2560)
    _ = model(x)
    
    # Check telemetry was collected
    assert model._pre_cap_n > 0, "Telemetry should have collected samples"
    assert model._pre_cap_max > 0, "Telemetry should have max value"
    
    # Reset
    model.reset_pre_cap_stats()
    assert model._pre_cap_n == 0, "Telemetry should be reset"


def test_load_layer_norms():
    """Test loading layer norms from calibration."""
    import json
    import tempfile
    import os
    
    # Create dummy calibration
    calibration = {
        "layers": {
            "13": {"median_norm": 5.5},
            "15": {"median_norm": 6.0},
            "18": {"median_norm": 6.5},
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(calibration, f)
        temp_path = f.name
    
    try:
        from src.steering.steernet_v5 import load_layer_norms
        norms = load_layer_norms(temp_path)
        
        assert norms == {13: 5.5, 15: 6.0, 18: 6.5}
    finally:
        os.unlink(temp_path)


def test_steernet_v5_rank_fixed():
    """Test that rank is fixed at 64."""
    model = RSCSteerNetV5(
        hidden_dim=2560,
        rank=64,
        layer_idx=13,
    )
    
    assert model.rank == 64
    assert model.down.out_features == 64
    assert model.up.in_features == 64
