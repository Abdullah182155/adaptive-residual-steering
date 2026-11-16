from types import SimpleNamespace
import json
import torch

from src.steering.steernet import SteerNetV2, load_layer_norms


def test_v2_zero_initialization_produces_exact_identity_delta():
    network = SteerNetV2(hidden_dim=12, rank=16, layer_norm_ref=20.0)
    hidden = torch.randn(2, 5, 12)

    delta = network(hidden)

    assert torch.count_nonzero(network.up_proj.weight) == 0
    assert torch.count_nonzero(delta) == 0


def test_v2_norm_cap_is_per_token_and_differentiable():
    network = SteerNetV2(hidden_dim=12, rank=4, rho_max=0.10, layer_norm_ref=2.0)
    with torch.no_grad():
        network.down_proj.weight.fill_(0.25)
        network.up_proj.weight.fill_(10.0)
    hidden = torch.randn(2, 3, 12, requires_grad=True)

    delta = network(hidden, is_decode=True)
    delta.norm(dim=-1).sum().backward()

    assert torch.all(delta.norm(dim=-1) <= 0.2 + 1e-5)
    assert network.up_proj.weight.grad is not None
    assert torch.isfinite(network.up_proj.weight.grad).all()
    assert torch.isfinite(hidden.grad).all()


def test_v2_shields_prefill_sink_and_honors_prompt_mask():
    network = SteerNetV2(hidden_dim=8, rank=2, layer_norm_ref=5.0)
    with torch.no_grad():
        network.up_proj.weight.fill_(0.1)

    delta = network(
        torch.ones(1, 3, 8),
        injection_mask=torch.tensor([[False, True, False]]),
    )

    assert torch.count_nonzero(delta[:, 0]) == 0
    assert torch.count_nonzero(delta[:, 1]) > 0
    assert torch.count_nonzero(delta[:, 2]) == 0


def test_norm_loader_selects_p99_without_median_fallback(tmp_path):
    path = tmp_path / "norms.json"
    path.write_text(
        json.dumps({"layers": {"12": {"median_norm": 10.0, "p99_norm": 20.0}}}),
        encoding="utf-8",
    )

    assert load_layer_norms(str(path), statistic="p99_norm") == {12: 20.0}
    assert load_layer_norms(str(path), statistic="median_norm") == {12: 10.0}


def test_v2_config_and_wrapper_select_calibrated_v2_modules(tmp_path):
    from torch import nn
    from src.configs.config import RSCConfig
    from src.models.wrapper import Phi2WithRSC

    calibration = tmp_path / "norms.json"
    calibration.write_text(
        json.dumps({
            "layers": {
                str(layer): {"median_norm": 4.0, "p99_norm": 10.0 + layer}
                for layer in (12, 15, 18, 21)
            }
        }),
        encoding="utf-8",
    )

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(hidden_size=8)
            self.model = SimpleNamespace(layers=nn.ModuleList([nn.Identity() for _ in range(22)]))

    cfg = RSCConfig(steernet_arch="v2", layer_norms_calibration=str(calibration))
    cfg.validate()
    wrapped = Phi2WithRSC(Backbone(), cfg, torch.device("cpu"))

    assert cfg.fixed_layers == [12, 15, 18, 21]
    assert cfg.lora_rank == 16 and cfg.rho_max == 0.10
    assert all(isinstance(wrapped.steer_nets[str(i)], SteerNetV2) for i in cfg.fixed_layers)
    assert wrapped.steer_nets["12"].layer_norm_ref == 22.0
