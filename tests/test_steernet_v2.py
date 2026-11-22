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


def test_v2_gain_is_applied_before_the_relative_norm_cap():
    network = SteerNetV2(hidden_dim=4, rank=2, rho_max=0.10, layer_norm_ref=10.0)
    with torch.no_grad():
        network.down_proj.weight.fill_(0.1)
        network.up_proj.weight.fill_(0.01)
    hidden = torch.ones(2, 3, 4)

    unit = network(hidden, is_decode=True, gain=torch.tensor([1.0, 1.0]))
    doubled = network(hidden, is_decode=True, gain=torch.tensor([2.0, 2.0]))

    assert torch.allclose(doubled, unit * 2.0, atol=1e-6)
    assert torch.all(doubled.norm(dim=-1) <= 1.0 + 1e-6)


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


def test_route_bank_installs_independent_experts_and_uses_selected_adapter(tmp_path):
    from torch import nn
    from src.configs.config import RSCConfig
    from src.models.wrapper import Phi2WithRSC

    calibration = tmp_path / "norms.json"
    calibration.write_text(
        json.dumps({
            "layers": {
                str(layer): {"p99_norm": 20.0}
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
    model = Phi2WithRSC(Backbone(), cfg, torch.device("cpu")).eval()
    model.add_route_expert([12, 15])
    route_state = model.route_expert_state([12, 15])
    model.install_route_experts([[], [12, 15]], {"route_12_15": route_state})

    expert = model.bank_experts["route_12_15"]["12"]
    shared = model.steer_nets["12"]
    assert expert.up_proj.weight.data_ptr() != shared.up_proj.weight.data_ptr()
    with torch.no_grad():
        expert.down_proj.weight.fill_(0.1)
        expert.up_proj.weight.fill_(0.1)

    model.set_route_experts([[12, 15]])
    model.set_routing_override(
        "matrix", torch.tensor([[True, True, False, False]], dtype=torch.bool)
    )
    model._prepare_forward(
        None,
        (),
        {
            "input_ids": torch.ones((1, 3), dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        },
    )
    hidden = torch.ones((1, 3, 8))
    steered = model._apply_steering(0, shared, hidden)
    inactive_layer = model._apply_steering(2, model.steer_nets["18"], hidden)

    assert not torch.equal(steered[:, 1:], hidden[:, 1:])
    assert torch.equal(inactive_layer, hidden)

    model.set_route_experts(None)
    shared_only = model._apply_steering(0, shared, hidden)
    assert torch.equal(shared_only, hidden)
