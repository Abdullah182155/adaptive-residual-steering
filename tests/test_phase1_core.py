"""Fast contracts for the production Phase 1 ARS surface."""
import json
from types import SimpleNamespace
import torch
from scripts.run_ars_pipeline import build_config, parse_args
from src.configs.config import RSCConfig, decode_only_for_shot, shot_aware_scale_map
from src.deploy.calibrate import measure_envelope, write_contract
from src.evaluation.answers import extract_numeric_answer
from src.models.wrapper import Phi2WithRSC
from src.steering.steernet import CausalDWConv1d, SteerNet


def test_certified_config_physics_and_decode_policy():
    cfg = RSCConfig()
    assert cfg.fixed_layers == [13, 20, 22]
    assert cfg.steer_kappa == 0.04
    assert cfg.steer_scale == 0.30
    assert cfg.k_scale_mode == "sqrt"
    assert shot_aware_scale_map(cfg) == {0: 1.0, 2: 0.50, 4: 0.35}
    assert decode_only_for_shot(0, cfg) is False
    assert decode_only_for_shot(2, cfg) is True


def test_causal_convolution_dirac_initialization_and_cache_priming():
    convolution = CausalDWConv1d(3, kernel_size=4)
    assert torch.allclose(
        convolution.weight[:, :, -1], torch.ones_like(convolution.weight[:, :, -1])
    )
    convolution(torch.randn(1, 5, 3))
    assert convolution._cache_initialized
    assert convolution._cache.shape == (1, 3, 3)
    assert convolution(torch.randn(1, 1, 3), is_decode=True).shape == (1, 1, 3)


def test_steernet_clamps_using_active_route_size():
    network = SteerNet(8, rank=4, layer_idx=13, layer_norms={13: 10.0}, kappa=0.04)
    delta = network(torch.randn(2, 4, 8), active_k=torch.tensor([3, 3]))
    assert float(delta.norm(dim=-1).max()) <= 0.04 * 10.0 / (3**0.5) + 1e-5


def test_prompt_shielding_keeps_prefill_pristine():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = SimpleNamespace(
        fixed_layers=[0],
        lora_rank=4,
        lora_alpha=32.0,
        lora_dropout=0.0,
        steer_kappa=0.04,
        layer_norms_calibration="",
        prompt_shielding=True,
        steer_scale=0.30,
    )
    backbone = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    ).eval()
    model = Phi2WithRSC(backbone, cfg, torch.device("cpu")).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(input_ids)
    with torch.no_grad():
        model.set_routing_override("none")
        baseline = model(input_ids=input_ids, attention_mask=mask).logits
        model.set_routing_override("only", [0])
        model.set_prompt_shield_mask(torch.ones_like(input_ids, dtype=torch.bool))
        shielded = model(input_ids=input_ids, attention_mask=mask).logits
    assert torch.allclose(baseline, shielded, atol=1e-6)


def test_canonical_and_lenient_answer_extraction():
    assert extract_numeric_answer("Work. #### 12") == 12.0
    assert extract_numeric_answer("Work\nThere are 19 left", lenient=True) == 19.0


def test_s5_contract_records_fixed_route(tmp_path):
    cfg = RSCConfig()
    rows = [{"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1])}]

    class Model:
        guard_state = {"rel_max": 0.12, "nonfinite": False}

        def set_routing_override(self, *args):
            self.route = args

        def set_prompt_shield_mask(self, *_):
            return None

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace()

    envelope = measure_envelope(Model(), rows, cfg, torch.device("cpu"))
    path = tmp_path / "contract.json"
    contract = write_contract(path, cfg=cfg, envelope=envelope)
    assert contract["fixed_layers"] == [13, 20, 22]
    assert json.loads(path.read_text())["envelope"]["route"] == [13, 20, 22]


def test_pipeline_cli_smoke_configuration():
    args = parse_args(
        [
            "--stage",
            "S1",
            "--s1_epochs",
            "3",
            "--s1_lr",
            "0.002",
            "--s_fixed",
            "13,20,22",
            "--set",
            "shot_aware_deploy=false",
            "--set",
            "catalog_max_k=3",
        ]
    )
    cfg = build_config(args)
    assert cfg.phase1_epochs == 3
    assert cfg.phase1_lr == 0.002
    assert cfg.shot_aware_deploy is False
