"""Fast contracts for the production Phase 1 ARS surface."""
import json
from types import SimpleNamespace
import torch
from scripts.run_ars_pipeline import build_config, parse_args
from src.configs.config import RSCConfig
from src.deploy.calibrate import measure_envelope, write_contract
from src.evaluation.answers import extract_numeric_answer, score_tiers
from src.models.wrapper import Phi2WithRSC
from src.steering.steernet import CausalDWConv1d, SteerNet
from src.training.trainer import _trim_to_attended_tokens


def test_certified_config_physics_and_decode_policy():
    cfg = RSCConfig()
    assert cfg.fixed_layers == [8, 11, 13, 15, 17, 19, 20, 21, 22, 24]
    assert cfg.candidate_layers == [8, 11, 13, 15, 17, 19, 20, 21, 22, 24]
    assert cfg.steer_kappa == 0.04
    assert cfg.steer_scale == 0.30
    assert cfg.k_scale_mode == "sqrt"
    assert cfg.max_tok_len == 512
    assert cfg.phase1_lr == 5e-4
    assert cfg.loss_mode == "typed"
    assert cfg.train_shot_probs == [0.50, 0.10, 0.20, 0.05, 0.15]
    assert cfg.shield_policy == "context"
    assert cfg.mag_l1_weight == 0.20
    assert cfg.focal_gamma == 1.0
    assert cfg.steer_amplitude_k is None


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
    tiers = score_tiers("Final answer: 12", 12.0)
    assert tiers["strict"] and tiers["marker"] and tiers["lenient"]


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
    assert contract["fixed_layers"] == cfg.fixed_layers
    assert contract["version"] == 3
    assert json.loads(path.read_text())["envelope"]["route"] == cfg.fixed_layers


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
            "loss_mode=legacy",
            "--set",
            "catalog_max_k=3",
        ]
    )
    cfg = build_config(args)
    assert cfg.phase1_epochs == 3
    assert cfg.phase1_lr == 0.002
    assert cfg.loss_mode == "legacy"


def test_batch_trimming_keeps_auxiliary_masks_aligned():
    batch = {
        "input_ids": torch.arange(12).view(1, 12),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]),
        "labels": torch.ones(1, 12, dtype=torch.long),
    }
    extras = {"prompt_shield_mask": torch.ones(1, 12, dtype=torch.bool)}
    trimmed, trimmed_extras = _trim_to_attended_tokens(batch, extras)
    assert trimmed["input_ids"].shape[-1] == 4
    assert trimmed_extras["prompt_shield_mask"].shape[-1] == 4


def test_dataset_split_preparation_and_batching():
    from datasets import Dataset
    from scripts.run_ars_pipeline import prepare_dataset_split
    from src.training.trainer import _extract_batch_rows

    ds = Dataset.from_dict(
        {
            "input_ids": torch.randint(0, 100, (10, 8)),
            "attention_mask": torch.ones((10, 8)),
            "labels": torch.randint(0, 100, (10, 8)),
        }
    )
    ds.set_format("torch")
    split = prepare_dataset_split(ds, limit=6)
    assert isinstance(split, list)
    assert len(split) == 6
    assert isinstance(split[0], dict)

    batch = _extract_batch_rows(split, offset=0, batch_size=4)
    assert isinstance(batch, list)
    assert len(batch) == 4


def test_wrapper_preserves_single_element_tuple_outputs():
    from types import SimpleNamespace
    from transformers import LlamaConfig, LlamaForCausalLM
    from src.steering.steernet import SteerNet

    cfg = SimpleNamespace(
        fixed_layers=[0],
        lora_rank=4,
        lora_alpha=32.0,
        lora_dropout=0.0,
        steer_kappa=0.04,
        layer_norms_calibration="",
        prompt_shielding=False,
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
    model._selection = torch.tensor([[True]])
    model._attention_mask = torch.ones(1, 10, dtype=torch.bool)

    steer_net = SteerNet(hidden_dim=8, rank=4, kappa=0.04)
    single_tuple = (torch.zeros(1, 10, 8),)
    steered = model._apply_steering(0, steer_net, single_tuple)
    assert isinstance(steered, tuple)
    assert len(steered) == 1
    assert steered[0].shape == (1, 10, 8)
