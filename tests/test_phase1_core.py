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
    assert cfg.fixed_layers == [17, 20, 23, 26]
    assert cfg.candidate_layers == [8, 11, 13, 15, 17, 19, 20, 21, 22, 23, 24, 26]
    assert cfg.steer_kappa == 0.15
    assert cfg.steer_scale == 0.30
    assert cfg.k_scale_mode == "sqrt"
    assert cfg.max_tok_len == 768
    assert cfg.phase1_lr == 1.5e-4
    assert cfg.loss_mode == "typed"
    assert cfg.train_shot_probs == [0.50, 0.15, 0.20, 0.10, 0.05]  # V23: balanced shot distribution
    assert cfg.shield_policy == "prompt"
    assert cfg.diversify_prompts is False
    assert cfg.generation_repetition_penalty == 1.0
    assert cfg.steernet_use_mag_head is False
    assert cfg.mag_l1_weight == 0.0
    assert cfg.s1_answer_weight == 1.0
    assert cfg.on_policy is False
    assert cfg.focal_gamma == 2.0
    assert cfg.kl_weight == 0.3
    assert cfg.lambda_result == 0.40
    assert cfg.lambda_final == 0.30
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
    assert float(delta.norm(dim=-1).max().detach()) <= 0.04 * 10.0 / (3**0.5) + 1e-5


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


def test_prompt_shield_policy_masks_every_prefill_token_except_last():
    from src.data.prompts import build_prompt_with_shield

    class CharacterTokenizer:
        def __call__(self, text, return_tensors="pt", return_offsets_mapping=False):
            ids = torch.arange(1, len(text) + 1).unsqueeze(0)
            encoded = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            if return_offsets_mapping:
                encoded["offset_mapping"] = torch.tensor(
                    [[(idx, idx + 1) for idx in range(len(text))]]
                )
            return encoded

    _, encoded, shield = build_prompt_with_shield(
        CharacterTokenizer(), "What is 2 + 2?", policy="prompt"
    )
    assert "offset_mapping" not in encoded
    assert shield.sum().item() == encoded["input_ids"].shape[1] - 1
    assert not shield[-1]


def test_canonical_and_lenient_answer_extraction():
    assert extract_numeric_answer("Work. #### 12") == 12.0
    assert extract_numeric_answer("Work\nThere are 19 left", lenient=True) == 19.0
    tiers = score_tiers("Final answer: 12", 12.0)
    assert tiers["strict"] and tiers["marker"] and tiers["lenient"]


def test_arithmetic_verifier_uses_safe_expression_parser():
    from src.training.trainer import _verify_arithmetic_steps

    assert _verify_arithmetic_steps("3 * 60 = 180\n180 - 0.3*180 = 126")
    assert not _verify_arithmetic_steps("540 * 7 = 3000")
    assert _verify_arithmetic_steps("If x = 4, then the total = 20")


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
            "0.00015",
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
    assert cfg.phase1_lr == 1.5e-4
    assert cfg.loss_mode == "legacy"


def test_benchmark_subset_is_shuffled_and_repeatable(monkeypatch):
    import scripts.run_ars_pipeline as pipeline

    class FakeDataset:
        def __init__(self, examples):
            self.examples = list(examples)

        def __len__(self):
            return len(self.examples)

        def select(self, indices):
            return FakeDataset([self.examples[index] for index in indices])

        def __iter__(self):
            return iter(self.examples)

    examples = [
        {"question": f"question {index}", "answer": f"work #### {index}"}
        for index in range(12)
    ]
    monkeypatch.setattr(
        pipeline,
        "load_dataset",
        lambda *_args, **_kwargs: FakeDataset(examples),
    )

    first = pipeline.load_benchmark_rows(8, seed=123)
    repeated = pipeline.load_benchmark_rows(8, seed=123)
    other_seed = pipeline.load_benchmark_rows(8, seed=124)
    source_indices = [row["source_index"] for row in first]

    assert source_indices == [row["source_index"] for row in repeated]
    assert source_indices != [row["source_index"] for row in other_seed]
    assert len(set(source_indices)) == 8
    assert source_indices != list(range(8))
    assert all(row["answer"] == str(row["source_index"]) for row in first)


def test_benchmark_cli_has_reproducible_test_seed():
    from scripts.run_ars_pipeline import parse_args

    assert parse_args(["--stage", "bench"]).test_seed == 42
    assert parse_args(["--stage", "bench", "--test_seed", "7"]).test_seed == 7


def test_phase2_v2_config_has_locked_architecture_values():
    args = parse_args(["--stage", "bench", "--steernet_arch", "v2"])
    cfg = build_config(args)

    assert cfg.steernet_arch == "v2"
    assert cfg.fixed_layers == [12, 15, 18, 21]
    assert cfg.lora_rank == 16
    assert cfg.rho_max == 0.10
    assert cfg.steernet_norm_stat == "p99_norm"
    assert not cfg.steernet_use_mag_head
    assert not cfg.steernet_use_conv


def test_phase1_learning_rate_is_locked():
    import pytest

    with pytest.raises(ValueError, match="phase1_lr is locked at 1.5e-4"):
        RSCConfig(phase1_lr=2e-4).validate()

    with pytest.raises(ValueError, match="phase1_lr is locked at 1.5e-4"):
        build_config(parse_args(["--stage", "S1", "--s1_lr", "0.0002"]))


def test_build_model_forwards_explicit_attention_backend(monkeypatch):
    import scripts.run_ars_pipeline as pipeline

    calls = {}

    class Tokenizer:
        pad_token = "<pad>"
        pad_token_id = 0
        eos_token = "<eos>"

    class Backbone:
        def to(self, _device):
            return self

    class WrappedModel:
        def eval(self):
            return self

    monkeypatch.setenv("ARS_ATTENTION_IMPLEMENTATION", "eager")
    monkeypatch.setattr(
        pipeline.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: Tokenizer(),
    )
    monkeypatch.setattr(
        pipeline.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(pad_token_id=None),
    )

    def load_backbone(*args, **kwargs):
        calls.update(kwargs)
        return Backbone()

    monkeypatch.setattr(pipeline.AutoModelForCausalLM, "from_pretrained", load_backbone)
    monkeypatch.setattr(pipeline, "Phi2WithRSC", lambda *args, **kwargs: WrappedModel())

    pipeline.build_model(RSCConfig(), torch.device("cpu"))

    assert calls["attn_implementation"] == "eager"


def test_build_model_rejects_unsupported_attention_backend(monkeypatch):
    import pytest
    import scripts.run_ars_pipeline as pipeline

    monkeypatch.setenv("ARS_ATTENTION_IMPLEMENTATION", "flash")
    with pytest.raises(ValueError, match="ARS_ATTENTION_IMPLEMENTATION"):
        pipeline.build_model(RSCConfig(), torch.device("cpu"))


def test_benchmark_arms_load_their_own_steernet_checkpoints(tmp_path, monkeypatch):
    import scripts.run_ars_pipeline as pipeline

    phase1 = tmp_path / "steernet.pt"
    gated = tmp_path / "steernet_gated.pt"
    phase1.touch()
    gated.touch()
    cfg = SimpleNamespace(
        lora_save_path=str(phase1),
        steernet_gated_save_path=str(gated),
    )
    loaded = []
    monkeypatch.setattr(
        pipeline,
        "load_steer_weights",
        lambda model, path, cfg, device: loaded.append(path),
    )

    pipeline._load_arm_weights(object(), "fixed", cfg, torch.device("cpu"))
    pipeline._load_arm_weights(object(), "steer_only", cfg, torch.device("cpu"))
    pipeline._load_arm_weights(object(), "gate", cfg, torch.device("cpu"))
    pipeline._load_arm_weights(object(), "baseline", cfg, torch.device("cpu"))

    assert loaded == [str(phase1), str(phase1), str(gated)]


def test_steer_only_benchmark_arm_isolated_from_gate_and_router():
    from src.evaluation.benchmark import _configure_arm

    gate_provider = object()

    class Model:
        def __init__(self):
            self.cfg = SimpleNamespace(use_gate=True)
            self.gates = {"17": SimpleNamespace(_freeze_gate=False)}
            self.calls = []
            self._gate_provider = gate_provider

        def set_gate_provider(self, provider):
            self._gate_provider = provider
            self.calls.append(("gate_provider", provider))

        def set_routing_override(self, *args):
            self.calls.append(("route", args))

        def set_route_selector(self, selector):
            self.calls.append(("route_selector", selector))

        def set_intervention_gain(self, gain):
            self.calls.append(("gain", gain))

        def set_steer_scale(self, scale):
            self.calls.append(("steer_scale", scale))

        def set_control(self, mode):
            self.calls.append(("control", mode))

    model = Model()
    cfg = SimpleNamespace(fixed_layers=[17, 20, 23, 26], use_gate=True)
    _configure_arm(model, "steer_only", cfg, gate_provider)

    assert cfg.use_gate is False
    assert model.cfg.use_gate is False
    assert model._gate_provider is None
    assert model.gates["17"]._freeze_gate is True
    assert ("route", ("only", [17, 20, 23, 26])) in model.calls
    assert ("route_selector", None) in model.calls
    assert ("steer_scale", 1.0) in model.calls
    assert ("gain", 1.0) in model.calls
    assert ("control", None) in model.calls

    _configure_arm(model, "gate", cfg, gate_provider)
    assert cfg.use_gate is True
    assert model.cfg.use_gate is True
    assert model._gate_provider is gate_provider


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


def test_wrapper_applies_per_example_intensity_after_steernet_cap():
    from types import SimpleNamespace
    from transformers import LlamaConfig, LlamaForCausalLM
    from src.steering.steernet import SteerNet

    cfg = SimpleNamespace(
        fixed_layers=[0], lora_rank=4, lora_alpha=32.0, lora_dropout=0.0,
        steer_kappa=0.1, layer_norms_calibration="", prompt_shielding=False,
        steer_scale=0.30, amplitude_mode="relative",
    )
    backbone = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        )
    ).eval()
    model = Phi2WithRSC(backbone, cfg, torch.device("cpu")).eval()
    model._selection = torch.tensor([[True]])
    model._attention_mask = torch.ones(1, 4, dtype=torch.bool)
    hidden = torch.randn(1, 4, 8)
    steer = SteerNet(hidden_dim=8, rank=4, kappa=0.1, init_gain=0.1, dropout=0.0).eval()

    model.set_intervention_gains(torch.tensor([0.0]))
    zero = model._apply_steering(0, steer, hidden)
    model.set_intervention_gains(torch.tensor([1.0]))
    full = model._apply_steering(0, steer, hidden)
    model.set_intervention_gains(torch.tensor([0.5]))
    half = model._apply_steering(0, steer, hidden)

    full_delta = full - hidden
    assert torch.allclose(zero, hidden)
    assert torch.allclose(half - hidden, full_delta * 0.5, atol=1e-6)


def test_prompt_template_variants_preserve_answer_contract():
    from src.data.prompts import PROMPT_PREFACES, build_prompt_unified

    prompts = [build_prompt_unified("What is 3 + 4?", template_id=i) for i in range(3)]
    assert len(set(PROMPT_PREFACES)) == 3
    assert len(set(prompts)) == 3
    assert all(prompt.endswith("Question: What is 3 + 4?\nAnswer: Let's think step by step.\n") for prompt in prompts)


def test_steernet_norm_capped_below_ceiling():
    """V21: When MLP output is small (init), scale=1.0 — no capping."""
    net = SteerNet(hidden_dim=8, rank=4, kappa=0.08, amplitude_mode="norm_capped",
                   init_gain=0.02)
    h = torch.randn(2, 5, 8)
    delta = net(h, active_k=torch.tensor([4, 4]))
    centered_norm = (h - h.mean(dim=-1, keepdim=True)).norm(dim=-1, keepdim=True)
    max_norm = 0.08 / (4 ** 0.5) * centered_norm
    delta_norms = delta.norm(dim=-1, keepdim=True)
    assert (delta_norms <= max_norm + 1e-5).all(), "delta should be below ceiling at init"
    assert delta_norms.mean() > 0, "delta should be non-zero"


def test_steernet_norm_capped_at_ceiling():
    """V21: With large init, MLP output hits the ceiling."""
    net = SteerNet(hidden_dim=8, rank=4, kappa=0.08, amplitude_mode="norm_capped",
                   init_gain=5.0)
    h = torch.randn(2, 5, 8) * 10
    delta = net(h, active_k=torch.tensor([4, 4]))
    centered_norm = (h.float() - h.float().mean(dim=-1, keepdim=True)).norm(dim=-1, keepdim=True)
    max_norm = 0.08 / (4 ** 0.5) * centered_norm
    delta_norms = delta.norm(dim=-1, keepdim=True)
    assert (delta_norms <= max_norm + 1e-4).all(), "delta should be capped at ceiling"


def test_steernet_norm_capped_gradient_flow():
    """V21: Gradients flow through both direction and magnitude below ceiling."""
    net = SteerNet(hidden_dim=8, rank=4, kappa=0.08, amplitude_mode="norm_capped",
                   init_gain=0.02)
    net.train()
    h = torch.randn(2, 5, 8, requires_grad=False)
    delta = net(h, active_k=torch.tensor([4, 4]))
    loss = delta.sum()
    loss.backward()
    assert net.up.weight.grad is not None, "up.weight should have gradient"
    assert net.up.weight.grad.abs().sum() > 0, "gradient should be non-zero"


def test_config_validates_norm_capped():
    """V21: norm_capped is a valid amplitude_mode."""
    cfg = RSCConfig(amplitude_mode="norm_capped")
    cfg.validate()


def test_phase1_uses_identity_magnitude_by_default():
    """Phase 1 RFT pins the magnitude to one by omitting the magnitude head."""
    cfg = RSCConfig()
    assert cfg.steernet_use_mag_head is False
    assert cfg.amplitude_mode == "relative"
    assert cfg.rho_max == 0.08
    assert cfg.steernet_init_gain == 1.0
    assert cfg.on_policy_frac == 0.20
    assert cfg.gate_training_mode == "supervised"
    assert cfg.gate_supervised_temperature == 5.0


def test_v22_steernet_mag_head_gradient_always_flows():
    """V22: mag_head=True + relative mode — gradient flows through sigmoid, never killed by clamp."""
    net = SteerNet(hidden_dim=8, rank=4, kappa=0.04, amplitude_mode="relative",
                   use_mag_head=True, init_gain=1.0)
    net.train()
    h = torch.randn(2, 5, 8, requires_grad=False)
    delta = net(h, active_k=torch.tensor([4, 4]))
    loss = delta.sum()
    loss.backward()
    assert net.mag_head.weight.grad is not None, "mag_head.weight should have gradient"
    assert net.mag_head.weight.grad.abs().sum() > 0, "mag_head gradient should be non-zero"
    assert net.up.weight.grad is not None, "up.weight should have gradient"


def test_v22_steernet_mag_head_per_token_variation():
    """V22: Different tokens should have different magnitude values (per-token selectivity)."""
    net = SteerNet(hidden_dim=16, rank=8, kappa=0.04, amplitude_mode="relative",
                   use_mag_head=True, init_gain=1.0, mag_head_init_bias=0.0)
    net.eval()
    h = torch.randn(1, 10, 16) * 5
    delta = net(h, active_k=torch.tensor([4]))
    delta_norms = delta.norm(dim=-1)
    # With random inputs and mag_head bias=0.0, magnitudes should vary across tokens
    assert delta_norms[:, 1:].std() > 0, "delta norms should vary across tokens"


def test_v22_gate_xavier_init():
    """V22: gate fc2.weight uses Xavier init (not zeros) — breaks constant-output trap."""
    from src.configs.config import RSCConfig
    from src.gating.gate import RSCUsefulnessGate
    cfg = RSCConfig()
    gate = RSCUsefulnessGate(32, cfg)
    assert gate.fc2.weight.abs().sum() > 0, "fc2.weight should be non-zero (Xavier init)"


def test_param_groups_up_weight_decay():
    """Verify _param_groups assigns correct weight decay to up.weight based on mag_head."""
    from src.training.trainer import _param_groups
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.steer_nets = nn.ModuleDict({
                "17": nn.ModuleDict({
                    "up": nn.Linear(4, 8),
                    "mag_head": nn.Linear(8, 1),
                })
            })

    cfg = RSCConfig(steernet_use_mag_head=True, weight_decay=0.01)
    assert cfg.use_mag_head is True
    groups = _param_groups(DummyModel(), cfg)
    up_group = next(g for g in groups if any(p.shape == (8, 4) for p in g["params"]))
    assert up_group["weight_decay"] == 0.01

    cfg_no_mag = RSCConfig(steernet_use_mag_head=False, weight_decay=0.01)
    assert cfg_no_mag.use_mag_head is False
    groups_no_mag = _param_groups(DummyModel(), cfg_no_mag)
    up_group_no_mag = next(g for g in groups_no_mag if any(p.shape == (8, 4) for p in g["params"]))
    assert up_group_no_mag["weight_decay"] == 0.0
