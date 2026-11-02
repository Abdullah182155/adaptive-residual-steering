import json
from types import SimpleNamespace
import torch
from src.configs.config import RSCConfig, DEFAULT_SHOT_AWARE_SCALE
from src.deploy.calibrate import measure_envelope, write_contract
from src.evaluation.answers import extract_numeric_answer
from src.steering.steernet_v5 import CausalDWConv1d, RSCSteerNetV5
from src.models.wrapper import Phi2WithRSC

def test_phase1_config_is_fixed_route():
    assert RSCConfig().fixed_layers == [13, 20, 22]
    assert DEFAULT_SHOT_AWARE_SCALE == {0: 1.0, 2: 0.50, 4: 0.35}
def test_causal_convolution_prefill_primes_decode_cache():
    conv = CausalDWConv1d(3, kernel_size=4); conv(torch.randn(1, 5, 3))
    assert conv._cache_initialized and conv(torch.randn(1, 1, 3), is_decode=True).shape == (1, 1, 3)
def test_v5_cap_and_dirac_current_tap():
    net = RSCSteerNetV5(8, layer_idx=13, layer_norms={13: 10.0}, kappa=0.08)
    assert torch.allclose(net.conv.weight[:, :, -1], torch.ones_like(net.conv.weight[:, :, -1]))
    assert float(net(torch.randn(2, 4, 8)).norm(dim=-1).max()) <= net.C_l + 1e-5
def test_canonical_and_lenient_answers():
    assert extract_numeric_answer("Work. Final answer: 12") == 12.0
    assert extract_numeric_answer("Work\nThere are 19 left", lenient=True) == 19.0
def test_s5_contract_records_fixed_route(tmp_path):
    cfg = RSCConfig(); rows = [{"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1])}]
    class Model:
        guard_state = {"rel_max": 0.12}
        def set_routing_override(self, *args): self.route = args
        def eval(self): return self
        def __call__(self, **kwargs): return SimpleNamespace()
    envelope = measure_envelope(Model(), rows, cfg, torch.device("cpu")); path = tmp_path / "contract.json"
    assert write_contract(path, cfg=cfg, envelope=envelope)["fixed_layers"] == [13, 20, 22]
    assert json.loads(path.read_text())["envelope"]["route"] == [13, 20, 22]

def test_wrapper_shields_prompt_tokens():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = RSCConfig(fixed_layers=[0], candidate_layers=[0], lora_rank=4)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)).eval()
    model = Phi2WithRSC(base, cfg, torch.device("cpu")).eval()
    ids = torch.tensor([[1, 2, 3]]); mask = torch.ones_like(ids)
    with torch.no_grad():
        model.set_routing_override("none"); baseline = model(input_ids=ids, attention_mask=mask).logits
        model.set_routing_override("only", [0]); model.set_prompt_shield_mask(torch.ones_like(ids, dtype=torch.bool))
        shielded = model(input_ids=ids, attention_mask=mask).logits
    assert torch.allclose(baseline, shielded, atol=1e-6)
