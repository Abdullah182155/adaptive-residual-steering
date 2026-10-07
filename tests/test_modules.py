import sys
import os
import unittest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.configs.config import RSCConfig
from src.steering.steernet import RSCSteerNet
from src.gating.gate import RSCUsefulnessGate
from src.models.wrapper import Phi2WithRSC
from src.routing.router import JointLayerRouter
from src.routing.rloo import sample_k, sample_subset_plackett_luce, policy_entropy_bonus
from src.training import (
    train_rsc,
    train_phase1_steernet,
    train_router_bootstrap,
    run_staged_phase,
    save_steer_weights,
    load_steer_weights,
    run_pg_loss,
)

class TestARSModules(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = RSCConfig()
        self.hidden_dim = 64
        self.batch_size = 2
        self.seq_len = 8

    def test_steernet_shape(self):
        steer = RSCSteerNet(hidden_dim=self.hidden_dim, rank=16, lora_alpha=32.0)
        h = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        active_k = torch.tensor([2, 3])
        out = steer(h, active_k=active_k)
        self.assertEqual(out.shape, h.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_usefulness_gate(self):
        gate = RSCUsefulnessGate(hidden_dim=self.hidden_dim, cfg=self.cfg)
        h = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        delta = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        alpha = gate(h, delta)
        self.assertEqual(alpha.shape, (self.batch_size, self.seq_len, 1))
        self.assertTrue((alpha >= 0.0).all() and (alpha <= 1.0).all())

    def test_joint_router(self):
        target_layers = [10, 12, 14, 16]
        router = JointLayerRouter(target_layers, hidden_dim=self.hidden_dim, n_layers=20, max_active_layers=4)
        h = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        u, k_logits = router(h, mask)
        self.assertEqual(u.shape, (self.batch_size, len(target_layers)))
        self.assertEqual(k_logits.shape, (self.batch_size, 4))

    def test_rloo_sampling(self):
        k_logits = torch.randn(self.batch_size, 4)
        k, logp_k = sample_k(k_logits)
        self.assertEqual(k.shape, (self.batch_size,))
        self.assertTrue((k >= 1).all() and (k <= 4).all())

        u = torch.randn(self.batch_size, 4)
        subset_mask, logp_sub = sample_subset_plackett_luce(u, k)
        self.assertEqual(subset_mask.shape, (self.batch_size, 4))
        self.assertEqual(logp_sub.shape, (self.batch_size,))

    def test_config_attributes(self):
        cfg = RSCConfig()
        required_attrs = [
            "steer_finetune_lr", "steer_finetune_epochs", "steer_finetune_save_path",
            "gate_only_lr", "gate_only_epochs", "gate_only_save_path",
            "joint_finetune_epochs", "joint_finetune_lr_scale", "joint_finetune_router_lr_scale",
            "gate_retune_epochs", "gate_retune_lr", "phase2_save_path", "lora_save_path",
        ]
        for attr in required_attrs:
            self.assertTrue(hasattr(cfg, attr), f"Missing config attribute: {attr}")

    def test_routing_override_modes(self):
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Config", (), {"num_hidden_layers": 10, "hidden_size": 64})()
                self.model = type("Model", (), {"layers": nn.ModuleList([nn.Linear(64, 64) for _ in range(10)])})()
            def forward(self, *args, **kwargs):
                return None
        from src.models.wrapper import Phi2WithRSC
        dummy = DummyModel()
        cfg = RSCConfig()
        wrapper = Phi2WithRSC(dummy, cfg, device=torch.device("cpu"))
        wrapper.set_routing_override("all")
        self.assertEqual(wrapper._routing_override[0], "only")
        self.assertEqual(wrapper._routing_override[1], set(wrapper.target_layers))
        wrapper.set_routing_override("none")
        self.assertEqual(wrapper._routing_override[0], "none")
        wrapper.set_routing_override(None)
        self.assertIsNone(wrapper._routing_override)

    def test_cooperative_synergy(self):
        target_layers = [9, 11, 13, 15]
        router = JointLayerRouter(target_layers, hidden_dim=self.hidden_dim, n_layers=20, max_active_layers=4)
        h = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        u, k_logits = router(h, mask)
        self.assertIsNotNone(router._last_synergy_matrix)
        self.assertEqual(router._last_synergy_matrix.shape, (self.batch_size, 4, 4))
        # Diagonal must be 0 (no self-synergy)
        diag = torch.diagonal(router._last_synergy_matrix, dim1=1, dim2=2)
        self.assertTrue((diag == 0.0).all())
        # Test subset sampling with synergy
        k = torch.tensor([2, 3])
        subset, logp = sample_subset_plackett_luce(u, k, synergy_matrix=router._last_synergy_matrix)
        self.assertEqual(subset.shape, (self.batch_size, 4))
        self.assertEqual(subset.sum(dim=-1).tolist(), [2, 3])

    def test_prompt_shielding(self):
        class DummyLayer(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.linear = nn.Identity()
            def forward(self, h, *args, **kwargs):
                return h
        class DummyModel(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.config = type("Config", (), {"num_hidden_layers": 10, "hidden_size": hidden_dim})()
                self.embed = nn.Embedding(50, hidden_dim)
                self.model = type("Model", (), {"layers": nn.ModuleList([DummyLayer(hidden_dim) for _ in range(10)])})()
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                h = self.embed(input_ids)
                for layer in self.model.layers:
                    h = layer(h)
                return type("Output", (), {"logits": h})()

        dummy = DummyModel(self.hidden_dim)
        cfg = RSCConfig()
        wrapper = Phi2WithRSC(dummy, cfg, device=torch.device("cpu"))
        wrapper.eval()

        B, T = 1, 10
        # Pretend first 6 tokens are demonstrations, last 4 are question
        shield_mask = torch.zeros(B, T, dtype=torch.bool)
        shield_mask[:, :6] = True  # Shield first 6 tokens
        wrapper.set_prompt_shield_mask(shield_mask)

        # Force a steering layer active
        wrapper.set_routing_override("all")
        wrapper.set_gate_freeze(True, 1.0)

        # Hook to capture hidden states at the last steered layer
        steered_layer = wrapper.target_layers[-1]
        captured_h = {}
        def cap_hook(mod, inp, out):
            captured_h["out"] = out[0] if isinstance(out, tuple) else out
        h_handle = dummy.model.layers[steered_layer].register_forward_hook(cap_hook)

        inp_ids = torch.arange(T, dtype=torch.long).unsqueeze(0)
        attn = torch.ones(B, T, dtype=torch.long)
        _ = wrapper(input_ids=inp_ids, attention_mask=attn)
        h_handle.remove()

        out_h = captured_h["out"]
        # The first 6 tokens (shielded) must be unmodified baseline input
        baseline_h = dummy.embed(inp_ids)
        self.assertTrue(torch.allclose(out_h[:, :6, :], baseline_h[:, :6, :], atol=1e-5))
        # The last 4 tokens (unshielded) must have received residual steering
        self.assertFalse(torch.allclose(out_h[:, 6:, :], baseline_h[:, 6:, :], atol=1e-5))

    def test_gate_regularization_gradient_flow(self):
        gate = RSCUsefulnessGate(hidden_dim=self.hidden_dim, cfg=self.cfg)
        torch.nn.init.normal_(gate.fc2.weight, std=0.02)
        h = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, requires_grad=True)
        # Extreme magnitude delta to test anti-saturation
        delta = 100.0 * torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        alpha = gate(h, delta)
        self.assertEqual(alpha.shape, (self.batch_size, self.seq_len, 1))
        # Check output is strictly in (0, 1) and not NaN
        self.assertTrue(torch.all(alpha > 0.0) and torch.all(alpha < 1.0))
        self.assertFalse(torch.isnan(alpha).any())

        # Check backward gradient flow through aux alpha
        loss = gate._alpha_for_aux.sum()
        loss.backward()
        for name, param in gate.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Param {name} gradient is None")
                self.assertTrue(torch.isfinite(param.grad).all(), f"Param {name} gradient contains inf or NaN")
                self.assertFalse((param.grad == 0).all(), f"Param {name} gradient completely vanished to 0")

if __name__ == "__main__":
    unittest.main()

