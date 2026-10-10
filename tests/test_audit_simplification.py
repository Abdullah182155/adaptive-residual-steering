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

class DummyBackboneLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, *args, **kwargs):
        return self.linear(hidden_states)

class DummyModel(nn.Module):
    def __init__(self, hidden_dim: int = 64, n_layers: int = 10):
        super().__init__()
        self.layers = nn.ModuleList([DummyBackboneLayer(hidden_dim) for _ in range(n_layers)])

class DummyPhi2(nn.Module):
    def __init__(self, hidden_dim: int = 64, n_layers: int = 10):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_dim})()
        self.model = DummyModel(hidden_dim, n_layers)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        B = input_ids.shape[0] if input_ids is not None else 1
        T = input_ids.shape[1] if input_ids is not None else 8
        h = torch.randn(B, T, self.config.hidden_size)
        for layer in self.model.layers:
            h = layer(h)
        loss = torch.tensor(0.5, requires_grad=True)
        return type("Output", (), {"logits": h, "loss": loss})()

class TestAuditSimplification(unittest.TestCase):
    def setUp(self):
        self.hidden_dim = 64
        self.rank = 8
        self.alpha = 16.0

    def test_steernet_decoupled_magnitude_mode(self):
        # Direct mode (standard)
        net_direct = RSCSteerNet(
            self.hidden_dim, self.rank, self.alpha, magnitude_mode="direct"
        )
        h = torch.randn(2, 5, self.hidden_dim)
        out_direct = net_direct(h)
        self.assertEqual(out_direct.shape, (2, 5, self.hidden_dim))

        # Decoupled mode (direction normalized + bounded learnable magnitude)
        net_decoupled = RSCSteerNet(
            self.hidden_dim, self.rank, self.alpha, magnitude_mode="decoupled", max_magnitude=2.0
        )
        out_decoupled = net_decoupled(h)
        self.assertEqual(out_decoupled.shape, (2, 5, self.hidden_dim))
        
        # Test backprop through learnable log_magnitude
        loss = out_decoupled.sum()
        loss.backward()
        self.assertIsNotNone(net_decoupled.log_magnitude.grad)

    def test_token_gate_constant_mode_and_saturation(self):
        cfg = RSCConfig()
        cfg.gating_mode = "constant"
        cfg.gate_constant_value = 0.75
        gate = RSCUsefulnessGate(self.hidden_dim, cfg)

        h = torch.randn(2, 6, self.hidden_dim)
        delta = torch.randn(2, 6, self.hidden_dim)
        alpha = gate(h, delta)
        self.assertEqual(alpha.shape, (2, 6, 1))
        self.assertTrue(torch.allclose(alpha, torch.tensor(0.75)))

        metrics = gate.get_saturation_metrics()
        self.assertAlmostEqual(metrics["mean_alpha"], 0.75, places=2)
        self.assertEqual(metrics["low_saturation_pct"], 0.0)
        self.assertEqual(metrics["high_saturation_pct"], 0.0)

    def test_token_gate_learned_saturation_metrics(self):
        cfg = RSCConfig()
        cfg.gating_mode = "learned"
        gate = RSCUsefulnessGate(self.hidden_dim, cfg)

        h = torch.randn(2, 10, self.hidden_dim)
        delta = torch.randn(2, 10, self.hidden_dim)
        alpha = gate(h, delta)
        metrics = gate.get_saturation_metrics()
        self.assertIn("mean_alpha", metrics)
        self.assertIn("low_saturation_pct", metrics)
        self.assertIn("high_saturation_pct", metrics)
        self.assertGreaterEqual(metrics["mean_alpha"], 0.0)
        self.assertLessEqual(metrics["mean_alpha"], 1.0)

    def test_fixed_layers_configuration(self):
        cfg = RSCConfig()
        cfg.fixed_layers = [3, 5, 7]
        target_layers = cfg.rsc_layers_for(10)
        self.assertEqual(target_layers, [3, 5, 7])

    def test_model_wrapper_fixed_layers_routing(self):
        cfg = RSCConfig()
        cfg.fixed_layers = [3, 5]
        dummy_base = DummyPhi2(hidden_dim=self.hidden_dim, n_layers=10)
        wrapped = Phi2WithRSC(dummy_base, cfg)

        self.assertEqual(wrapped.target_layers, [3, 5])
        self.assertIsNotNone(wrapped._routing_override)
        mode, layers = wrapped._routing_override
        self.assertEqual(mode, "only")
        self.assertEqual(sorted(list(layers)), [3, 5])

        # Test forward pass with fixed layers
        inp = torch.randint(0, 100, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.bool)
        out = wrapped(input_ids=inp, attention_mask=mask)
        self.assertIsNotNone(out)

    def test_steernet_bounded_relative_norm(self):
        # Even with an artificially inflated lora_alpha, ||delta|| / ||h|| must never exceed max_relative_norm
        beta_max = 0.12
        net = RSCSteerNet(
            self.hidden_dim, self.rank, lora_alpha=5000.0, max_relative_norm=beta_max
        )
        h = torch.randn(2, 8, self.hidden_dim)
        delta = net(h)
        h_norm = torch.norm(h.float(), dim=-1, keepdim=True)
        delta_norm = torch.norm(delta.float(), dim=-1, keepdim=True)
        rel_ratio = delta_norm / (h_norm + 1e-7)
        self.assertTrue((rel_ratio <= beta_max + 1e-4).all(), f"Max relative ratio was {rel_ratio.max().item()} > {beta_max}")

    def test_steernet_orthogonal_projection(self):
        net = RSCSteerNet(
            self.hidden_dim, self.rank, self.alpha, orthogonal_projection=True
        )
        h = torch.randn(2, 8, self.hidden_dim)
        delta = net(h)
        dot_product = (delta * h).sum(dim=-1)
        self.assertTrue(torch.allclose(dot_product, torch.zeros_like(dot_product), atol=1e-4))

    def test_router_difficulty_awareness_and_dropout(self):
        from src.routing.router import JointLayerRouter
        router = JointLayerRouter(
            target_layers=[2, 4, 6],
            hidden_dim=self.hidden_dim,
            n_layers=10,
            max_active_layers=2,
            layer_dropout=0.5,
            difficulty_aware=True,
        )
        h = torch.randn(2, 8, self.hidden_dim)
        mask = torch.ones(2, 8, dtype=torch.bool)
        
        # Test evaluation mode (no dropout applied)
        router.eval()
        u_eval, k_eval = router(h, mask)
        self.assertEqual(u_eval.shape, (2, 3))
        self.assertFalse((u_eval <= -5000.0).any())

        # Test training mode (dropout can mask candidate layers)
        router.train()
        u_train, k_train = router(h, mask)
        self.assertEqual(u_train.shape, (2, 3))

if __name__ == "__main__":
    unittest.main()

