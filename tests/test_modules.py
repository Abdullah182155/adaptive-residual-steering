import sys
import os
import unittest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.configs.config import RSCConfig
from src.steering.steernet import RSCSteerNet
from src.gating.gate import RSCUsefulnessGate
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

if __name__ == "__main__":
    unittest.main()
