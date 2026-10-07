import unittest
import torch
import torch.nn as nn
from src.configs.config import RSCConfig
from src.steering.steernet import RSCSteerNet
from src.gating.gate import RSCUsefulnessGate
from src.routing.router import JointLayerRouter
from src.routing.rloo import sample_k, sample_subset_plackett_luce, policy_entropy_bonus

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

if __name__ == "__main__":
    unittest.main()
