import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.configs.config import RSCConfig
from src.routing.router import JointLayerRouter
from src.routing.distill import (
    build_soft_targets,
    balance_loss,
    consistency_loss,
    kl_q_p,
    distill_router,
    predict_classes,
    router_eval_metrics,
)

TARGET_LAYERS = [13, 15, 18, 20, 22]


class _StubTable:
    """Duck-typed CFTable: only the attributes distill.py reads."""

    def __init__(self, gain, pooled, scalars, n_shot, qid):
        self._gain = gain
        self.pooled = pooled
        self.scalars = scalars
        self.n_shot = n_shot
        self.qid = qid

    def gain(self):
        return self._gain


def make_planted_table(n_q=400, d=16, c=4, seed=0):
    """Each question's best class is a linear (argmax-of-coordinates) function of pooled."""
    rng = np.random.RandomState(seed)
    base = rng.randn(n_q, d).astype(np.float32)
    planted = rng.randint(0, c, size=n_q)
    base[np.arange(n_q), planted] += 2.5
    best = base[:, :c].argmax(axis=1)

    gain = -rng.uniform(0.2, 0.6, size=(n_q, c))
    gain[:, 0] = 0.0
    gain[np.arange(n_q), best] = np.where(best == 0, 0.0, 1.0)
    gain = gain.astype(np.float32)

    pooled0 = base
    pooled2 = base + 0.3 + 0.05 * rng.randn(n_q, d).astype(np.float32)   # shot-dependent offset
    scal = (0.1 * rng.rand(2 * n_q, 3)).astype(np.float32)
    return _StubTable(
        gain=np.concatenate([gain, gain], axis=0),
        pooled=np.concatenate([pooled0, pooled2], axis=0).astype(np.float32),
        scalars=scal,
        n_shot=np.concatenate([np.zeros(n_q, dtype=np.int64), np.full(n_q, 2, dtype=np.int64)]),
        qid=np.concatenate([np.arange(n_q), np.arange(n_q)]).astype(np.int64),
    )


class TestDistill(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = RSCConfig()
        self.cfg.router_catalog_save_path = os.path.join(self.tmp.name, "router.pt")

    def tearDown(self):
        self.tmp.cleanup()

    def test_build_soft_targets(self):
        rewards = torch.tensor([
            [0.5, 0.5, 0.0, -1.0],     # tie between classes 0 and 1
            [0.0, 1.0, 0.2, 0.3],      # unique clear winner (class 1)
        ])
        q, near = build_soft_targets(rewards, eps=0.01, tau=0.01)
        self.assertAlmostEqual(float(q[0, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(q[0, 1]), 0.5, places=6)
        self.assertAlmostEqual(float(q[0, 2:].sum()), 0.0, places=6)
        self.assertTrue(torch.allclose(q[1], torch.tensor([0.0, 1.0, 0.0, 0.0]), atol=1e-6))
        self.assertTrue(torch.allclose(q.sum(dim=1), torch.ones(2), atol=1e-6))
        self.assertEqual(near.sum(dim=1).tolist(), [2, 1])

    def test_balance_loss(self):
        p = torch.softmax(torch.randn(8, 5), dim=-1)
        none_tied = torch.zeros(8, dtype=torch.bool)
        self.assertEqual(float(balance_loss(p, none_tied)), 0.0)

        collapsed = torch.zeros(8, 5)
        collapsed[:, 0] = 1.0
        all_tied = torch.ones(8, dtype=torch.bool)
        collapsed_val = float(balance_loss(collapsed, all_tied))
        self.assertGreater(collapsed_val, 0.0)

        uniform = torch.full((8, 5), 0.2)
        self.assertGreater(collapsed_val, float(balance_loss(uniform, all_tied)))

    def test_kl_and_consistency_basics(self):
        q = torch.tensor([[0.5, 0.5, 0.0]])
        log_p = torch.log(torch.tensor([[0.5, 0.5, 1e-9]]))
        self.assertAlmostEqual(float(kl_q_p(q, log_p)), 0.0, places=5)
        p = torch.softmax(torch.randn(4, 6), dim=-1)
        self.assertAlmostEqual(float(consistency_loss(p, p)), 0.0, places=6)

    def test_planted_structure_distillation(self):
        d, c = 16, 4
        table = make_planted_table(n_q=400, d=d, c=c, seed=0)
        self.cfg.router_distill_epochs = 150
        self.cfg.router_distill_lr = 5e-3
        router = JointLayerRouter(TARGET_LAYERS, hidden_dim=d, n_layers=20,
                                  max_active_layers=4, n_catalog=c)
        report = distill_router(router, table, self.cfg, torch.device("cpu"), None,
                                seed=0, patience=40)
        self.assertGreaterEqual(report.top1_agreement, 0.9)
        self.assertLessEqual(report.regret_ratio, 0.5)
        self.assertTrue(os.path.exists(self.cfg.router_catalog_save_path))

    def test_class_mask_restricts_predictions(self):
        d, c, j = 16, 4, 2
        table = make_planted_table(n_q=120, d=d, c=c, seed=1)
        self.cfg.router_distill_epochs = 20
        self.cfg.router_distill_lr = 5e-3
        router = JointLayerRouter(TARGET_LAYERS, hidden_dim=d, n_layers=20,
                                  max_active_layers=4, n_catalog=c)
        report = distill_router(router, table, self.cfg, torch.device("cpu"), None,
                                class_mask=[0, j], seed=1, patience=20)
        for k, count in enumerate(report.class_hist):
            if k not in (0, j):
                self.assertEqual(count, 0)
        preds = predict_classes(router, table, np.arange(len(table.qid)), class_mask=[0, j])
        self.assertTrue(set(np.unique(preds).tolist()).issubset({0, j}))
        m = router_eval_metrics(router, table, np.arange(len(table.qid)), self.cfg, class_mask=[0, j])
        self.assertTrue(all(m["class_hist"][k] == 0 for k in range(c) if k not in (0, j)))

    def test_fresh_router_argmax_equals_zero_and_preserves_legacy_tests(self):
        d, c, B, T = 16, 6, 3, 8
        router = JointLayerRouter(TARGET_LAYERS, hidden_dim=d, n_layers=20,
                                  max_active_layers=4, n_catalog=c)
        self.assertTrue(torch.count_nonzero(router.catalog_head.weight) == 0)
        self.assertTrue(torch.count_nonzero(router.catalog_head.bias) == 0)

        h = torch.randn(B, T, d)
        mask = torch.ones(B, T, dtype=torch.bool)
        logits = router.catalog_logits(h, mask)
        self.assertEqual(tuple(logits.shape), (B, c))
        self.assertTrue(torch.count_nonzero(logits) == 0)
        self.assertTrue((logits.argmax(dim=-1) == 0).all())

        # legacy forward is untouched
        u, k_logits = router(h, mask)
        self.assertEqual(tuple(u.shape), (B, len(TARGET_LAYERS)))
        self.assertEqual(tuple(k_logits.shape), (B, 5))

        # routers built the old way have no catalog head
        legacy = JointLayerRouter(TARGET_LAYERS, hidden_dim=d, n_layers=20, max_active_layers=4)
        self.assertIsNone(legacy.catalog_head)
        u2, k2 = legacy(h, mask)
        self.assertEqual(tuple(u2.shape), (B, len(TARGET_LAYERS)))
        self.assertEqual(tuple(k2.shape), (B, 5))


if __name__ == "__main__":
    unittest.main()
