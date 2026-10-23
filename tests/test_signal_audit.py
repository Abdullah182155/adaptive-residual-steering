import os
import subprocess
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.configs.config import RSCConfig
from src.analysis.signal_audit import compute_shot_audit, compute_signal_audit


class _StubTable:
    """Duck-typed CFTable: only gain(), catalog, shots, heldout are used by the audit."""

    def __init__(self, gain, catalog=None, shots=None, heldout=None):
        self._g = np.asarray(gain, dtype=np.float64)
        N, C = self._g.shape
        self.catalog = catalog if catalog is not None else [(i,) for i in range(C)]
        self.shots = np.zeros(N, dtype=int) if shots is None else shots
        self.heldout = np.zeros(N, dtype=bool) if heldout is None else heldout

    def gain(self):
        return self._g


class TestSignalAudit(unittest.TestCase):
    def setUp(self):
        self.cfg = RSCConfig()
        self.cfg.audit_n_boot = 300

    def test_zero_gain_rational_abstention(self):
        t = _StubTable(np.zeros((40, 5)), catalog=[(), (1,), (2,), (1, 2), (3,)])
        res = compute_signal_audit(t, self.cfg)
        a = res.per_shot[0]
        self.assertEqual(a.g_fixed_star, 0.0)
        self.assertEqual(a.headroom_H, 0.0)
        self.assertIn("SWITCH_TRAINING_TARGET", res.flags)
        self.assertEqual(a.k0_diagnosis, "rational_abstention")
        self.assertTrue(res.decision_1.startswith("BRANCH_A"))

    def test_planted_structure_headroom(self):
        g = np.zeros((200, 2))
        g[:100, 0] = 0.05
        g[100:, 1] = 0.05
        a = compute_shot_audit(_StubTable(g), self.cfg, seed=1)
        self.assertAlmostEqual(a.g_fixed_star, 0.025, delta=1e-5)
        self.assertAlmostEqual(a.g_oracle, 0.05, delta=1e-5)
        self.assertAlmostEqual(a.headroom_H, 0.025, delta=1e-5)
        # H* is biased slightly low under re-selection, so allow a small tolerance
        self.assertLessEqual(a.headroom_ci[0], 0.025 + 1e-3)
        self.assertGreaterEqual(a.headroom_ci[1], 0.025 - 1e-3)

    def test_crossfit_conservative(self):
        rng = np.random.default_rng(0)
        g = rng.normal(0.0, 0.05, size=(100, 8))
        a = compute_shot_audit(_StubTable(g), self.cfg, seed=3)
        self.assertLessEqual(a.g_fixed_crossfit, a.g_fixed_star + 1e-12)

    def test_degenerate_table_does_not_raise(self):
        a = compute_shot_audit(_StubTable(np.zeros((1, 1))), self.cfg)
        self.assertEqual(a.n, 1)
        compute_signal_audit(_StubTable(np.zeros((1, 1))), self.cfg)

    def test_cli_help(self):
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "run_signal_audit.py"), "--help"],
            capture_output=True, text=True, cwd=ROOT,
        )
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
