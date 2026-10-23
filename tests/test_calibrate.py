import contextlib
import json
import os
import sys
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.deploy import calibrate as C

DEV = torch.device("cpu")


class FakeModel:
    def __init__(self):
        self.target_layers = [10, 12, 14, 16]
        self.override = None
        self.theta    = None
        self.gate     = None
        self.guards   = None
        self.guard_state = {}
        self.rel = {}

    @contextlib.contextmanager
    def counterfactual_mode(self):
        yield

    def eval(self):
        return self

    def set_routing_override(self, mode=None, layers=None):
        self.override = None if mode is None else (mode, frozenset(layers or ()))

    def set_gate_mode(self, mode, scale=None):
        self.gate = (mode, scale)

    def set_theta_abs(self, th):
        self.theta = th

    def set_guards(self, **kw):
        self.guards = kw

    def set_router_policy(self, p):
        self.policy = p

    def set_router_inference_mode(self, m):
        self.inf = m

    def set_router_context_mask(self, m):
        pass

    def reset_guard_state(self):
        self.guard_state = {}

    def reset_routing_diagnostics(self):
        pass

    def routing_k_distribution(self):
        return {0: 0.25}

    def __call__(self, input_ids=None, attention_mask=None):
        self.guard_state = {"rel_norms": {k: torch.tensor(v) for k, v in self.rel.items()}}
        B, T = input_ids.shape
        return SimpleNamespace(logits=torch.zeros(B, T, 5))


def cfg_ns(**kw):
    base = dict(
        calib_nll_tie_eps=0.003,
        steer_scale=1.0,
        calib_theta_grid=(0.1, 0.5, 0.9, 1.01),
        calib_envelope_quantile=0.999,
        calib_batch_size=2,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def rows(n=10):
    return [
        {
            "input_ids":      torch.zeros(4, dtype=torch.long),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels":         torch.full((4,), -100),
        }
        for _ in range(n)
    ]


def acc_vec(a, n):
    k = int(round(a * n))
    return [True] * k + [False] * (n - k)


def full_fields(**over):
    f = dict(
        steer_scale=1.0,
        k_scale_mode="sqrt",
        steernet_k_conditioning=True,
        catalog={"layers": [10, 12], "max_k": 4},
        s_fixed_star=[12],
        fixed_evidence={"n": 120},
        theta_abs=0.5,
        theta_table=[{"theta": 0.5, "nll_gain": 0.1}],
        router={"policy": "catalog", "inference_mode": "argmax", "ckpt": "r.pt", "sha256": "a"},
        gate={"mode": "learned", "adopted": "learned", "evidence": {}, "ckpt": "g.pt", "sha256": "b"},
        envelope={"rel_norm_cap": 0.02, "quantile": 0.999, "p50": 0.01,
                  "p99": 0.02, "max": 0.03, "per_layer": {}},
        guards={"k_max": 4, "finite_check": True, "revert_to_k0": True},
        steernet={"ckpt": "s.pt", "sha256": "c"},
        selection_split={"n": 120, "ids_hash": "h"},
        manifest_path="m.json",
    )
    f.update(over)
    return f


# --------------------------------------------------------------------------- tests

class TestFixedSubset(unittest.TestCase):
    NLL     = {frozenset({10}): 1.000, frozenset({12}): 1.001,
               frozenset({10, 12}): 1.002, frozenset({14}): 1.5}
    catalog = [(10,), (12,), (10, 12), (14,), ()]

    def _run(self, accs):
        def fake_nll(model, r, cfg, dev):
            return np.full(len(r), self.NLL[model.override[1]])

        def fake_acc(model, r, cfg, dev):
            return acc_vec(accs[model.override[1]], len(r))

        with mock.patch.object(C, "_row_nll",          side_effect=fake_nll), \
             mock.patch.object(C, "paired_greedy_eval", side_effect=fake_acc):
            return C.choose_fixed_subset(FakeModel(), rows(10), self.catalog, cfg_ns(), DEV)

    def test_accuracy_then_smaller_k(self):
        s, ev = self._run({
            frozenset({10}): .5, frozenset({12}): .6, frozenset({10, 12}): .6,
        })
        self.assertEqual(s, (12,))
        self.assertEqual(len(ev["candidates"]), 3)  # (14,) is outside the NLL tie band

    def test_full_tie_goes_to_small_k_then_nll(self):
        s, _ = self._run({
            frozenset({10}): .6, frozenset({12}): .6, frozenset({10, 12}): .6,
        })
        self.assertEqual(s, (10,))


class TestThetaSearch(unittest.TestCase):
    GAIN = {0.1: .01, 0.5: .05, 0.9: .05, 1.01: .05}

    def _run(self, acc):
        def fake_nll(model, r, cfg, dev):
            if model.override is not None and model.override[0] == "none":
                return np.full(len(r), 2.0)
            return np.full(len(r), 2.0 - self.GAIN[model.theta])

        def fake_acc(model, r, cfg, dev):
            return acc_vec(acc[model.theta], len(r))

        with mock.patch.object(C, "_row_nll",          side_effect=fake_nll), \
             mock.patch.object(C, "paired_greedy_eval", side_effect=fake_acc):
            return C.search_theta_abs(FakeModel(), rows(10), cfg_ns(), DEV)

    def test_ties_break_by_accuracy(self):
        th, table = self._run({0.1: .5, 0.5: .6, 0.9: .6, 1.01: .5})
        self.assertEqual(th, 0.9)
        self.assertEqual(len(table), 4)

    def test_ties_break_to_larger_theta(self):
        th, _ = self._run({0.1: .5, 0.5: .6, 0.9: .6, 1.01: .6})
        self.assertEqual(th, 1.01)


class TestGateAdoption(unittest.TestCase):
    def test_truth_table(self):
        n = 100
        cases = [
            (-0.01, 0.60, 0.60, "learned"),
            (-0.01, 0.59, 0.60, "learned"),    # within tol of one example
            (-0.01, 0.58, 0.60, "passthrough"), # accuracy loss beyond tol
            ( 0.00, 0.70, 0.60, "passthrough"), # NLL not strictly better
            ( 0.02, 0.70, 0.60, "passthrough"),
        ]
        for diff, ag, au, want in cases:
            got, ev = C.decide_gate_adoption(
                {"mean_nll_diff": diff, "n": n},
                {"acc_gated": ag, "acc_ungated": au, "n": n},
                tol=1,
            )
            self.assertEqual(got, want, (diff, ag, au))
            self.assertEqual(ev["decision"], want)


class TestContract(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p       = os.path.join(d, "deployment_contract.json")
            written = C.write_contract(p, **full_fields())
            loaded  = C.load_contract(p)
            self.assertEqual(written, loaded)
            self.assertEqual(loaded["schema_version"], C.SCHEMA_VERSION)
            self.assertIn("created",    loaded)
            self.assertIn("git_commit", loaded)

    def test_missing_field_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                C.write_contract(os.path.join(d, "c.json"), steer_scale=1.0)
            f = full_fields()
            f["guards"] = {"k_max": 4}
            with self.assertRaises(ValueError):
                C.write_contract(os.path.join(d, "c.json"), **f)

    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.bin")
            with open(p, "wb") as f:
                f.write(b"abc")
            self.assertEqual(
                C.sha256_file(p),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


class TestApplyContract(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as d:
            self.c = C.write_contract(os.path.join(d, "c.json"), **full_fields())

    def test_fixed(self):
        m = FakeModel()
        C.apply_contract(m, self.c, "fixed")
        self.assertEqual(m.override, ("only", frozenset({12})))
        self.assertEqual(m.gate, ("const", 1.0))
        self.assertTrue(m.guards["enabled"])

    def test_router(self):
        m = FakeModel()
        C.apply_contract(m, self.c, "router")
        self.assertIsNone(m.override)
        self.assertEqual(m.theta, 0.5)
        self.assertEqual(m.gate, ("const", 1.0))
        self.assertTrue(m.guards["enabled"])

    def test_full_ars_learned_and_passthrough(self):
        m = FakeModel()
        C.apply_contract(m, self.c, "full_ars")
        self.assertEqual(m.gate[0], "learned")
        self.assertEqual(m.theta, 0.5)
        self.assertTrue(m.guards["enabled"])
        c2 = json.loads(json.dumps(self.c))
        c2["gate"]["adopted"] = "passthrough"
        C.apply_contract(m, c2, "full_ars")
        self.assertEqual(m.gate, ("const", 1.0))

    def test_raw_ars(self):
        m = FakeModel()
        C.apply_contract(m, self.c, "raw_ars")
        self.assertEqual(m.theta, 1.01)
        self.assertEqual(m.gate[0], "learned")
        self.assertFalse(m.guards["enabled"])

    def test_bad_arm(self):
        with self.assertRaises(ValueError):
            C.apply_contract(FakeModel(), self.c, "all")


class TestEnvelope(unittest.TestCase):
    def test_quantiles_no_warning(self):
        m     = FakeModel()
        m.rel = {
            10: np.linspace(0.001, 0.02, 500).tolist(),
            12: np.linspace(0.001, 0.03, 500).tolist(),
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            env = C.measure_envelope(m, rows(3), cfg_ns(), DEV)
        self.assertFalse(any("W_ENVELOPE_MISMATCH" in str(x.message) for x in w))
        self.assertLessEqual(env["p50"], env["p99"])
        self.assertLessEqual(env["p99"],          env["rel_norm_cap"] + 1e-12)
        self.assertLessEqual(env["rel_norm_cap"],  env["max"] + 1e-12)
        self.assertAlmostEqual(env["max"], 0.03, places=6)
        self.assertEqual(set(env["per_layer"]), {"10", "12"})
        self.assertEqual(env["quantile"], 0.999)

    def test_mismatch_warns(self):
        m     = FakeModel()
        m.rel = {10: np.linspace(0.01, 0.3, 500).tolist()}
        with self.assertWarns(UserWarning):
            env = C.measure_envelope(m, rows(2), cfg_ns(), DEV)
        self.assertTrue(any("W_ENVELOPE_MISMATCH" in s for s in env["warnings"]))


if __name__ == "__main__":
    unittest.main()
