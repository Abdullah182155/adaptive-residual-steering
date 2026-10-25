import contextlib
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import benchmark as B
from src.evaluation.output_cache import (
    EMPTY_ROUTE_KEY,
    OutputCache,
    canonical_route_key,
)
from src.evaluation.stats import (
    compare_to_control,
    holm_bonferroni,
    mcnemar_exact,
    paired_bootstrap_ci,
)


# --------------------------------------------------------------------------- stats

class TestStats(unittest.TestCase):
    def test_mcnemar_known_values(self):
        ctrl  = [True] * 2 + [False] * 8 + [True] * 5  # c=2 (ctrl only), b=8 (treat only)
        treat = [False] * 2 + [True] * 8 + [True] * 5
        r = mcnemar_exact(ctrl, treat)
        self.assertEqual((r["b"], r["c"]), (8, 2))
        self.assertAlmostEqual(r["p"], 112 / 1024)

    def test_mcnemar_edges(self):
        self.assertEqual(mcnemar_exact([True, False], [True, False])["p"], 1.0)
        r = mcnemar_exact([False] * 10, [True] * 10)
        self.assertAlmostEqual(r["p"], 2 / 1024)
        r = mcnemar_exact([False] * 700, [True] * 700)  # no underflow crash at large n
        self.assertGreater(r["p"], 0.0)
        self.assertLess(r["p"], 1e-100)
        self.assertEqual(
            mcnemar_exact([True] * 5 + [False] * 5, [False] * 5 + [True] * 5)["p"], 1.0
        )

    def test_bootstrap(self):
        r = paired_bootstrap_ci([False] * 50, [True] * 50, n_boot=300, seed=1)
        self.assertAlmostEqual(r["delta_pp"], 100.0)
        self.assertAlmostEqual(r["ci_lo"],    100.0)
        self.assertAlmostEqual(r["ci_hi"],    100.0)
        z = paired_bootstrap_ci([True, False] * 20, [True, False] * 20, n_boot=300)
        self.assertEqual((z["delta_pp"], z["ci_lo"], z["ci_hi"]), (0.0, 0.0, 0.0))
        rng = np.random.default_rng(0)
        c   = rng.random(200) < 0.4
        t   = rng.random(200) < 0.6
        a   = paired_bootstrap_ci(c, t, n_boot=500, seed=7)
        b   = paired_bootstrap_ci(c, t, n_boot=500, seed=7)
        self.assertEqual(a, b)
        self.assertLess(a["ci_lo"],    a["delta_pp"])
        self.assertGreater(a["ci_hi"], a["delta_pp"])

    def test_holm(self):
        adj = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.2})
        self.assertAlmostEqual(adj["a"], 0.04)
        self.assertAlmostEqual(adj["c"], 0.09)
        self.assertAlmostEqual(adj["b"], 0.09)
        self.assertAlmostEqual(adj["d"], 0.2)

    def test_compare_to_control(self):
        ctrl = [False] * 40 + [True] * 10
        out  = compare_to_control(
            ctrl, {"x": [True] * 40 + [True] * 10, "y": ctrl}, n_boot=200, seed=0
        )
        self.assertTrue(out["x"]["significant"])
        self.assertFalse(out["y"]["significant"])
        self.assertEqual(out["y"]["p"], 1.0)


# --------------------------------------------------------------------------- cache

class TestCache(unittest.TestCase):
    def test_canonical_empty(self):
        self.assertEqual(canonical_route_key([], "learned", 0.5), EMPTY_ROUTE_KEY)
        self.assertEqual(canonical_route_key((), "const",  1.0),  EMPTY_ROUTE_KEY)
        self.assertEqual(canonical_route_key([12, 10], "const", 1.0), ((10, 12), "const", 1.0))

    def test_hit_miss_persist_and_namespace(self):
        with tempfile.TemporaryDirectory() as d:
            c  = OutputCache(d, "fp")
            k  = canonical_route_key([10], "const", 1.0)
            self.assertIsNone(c.get("p", k, "guards_on"))
            c.put("p", k, "guards_on", {"text": "t", "tripped": False})
            self.assertEqual(c.get("p", k, "guards_on")["text"], "t")
            self.assertIsNone(c.get("p", k, "guards_off"))              # namespaced
            self.assertIsNone(c.get("q", k, "guards_on"))               # prompt-specific
            self.assertIsNone(
                c.get("p", canonical_route_key([12], "const", 1.0), "guards_on")
            )
            c.put("p", EMPTY_ROUTE_KEY, "guards_on", {"text": "base", "tripped": False})
            self.assertEqual(
                c.get("p", canonical_route_key([], "learned", 0.3), "guards_off")["text"],
                "base",
            )
            c2 = OutputCache(d, "fp")   # survives restart
            self.assertEqual(c2.get("p", k, "guards_on")["text"], "t")
            self.assertEqual(len(c2), 2)
            self.assertEqual(c.stats()["hits"] >= 2, True)


# --------------------------------------------------------------------------- helpers

def contract_dict(adopted="learned"):
    return {
        "steer_scale":  1.0,
        "k_scale_mode": "sqrt",
        "s_fixed_star": [12],
        "theta_abs":    0.5,
        "router": {"policy": "catalog", "inference_mode": "argmax"},
        "gate":   {"adopted": adopted},
        "envelope": {"rel_norm_cap": 0.02},
        "guards": {"k_max": 4, "finite_check": True, "revert_to_k0": True},
    }


class FakeModel:
    def __init__(self):
        self.override = None
        self.guards   = {"enabled": False}
        self.gate     = None
        self.theta    = None

    def set_routing_override(self, mode=None, layers=None):
        self.override = None if mode is None else (mode, frozenset(layers or ()))

    def set_guards(self, **kw):
        self.guards = {**self.guards, **kw}

    def set_gate_mode(self, mode, scale=None):
        self.gate = (mode, scale)

    def set_theta_abs(self, th):
        self.theta = th

    def set_router_policy(self, p):
        pass

    def set_router_inference_mode(self, m):
        pass


# --------------------------------------------------------------------------- run_arm

class TestRunArm(unittest.TestCase):
    PROMPTS = ["p0", "p1", "p2"]

    def setUp(self):
        self.cfg      = SimpleNamespace(bench_max_new_tokens=16, bench_batch_size=((0, 8),))
        self.contract = contract_dict()
        self.tmp      = tempfile.TemporaryDirectory()
        self.cache    = OutputCache(self.tmp.name, "t")
        self.model    = FakeModel()
        self.generated: list = []

        def fake_gen(model, tok, prompts, device, max_new, bs, desc=""):
            self.generated.append(list(prompts))
            is_base = model.override is not None and model.override[0] == "none"
            on      = bool(model.guards.get("enabled"))
            texts   = [("base:" if is_base else "steered:") + p for p in prompts]
            trip    = [(not is_base) and on and p == "p1" for p in prompts]
            return texts, trip

        self.p_gen   = mock.patch.object(B, "_generate_batch_full", side_effect=fake_gen)
        self.p_route = mock.patch.object(B, "predict_routes", return_value=[(), (10,), (12,)])
        self.p_gen.start()
        self.p_route.start()
        self.addCleanup(self.p_gen.stop)
        self.addCleanup(self.p_route.stop)
        self.addCleanup(self.tmp.cleanup)

    def _run(self, which):
        return B.run_arm(
            self.model, None, self.PROMPTS, self.contract,
            self.cfg, "cpu", which, self.cache,
        )

    def test_empty_route_shares_baseline_generation(self):
        base = self._run(None)
        self.assertEqual(base["texts"], ["base:p0", "base:p1", "base:p2"])
        self.assertEqual(sum(len(g) for g in self.generated), 3)
        self.generated.clear()
        r = self._run("router")
        self.assertEqual(self.generated, [["p1", "p2"]])  # p0 (empty route) from cache
        self.assertEqual(r["texts"][0], "base:p0")
        self.assertEqual(r["routes"], [[], [10], [12]])

    def test_guard_revert_vs_raw(self):
        self._run(None)
        full = self._run("full_ars")
        self.assertEqual(full["tripped"],  [False, True, False])
        self.assertEqual(full["reverted"], [False, True, False])
        self.assertEqual(full["texts"][1], "base:p1")   # K=0 revert
        self.assertEqual(full["texts"][2], "steered:p2")
        raw = self._run("raw_ars")
        self.assertEqual(raw["reverted"], [False, False, False])
        self.assertEqual(raw["tripped"],  [False, False, False])
        self.assertEqual(raw["texts"][1], "steered:p1")  # guards off: no revert

    def test_second_pass_is_fully_cached(self):
        self._run(None)
        self._run("router")
        self.generated.clear()
        self._run("router")
        self.assertEqual(self.generated, [])

    def test_fixed_arm_uses_contract_subset(self):
        self._run(None)
        self.generated.clear()
        r = self._run("fixed")
        self.assertEqual(r["routes"], [[12], [12], [12]])
        self.assertEqual(sum(len(g) for g in self.generated), 3)


# --------------------------------------------------------------------------- assemble

class TestAssemble(unittest.TestCase):
    def test_block_and_extraction(self):
        golds = [1.0, 2.0, 3.0, 4.0]
        mk    = lambda ts, rs: {
            "texts": ts, "routes": rs, "tripped": [False] * 4, "reverted": [False] * 4
        }
        base = mk(["#### 1", "#### 9", "#### 3", "#### 9"], [[]] * 4)
        full = mk(["#### 1", "#### 2", "#### 3", "#### 9"], [[10], [10, 12], [], [12]])
        meta = {
            k: {"label": k, "gate_mode": "const", "guards": True}
            for k in ("baseline", "full_ars")
        }
        blk = B.assemble_shot_block(
            0, golds, {"baseline": base, "full_ars": full}, meta, n_boot=100, seed=0
        )
        self.assertEqual(blk["arms"]["baseline"]["accuracy_strict"], 50.0)
        self.assertEqual(blk["arms"]["full_ars"]["accuracy_strict"], 75.0)
        st = blk["stats"]["strict"]["full_ars"]
        self.assertEqual((st["b"], st["c"]), (1, 0))
        self.assertAlmostEqual(blk["arms"]["full_ars"]["summary"]["abstain_rate"], 0.25)
        self.assertEqual(B.extract_strict("so Final answer: 1,234."), 1234.0)
        self.assertEqual(B.extract_strict("Therefore, Final Answer: 56."), 56.0)
        self.assertEqual(B.extract_strict("The final answer is: 78."), 78.0)
        self.assertEqual(B.extract_strict("So the Final Answer is 90."), 90.0)
        self.assertIsNone(B.extract_strict("it is 5 maybe"))


if __name__ == "__main__":
    unittest.main()
