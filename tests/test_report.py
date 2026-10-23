import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import make_report as MR
from src.evaluation.benchmark import ARM_SPEC, assemble_shot_block


def synthetic_record(n=60, seed=0):
    rng   = random.Random(seed)
    golds = [float(i) for i in range(n)]
    arm_results, arm_meta = {}, {}
    p_ok = {"baseline": 0.4, "fixed": 0.45, "router": 0.5, "full_ars": 0.55, "raw_ars": 0.35}
    for key, label, which in ARM_SPEC:
        texts, routes = [], []
        for g in golds:
            ok = rng.random() < p_ok[key]
            texts.append(f"#### {int(g) if ok else int(g) + 1}")
            routes.append([] if key == "baseline" else rng.choice([[], [10], [10, 12]]))
        arm_results[key] = {
            "texts":    texts,
            "routes":   routes,
            "tripped":  [False] * n,
            "reverted": [False] * n,
        }
        arm_meta[key] = {"label": label, "gate_mode": "const_x", "guards": key != "baseline"}
    blk = assemble_shot_block(0, golds, arm_results, arm_meta, n_boot=200, seed=1)
    return {
        "schema_version": 1,
        "created":    "t",
        "git_commit": "abc",
        "model_name": "m",
        "config": {"bench_n_boot": 200, "seed": 1, "bench_noninf_margin_pp": 1.0},
        "contract": {
            "s_fixed_star": [12], "theta_abs": 0.5,
            "gate_adopted": "learned", "rel_norm_cap": 0.02,
        },
        "shots":     {"0": blk},
        "secondary": {},
    }


class TestReport(unittest.TestCase):
    def test_tables(self):
        rec    = synthetic_record()
        tables = MR.build_tables(rec["shots"]["0"], n_boot=200)
        self.assertEqual(set(tables), {"accuracy", "paired_strict", "paired_lenient", "routing"})
        self.assertEqual(len(tables["accuracy"]["rows"]),      5)
        self.assertEqual(len(tables["paired_strict"]["rows"]), 4)  # four treatments vs baseline
        self.assertEqual(len(tables["routing"]["columns"]),    7)
        md = MR.to_markdown(tables["paired_strict"])
        self.assertIn("Holm p",     md)
        self.assertIn("McNemar p",  md)

    def test_stats_recomputed_when_missing(self):
        rec    = synthetic_record()
        blk    = rec["shots"]["0"]
        stored = blk["stats"]["strict"]["full_ars"]["b"]
        blk["stats"] = {}
        st = MR.ensure_stats(blk, n_boot=100, seed=1)
        self.assertEqual(st["strict"]["full_ars"]["b"], stored)

    def test_latex_and_files(self):
        self.assertEqual(MR.latex_escape("a_b 5% & c"), r"a\_b 5\% \& c")
        with tempfile.TemporaryDirectory() as d:
            paths = MR.write_report(synthetic_record(), d)
            md    = os.path.join(d, "REPORT.md")
            self.assertIn(md, paths)
            text = open(md, encoding="utf-8").read()
            self.assertIn("Full ARS",           text)
            self.assertIn("Deployment contract", text)
            tex = [p for p in paths if p.endswith(".tex")]
            self.assertEqual(len(tex), 4)
            body = open(tex[0], encoding="utf-8").read()
            self.assertIn(r"\begin{tabular}", body)
            self.assertIn(r"\toprule",        body)
            self.assertIn(r"\bottomrule",     body)
            self.assertTrue(os.path.isdir(os.path.join(d, "tables")))


if __name__ == "__main__":
    unittest.main()
