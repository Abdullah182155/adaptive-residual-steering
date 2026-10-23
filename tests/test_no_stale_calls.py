import os
import re
import unittest

ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SELF      = os.path.abspath(__file__)
SCAN_DIRS = ["src", "scripts", "tests", "docs", "notebooks"]
SCAN_FILES = ["README.md"]
EXTS      = (".py", ".md", ".ipynb")

STALE_ALL  = re.compile(
    r"""(set_routing_override\(\s*\\?["']all\\?["']|_routing_override\s*=\s*\(?\s*["']all["'])"""
)
STALE_INIT = re.compile(r"-\s*6\.0")
INIT_WORDS = ("init", "sigma", "σ", "α", "alpha", "bias")


def _files():
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dp, dn, fn in os.walk(base):
            dn[:] = [x for x in dn if x not in {"__pycache__", ".ipynb_checkpoints"}]
            for f in fn:
                p = os.path.join(dp, f)
                if f.endswith(EXTS) and os.path.abspath(p) != SELF:
                    yield p
    for f in SCAN_FILES:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            yield p


class TestNoStaleCalls(unittest.TestCase):
    def test_no_all_override(self):
        bad = []
        for p in _files():
            with open(p, encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if STALE_ALL.search(line):
                        bad.append(f"{os.path.relpath(p, ROOT)}:{i}: {line.strip()[:100]}")
        self.assertEqual(bad, [], 'stale routing override "all" found:\n' + "\n".join(bad))

    def test_no_stale_gate_init_claim(self):
        bad = []
        for p in _files():
            with open(p, encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    low = line.lower()
                    if STALE_INIT.search(line) and any(w in low for w in INIT_WORDS):
                        bad.append(f"{os.path.relpath(p, ROOT)}:{i}: {line.strip()[:100]}")
        self.assertEqual(bad, [], "stale -6.0 gate-init claim found:\n" + "\n".join(bad))

    def test_methodology_documents_current_init(self):
        p = os.path.join(ROOT, "docs", "methodology.md")
        if os.path.exists(p):
            text = open(p, encoding="utf-8").read()
            self.assertIn("0.88", text)


if __name__ == "__main__":
    unittest.main()
