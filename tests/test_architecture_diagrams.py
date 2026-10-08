import os
import unittest
from scripts.check_architecture_diagrams import (
    compute_arch_hashes,
    compute_combined_hash,
    check_integrity,
    ARCH_FILES,
    DIAGRAM_FILES,
    get_repo_root,
)

class TestArchitectureDiagrams(unittest.TestCase):
    def setUp(self):
        self.root = get_repo_root()

    def test_arch_files_exist(self):
        for rel in ARCH_FILES:
            full = os.path.join(self.root, rel)
            self.assertTrue(os.path.exists(full), f"Architectural source file missing: {rel}")

    def test_diagram_files_exist(self):
        for rel in DIAGRAM_FILES:
            full = os.path.join(self.root, rel)
            self.assertTrue(os.path.exists(full), f"Diagram artifact missing: {rel}")

    def test_hash_computation(self):
        hashes = compute_arch_hashes(self.root)
        self.assertEqual(len(hashes), len(ARCH_FILES))
        for k, v in hashes.items():
            self.assertNotEqual(v, "missing")
            self.assertEqual(len(v), 64)  # SHA-256 hex length

        comb1 = compute_combined_hash(hashes)
        comb2 = compute_combined_hash(hashes)
        self.assertEqual(comb1, comb2)
        self.assertEqual(len(comb1), 64)

    def test_integrity_check_cached(self):
        is_valid, reason, info = check_integrity(self.root)
        self.assertTrue(is_valid, f"Expected architecture cache to be valid, got: {reason}")
        self.assertIn("current_combined", info)
        self.assertEqual(info["current_combined"], info["cached_combined"])


if __name__ == "__main__":
    unittest.main()
