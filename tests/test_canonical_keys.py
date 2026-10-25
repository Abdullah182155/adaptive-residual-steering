import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.keys import canonical_token_key, parse_token_key


class TestCanonicalKeys(unittest.TestCase):
    def test_canonical_token_key_formats(self):
        # Raw integer qid
        self.assertEqual(canonical_token_key("train", 15, 0), "train:15:0")
        self.assertEqual(canonical_token_key("train", "15", 0), "train:15:0")

        # Prefixed qid
        self.assertEqual(canonical_token_key("train", "train:15", 0), "train:15:0")
        self.assertEqual(canonical_token_key("eval", "eval:42", 8), "eval:42:8")
        self.assertEqual(canonical_token_key("", "train:99", 4), "train:99:4")

        # Defaults
        self.assertEqual(canonical_token_key(None, "10", None), "train:10:0")

    def test_parse_token_key(self):
        split, qid, shot = parse_token_key("train:15:0")
        self.assertEqual(split, "train")
        self.assertEqual(qid, "15")
        self.assertEqual(shot, 0)

        split2, qid2, shot2 = parse_token_key("eval:99:8")
        self.assertEqual(split2, "eval")
        self.assertEqual(qid2, "99")
        self.assertEqual(shot2, 8)


if __name__ == "__main__":
    unittest.main()
