import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.multi_dataset import (
    EvalBenchmarkConfig,
    build_benchmark_prompt,
    extract_numeric_answer,
    extract_choice_answer,
    evaluate_sample_answer,
    load_benchmark_dataset,
    BENCHMARK_EXEMPLARS,
)

class TestMultiDatasetBenchmark(unittest.TestCase):
    def test_config_defaults(self):
        cfg = EvalBenchmarkConfig()
        self.assertEqual(len(cfg.datasets), 5)
        self.assertIn("gsm8k", cfg.datasets)
        self.assertIn("svamp", cfg.datasets)
        self.assertIn("arc", cfg.datasets)
        self.assertIn("math500", cfg.datasets)
        self.assertIn("gsm_plus", cfg.datasets)
        self.assertEqual(cfg.shot_counts, [0, 2, 4, 8])
        self.assertTrue(cfg.prompt_shielding)

    def test_few_shot_prompt_builder(self):
        for ds_name in ["gsm8k", "svamp", "arc", "math500", "gsm_plus"]:
            # 0-shot
            p0 = build_benchmark_prompt(ds_name, "What is 2 + 2?", n_shot=0)
            self.assertIn("Question: What is 2 + 2?", p0)
            self.assertIn("Answer: Let's think step by step.", p0)
            self.assertNotIn("Step 1:", p0)

            # 2-shot
            p2 = build_benchmark_prompt(ds_name, "What is 2 + 2?", n_shot=2)
            self.assertEqual(p2.count("Question:"), 3)  # 2 exemplars + 1 test question

            # 4-shot
            p4 = build_benchmark_prompt(ds_name, "What is 2 + 2?", n_shot=4)
            self.assertEqual(p4.count("Question:"), 5)

            # 8-shot
            p8 = build_benchmark_prompt(ds_name, "What is 2 + 2?", n_shot=8)
            self.assertEqual(p8.count("Question:"), 9)

    def test_extract_numeric_answer(self):
        self.assertEqual(extract_numeric_answer("Let's calculate.\n#### 42"), 42.0)
        self.assertEqual(extract_numeric_answer("The result is \\boxed{100}."), 100.0)
        self.assertEqual(extract_numeric_answer("Final answer: 25.5"), 25.5)
        self.assertEqual(extract_numeric_answer("The answer is $12,500"), 12500.0)
        self.assertEqual(extract_numeric_answer("Therefore the total is 14."), 14.0)
        self.assertEqual(extract_numeric_answer("x = -7"), -7.0)
        self.assertIsNone(extract_numeric_answer("No numbers here"))

    def test_extract_choice_answer(self):
        self.assertEqual(extract_choice_answer("Final answer: (A)"), "A")
        self.assertEqual(extract_choice_answer("The correct answer is B."), "B")
        self.assertEqual(extract_choice_answer("Thus \\boxed{C}"), "C")
        self.assertEqual(extract_choice_answer("Therefore, (D) is correct."), "D")
        self.assertEqual(extract_choice_answer("The option is (B)"), "B")

    def test_evaluate_sample_answer_numeric(self):
        resp_corr = "Step 1: Calculate.\nFinal answer: 42"
        res = evaluate_sample_answer("gsm8k", resp_corr, 42.0)
        self.assertTrue(res["correct"])
        self.assertEqual(res["extracted"], 42.0)
        self.assertTrue(res["has_answer"])

        resp_incorr = "Step 1: Calculate.\nFinal answer: 10"
        res_fail = evaluate_sample_answer("gsm8k", resp_incorr, 42.0)
        self.assertFalse(res_fail["correct"])
        self.assertEqual(res_fail["extracted"], 10.0)

    def test_evaluate_sample_answer_multiple_choice(self):
        resp_corr = "Step 1: Plants use light.\nFinal answer: B"
        res = evaluate_sample_answer("arc", resp_corr, "B")
        self.assertTrue(res["correct"])
        self.assertEqual(res["extracted"], "B")

        # Digital mapping (e.g. 2 -> B)
        res_digit = evaluate_sample_answer("arc", "Final answer: 2", "B")
        self.assertTrue(res_digit["correct"])

    def test_load_benchmark_dataset_fallback(self):
        for name in ["gsm8k", "svamp", "arc", "math500", "gsm_plus"]:
            data = load_benchmark_dataset(name, n_samples=3, use_fallback_if_failed=True)
            self.assertEqual(len(data), 3)
            for item in data:
                self.assertIn("question", item)
                self.assertIn("gold", item)
                self.assertIn("dataset", item)
                self.assertIsNotNone(item["gold"])

if __name__ == "__main__":
    unittest.main()
