import unittest
from unittest.mock import MagicMock
import torch

from src.evaluation.multi_benchmark_runner import (
    evaluate_model_condition,
    run_fair_multishot_benchmark,
    format_benchmark_summary_table,
    export_latex_summary_table,
)
from src.evaluation.multi_dataset import EvalBenchmarkConfig

class MockBatchEncoding(dict):
    def to(self, device):
        res = MockBatchEncoding()
        for k, v in self.items():
            res[k] = v.to(device) if hasattr(v, "to") else v
        return res


class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, return_tensors=None, **kwargs):
        tokens = [10, 20, 30]
        ids = torch.tensor([tokens])
        res = MockBatchEncoding()
        res["input_ids"] = ids
        res["attention_mask"] = torch.ones_like(ids)
        return res

    def encode(self, text, add_special_tokens=False):
        return [10, 20]

    def decode(self, token_ids, skip_special_tokens=True):
        return "Final answer: 42"


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.routing_override = None
        self.gate_freeze = False
        self.gate_freeze_value = 1.0
        self.prompt_shield_mask = None

    def set_routing_override(self, mode=None, layers=None):
        self.routing_override = mode

    def set_gate_freeze(self, freeze: bool, value: float = 1.0):
        self.gate_freeze = freeze
        self.gate_freeze_value = value

    def set_prompt_shield_mask(self, mask):
        self.prompt_shield_mask = mask

    def generate(self, input_ids, **kwargs):
        # Return mock token generation
        device = input_ids.device
        added = torch.tensor([[40, 50, 60]], device=device)
        return torch.cat([input_ids, added], dim=1)


class TestMultiBenchmarkRunner(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.tokenizer = MockTokenizer()
        self.device = torch.device("cpu")

    def test_evaluate_model_condition_baseline(self):
        samples = [{"question": "What is 40 + 2?", "gold": 42.0}]
        res = evaluate_model_condition(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            dataset_name="gsm8k",
            samples=samples,
            n_shot=0,
            condition="baseline",
        )
        # Check that routing was reset in finally
        self.assertIsNone(self.model.routing_override)
        self.assertFalse(self.model.gate_freeze)
        self.assertEqual(res["summary"]["accuracy"], 100.0)
        self.assertEqual(res["summary"]["correct"], 1)

    def test_evaluate_model_condition_steernet(self):
        samples = [{"question": "What is 40 + 2?", "gold": 42.0}]
        res = evaluate_model_condition(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            dataset_name="gsm8k",
            samples=samples,
            n_shot=2,
            condition="steernet",
        )
        self.assertEqual(res["summary"]["condition"], "steernet")
        self.assertEqual(res["summary"]["n_shot"], 2)

    def test_run_fair_multishot_benchmark_mock(self):
        cfg = EvalBenchmarkConfig(
            datasets=["gsm8k"],
            samples_per_dataset={"gsm8k": 2},
            shot_counts=[0, 2],
        )
        res = run_fair_multishot_benchmark(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            config=cfg,
            conditions=("baseline", "ars"),
            verbose=False,
        )
        tree = res["benchmark_results"]
        self.assertIn("gsm8k", tree)
        self.assertIn("baseline", tree["gsm8k"])
        self.assertIn("ars", tree["gsm8k"])
        self.assertIn("0_shot", tree["gsm8k"]["baseline"])
        self.assertIn("2_shot", tree["gsm8k"]["ars"])
        self.assertEqual(len(res["metrics_summary"]), 4)  # 1 dataset * 2 shots * 2 conditions

    def test_format_benchmark_summary_table(self):
        tree = {
            "gsm8k": {
                "baseline": {"0_shot": 37.0, "2_shot": 40.0},
                "steernet": {"0_shot": 41.0, "2_shot": 43.0},
                "ars": {"0_shot": 47.0, "2_shot": 53.33},
            }
        }
        table_str = format_benchmark_summary_table(tree)
        self.assertIn("GSM8K", table_str)
        self.assertIn("37.00%", table_str)
        self.assertIn("+10.00%", table_str)
        self.assertIn("+13.33%", table_str)

    def test_export_latex_summary_table(self):
        tree = {
            "gsm8k": {
                "baseline": {"0_shot": 37.0},
                "steernet": {"0_shot": 41.0},
                "ars": {"0_shot": 47.0},
            }
        }
        latex_str = export_latex_summary_table(tree)
        self.assertIn(r"\begin{table*}", latex_str)
        self.assertIn(r"\toprule", latex_str)
        self.assertIn(r"\textbf{GSM8K}", latex_str)
        self.assertIn(r"\textbf{+10.00\%}", latex_str)
        self.assertIn(r"\end{table*}", latex_str)


if __name__ == "__main__":
    unittest.main()
