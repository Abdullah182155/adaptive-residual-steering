import os
import shutil
import tempfile
import unittest
import torch
import torch.nn as nn

from src.configs.config import RSCConfig
from src.experiments.registry import (
    ExperimentRegistry,
    ExperimentRun,
)

class TestExperimentRegistry(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ExperimentRegistry(base_dir=self.temp_dir)
        self.cfg = RSCConfig()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_and_list_experiment(self):
        run = self.registry.create_experiment(tag="test_phi2", config=self.cfg, notes="Initial test run")
        self.assertTrue(run.experiment_id.startswith("exp_"))
        self.assertIn("test_phi2", run.experiment_id)
        self.assertEqual(run.tag, "test_phi2")

        # Verify registry listing
        runs_list = self.registry.list_experiments()
        self.assertEqual(len(runs_list), 1)
        self.assertEqual(runs_list[0]["experiment_id"], run.experiment_id)
        self.assertEqual(runs_list[0]["status"], "running")

    def test_log_training_and_diagnostics(self):
        run = self.registry.create_experiment(tag="test_logging", config=self.cfg)
        
        # Log training phase
        run.log_training_phase("phase1", {"best_val": 0.5798, "epochs": 2})
        run.log_training_phase("phase2d", {"best_acc": 0.610, "best_val": 0.6215})
        
        # Log routing diagnostics
        run.log_routing_diagnostics({
            "selection_rates": {"9": 0.874, "15": 0.254, "20": 0.375},
            "k_distribution": {"1": 0.027, "2": 0.043, "3": 0.884, "4": 0.046},
            "win_rate_vs_base": 0.65,
        })

        # Log benchmark results across multi-shot
        run.log_benchmark_result(
            stage_name="Stage C - Full ARS",
            accuracy=47.0,
            n_shot=0,
            dataset="gsm8k",
            tokens_per_sec=25.13,
        )
        run.log_benchmark_result(
            stage_name="Stage C - Full ARS",
            accuracy=53.33,
            n_shot=2,
            dataset="gsm8k",
            tokens_per_sec=24.37,
        )

        run.finalize(status="completed", summary_notes="Beat baseline by +16.3% on 2-shot")

        # Reload from disk via registry
        reloaded = self.registry.get_experiment(run.experiment_id)
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.data["status"], "completed")
        self.assertEqual(reloaded.data["training_logs"]["phase2d"]["best_acc"], 0.610)
        self.assertEqual(reloaded.data["routing_diagnostics"]["win_rate_vs_base"], 0.65)
        self.assertEqual(
            reloaded.data["benchmark_results"]["gsm8k"]["Stage C - Full ARS"]["2_shot"]["accuracy"],
            53.33,
        )

    def test_save_and_load_weights(self):
        run = self.registry.create_experiment(tag="test_weights")
        
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.steer_net_0 = nn.Linear(8, 8)
                self.gate_0 = nn.Linear(8, 1)
        
        model = DummyModel()
        saved_path = run.save_weights(model, filename="final_ars_weights.pt")
        self.assertTrue(os.path.exists(saved_path))
        self.assertEqual(run.data["artifacts"]["weights_path"], saved_path)

        # Perturb weights
        with torch.no_grad():
            model.steer_net_0.weight.fill_(999.0)

        # Restore from experiment
        success = run.load_weights(model, filename="final_ars_weights.pt")
        self.assertTrue(success)
        self.assertFalse((model.steer_net_0.weight == 999.0).all())

if __name__ == "__main__":
    unittest.main()
