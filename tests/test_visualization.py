import os
import shutil
import tempfile
import unittest
import numpy as np
import matplotlib.pyplot as plt

from src.visualization import (
    plot_multishot_scaling,
    plot_synergy_heatmap,
    plot_layer_selection_and_k,
    plot_training_progression,
    generate_experiment_report_charts,
)

class TestARSVisualizations(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        plt.close("all")
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_plot_multishot_scaling(self):
        data = {
            "Baseline": {"0_shot": 37.0, "2_shot": 37.5},
            "SteerNet Only": {"0_shot": 41.0, "2_shot": 43.0},
            "Full ARS": {"0_shot": 47.0, "2_shot": 53.33},
        }
        save_path = os.path.join(self.temp_dir, "scaling.png")
        fig = plot_multishot_scaling(data, dataset_name="GSM8K", save_path=save_path)
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(save_path))
        self.assertGreater(os.path.getsize(save_path), 1000)

    def test_plot_synergy_heatmap(self):
        syn_matrix = np.array([
            [0.0, 0.45, 0.50, -0.20],
            [0.45, 0.0, -0.30, -0.25],
            [0.50, -0.30, 0.0, 0.60],
            [-0.20, -0.25, 0.60, 0.0],
        ])
        labels = ["L9", "L11", "L15", "L20"]
        save_path = os.path.join(self.temp_dir, "synergy.png")
        fig = plot_synergy_heatmap(syn_matrix, layer_labels=labels, save_path=save_path)
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(save_path))
        self.assertGreater(os.path.getsize(save_path), 1000)

    def test_plot_layer_selection_and_k(self):
        sel_rates = {"9": 0.87, "11": 0.27, "15": 0.25, "20": 0.38}
        k_dist = {"1": 0.03, "2": 0.04, "3": 0.88, "4": 0.05}
        save_path = os.path.join(self.temp_dir, "layers.png")
        fig = plot_layer_selection_and_k(sel_rates, k_dist, save_path=save_path)
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(save_path))
        self.assertGreater(os.path.getsize(save_path), 1000)

    def test_plot_training_progression(self):
        logs = {
            "phase1": {"best_val": 0.5798, "best_acc": 0.41},
            "phase2a": {"best_val": 0.5771, "best_acc": 0.56},
            "phase2b": {"best_val": 0.6318, "best_acc": 0.56},
            "phase2c": {"best_val": 0.6291, "best_acc": 0.58},
            "phase2d": {"best_val": 0.6215, "best_acc": 0.61},
        }
        save_path = os.path.join(self.temp_dir, "training.png")
        fig = plot_training_progression(logs, save_path=save_path)
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(save_path))
        self.assertGreater(os.path.getsize(save_path), 1000)

    def test_generate_experiment_report_charts(self):
        record = {
            "experiment_id": "test_exp",
            "benchmark_results": {
                "gsm8k": {
                    "Baseline": {"0_shot": {"accuracy": 37.0}},
                    "Full ARS": {"0_shot": {"accuracy": 47.0}, "2_shot": {"accuracy": 53.33}},
                }
            },
            "routing_diagnostics": {
                "selection_rates": {"9": 0.87, "15": 0.25},
                "k_distribution": {"1": 0.03, "3": 0.88},
            },
            "training_logs": {
                "phase1": {"best_val": 0.5798, "best_acc": 0.41},
                "phase2d": {"best_val": 0.6215, "best_acc": 0.61},
            },
        }
        out_dir = os.path.join(self.temp_dir, "charts_out")
        charts = generate_experiment_report_charts(record, output_dir=out_dir)
        self.assertEqual(len(charts), 3)
        for c in charts:
            self.assertTrue(os.path.exists(c))

if __name__ == "__main__":
    unittest.main()
