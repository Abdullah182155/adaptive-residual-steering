import sys
import os
import unittest
from unittest.mock import patch
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.configs.config import (
    RSCConfig,
    assert_engine_v32_kaggle_profile,
    make_engine_v32_kaggle_config,
    make_engine_v40_kaggle_config,
)
from src.steering.steernet import RSCSteerNet, post_cap_ratio_from_pre_cap_q
from src.gating.gate import RSCUsefulnessGate, token_competitive_allocation
from src.models.wrapper import Phi2WithRSC
from src.routing.router import JointLayerRouter
from src.routing.rloo import (
    sample_k,
    sample_subset_plackett_luce,
    policy_entropy_bonus,
    mutual_information_diversity_loss,
    counterfactual_k_preference_loss,
    capped_route_tariff,
    per_example_answer_nll,
    PairEvidenceTable,
)
from src.training import (
    train_rsc,
    train_phase1_steernet,
    train_router_bootstrap,
    run_staged_phase,
    save_steer_weights,
    load_steer_weights,
    run_pg_loss,
)
from src.training.trainer import (
    _small_run_waiver,
    _optimizer_has_gradients,
    enforce_gpu_memory_budget,
    load_phase2_training_state,
    _phase1_epoch_generation_screen,
    _phase1_confirmation_verdict,
    _phase1_pair_is_safe,
    phase1_steernet_acceptance,
    save_phase2_training_state,
)
from src.training.losses import LRReducer, log_gpu_memory, compute_route_activity_floor_loss
from src.evaluation.deployment import adaptive_vs_fixed_promotion

class TestARSModules(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = RSCConfig()
        self.hidden_dim = 64
        self.batch_size = 2
        self.seq_len = 8

    def test_steernet_shape(self):
        steer = RSCSteerNet(hidden_dim=self.hidden_dim, rank=16, lora_alpha=32.0)
        h = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        active_k = torch.tensor([2, 3])
        out = steer(h, active_k=active_k)
        self.assertEqual(out.shape, h.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_usefulness_gate(self):
        gate = RSCUsefulnessGate(hidden_dim=self.hidden_dim, cfg=self.cfg)
        h = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        delta = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        alpha = gate(h, delta)
        self.assertEqual(alpha.shape, (self.batch_size, self.seq_len, 1))
        self.assertTrue((alpha >= 0.0).all() and (alpha <= 1.0).all())
        # Health telemetry must exclude non-intervened/prompt-shielded tokens.
        intervention_mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.bool)
        intervention_mask[:, 3:] = True
        gate.record_diagnostics(alpha, intervention_mask)
        health = gate.get_saturation_metrics()
        self.assertEqual(health["intervened_tokens"], self.batch_size * (self.seq_len - 3))
        self.assertGreaterEqual(health["token_std"], 0.0)

    def test_gate_prefill_matches_streaming_decode(self):
        """The causal gate must produce identical values in prefill and decode."""
        cfg = RSCConfig(gate_dropout=0.0)
        gate = RSCUsefulnessGate(hidden_dim=self.hidden_dim, cfg=cfg).eval()
        h = torch.randn(1, 6, self.hidden_dim)
        delta = torch.randn_like(h)

        gate.reset_conv_cache()
        prefill = gate(h, delta).detach()
        gate.reset_conv_cache()
        streamed = torch.cat([gate(h[:, i:i + 1], delta[:, i:i + 1]).detach() for i in range(h.shape[1])], dim=1)
        self.assertTrue(torch.allclose(prefill, streamed, atol=1e-6, rtol=1e-5))

    def test_joint_router(self):
        target_layers = [10, 12, 14, 16]
        router = JointLayerRouter(target_layers, hidden_dim=self.hidden_dim, n_layers=20, max_active_layers=4)
        h = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        u, k_logits = router(h, mask)
        self.assertEqual(u.shape, (self.batch_size, len(target_layers)))
        self.assertEqual(k_logits.shape, (self.batch_size, 5))  # K in {0, 1, 2, 3, 4}

    def test_rloo_sampling(self):
        k_logits = torch.randn(self.batch_size, 4)
        k, logp_k = sample_k(k_logits)
        self.assertEqual(k.shape, (self.batch_size,))
        self.assertTrue((k >= 1).all() and (k <= 4).all())

        u = torch.randn(self.batch_size, 4)
        subset_mask, logp_sub = sample_subset_plackett_luce(u, k)
        self.assertEqual(subset_mask.shape, (self.batch_size, 4))
        self.assertEqual(logp_sub.shape, (self.batch_size,))

    def test_phase1_screen_uses_matched_baseline_accuracy(self):
        """Screen-only Phase 1 telemetry must not compare a candidate to 0.0."""
        class ScreenOnlyModel:
            target_layers = [11]
            _routing_override = None

            def set_routing_override(self, mode=None, layers=None):
                self._routing_override = None if mode is None else (mode, layers)

            def set_gate_freeze(self, freeze, value=1.0):
                pass

        cfg = RSCConfig(
            phase1_acceptance_examples=16,
            phase1_acceptance_accuracy_examples=0,
            phase1_generation_screen_examples=8,
            phase1_successive_halving_examples=8,
            phase1_generation_confirmation_examples=8,
            phase1_gate_scale_grid=(1.0,),
            phase1_min_train_nll_reduction=0.05,
            phase1_min_val_nll_reduction=0.05,
        )
        rows = [{"question": "q", "final_num": 1.0}] * 16
        screen = {"accuracy": 0.5, "answer_present": 1.0, "median_tokens": 10.0}
        with patch("src.training.trainer.evaluate_loss", side_effect=[1.0, 1.0, 0.8, 0.8]), patch(
            "src.training.trainer._phase1_zero_shot_generation_probe",
            side_effect=[screen, screen, screen, screen],
        ):
            report = phase1_steernet_acceptance(
                ScreenOnlyModel(), None, rows, rows, cfg, torch.device("cpu")
            )

        self.assertEqual(report["accuracy_comparison"], "disjoint_generation_confirmation")
        self.assertEqual(report["accuracy_comparison_baseline"], 0.5)
        self.assertEqual(report["validation_accuracy_gain"], 0.0)

    def test_phase1_rejects_shortened_completion_candidates(self):
        """A candidate below the CoT retention floor cannot win the screen."""
        class ScreenOnlyModel:
            target_layers = [11]
            _routing_override = None

            def set_routing_override(self, mode=None, layers=None):
                self._routing_override = None if mode is None else (mode, layers)

            def set_gate_freeze(self, freeze, value=1.0):
                pass

        cfg = RSCConfig(
            phase1_acceptance_examples=16,
            phase1_acceptance_accuracy_examples=0,
            phase1_generation_screen_examples=8,
            phase1_successive_halving_examples=8,
            phase1_generation_confirmation_examples=8,
            phase1_gate_scale_grid=(1.0,),
            continue_on_diagnostic_failure=True,
        )
        rows = [{"question": "q", "final_num": 1.0}] * 16
        baseline = {"accuracy": 0.5, "answer_present": 1.0, "median_tokens": 100.0}
        truncated = {"accuracy": 1.0, "answer_present": 1.0, "median_tokens": 84.0}
        with patch("src.training.trainer.evaluate_loss", side_effect=[1.0, 1.0, 0.8, 0.8]), patch(
            "src.training.trainer._phase1_zero_shot_generation_probe",
            side_effect=[baseline, truncated],
        ):
            report = phase1_steernet_acceptance(
                ScreenOnlyModel(), None, rows, rows, cfg, torch.device("cpu")
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["status"], "FAIL_NO_SAFE_PAIR")
        self.assertEqual(report["truncation_rejected_layers"], [11])

    def test_phase1_joint_scale_search_keeps_safe_low_alpha_pair(self):
        """A full-strength failure must not discard the same layer at a safe scale."""
        class ScreenOnlyModel:
            target_layers = [11]
            _routing_override = None

            def set_routing_override(self, mode=None, layers=None):
                self._routing_override = None if mode is None else (mode, layers)

            def set_gate_freeze(self, freeze, value=1.0):
                pass

        cfg = RSCConfig(
            phase1_acceptance_examples=16,
            phase1_generation_screen_examples=8,
            phase1_successive_halving_examples=8,
            phase1_generation_confirmation_examples=8,
            phase1_gate_scale_grid=(0.2, 1.0),
            continue_on_diagnostic_failure=False,
        )
        rows = [{"question": "q", "final_num": 1.0}] * 16
        baseline = {"accuracy": 0.5, "answer_present": 1.0, "median_tokens": 100.0}
        safe_low_alpha = {"accuracy": 0.5, "answer_present": 1.0, "median_tokens": 100.0}
        winning_confirmation = {"accuracy": 0.625, "answer_present": 1.0, "median_tokens": 100.0}
        unsafe_full_alpha = {"accuracy": 1.0, "answer_present": 1.0, "median_tokens": 84.0}
        with patch("src.training.trainer.evaluate_loss", side_effect=[1.0, 1.0, 0.8, 0.8]), patch(
            "src.training.trainer._phase1_zero_shot_generation_probe",
            side_effect=[baseline, safe_low_alpha, unsafe_full_alpha, baseline, winning_confirmation],
        ):
            report = phase1_steernet_acceptance(
                ScreenOnlyModel(), None, rows, rows, cfg, torch.device("cpu")
            )

        self.assertTrue(report["passed"])
        self.assertEqual(report["selected_layer"], 11)
        self.assertEqual(report["fixed_steer_gate_scale"], 0.2)
        self.assertTrue(report["selected_calibration"]["safe"])
        self.assertEqual(report["truncation_rejected_layers"], [])
        self.assertEqual(report["calibration_ranking"][0]["layer"], 11)
        self.assertIn("accuracy > answer_present", report["selection_reason"])

    def test_phase1_confirmation_falls_back_to_next_ranked_candidate(self):
        """A failed top confirmation route must not discard the next safe route."""
        class ScreenOnlyModel:
            target_layers = [11, 24]
            _routing_override = None

            def set_routing_override(self, mode=None, layers=None):
                self._routing_override = None if mode is None else (mode, layers)

            def set_gate_freeze(self, freeze, value=1.0):
                pass

        cfg = RSCConfig(
            phase1_acceptance_examples=16,
            phase1_generation_screen_examples=8,
            phase1_successive_halving_examples=8,
            phase1_generation_confirmation_examples=8,
            phase1_gate_scale_grid=(0.6,),
            phase1_max_confirmation_candidates=2,
            continue_on_diagnostic_failure=False,
        )
        rows = [{"question": "q", "final_num": 1.0}] * 16
        baseline_calibration = {"accuracy": 0.50, "answer_present": 1.0, "median_tokens": 100.0}
        top_calibration = {"accuracy": 0.75, "answer_present": 1.0, "median_tokens": 100.0}
        fallback_calibration = {"accuracy": 0.625, "answer_present": 1.0, "median_tokens": 100.0}
        baseline_confirmation = {"accuracy": 0.75, "answer_present": 1.0, "median_tokens": 100.0}
        top_confirmation = {"accuracy": 0.50, "answer_present": 1.0, "median_tokens": 100.0}
        fallback_confirmation = {"accuracy": 0.875, "answer_present": 1.0, "median_tokens": 100.0}
        with patch(
            "src.training.trainer.evaluate_loss", side_effect=[1.0, 1.0, 0.8, 0.8, 0.8, 0.8]
        ), patch(
            "src.training.trainer._phase1_zero_shot_generation_probe",
            side_effect=[
                baseline_calibration, top_calibration, fallback_calibration,
                baseline_confirmation, top_confirmation, fallback_confirmation,
            ],
        ):
            report = phase1_steernet_acceptance(
                ScreenOnlyModel(), None, rows, rows, cfg, torch.device("cpu")
            )

        self.assertTrue(report["passed"])
        self.assertEqual(report["selected_layer"], 24)
        self.assertEqual(len(report["confirmation_trials"]), 2)
        self.assertFalse(report["confirmation_trials"][0]["passed"])
        self.assertTrue(report["confirmation_trials"][1]["passed"])

    def test_phase1_confirmation_records_one_question_noise_without_promotion(self):
        """A one-item compact-confirmation loss is diagnostic, never a promotion."""
        class ScreenOnlyModel:
            target_layers = [11]
            _routing_override = None

            def set_routing_override(self, mode=None, layers=None):
                self._routing_override = None if mode is None else (mode, layers)

            def set_gate_freeze(self, freeze, value=1.0):
                pass

        cfg = RSCConfig(
            phase1_acceptance_examples=40,
            phase1_generation_screen_examples=20,
            phase1_successive_halving_examples=20,
            phase1_generation_confirmation_examples=20,
            phase1_gate_scale_grid=(0.6,),
            phase1_allow_confirmation_noise_tolerance=True,
            continue_on_diagnostic_failure=False,
        )
        rows = [{"question": "q", "final_num": 1.0}] * 40
        baseline_calibration = {"accuracy": 0.45, "answer_present": 1.0, "median_tokens": 100.0}
        candidate_calibration = {"accuracy": 0.60, "answer_present": 1.0, "median_tokens": 100.0}
        baseline_confirmation = {"accuracy": 0.65, "answer_present": 1.0, "median_tokens": 100.0}
        candidate_confirmation = {"accuracy": 0.60, "answer_present": 1.0, "median_tokens": 100.0}
        with patch(
            "src.training.trainer.evaluate_loss", side_effect=[1.0, 1.0, 0.8, 0.8]
        ), patch(
            "src.training.trainer._phase1_zero_shot_generation_probe",
            side_effect=[baseline_calibration, candidate_calibration, baseline_confirmation, candidate_confirmation],
        ):
            report = phase1_steernet_acceptance(
                ScreenOnlyModel(), None, rows, rows, cfg, torch.device("cpu")
            )

        self.assertFalse(report["passed"])
        self.assertTrue(report["confirmation_trials"][0]["noise_tolerance_used"])
        self.assertAlmostEqual(report["confirmation_trials"][0]["combined_accuracy_gain"], 0.05)

    def test_phase1_confirmation_fails_closed_when_every_candidate_regresses(self):
        """Fallback does not weaken fail-closed promotion when every route fails."""
        class ScreenOnlyModel:
            target_layers = [11, 24]
            _routing_override = None

            def set_routing_override(self, mode=None, layers=None):
                self._routing_override = None if mode is None else (mode, layers)

            def set_gate_freeze(self, freeze, value=1.0):
                pass

        cfg = RSCConfig(
            phase1_acceptance_examples=16,
            phase1_generation_screen_examples=8,
            phase1_successive_halving_examples=8,
            phase1_generation_confirmation_examples=8,
            phase1_gate_scale_grid=(0.6,),
            phase1_max_confirmation_candidates=2,
            continue_on_diagnostic_failure=True,
        )
        rows = [{"question": "q", "final_num": 1.0}] * 16
        calibration_baseline = {"accuracy": 0.50, "answer_present": 1.0, "median_tokens": 100.0}
        first_calibration = {"accuracy": 0.75, "answer_present": 1.0, "median_tokens": 100.0}
        second_calibration = {"accuracy": 0.625, "answer_present": 1.0, "median_tokens": 100.0}
        confirmation_baseline = {"accuracy": 0.75, "answer_present": 1.0, "median_tokens": 100.0}
        regression = {"accuracy": 0.50, "answer_present": 1.0, "median_tokens": 100.0}
        with patch(
            "src.training.trainer.evaluate_loss", side_effect=[1.0, 1.0, 0.8, 0.8, 0.8, 0.8]
        ), patch(
            "src.training.trainer._phase1_zero_shot_generation_probe",
            side_effect=[
                calibration_baseline, first_calibration, second_calibration,
                confirmation_baseline, regression, regression,
            ],
        ):
            report = phase1_steernet_acceptance(
                ScreenOnlyModel(), None, rows, rows, cfg, torch.device("cpu")
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(len(report["confirmation_trials"]), 2)
        self.assertTrue(all("confirmation_accuracy" in trial["failure_reasons"] for trial in report["confirmation_trials"]))

    def test_epoch_screen_requires_matched_generation_baseline(self):
        """A CoT-safe route that loses to baseline cannot select a checkpoint."""
        class ProbeModel:
            target_layers = [11]
            _router_warmup = True

            def set_router_warmup(self, enabled):
                self._router_warmup = bool(enabled)

        cfg = RSCConfig(
            phase1_epoch_generation_screen_examples=6,
            phase1_epoch_gate_scale_grid=(0.2,),
        )
        baseline = {"accuracy": 0.5, "answer_present": 1.0, "median_tokens": 100.0, "pre_cap_norm": {}}
        regressing = {"accuracy": 1 / 3, "answer_present": 1.0, "median_tokens": 100.0, "pre_cap_norm": {"p95": 0.9}}
        model = ProbeModel()
        with patch("src.training.trainer._phase1_probe_with_norm", side_effect=[baseline, regressing]):
            result = _phase1_epoch_generation_screen(
                model, object(), [{"question": "q"}] * 6, cfg, torch.device("cpu")
            )

        self.assertEqual(result["status"], "NO_NONREGRESSING_SAFE_PAIR")
        self.assertIsNone(result["best"])
        self.assertEqual(len(result["safe_pairs"]), 1)
        self.assertTrue(model._router_warmup)

    def test_phase1_pair_uses_postcap_envelope_and_rejects_invalid_displacement(self):
        """Raw saturation is diagnostic; impossible realized displacement is rejected."""
        cfg = RSCConfig()
        safe, ratio, q95 = _phase1_pair_is_safe(
            {"median_tokens": 100.0, "pre_cap_norm": {"p95": 1.16}}, 100.0, cfg, gate_scale=0.70,
        )
        self.assertTrue(safe)
        self.assertEqual(ratio, 1.0)
        self.assertEqual(q95, 1.16)
        runaway, _, _ = _phase1_pair_is_safe(
            {"median_tokens": 100.0, "pre_cap_norm": {"p95": 1.51, "post_cap_p95": 1.10}},
            100.0, cfg, gate_scale=0.95,
        )
        self.assertFalse(runaway)

    def test_phase1_calibration_default_uses_twelve_rows(self):
        """The joint search has enough calibration rows to reduce tie noise."""
        self.assertEqual(RSCConfig().phase1_generation_screen_examples, 12)

    def test_router_mutual_information_diversity(self):
        # Input-dependent, confident decisions must score better than one
        # globally collapsed route under the batch diversity objective.
        collapsed_k = torch.tensor([[8.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
        collapsed_l = torch.tensor([[8.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
        varied_k = torch.tensor([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
        varied_l = torch.tensor([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
        collapsed_losses = mutual_information_diversity_loss(collapsed_k, collapsed_l)
        varied_losses = mutual_information_diversity_loss(varied_k, varied_l)
        self.assertLess(varied_losses[0], collapsed_losses[0])
        self.assertLess(varied_losses[1], collapsed_losses[1])

    def test_counterfactual_k_preference_trains_unsteered_when_it_wins(self):
        # K=0 is not prescribed: it receives the largest target probability
        # only because its observed counterfactual reward is largest.
        logits = torch.zeros(2, 5, requires_grad=True)
        k_actions = torch.tensor([[1, 2, 4], [1, 3, 4]])
        loss = counterfactual_k_preference_loss(
            logits, k_actions,
            reward_0=torch.tensor([0.0, 0.0]),
            action_rewards=torch.tensor([[-1.0, -1.0, -2.0], [-1.0, -1.0, -2.0]]),
        )
        loss.backward()
        self.assertLess(logits.grad[:, 0].mean().item(), 0.0)

    def test_answer_tail_weight_penalizes_final_answer_errors(self):
        """A copied rationale cannot hide an incorrect completion tail."""
        logits = torch.full((1, 5, 6), -6.0)
        labels = torch.tensor([[-100, 1, 2, 3, 4]])
        logits[0, 0, 1] = 6.0  # first target correct
        logits[0, 1, 2] = 6.0  # second target correct
        logits[0, 2, 0] = 6.0  # final-tail targets deliberately wrong
        logits[0, 3, 0] = 6.0
        ordinary = per_example_answer_nll(logits, labels)
        answer_aware = per_example_answer_nll(logits, labels, answer_tail_tokens=2, answer_tail_weight=4.0)
        self.assertGreater(answer_aware.item(), ordinary.item())

    def test_config_attributes(self):
        cfg = RSCConfig()
        required_attrs = [
            "steer_finetune_lr", "steer_finetune_epochs", "steer_finetune_save_path",
            "gate_only_lr", "gate_only_epochs", "gate_only_save_path",
            "joint_finetune_epochs", "joint_finetune_lr_scale", "joint_finetune_router_lr_scale",
            "gate_retune_epochs", "gate_retune_lr", "phase2_save_path", "lora_save_path",
            "prompt_shielding", "lambda_synergy", "router_balance_weight", "lambda_cost",
            "beta_entropy_k", "beta_entropy_subset", "min_reliable_validation_examples",
            "allow_inconclusive_small_runs", "router_preflight_examples",
            "router_fail_fast_on_collapse", "lambda_router_k_preference",
            "continue_on_diagnostic_failure", "answer_tail_tokens",
            "answer_tail_weight", "router_synergy_limit", "lambda_pg",
        ]
        for attr in required_attrs:
            self.assertTrue(hasattr(cfg, attr), f"Missing config attribute: {attr}")
        self.assertEqual(cfg.lambda_pg, 0.0)

    def test_tiny_holdout_marks_audit_inconclusive_not_passed(self):
        cfg = RSCConfig(min_reliable_validation_examples=16, allow_inconclusive_small_runs=True)
        self.assertTrue(_small_run_waiver(4, cfg))
        self.assertFalse(_small_run_waiver(16, cfg))
        cfg.allow_inconclusive_small_runs = False
        self.assertFalse(_small_run_waiver(4, cfg))

    def test_optimizer_gradient_guard_handles_a_legal_k_zero_window(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.assertFalse(_optimizer_has_gradients([parameter]))
        (parameter * 2.0).backward()
        self.assertTrue(_optimizer_has_gradients([parameter]))

    def test_routing_override_modes(self):
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Config", (), {"num_hidden_layers": 10, "hidden_size": 64})()
                self.model = type("Model", (), {"layers": nn.ModuleList([nn.Linear(64, 64) for _ in range(10)])})()
            def forward(self, *args, **kwargs):
                return None
        from src.models.wrapper import Phi2WithRSC
        dummy = DummyModel()
        cfg = RSCConfig()
        wrapper = Phi2WithRSC(dummy, cfg, device=torch.device("cpu"))
        wrapper.set_routing_override("only", wrapper.target_layers)
        self.assertEqual(wrapper._routing_override[0], "only")
        self.assertEqual(wrapper._routing_override[1], set(wrapper.target_layers))
        wrapper.set_routing_override("none")
        self.assertEqual(wrapper._routing_override[0], "none")
        wrapper.set_routing_override(None)
        self.assertIsNone(wrapper._routing_override)

    def test_argmax_router_keeps_k_zero_and_is_deterministic(self):
        class DummyLayer(nn.Module):
            def forward(self, h, *args, **kwargs):
                return h

        class DummyModel(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.config = type("Config", (), {"num_hidden_layers": 10, "hidden_size": hidden_dim})()
                self.embed = nn.Embedding(50, hidden_dim)
                self.model = type("Model", (), {"layers": nn.ModuleList([DummyLayer() for _ in range(10)])})()
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                h = self.embed(input_ids)
                for layer in self.model.layers:
                    h = layer(h)
                return type("Output", (), {"logits": h})()

        wrapper = Phi2WithRSC(DummyModel(self.hidden_dim), RSCConfig(), device=torch.device("cpu")).eval()
        with torch.no_grad():
            wrapper.router.k_head.weight.zero_()
            wrapper.router.k_head.bias.fill_(-10.0)
            wrapper.router.k_head.bias[0] = 10.0
        ids = torch.tensor([[1, 2, 3, 4]])
        baseline = wrapper.base_model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        out_a = wrapper(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        route_a = wrapper._precomputed_routes.clone()
        out_b = wrapper(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        self.assertEqual(int(wrapper._desired_k.item()), 0)
        self.assertEqual(int(route_a.sum().item()), 0)
        self.assertTrue(torch.equal(route_a, wrapper._precomputed_routes))
        self.assertTrue(torch.equal(out_a, out_b))
        self.assertTrue(torch.equal(out_a, baseline))

    def test_fixed_route_controls_do_not_pollute_router_telemetry(self):
        class DummyLayer(nn.Module):
            def forward(self, h, *args, **kwargs):
                return h

        class DummyModel(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.config = type("Config", (), {"num_hidden_layers": 10, "hidden_size": hidden_dim})()
                self.embed = nn.Embedding(50, hidden_dim)
                self.model = type("Model", (), {"layers": nn.ModuleList([DummyLayer() for _ in range(10)])})()
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                h = self.embed(input_ids)
                for layer in self.model.layers:
                    h = layer(h)
                return type("Output", (), {"logits": h})()

        wrapper = Phi2WithRSC(DummyModel(self.hidden_dim), RSCConfig(), device=torch.device("cpu")).eval()
        wrapper.set_routing_override("balanced")
        wrapper.set_gate_freeze(True, 1.0)
        ids = torch.tensor([[1, 2, 3, 4]])
        _ = wrapper(input_ids=ids, attention_mask=torch.ones_like(ids))
        diag = wrapper.routing_diagnostics()
        self.assertTrue(all(row["seen_forwards"] == 0 for row in diag.values()))
        self.assertTrue(all(row["selected_forwards"] == 0 for row in diag.values()))

    def test_explicit_routing_override_beats_warmup(self):
        """Audit controls must not be silently replaced by Phase-1 warmup routes."""
        class DummyLayer(nn.Module):
            def forward(self, h, *args, **kwargs):
                return h

        class DummyModel(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.config = type("Config", (), {"num_hidden_layers": 10, "hidden_size": hidden_dim})()
                self.embed = nn.Embedding(50, hidden_dim)
                self.model = type("Model", (), {"layers": nn.ModuleList([DummyLayer() for _ in range(10)])})()

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                h = self.embed(input_ids)
                for layer in self.model.layers:
                    h = layer(h)
                return type("Output", (), {"logits": h})()

        wrapper = Phi2WithRSC(DummyModel(self.hidden_dim), RSCConfig(), device=torch.device("cpu")).eval()
        wrapper.set_router_warmup(True)
        wrapper.set_gate_freeze(True, 1.0)
        ids = torch.tensor([[1, 2, 3, 4]])
        attn = torch.ones_like(ids)
        baseline = wrapper.base_model(input_ids=ids, attention_mask=attn).logits

        wrapper.set_routing_override("none")
        no_route = wrapper(input_ids=ids, attention_mask=attn).logits
        self.assertEqual(int(wrapper._precomputed_routes.sum().item()), 0)
        self.assertTrue(torch.equal(no_route, baseline))

        selected_layer = wrapper.target_layers[0]
        with torch.no_grad():
            wrapper.steer_nets[str(selected_layer)].up.weight.fill_(0.1)
        wrapper.set_routing_override("only", [selected_layer])
        routed = wrapper(input_ids=ids, attention_mask=attn).logits
        selected = wrapper._precomputed_routes[0]
        self.assertEqual(int(selected.sum().item()), 1)
        self.assertTrue(bool(selected[wrapper.target_layers.index(selected_layer)].item()))
        self.assertFalse(torch.equal(routed, baseline))

    def test_cooperative_synergy(self):
        target_layers = [9, 11, 13, 15]
        router = JointLayerRouter(target_layers, hidden_dim=self.hidden_dim, n_layers=20, max_active_layers=4)
        h = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        u, k_logits = router(h, mask)
        self.assertIsNotNone(router._last_synergy_matrix)
        self.assertEqual(router._last_synergy_matrix.shape, (self.batch_size, 4, 4))
        # Diagonal must be 0 (no self-synergy)
        diag = torch.diagonal(router._last_synergy_matrix, dim1=1, dim2=2)
        self.assertTrue((diag == 0.0).all())
        # Test subset sampling with synergy
        k = torch.tensor([2, 3])
        subset, logp = sample_subset_plackett_luce(u, k, synergy_matrix=router._last_synergy_matrix)
        self.assertEqual(subset.shape, (self.batch_size, 4))
        self.assertEqual(subset.sum(dim=-1).tolist(), [2, 3])

    def test_prompt_shielding(self):
        class DummyLayer(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.linear = nn.Identity()
            def forward(self, h, *args, **kwargs):
                return h
        class DummyModel(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.config = type("Config", (), {"num_hidden_layers": 10, "hidden_size": hidden_dim})()
                self.embed = nn.Embedding(50, hidden_dim)
                self.model = type("Model", (), {"layers": nn.ModuleList([DummyLayer(hidden_dim) for _ in range(10)])})()
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                h = self.embed(input_ids)
                for layer in self.model.layers:
                    h = layer(h)
                return type("Output", (), {"logits": h})()

        dummy = DummyModel(self.hidden_dim)
        cfg = RSCConfig()
        cfg.router_pairwise_probe_probability = 1.0
        wrapper = Phi2WithRSC(dummy, cfg, device=torch.device("cpu"))
        wrapper.eval()

        B, T = 1, 10
        # Pretend first 6 tokens are demonstrations, last 4 are question
        shield_mask = torch.zeros(B, T, dtype=torch.bool)
        shield_mask[:, :6] = True  # Shield first 6 tokens
        wrapper.set_prompt_shield_mask(shield_mask)

        # Force a steering layer active
        wrapper.set_routing_override("only", wrapper.target_layers)
        wrapper.set_gate_freeze(True, 1.0)
        # Ensure non-zero steering intervention (since identity init starts at delta=0)
        with torch.no_grad():
            for sn in wrapper.steer_nets.values():
                sn.up.weight.fill_(0.1)
                sn.mag_head.bias.fill_(0.0)

        # Hook to capture hidden states at the last steered layer
        steered_layer = wrapper.target_layers[-1]
        captured_h = {}
        def cap_hook(mod, inp, out):
            captured_h["out"] = out[0] if isinstance(out, tuple) else out
        h_handle = dummy.model.layers[steered_layer].register_forward_hook(cap_hook)

        inp_ids = torch.arange(T, dtype=torch.long).unsqueeze(0)
        attn = torch.ones(B, T, dtype=torch.long)
        _ = wrapper(input_ids=inp_ids, attention_mask=attn)
        h_handle.remove()

        out_h = captured_h["out"]
        # The first 6 tokens (shielded) must be unmodified baseline input
        baseline_h = dummy.embed(inp_ids)
        self.assertTrue(torch.allclose(out_h[:, :6, :], baseline_h[:, :6, :], atol=1e-5))
        # The last 4 tokens (unshielded) must have received residual steering
        self.assertFalse(torch.allclose(out_h[:, 6:, :], baseline_h[:, 6:, :], atol=1e-5))

    def test_gate_regularization_gradient_flow(self):
        gate = RSCUsefulnessGate(hidden_dim=self.hidden_dim, cfg=self.cfg)
        torch.nn.init.normal_(gate.fc2.weight, std=0.02)
        h = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, requires_grad=True)
        # Extreme magnitude delta to test anti-saturation
        delta = 100.0 * torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        alpha = gate(h, delta)
        self.assertEqual(alpha.shape, (self.batch_size, self.seq_len, 1))
        # Check output is strictly in (0, 1) and not NaN
        self.assertTrue(torch.all(alpha > 0.0) and torch.all(alpha < 1.0))
        self.assertFalse(torch.isnan(alpha).any())

        # Check backward gradient flow through aux alpha
        loss = gate._alpha_for_aux.sum()
        loss.backward()
        for name, param in gate.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Param {name} gradient is None")
                self.assertTrue(torch.isfinite(param.grad).all(), f"Param {name} gradient contains inf or NaN")
                self.assertFalse((param.grad == 0).all(), f"Param {name} gradient completely vanished to 0")

    def test_checkpoint_evaluation_both_router_modes(self):
        from src.training.trainer import _evaluate_and_maybe_checkpoint
        class DummyLayer(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, h, *args, **kwargs):
                return h
        class DummyModel(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.config = type("Config", (), {"num_hidden_layers": 10, "hidden_size": hidden_dim})()
                self.embed = nn.Embedding(50, hidden_dim)
                self.model = type("Model", (), {"layers": nn.ModuleList([DummyLayer() for _ in range(10)])})()
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                h = self.embed(input_ids)
                for layer in self.model.layers:
                    h = layer(h)
                return type("Output", (), {"logits": h, "loss": torch.tensor(0.5)})()
            def generate(self, input_ids=None, **kwargs):
                return input_ids

        dummy = DummyModel(self.hidden_dim)
        cfg = RSCConfig()
        cfg.accuracy_eval_n = 2
        wrapper = Phi2WithRSC(dummy, cfg, device=torch.device("cpu"))

        eval_data = [
            {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1]), "labels": torch.tensor([-100, 2, 3])},
            {"input_ids": torch.tensor([4, 5, 6]), "attention_mask": torch.tensor([1, 1, 1]), "labels": torch.tensor([-100, 5, 6])},
        ]
        tokenizer = type("Tokenizer", (), {
            "pad_token_id": 0, "eos_token_id": 1,
            "decode": lambda *a, **k: "#### 42",
            "encode": lambda *a, **k: [1, 2],
        })()

        # 1. train_router=False (must NOT raise UnboundLocalError)
        best_acc, best_val, state, p_ctr, improved, cur_val, cur_acc = _evaluate_and_maybe_checkpoint(
            wrapper, tokenizer, eval_data, cfg, torch.device("cpu"), "./test_ckpt.pt",
            best_acc=0.0, best_val=float("inf"), patience_ctr=0, train_router=False, label="test_no_router",
        )
        self.assertIsNotNone(cur_val)

        # 2. train_router=True (must include active layer metrics without error)
        best_acc, best_val, state, p_ctr, improved, cur_val, cur_acc = _evaluate_and_maybe_checkpoint(
            wrapper, tokenizer, eval_data, cfg, torch.device("cpu"), "./test_ckpt.pt",
            best_acc=0.0, best_val=float("inf"), patience_ctr=0, train_router=True, label="test_with_router",
        )
        self.assertIsNotNone(cur_val)
        if os.path.exists("./test_ckpt.pt"):
            try:
                os.remove("./test_ckpt.pt")
            except OSError:
                pass
        if os.path.exists("./test_ckpt.pt.manifest.json"):
            try:
                os.remove("./test_ckpt.pt.manifest.json")
            except OSError:
                pass

    def test_normalized_synergy_and_cooperation_analysis(self):
        """Ensure synergy matrix is strictly bounded in [-1, 1] and cooperation analysis works."""
        target_layers = [9, 11, 13, 15]
        router = JointLayerRouter(target_layers, hidden_dim=self.hidden_dim, n_layers=20, max_active_layers=4)
        h = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        u, k_logits = router(h, mask)
        syn_mat = router.get_layer_synergy_matrix()
        self.assertIsNotNone(syn_mat)
        self.assertTrue((syn_mat >= -router.synergy_limit).all() and (syn_mat <= router.synergy_limit).all())
        # Diagonal must be 0
        diag = torch.diagonal(syn_mat, dim1=-2, dim2=-1)
        self.assertTrue((diag == 0.0).all())

        analysis = router.get_cooperation_analysis()
        self.assertIn("top_synergy_pairs", analysis)
        self.assertIn("top_antagonistic_pairs", analysis)
        self.assertIn("mean_synergy", analysis)
        self.assertIsInstance(analysis["top_synergy_pairs"], list)

    def test_anchor_grounded_rloo_loss(self):
        """Ensure exhaustive-K RLOO loss runs with baseline anchor and synergy alignment."""
        class DummyLayer(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
            def forward(self, h, *args, **kwargs):
                return h
        class DummyModel(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.config = type("Config", (), {"num_hidden_layers": 10, "hidden_size": hidden_dim})()
                self.embed = nn.Embedding(50, hidden_dim)
                self.model = type("Model", (), {"layers": nn.ModuleList([DummyLayer(hidden_dim) for _ in range(10)])})()
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                h = self.embed(input_ids)
                for layer in self.model.layers:
                    h = layer(h)
                return type("Output", (), {"logits": h, "loss": torch.tensor(0.5)})()

        dummy = DummyModel(self.hidden_dim)
        cfg = RSCConfig()
        wrapper = Phi2WithRSC(dummy, cfg, device=torch.device("cpu"))

        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        mask = torch.ones_like(ids)
        lbls = torch.tensor([[-100, 2, 3, 4], [-100, 6, 7, 8]])
        prompt_mask = mask.bool() & lbls.eq(-100)

        # Trigger forward to cache pooled features
        _ = wrapper(input_ids=ids, attention_mask=mask)
        from src.routing.rloo import compute_router_rloo_loss
        loss, diag = compute_router_rloo_loss(wrapper, ids, mask, lbls, cfg, prompt_mask=prompt_mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("win_rate_vs_base", diag)
        self.assertIn("synergy_align_loss", diag)
        self.assertIn("mean_reward_delta", diag)
        self.assertEqual(diag["counterfactual_k_actions"], [1, 2, 3, 4])
        # Pairwise probes are auxiliary counterfactuals: they must not leak
        # their last fixed route into the next ordinary training forward.
        self.assertIsNone(wrapper._routing_override)
        self.assertTrue(wrapper.training)

    def test_rloo_refreshes_router_cache_for_current_example(self):
        """Bootstrap RLOO must not use the pooled state of a preceding sample."""
        class DummyLayer(nn.Module):
            def forward(self, h, *args, **kwargs):
                return h

        class DummyModel(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.config = type("Config", (), {"num_hidden_layers": 10, "hidden_size": hidden_dim})()
                self.embed = nn.Embedding(50, hidden_dim)
                self.model = type("Model", (), {"layers": nn.ModuleList([DummyLayer() for _ in range(10)])})()
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                h = self.embed(input_ids)
                for layer in self.model.layers:
                    h = layer(h)
                return type("Output", (), {"logits": h})()

        wrapper = Phi2WithRSC(DummyModel(self.hidden_dim), RSCConfig(), device=torch.device("cpu"))
        stale_ids = torch.tensor([[1, 2, 3, 4]])
        current_ids = torch.tensor([[5, 6, 7, 8]])
        mask = torch.ones_like(current_ids)
        labels = torch.tensor([[-100, 6, 7, 8]])
        _ = wrapper(input_ids=stale_ids, attention_mask=torch.ones_like(stale_ids))
        stale_pooled = wrapper.router._last_pooled.clone()
        from src.routing.rloo import compute_router_rloo_loss
        loss, _ = compute_router_rloo_loss(wrapper, current_ids, mask, labels, RSCConfig())
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.allclose(wrapper.router._last_pooled, stale_pooled, atol=1e-6))

    def test_joint_backward_no_graph_revisit(self):
        """Ensure joint training (steer + gate) executes cleanly without graph reuse error."""
        from src.training.losses import compute_l1_gate, compute_antisat_gate
        gate = RSCUsefulnessGate(hidden_dim=self.hidden_dim, cfg=self.cfg)
        steer = RSCSteerNet(hidden_dim=self.hidden_dim, rank=16, lora_alpha=32.0)

        h = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, requires_grad=True)
        delta = steer(h, active_k=torch.tensor([1, 1]))
        alpha = gate(h, delta)
        h_out = h + alpha.detach() * delta
        task_loss = h_out.sum()

        class ModelMock:
            gates = {"0": gate}

        aux_loss = compute_l1_gate(ModelMock()) + compute_antisat_gate(ModelMock())
        micro_loss = task_loss + aux_loss
        micro_loss.backward()

        self.assertIsNotNone(steer.down1.weight.grad)
        self.assertTrue(torch.isfinite(steer.down1.weight.grad).all())
        self.assertIsNotNone(gate.fc1.weight.grad)
        self.assertTrue(torch.isfinite(gate.fc1.weight.grad).all())

if __name__ == "__main__":
    unittest.main()




class TestAnswerExtractor(unittest.TestCase):
    """ISSUE-13: canonical extract_numeric_answer word-boundary tests."""

    def setUp(self):
        from src.evaluation.answers import extract_numeric_answer as _ex
        self.ex = _ex

    def test_gsm8k_hash_marker(self):
        self.assertEqual(self.ex("Step 1: x\n#### 18"), 18)

    def test_comma_number(self):
        self.assertEqual(self.ex("#### 1,234."), 1234)

    def test_dollar_prefix(self):
        self.assertEqual(self.ex("Final answer: $6"), 6)

    def test_fallback_last_equals(self):
        self.assertEqual(self.ex("she gave this 5 away, so 7 - 2 = 5 apples"), 5)
        self.assertEqual(self.ex("his 12 books and 3 + 4 = 7"), 7)

    def test_no_number_returns_none(self):
        self.assertIsNone(self.ex("no numbers here"))

    def test_is_substring_bug_fixed(self):
        # No explicit marker and no '=' in "this 5" → None (old bug returned 5)
        result = self.ex("she gave this 5 away without any equals")
        self.assertIsNone(result)

    def test_boxed_marker(self):
        self.assertEqual(self.ex(r"answer: \boxed{42}"), 42)

    def test_the_answer_is_marker(self):
        self.assertEqual(self.ex("The answer is: 100"), 100)

    def test_negative_number(self):
        self.assertEqual(self.ex("#### -5"), -5)


class TestStopAndExtraction(unittest.TestCase):
    """A-1 run-analysis: cut_continuation, should_stop_on_repeat, extraction priority."""

    def setUp(self):
        from src.evaluation.answers import (
            cut_continuation, should_stop_on_repeat, extract_numeric_answer,
        )
        self.cut = cut_continuation
        self.stop = should_stop_on_repeat
        self.ex = extract_numeric_answer

    def test_cut_continuation_removes_follow_up(self):
        text = "Final answer: 42\nQuestion: There are 10 apples..."
        self.assertEqual(self.cut(text), "Final answer: 42")

    def test_cut_continuation_no_follow_up_unchanged(self):
        text = "Step 1: x = 5\nFinal answer: 5"
        self.assertEqual(self.cut(text), text)

    def test_should_stop_on_repeat_detects_loop(self):
        repeated = "She has 10 apples left after giving some away.\n" * 4
        self.assertTrue(self.stop(repeated))

    def test_should_stop_on_repeat_no_loop(self):
        text = "Step 1: x = 5\nStep 2: y = 10\nStep 3: z = 15\n"
        self.assertFalse(self.stop(text))

    def test_extract_indras_final_answer_line(self):
        # 'Final answer: Indras has 6 letters and her sister has 13 letters'
        # should return 13 (last number in the Final answer line)
        result = self.ex("Final answer: Indras has 6 letters and her sister has 13 letters")
        self.assertEqual(result, 13)

    def test_extract_no_hallucinated_tail(self):
        # The #### in the follow-up question should NOT override the real answer
        text = "Final answer: 10\nQuestion: Now there are 9. #### 9"
        self.assertEqual(self.ex(text), 10)  # cut_continuation removes the tail

    def test_extract_equals_beats_therefore(self):
        # '=' scan should win over 'therefore' heuristic
        text = "Therefore, she has many apples. The total is = 7"
        self.assertEqual(self.ex(text), 7)


class TestPairedCI(unittest.TestCase):
    """A-2: paired_delta_ci correctness."""

    def setUp(self):
        from src.evaluation.deployment import paired_delta_ci
        self.ci = paired_delta_ci

    def test_zero_diff_ci_contains_zero(self):
        flags = [1, 0, 1, 0, 1, 1, 0, 1, 0, 0] * 10
        d, lo, hi = self.ci(flags, flags, n_boot=500)
        self.assertAlmostEqual(d, 0.0)
        self.assertLessEqual(lo, 0.0)
        self.assertGreaterEqual(hi, 0.0)

    def test_large_diff_ci_positive(self):
        a = [0] * 100
        b = [1] * 100
        d, lo, hi = self.ci(a, b, n_boot=500)
        self.assertAlmostEqual(d, 1.0)
        self.assertGreater(lo, 0.5)

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            self.ci([1, 0], [1, 0, 1])

    def test_clustered_views_preserve_point_estimate(self):
        # Two shots of the same questions are correlated.  The clustered path
        # must retain the pooled point estimate while resampling questions.
        baseline = [[0, 1], [1, 0], [0, 0], [1, 1]]
        ars = [[1, 1], [1, 0], [1, 0], [1, 1]]
        d, lo, hi = self.ci(baseline, ars, n_boot=500, seed=7)
        self.assertAlmostEqual(d, 0.25)
        self.assertLessEqual(lo, d)
        self.assertGreaterEqual(hi, d)

    def test_clustered_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.ci([[1, 0]], [[1], [0]])


class TestRun8FailClosedDefaults(unittest.TestCase):
    """Run 8 must not silently continue after an invalid routing diagnostic under strict policy."""

    def test_decision_defaults_fail_closed(self):
        cfg_strict = RSCConfig(failure_policy="strict")
        self.assertFalse(cfg_strict.continue_on_diagnostic_failure)
        self.assertTrue(cfg_strict.skip_adaptive_phases_when_phase1_fails)
        self.assertTrue(cfg_strict.skip_adaptive_phases_when_router_fails)
        self.assertTrue(cfg_strict.router_fail_fast_on_collapse)
        self.assertEqual(cfg_strict.deployment_min_examples, 120)

        # M0: warn policy coerces legacy fail-closed flags so runs never stall
        cfg_warn = RSCConfig(failure_policy="warn")
        self.assertTrue(cfg_warn.continue_on_diagnostic_failure)
        self.assertFalse(cfg_warn.skip_adaptive_phases_when_router_fails)
        self.assertFalse(cfg_warn.router_fail_fast_on_collapse)


class TestSteerNetIdentityInit(unittest.TestCase):
    """ISSUE-5: SteerNet must produce near-zero delta at initialization."""

    def test_identity_at_init(self):
        """up.weight = zeros means delta ~ 0 for all inputs at init."""
        steer = RSCSteerNet(hidden_dim=64, rank=8, lora_alpha=16.0, magnitude_mode='direct')
        steer.eval()
        with torch.no_grad():
            h = torch.randn(2, 10, 64)
            delta = steer(h, active_k=torch.tensor([1, 2]))
        self.assertLess(
            delta.abs().max().item(), 1e-4,
            f"Expected near-zero delta at init, got max={delta.abs().max().item():.6f}"
        )

    def test_phase1_route_width_config_is_independent_k1(self):
        """Phase 1 must establish independent capacity before K>=2 cooperation."""
        cfg = self.cfg if hasattr(self, 'cfg') else RSCConfig()
        self.assertTrue(hasattr(cfg, 'router_warmup_random_subsets'))
        self.assertTrue(hasattr(cfg, 'router_warmup_max_k'))
        self.assertEqual(cfg.router_warmup_max_k, 1)
        self.assertFalse(cfg.router_warmup_random_subsets)


class TestGateStreamingTrainedConv(unittest.TestCase):
    """ISSUE-17: Gate prefill and streaming decode must match even after conv weights are trained/perturbed."""

    def test_gate_streaming_matches_prefill_with_trained_conv(self):
        import torch
        from src.configs.config import RSCConfig
        from src.gating.gate import RSCUsefulnessGate
        gate = RSCUsefulnessGate(hidden_dim=64, cfg=RSCConfig(gate_dropout=0.0)).eval()
        with torch.no_grad():
            gate.causal_conv.weight.normal_(0, 0.5)
            gate.causal_conv.bias.normal_(0, 0.1)
        h = torch.randn(1, 6, 64)
        delta = torch.randn_like(h)
        gate.reset_conv_cache()
        prefill = gate(h, delta).detach()
        gate.reset_conv_cache()
        streamed = torch.cat([gate(h[:, i:i + 1], delta[:, i:i + 1]).detach() for i in range(h.shape[1])], dim=1)
        self.assertTrue(torch.allclose(prefill, streamed, atol=1e-5, rtol=1e-4))


class TestGateDegeneracyAndSaturation(unittest.TestCase):
    """R2/R3/R5: Tests for smooth saturation envelope, gate degeneracy detection, and trim_batch."""

    def test_gate_is_degenerate_detection(self):
        from src.training.trainer import gate_is_degenerate
        # Degenerate: alpha saturated with negligible variance
        diag_degen = {
            "gate": {
                "18": {"intervened_tokens": 100, "token_std": 0.0001, "high_saturation_pct": 99.8},
                "20": {"intervened_tokens": 100, "token_std": 0.0002, "high_saturation_pct": 100.0},
            }
        }
        self.assertTrue(gate_is_degenerate(diag_degen))

        # Healthy: lower saturation and real variance
        diag_healthy = {
            "gate": {
                "18": {"intervened_tokens": 100, "token_std": 0.05, "high_saturation_pct": 40.0},
                "20": {"intervened_tokens": 100, "token_std": 0.08, "high_saturation_pct": 35.0},
            }
        }
        self.assertFalse(gate_is_degenerate(diag_healthy))

    def test_steernet_smooth_saturation_envelope(self):
        from src.steering.steernet import RSCSteerNet
        steer = RSCSteerNet(hidden_dim=64, rank=8, lora_alpha=16.0, max_relative_norm=0.04, magnitude_mode="direct")
        steer.eval()
        # Set nonzero weights so delta is large
        with torch.no_grad():
            steer.up.weight.normal_(0, 1.0)
            steer.mag_head.bias.fill_(5.0)  # large magnitude
            h = torch.randn(2, 10, 64)
            delta = steer(h, active_k=torch.tensor([1, 1]))
            h_norm = torch.norm(h.float(), dim=-1, keepdim=True)
            delta_norm = torch.norm(delta.float(), dim=-1, keepdim=True)
            ratio = delta_norm / (h_norm + 1e-7)
            # Smooth saturation must keep ratio strictly bounded below max_relative_norm (0.04)
            self.assertLess(ratio.max().item(), 0.0401)
            # Telemetry _clip_stats should record clipped tokens
            self.assertGreater(steer._clip_stats, 0.0)

    def test_trim_batch(self):
        from src.training.trainer import _trim_batch
        # Batch of 2 items, max length 10, real length 4
        ids = torch.zeros((2, 10), dtype=torch.long)
        mask = torch.zeros((2, 10), dtype=torch.long)
        lbls = torch.full((2, 10), -100, dtype=torch.long)
        shield = torch.zeros((2, 10), dtype=torch.long)
        # Item 0 has length 3, Item 1 has length 4
        mask[0, :3] = 1
        mask[1, :4] = 1
        t_ids, t_mask, t_lbls, t_shield = _trim_batch(ids, mask, lbls, shield)
        self.assertEqual(t_ids.shape[1], 4)
        self.assertEqual(t_mask.shape[1], 4)
        self.assertEqual(t_lbls.shape[1], 4)
        self.assertEqual(t_shield.shape[1], 4)

    def test_deployment_confirmation_bounds_small_eval(self):
        # Regression test for IndexError when len(eval_data) < 16 (e.g. 4)
        eval_data = [
            {"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1])}
            for _ in range(4)
        ]
        deployment_n = min(len(eval_data), 300)
        start_idx = max(0, len(eval_data) - deployment_n)
        confirm_rows = [eval_data[i] for i in range(start_idx, len(eval_data))]
        self.assertEqual(len(confirm_rows), 4)

        # Empty eval_data
        empty_data = []
        deployment_n_empty = min(len(empty_data), 300)
        start_idx_empty = max(0, len(empty_data) - deployment_n_empty)
        confirm_rows_empty = [empty_data[i] for i in range(start_idx_empty, len(empty_data))]
        self.assertEqual(len(confirm_rows_empty), 0)


class TestP0HardeningAndDiagnostics(unittest.TestCase):
    """P0 unit tests for Run 3 forensic fixes."""

    def test_lenient_concluding_statement_extraction(self):
        from src.evaluation.answers import (
            extract_numeric_answer,
            extract_numeric_answer_lenient,
            rescore_answers,
        )
        text = "Step 5: Together, they have 13 letters."
        # Strict mode must return None (no equality or final answer marker)
        self.assertIsNone(extract_numeric_answer(text, lenient=False))
        # Lenient mode must extract 13 from the last non-empty line
        self.assertEqual(extract_numeric_answer(text, lenient=True), 13.0)
        self.assertEqual(extract_numeric_answer_lenient(text), 13.0)

        # Rescore test
        details = [
            {"condition": "steernet", "n_shot": 0, "response": text, "gold": 13},
            {"condition": "steernet", "n_shot": 0, "response": "She has no letters.", "gold": 13},
        ]
        strict_scores = rescore_answers(details, lenient=False)
        self.assertEqual(strict_scores[("steernet", 0)], 0.0)
        lenient_scores = rescore_answers(details, lenient=True)
        self.assertEqual(lenient_scores[("steernet", 0)], 50.0)

    def test_reserved_confirmation_split_isolation(self):
        # 100 eval items, deployment_examples=30
        eval_data = list(range(100))
        n_dep = min(30, len(eval_data) // 2)
        eval_sel = [eval_data[i] for i in range(len(eval_data) - n_dep)]
        confirm_rows = [eval_data[i] for i in range(len(eval_data) - n_dep, len(eval_data))]

        self.assertEqual(len(eval_sel), 70)
        self.assertEqual(len(confirm_rows), 30)
        # Sets must be strictly disjoint
        self.assertTrue(set(eval_sel).isdisjoint(set(confirm_rows)))

    def test_route_entropy_calculation(self):
        from src.evaluation.multi_benchmark_runner import route_entropy_by_shot
        # 1. Collapsed routes (100% same route) -> entropy 0.0
        collapsed_details = [
            {"condition": "steernet", "n_shot": 0, "route": (11, 13)},
            {"condition": "steernet", "n_shot": 0, "route": (11, 13)},
            {"condition": "steernet", "n_shot": 0, "route": (11, 13)},
        ]
        ent_collapsed = route_entropy_by_shot(collapsed_details, condition="steernet")
        self.assertEqual(ent_collapsed[0], 0.0)

        # 2. Diverse routes (50% route A, 50% route B) -> entropy 1.0 bit
        diverse_details = [
            {"condition": "steernet", "n_shot": 2, "route": (11, 13)},
            {"condition": "steernet", "n_shot": 2, "route": (18, 20)},
        ]
        ent_diverse = route_entropy_by_shot(diverse_details, condition="steernet")
        self.assertAlmostEqual(ent_diverse[2], 1.0, places=3)

    def test_pre_cap_norm_ratio_tracking_without_sync(self):
        import torch
        from src.steering.steernet import RSCSteerNet
        from src.training.trainer import collect_pre_cap_ratio

        steer = RSCSteerNet(hidden_dim=32, rank=4, lora_alpha=8.0, max_relative_norm=0.04)
        steer.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            steer.up.weight.normal_(0, 2.0)
            steer.mag_head.bias.fill_(6.0)
            h = torch.randn(2, 5, 32)
            delta = steer(h, active_k=torch.tensor([1, 1]))

        # _last_pre_cap_ratio must be populated
        self.assertIsNotNone(steer._last_pre_cap_ratio)
        self.assertGreater(steer._last_pre_cap_ratio.item(), 1.0)

        # Test model-level collector
        dummy_model = torch.nn.Module()
        dummy_model.steer_nets = torch.nn.ModuleDict({"11": steer})
        ratios = collect_pre_cap_ratio(dummy_model)
        self.assertIn(11, ratios)
        val = ratios[11]["mean"] if isinstance(ratios[11], dict) else ratios[11]
        self.assertGreater(val, 1.0)


class TestRun4HardeningAndDiagnostics(unittest.TestCase):
    """Forensic verification tests for Run 4 hardening and telemetry (WP1, WP2, WP3)."""

    def test_pre_cap_report_accumulation_and_reset(self):
        """WP1: pre_cap_report accumulates across multiple forwards and resets cleanly."""
        from src.steering.steernet import RSCSteerNet, pre_cap_report
        steer = RSCSteerNet(hidden_dim=32, rank=4, lora_alpha=8.0, max_relative_norm=0.04)
        steer.eval()
        dummy_model = torch.nn.Module()
        dummy_model.steer_nets = torch.nn.ModuleDict({"11": steer})

        with torch.no_grad():
            steer.up.weight.normal_(0, 1.0)
            steer.mag_head.bias.fill_(6.0)
            h1 = torch.randn(2, 5, 32)
            _ = steer(h1, active_k=torch.tensor([1, 1]))
            h2 = torch.randn(3, 4, 32)
            _ = steer(h2, active_k=torch.tensor([1, 1, 1]))

        report = pre_cap_report(dummy_model, reset=True)
        self.assertIn(11, report)
        self.assertEqual(report[11]["tokens"], 2 * 5 + 3 * 4)
        self.assertGreater(report[11]["mean"], 0.0)
        self.assertGreaterEqual(report[11]["max"], report[11]["mean"])

        # Reset should have cleared stats
        empty_report = pre_cap_report(dummy_model, reset=False)
        self.assertEqual(empty_report, {})

    def test_route_entropy_negative_zero_clamp_and_empty_route(self):
        """WP3: route_entropy_by_shot clamps -0.0 to 0.0 and suppresses collapse on empty deployed route."""
        from src.evaluation.multi_benchmark_runner import route_entropy_by_shot
        details = [
            {"condition": "steernet", "n_shot": 0, "route": (11, 15)},
            {"condition": "steernet", "n_shot": 0, "route": (11, 15)},
            {"condition": "steernet", "n_shot": 2, "route": (11, 15)},
            {"condition": "ars", "n_shot": 0, "route": ()},
            {"condition": "ars", "n_shot": 0, "route": ()},
        ]
        entropies = route_entropy_by_shot(details, condition="steernet")
        self.assertIn(0, entropies)
        self.assertEqual(entropies[0], 0.0)
        # Verify it is positive zero, not -0.0
        self.assertFalse(str(entropies[0]).startswith("-"))

        # ARS with empty route (deployed mode) should have entropy 0.0 without crash
        ars_entropies = route_entropy_by_shot(details, condition="ars")
        self.assertEqual(ars_entropies.get(0), 0.0)

    def test_chain_stats_and_formatting(self):
        """WP2: chain_stats computes steps, characters, tokens, and strict vs lenient accuracy."""
        from src.evaluation.multi_benchmark_runner import chain_stats, format_chain_stats_table
        details = [
            {
                "condition": "baseline",
                "n_shot": 0,
                "steps": 3,
                "chars": 150,
                "tokens": 40,
                "correct": True,
                "lenient_correct": True,
            },
            {
                "condition": "baseline",
                "n_shot": 0,
                "steps": 4,
                "chars": 200,
                "tokens": 50,
                "correct": False,
                "lenient_correct": True,  # Concluding statement saved under lenient
            },
        ]
        stats = chain_stats(details)
        self.assertIn(("baseline", 0), stats)
        b_stat = stats[("baseline", 0)]
        self.assertEqual(b_stat["count"], 2)
        self.assertEqual(b_stat["mean_steps"], 3.5)
        self.assertEqual(b_stat["mean_chars"], 175.0)
        self.assertEqual(b_stat["mean_tokens"], 45.0)
        self.assertEqual(b_stat["accuracy_strict"], 50.0)
        self.assertEqual(b_stat["accuracy_lenient"], 100.0)

        tbl = format_chain_stats_table(stats)
        self.assertIn("baseline", tbl)
        self.assertIn("0-shot", tbl)
        self.assertIn("50.0%", tbl)
        self.assertIn("100.0%", tbl)

    def test_dual_extractor_table_formatting(self):
        """WP2: format_dual_extractor_table computes difference between strict and lenient."""
        from src.evaluation.multi_benchmark_runner import format_dual_extractor_table
        p_strict = {("gsm8k", "baseline", 0): [1, 0, 0, 0]}
        p_lenient = {("gsm8k", "baseline", 0): [1, 1, 0, 0]}
        tbl = format_dual_extractor_table(p_strict, p_lenient)
        self.assertIn("GSM8K", tbl)
        self.assertIn("25.00%", tbl)
        self.assertIn("50.00%", tbl)
        self.assertIn("+25.00pp", tbl)

    def test_alpha_jitter_floor_with_flags_fn(self):
        """WP2: alpha_jitter_floor measures decoding flip rate under tiny alpha jitter."""
        from src.evaluation.deployment import alpha_jitter_floor
        # Reference alpha=1.0: 100 items all 1
        # alpha=0.999: item 0 flipped to 0 (1 flip = 1.0% flip rate)
        # alpha=0.998: items 0 and 1 flipped to 0 (2 flips = 2.0% flip rate)
        def mock_flags(alpha):
            if alpha == 1.0:
                return [1] * 100
            elif alpha == 0.999:
                return [0] + [1] * 99
            elif alpha == 0.998:
                return [0, 0] + [1] * 98
            return [1] * 100

        res = alpha_jitter_floor(flags_fn=mock_flags, alphas=(1.0, 0.999, 0.998))
        self.assertIn("flip_rates", res)
        self.assertEqual(res["flip_rates"][(1.0, 0.999)], 0.01)
        self.assertEqual(res["flip_rates"][(1.0, 0.998)], 0.02)
        self.assertEqual(res["noise_floor_pp"], 2.0)

    def test_export_latex_summary_table_respects_significance(self):
        """WP3: export_latex_summary_table does not bold non-significant cells when per_item_correct is provided."""
        from src.evaluation.multi_benchmark_runner import export_latex_summary_table
        tree = {
            "gsm8k": {
                "baseline": {"0_shot": 56.0},
                "fixed_steernet": {"0_shot": 60.0},
                "steernet": {"0_shot": 53.0},
                "ars": {"0_shot": 60.0},  # Tied with fixed_steernet, but difference vs baseline has lo <= 0
            }
        }
        # Baseline: 56 ones, 44 zeros
        # ARS: 60 ones, 40 zeros, but overlapping such that CI contains 0
        base_flags = [1] * 56 + [0] * 44
        ars_flags = [1] * 54 + [0] * 2 + [1] * 6 + [0] * 38  # delta +4, but CI [-5, +13] contains 0
        per_item = {
            ("gsm8k", "baseline", 0): base_flags,
            ("gsm8k", "ars", 0): ars_flags,
        }
        latex_tbl = export_latex_summary_table(tree, per_item_correct=per_item)
        # Since CI contains 0 (lo <= 0), ARS cell '60.00%' must NOT be bolded
        self.assertNotIn(r"\textbf{60.00\%}", latex_tbl)


class TestEngineV2OutcomeSupervision(unittest.TestCase):
    """Engine v2: Outcome-supervised ARS, calculation masks, execution rewards, and gate budget."""

    def test_is_calc_token_id(self):
        """Digits and math symbols should be classified as 1; English words and Step as 0."""
        from src.data.dataset import _is_calc_token_id

        class MockTokenizer:
            def decode(self, token_ids):
                mapping = {
                    10: "Step",
                    11: "Indras",
                    12: "42",
                    13: "+",
                    14: "=",
                    15: "apples",
                    16: "$100",
                }
                return mapping.get(token_ids[0], "")

        tok = MockTokenizer()
        self.assertEqual(_is_calc_token_id(tok, 10), 0)  # "Step"
        self.assertEqual(_is_calc_token_id(tok, 11), 0)  # "Indras"
        self.assertEqual(_is_calc_token_id(tok, 12), 1)  # "42"
        self.assertEqual(_is_calc_token_id(tok, 13), 1)  # "+"
        self.assertEqual(_is_calc_token_id(tok, 14), 1)  # "="
        self.assertEqual(_is_calc_token_id(tok, 15), 0)  # "apples"
        self.assertEqual(_is_calc_token_id(tok, 16), 1)  # "$100"

    def test_per_example_execution_reward_distinguishes_correctness(self):
        """Execution reward must be higher for correct predictions on calculation tokens than incorrect ones."""
        from src.routing.rloo import per_example_execution_reward

        B, T, V = 2, 8, 20
        # Row 0: correct predictions; Row 1: incorrect predictions
        labels = torch.tensor([
            [-100, -100, 5, 6, 7, 8, 9, 10],
            [-100, -100, 5, 6, 7, 8, 9, 10],
        ], dtype=torch.long)
        calc_mask = torch.tensor([
            [0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1],
        ], dtype=torch.long)

        # Logits for row 0 predict target labels; row 1 predicts 0 (wrong)
        logits = torch.zeros(B, T, V)
        for t in range(T - 1):
            lbl0 = labels[0, t + 1].item()
            if lbl0 != -100:
                logits[0, t, lbl0] = 10.0
            logits[1, t, 0] = 10.0  # wrong prediction

        rewards = per_example_execution_reward(
            logits, labels, calc_mask=calc_mask, outcome_weight=1.0, nll_weight=0.1
        )
        self.assertEqual(rewards.shape, (2,))
        self.assertGreater(rewards[0].item(), rewards[1].item())
        self.assertGreater(rewards[0].item(), 0.5)
        self.assertLess(rewards[1].item(), 0.0)

    def test_per_example_answer_nll_calc_mode(self):
        """When calc_target_mode == 'calculation', non-calc tokens should not contribute to NLL."""
        from src.routing.rloo import per_example_answer_nll

        B, T, V = 1, 6, 10
        labels = torch.tensor([[-100, 2, 3, 4, 5, 6]], dtype=torch.long)
        # Only tokens 4 and 5 are calculation tokens
        calc_mask = torch.tensor([[0, 0, 0, 1, 1, 0]], dtype=torch.long)

        # Logits that are perfect on calc tokens (4 and 5) but terrible on tokens 2, 3, 6
        logits = torch.zeros(B, T, V)
        logits[0, 2, 4] = 20.0  # perfect on token index 3 (value 4)
        logits[0, 3, 5] = 20.0  # perfect on token index 4 (value 5)
        # Tokens 2 and 3 and 6 predict index 0
        logits[0, 0, 0] = 20.0
        logits[0, 1, 0] = 20.0
        logits[0, 4, 0] = 20.0

        nll_calc = per_example_answer_nll(
            logits, labels, calc_mask=calc_mask, calc_target_mode="calculation"
        )
        # Since calc tokens are perfectly predicted, nll_calc should be near 0
        self.assertLess(nll_calc.item(), 0.05)

    def test_compute_gate_budget_loss(self):
        """Budget loss pulls total route mass towards target_sparsity (e.g. 0.20)."""
        from src.training.losses import compute_gate_budget_loss

        class DummyGate(nn.Module):
            def __init__(self, alpha_val):
                super().__init__()
                self.p = nn.Parameter(torch.tensor(1.0))
                self._alpha_for_aux = torch.full((2, 10, 1), alpha_val)
                self._valid_mask = torch.ones(2, 10, dtype=torch.bool)

        class DummyModel(nn.Module):
            def __init__(self, alpha_val):
                super().__init__()
                self.gates = nn.ModuleDict({"11": DummyGate(alpha_val)})

        # Perfect target alpha = 0.20 -> loss ≈ 0
        m_ideal = DummyModel(0.20)
        loss_ideal = compute_gate_budget_loss(m_ideal, target_sparsity=0.20)
        self.assertAlmostEqual(loss_ideal.item(), 0.0, places=4)

        # Saturated alpha = 1.0 -> loss = (1.0 - 0.20)^2 = 0.64
        m_sat = DummyModel(1.0)
        loss_sat = compute_gate_budget_loss(m_sat, target_sparsity=0.20)
        self.assertAlmostEqual(loss_sat.item(), 0.64, places=4)

        # Two selected layers splitting the same route budget must still be
        # ideal at total mass 0.20, rather than being penalized per layer.
        m_split = DummyModel(0.10)
        m_split.gates["13"] = DummyGate(0.10)
        loss_split = compute_gate_budget_loss(m_split, target_sparsity=0.20)
        self.assertAlmostEqual(loss_split.item(), 0.0, places=4)

    def test_trim_batch_with_calc_mask(self):
        """_trim_batch trims both standard tensors and calc_mask to the same sequence length."""
        from src.training.trainer import _trim_batch

        ids = torch.ones(2, 10, dtype=torch.long)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
        lbls = torch.ones(2, 10, dtype=torch.long)
        shield = torch.zeros(2, 10, dtype=torch.long)
        calc_mask = torch.ones(2, 10, dtype=torch.long)

        # Max real sequence length is 4
        ids_t, mask_t, lbls_t, shield_t, calc_t = _trim_batch(ids, mask, lbls, shield, calc_mask)
        self.assertEqual(ids_t.shape[1], 4)
        self.assertEqual(calc_t.shape[1], 4)
        self.assertEqual(mask_t.shape[1], 4)

    def test_gsm8k_to_cot_returns_splits(self):
        """gsm8k_to_cot must return a tuple (train_split, test_split) containing calc_mask."""
        from src.data.dataset import gsm8k_to_cot
        from src.configs.config import RSCConfig

        class MockTokenizer:
            pad_token_id = 0
            eos_token_id = 1
            def __call__(self, texts, truncation=True, max_length=128, padding="max_length"):
                if isinstance(texts, str):
                    texts = [texts]
                ids = [[10, 11, 12, 13, 0] + [0] * (max_length - 5) for _ in texts]
                masks = [[1, 1, 1, 1, 1] + [0] * (max_length - 5) for _ in texts]
                return {"input_ids": ids, "attention_mask": masks}
            def encode(self, text, add_special_tokens=False):
                return [10, 11, 12]
            def decode(self, token_ids, skip_special_tokens=True):
                return "Step 1: 42"

        cfg = RSCConfig()
        cfg.n_samples = 5
        # Dataset formatting is unit-tested against an in-memory fixture so
        # the suite never depends on Hub availability or a local cache.
        from datasets import Dataset
        fixture = Dataset.from_dict({
            "question": [f"Question {i}" for i in range(5)],
            "answer": ["Step 1: 1 + 1 = 2. #### 2" for _ in range(5)],
        })
        with patch("src.data.dataset.load_dataset", return_value=fixture):
            train_d, eval_d = gsm8k_to_cot(tokenizer=MockTokenizer(), cfg=cfg, n_samples=5)
        self.assertIsNotNone(train_d)
        self.assertIsNotNone(eval_d)
        self.assertGreater(len(train_d), 0)
        self.assertIn("calc_mask", train_d[0])


class TestEngineV21NormAndK0(unittest.TestCase):
    """Engine v2.1 tests: cap-aware norm penalty, route budget power 1.0, and K=0 margin preference."""

    def test_steernet_pre_cap_norm_differentiability(self):
        """Pre-cap ratio tensor last_pre_cap_q must be differentiable and shape [B, T, 1]."""
        from src.steering.steernet import RSCSteerNet
        torch.manual_seed(42)
        sn = RSCSteerNet(hidden_dim=32, rank=4, lora_alpha=8.0, max_relative_norm=0.02)
        sn.train()
        h = torch.randn(2, 5, 32, requires_grad=True)
        # Ensure delta is non-zero by nudging up.weight
        with torch.no_grad():
            sn.up.weight.fill_(0.1)
        delta = sn(h)
        self.assertIsNotNone(sn.last_pre_cap_q)
        self.assertEqual(sn.last_pre_cap_q.shape, (2, 5, 1))
        self.assertTrue(sn.last_pre_cap_q.requires_grad)

    def test_compute_steer_norm_loss_penalizes_excess(self):
        """Excess ratio q > q_hi must produce positive Huber loss and backpropagate to weights."""
        from src.steering.steernet import RSCSteerNet
        from src.training.losses import compute_steer_norm_loss
        torch.manual_seed(42)

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.steer_nets = nn.ModuleDict({
                    "11": RSCSteerNet(hidden_dim=32, rank=4, lora_alpha=8.0, max_relative_norm=0.02)
                })

        model = MockModel()
        sn = model.steer_nets["11"]
        sn.train()
        h = torch.randn(2, 5, 32)
        # Force raw norm to be large so q > q_hi (e.g. q ~ 2.0)
        with torch.no_grad():
            sn.up.weight.fill_(1.5)
        _ = sn(h)
        sn._last_valid_mask = torch.ones(2, 5, dtype=torch.bool)

        loss, stats = compute_steer_norm_loss(model, q_hi=0.75, huber_delta=0.10)
        self.assertGreater(loss.item(), 0.0)
        self.assertGreater(stats["q_mean"], 0.75)
        loss.backward()
        self.assertIsNotNone(sn.up.weight.grad)
        self.assertFalse(torch.all(sn.up.weight.grad == 0))

    def test_symmetric_k0_preference(self):
        """K=0 wins only because its measured reward is larger, without a margin hinge."""
        from src.routing.rloo import counterfactual_k_preference_loss
        torch.manual_seed(42)
        B, n_actions = 4, 5
        k_logits = torch.zeros(B, n_actions, requires_grad=True)
        k_actions = torch.tensor([[1, 2, 3, 4]] * B, dtype=torch.long)
        # Baseline reward = 1.0, steered rewards = 0.9 (worse than baseline).
        reward_0 = torch.ones(B)
        action_rewards = torch.full((B, 4), 0.9)

        loss = counterfactual_k_preference_loss(
            k_logits, k_actions, reward_0, action_rewards, temperature=0.10
        )
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        # Gradient on k_logits[:, 0] (z_0) should be negative (pushing z_0 higher to lower loss)
        self.assertLess(k_logits.grad[:, 0].mean().item(), 0.0)
        # Gradient on steered actions should be positive (pushing them down)
        self.assertGreater(k_logits.grad[:, 1:].mean().item(), 0.0)

    def test_equal_k_counterfactual_rewards_have_no_k0_bias(self):
        """Equal action values keep the symmetric distillation gradient neutral."""
        k_logits = torch.zeros(2, 5, requires_grad=True)
        k_actions = torch.tensor([[1, 2, 3, 4]] * 2, dtype=torch.long)
        loss = counterfactual_k_preference_loss(
            k_logits,
            k_actions,
            reward_0=torch.ones(2),
            action_rewards=torch.ones(2, 4),
            temperature=0.10,
        )
        loss.backward()
        self.assertTrue(torch.allclose(k_logits.grad, torch.zeros_like(k_logits.grad), atol=1e-6))

    def test_balanced_route_budget_power(self):
        """Engine v2.2 keeps multi-layer residuals viable with K**(-0.5)."""
        from src.configs.config import RSCConfig
        cfg = RSCConfig()
        self.assertEqual(cfg.steer_route_budget_power, 0.5)
        self.assertEqual(cfg.steer_max_relative_norm, 0.005)  # Remediation: adjusted to match measured envelope
        self.assertEqual(cfg.steer_loss_target, "calculation_weighted")
        self.assertEqual(cfg.calc_token_weight, 1.35)
        self.assertEqual(cfg.answer_tail_weight, 1.10)
        self.assertEqual(cfg.lora_dropout, 0.0)
        self.assertEqual(cfg.bootstrap_steer_compat_lr_scale, 0.05)
        self.assertEqual(cfg.router_k0_margin, 0.0)
        self.assertEqual(cfg.lambda_gate_diversity, 0.20)

    def test_multi_step_gradient_accumulation_no_graph_leakage(self):
        """Phase 1 multi-step accumulation with varying layer subsets must not retain backward graphs."""
        from src.models.wrapper import Phi2WithRSC
        from src.configs.config import RSCConfig

        class DummyLayer(nn.Module):
            def forward(self, h, *args, **kwargs):
                return h

        class DummyModel(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.config = type("Config", (), {"num_hidden_layers": 10, "hidden_size": hidden_dim})()
                self.embed = nn.Embedding(50, hidden_dim)
                self.model = type("Model", (), {"layers": nn.ModuleList([DummyLayer() for _ in range(10)])})()

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                h = self.embed(input_ids)
                for layer in self.model.layers:
                    h = layer(h)
                return type("Output", (), {"logits": h})()

        cfg = RSCConfig(router_warmup_random_subsets=True, router_warmup_max_k=2)
        dummy = DummyModel(hidden_dim=32)
        wrapper = Phi2WithRSC(dummy, cfg, device=torch.device("cpu"))
        wrapper.train()
        wrapper.set_gate_freeze(True, 1.0)
        wrapper._router_warmup = True

        # Ensure SteerNet up.weight is non-zero so norm loss is active
        for sn in wrapper.steer_nets.values():
            with torch.no_grad():
                sn.up.weight.fill_(1.0)

        # Simulate 3 microbatches of gradient accumulation without optimizer.zero_grad()
        for step in range(3):
            ids = torch.randint(0, 50, (2, 8))
            mask = torch.ones(2, 8, dtype=torch.bool)
            out = wrapper(input_ids=ids, attention_mask=mask)
            loss_norm, _ = wrapper.compute_active_steer_norm_loss(q_hi=0.75, huber_delta=0.10)
            loss = out.logits.sum() + loss_norm
            # If graph leakage occurs across microbatches, this raises RuntimeError on step 1 or 2
            loss.backward()


class TestEngineV3RoutingAndMemory(unittest.TestCase):
    """Regression coverage for Engine v3 contracts."""

    def test_masked_route_allocation_is_competitive_and_safe_for_k_zero(self):
        from src.gating.gate import masked_route_allocation

        logits = torch.tensor([[2.0, 1.0, -3.0], [1.0, 2.0, 3.0]])
        selected = torch.tensor([[True, True, False], [False, False, False]])
        allocation = masked_route_allocation(logits, selected, temperature=1.0)
        self.assertTrue(torch.allclose(allocation[0, :2].sum(), torch.tensor(1.0)))
        self.assertEqual(float(allocation[0, 2]), 0.0)
        self.assertTrue(torch.equal(allocation[1], torch.zeros(3)))

    def test_phase1b_eligibility_masks_learned_routes_but_not_audit_overrides(self):
        """Rejected/non-coadapted layers cannot re-enter learned routing."""
        class DummyLayer(nn.Module):
            def forward(self, h, *args, **kwargs):
                return h

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Config", (), {"num_hidden_layers": 10, "hidden_size": 16})()
                self.embed = nn.Embedding(32, 16)
                self.model = type("Model", (), {"layers": nn.ModuleList([DummyLayer() for _ in range(10)])})()

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                h = self.embed(input_ids)
                for layer in self.model.layers:
                    h = layer(h)
                return type("Output", (), {"logits": h})()

        wrapper = Phi2WithRSC(DummyModel(), RSCConfig(), device=torch.device("cpu"))
        eligible = wrapper.target_layers[-2:]
        wrapper.set_eligible_routing_layers(eligible)
        raw = torch.zeros(1, len(wrapper.target_layers))
        masked = wrapper.mask_routing_logits(raw)
        self.assertTrue((masked[0, :-2] < -1e3).all())
        self.assertTrue(torch.isfinite(masked[0, -2:]).all())
        k_logits = torch.zeros(1, wrapper.cfg.max_active_layers + 1)
        masked_k = wrapper.mask_routing_k_logits(k_logits)
        self.assertTrue((masked_k[0, 3:] < -1e3).all())

        # Explicit measurements retain full catalog access by design.
        rejected_layer = wrapper.target_layers[0]
        wrapper.set_routing_override("only", [rejected_layer])
        self.assertEqual(wrapper._routing_override[1], {rejected_layer})

    def test_v31_low_scale_grid_covers_early_layer_calibration(self):
        """The default grid must expose conservative early/middle amplitudes."""
        grid = RSCConfig().phase1_gate_scale_grid
        self.assertEqual(grid, (0.25, 0.35, 0.50, 0.70))

    def test_v31_effective_norm_contract_accepts_safe_low_scale(self):
        """Raw tails are acceptable only when alpha-scaled displacement is safe."""
        cfg = RSCConfig(
            phase1_max_generation_p95_norm_ratio=1.50,
            phase1_max_effective_generation_p95_norm_ratio=0.60,
        )
        result = {"median_tokens": 100.0, "pre_cap_norm": {"p95": 1.40}}
        self.assertTrue(_phase1_pair_is_safe(result, 100.0, cfg, gate_scale=0.70)[0])
        self.assertFalse(_phase1_pair_is_safe(result, 100.0, cfg, gate_scale=0.80)[0])

    def test_v31_tariff_zero_at_bootstrap_start(self):
        """The first bootstrap update has no abrupt K-cost policy shock."""
        tariff = capped_route_tariff(
            torch.tensor([0.04]), torch.tensor([[1, 2]]),
            torch.tensor([[0.10, 0.20]]), tariff_scale=0.0,
            max_positive_gain_fraction=0.50,
        )
        self.assertTrue(torch.equal(tariff, torch.zeros_like(tariff)))

    def test_v31_tariff_cannot_reverse_positive_route_gain(self):
        """A K=2 route that beats K=0 keeps a strictly positive net gain."""
        raw_delta = torch.tensor([[0.08, 0.10]])
        tariff = capped_route_tariff(
            torch.tensor([0.10]), torch.tensor([[1, 2]]), raw_delta,
            tariff_scale=1.0, max_positive_gain_fraction=0.50,
        )
        self.assertTrue(torch.all(raw_delta - tariff > 0.0))
        self.assertAlmostEqual(float(tariff[0, 1]), 0.05, places=6)

    def test_v31_finite_preference_with_masked_k_logits(self):
        """Finite sentinels keep K distillation differentiable under eligibility masks."""
        logits = torch.tensor([[0.2, 0.1, -1.0e4, -1.0e4]], requires_grad=True)
        loss = counterfactual_k_preference_loss(
            logits, torch.tensor([[1, 2]]), torch.tensor([0.2]),
            torch.tensor([[0.3, 0.1]]), temperature=0.10,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_v31_cooperation_report_omits_nonfinite_masked_pairs(self):
        """Masked/invalid pairs cannot poison synergy telemetry with NaN."""
        router = JointLayerRouter([11, 15, 22, 24], hidden_dim=8, n_layers=32, max_active_layers=4)
        router._last_synergy_matrix = torch.tensor([[
            [0.0, float("nan"), 0.2, 0.1],
            [float("nan"), 0.0, -0.1, 0.3],
            [0.2, -0.1, 0.0, 0.4],
            [0.1, 0.3, 0.4, 0.0],
        ]])
        report = router.get_cooperation_analysis(eligible_layers=[11, 22, 24])
        self.assertTrue(all(torch.isfinite(torch.tensor(pair[2])) for pair in report["top_synergy_pairs"]))
        self.assertTrue(torch.isfinite(torch.tensor(report["mean_synergy"])))

    def test_v31_cross_band_pool_prioritises_early_middle_late(self):
        """Phase 1B retains distinct physical depth bands before extra ranks."""
        from src.training.trainer import select_phase1b_candidate_layers

        class PoolModel:
            target_layers = [11, 13, 15, 18, 22, 24]

        report = {
            "calibration_ranking": [
                {"layer": 24, "gate_scale": 0.70, "safe": True},
                {"layer": 11, "gate_scale": 0.35, "safe": True},
                {"layer": 15, "gate_scale": 0.50, "safe": True},
                {"layer": 22, "gate_scale": 0.70, "safe": True},
            ]
        }
        layers, _, pool = select_phase1b_candidate_layers(
            PoolModel(), report, RSCConfig(phase1b_candidate_layers=4),
        )
        self.assertEqual(layers[:3], [11, 15, 24])
        self.assertEqual(pool["depth_bands"], ["early", "middle", "late"])

    def test_v31_preflight_requires_batch_union_depth_bands(self):
        """Two layers in one band do not satisfy the deterministic preflight contract."""
        from src.training.trainer import audit_router_policy_health

        class FakeRouterModel:
            _routing_override = None
            _router_inference_mode = "argmax"
            def reset_routing_diagnostics(self): pass
            def eval(self): return self
            def set_router_inference_mode(self, mode): self._router_inference_mode = mode
            def set_routing_override(self, mode=None, layers=None): self._routing_override = None if mode is None else (mode, layers)
            def set_prompt_shield_mask(self, mask): pass
            def set_router_context_mask(self, mask): pass
            def __call__(self, **kwargs): return None
            def intervention_diagnostics(self):
                return {
                    "realized_k_distribution": {0: 0.1, 1: 0.7, 2: 0.2},
                    "realized_k_support": 3, "realized_k_entropy": 0.5,
                    "active_layer_count": 2, "active_depth_band_count": 1,
                    "layer_selection_entropy": 0.5, "gate": {}, "steering": {},
                }

        row = {"input_ids": torch.ones(4, dtype=torch.long), "attention_mask": torch.ones(4, dtype=torch.long), "labels": torch.tensor([-100, -100, 1, 2])}
        cfg = RSCConfig(allow_inconclusive_small_runs=False, router_preflight_examples=2, continue_on_diagnostic_failure=True)
        result = audit_router_policy_health(FakeRouterModel(), [row, row], cfg, torch.device("cpu"))
        self.assertFalse(result["passed"])
        self.assertTrue(any("depth bands" in failure for failure in result["failures"]))

    def test_v31_router_rejection_preserves_certified_fixed_steernet(self):
        """An adaptive rejection must retain the confirmed fixed control."""
        from src.training.trainer import _fallback_to_certified_fixed_steernet

        class FallbackModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.eligible = "unset"
                self.override = None
                self.gate = None
            def set_eligible_routing_layers(self, layers): self.eligible = layers
            def set_router_warmup(self, value): self.warmup = value
            def set_routing_override(self, mode, layers=None): self.override = (mode, layers)
            def set_gate_freeze(self, frozen, scale): self.gate = (frozen, scale)

        model = FallbackModel()
        cfg = RSCConfig(fixed_steer_eval_layers=[24], fixed_steer_eval_gate_scale=0.70)
        logs = {}
        with patch("src.training.trainer.save_ars_checkpoint"):
            _fallback_to_certified_fixed_steernet(model, cfg, logs, {}, "preflight rejected")
        self.assertEqual(logs["adaptive_phases_status"], "FALLBACK_TO_CERTIFIED_FIXED_STEERNET")
        self.assertEqual(model.override, ("only", [24]))
        self.assertEqual(model.gate, (True, 0.70))

    def test_phase1_report_cache_requires_matching_checkpoint_provenance(self):
        """Resume must reject a cache copied from a different checkpoint."""
        import json
        import tempfile
        from src.training.trainer import _cache_phase1_report, _phase1_report_matches_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "phase1.pt")
            manifest = checkpoint + ".manifest.json"
            with open(manifest, "w", encoding="utf-8") as handle:
                json.dump({"state_sha256": "state-a", "git_commit": "abc", "config": {"rank": 16}}, handle)
            report_path = os.path.join(directory, "phase1_report.json")
            cached = _cache_phase1_report({"passed": True}, report_path, checkpoint)
            self.assertTrue(_phase1_report_matches_checkpoint(cached, checkpoint))

            with open(manifest, "w", encoding="utf-8") as handle:
                json.dump({"state_sha256": "state-b", "git_commit": "abc", "config": {"rank": 16}}, handle)
            self.assertFalse(_phase1_report_matches_checkpoint(cached, checkpoint))

    def test_none_override_trace_reports_realized_k_zero(self):
        """A baseline control must not be mislabeled as a K=1 route."""
        class DummyLayer(nn.Module):
            def forward(self, h, *args, **kwargs):
                return h

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Config", (), {"num_hidden_layers": 10, "hidden_size": 16})()
                self.embed = nn.Embedding(32, 16)
                self.model = type("Model", (), {"layers": nn.ModuleList([DummyLayer() for _ in range(10)])})()
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                h = self.embed(input_ids)
                for layer in self.model.layers:
                    h = layer(h)
                return type("Output", (), {"logits": h})()

        from src.models.wrapper import Phi2WithRSC
        wrapper = Phi2WithRSC(DummyModel(), RSCConfig(), device=torch.device("cpu"))
        wrapper.set_routing_trace(True)
        wrapper.set_routing_override("none")
        _ = wrapper(input_ids=torch.ones(1, 4, dtype=torch.long), attention_mask=torch.ones(1, 4, dtype=torch.long))
        first = wrapper.last_routing_trace[wrapper.target_layers[0]]
        self.assertEqual(first["desired_k"], [0])
        self.assertEqual(first["realized_k"], [0])

    def test_pre_cap_reservoir_is_bounded_and_cpu_resident(self):
        from src.steering.steernet import RSCSteerNet, pre_cap_report

        steer = RSCSteerNet(16, 4, 8.0, max_relative_norm=0.04, telemetry_reservoir_size=7)
        steer.eval()
        with torch.no_grad():
            for _ in range(4):
                steer(torch.randn(2, 6, 16), active_k=torch.tensor([1, 1]))
        self.assertLessEqual(sum(value.numel() for value in steer._pre_cap_values), 7)
        self.assertTrue(all(value.device.type == "cpu" for value in steer._pre_cap_values))
        dummy = nn.Module()
        dummy.steer_nets = nn.ModuleDict({"11": steer})
        self.assertEqual(pre_cap_report(dummy)[11]["tokens"], 48)

    def test_router_preflight_rejects_single_layer_without_k_two_support(self):
        from src.training.trainer import audit_router_policy_health
        from src.configs.config import RSCConfig

        class FakeRouterModel:
            _routing_override = None
            _router_inference_mode = "argmax"
            def reset_routing_diagnostics(self): pass
            def eval(self): return self
            def set_router_inference_mode(self, mode): self._router_inference_mode = mode
            def set_routing_override(self, mode=None, layers=None): self._routing_override = None if mode is None else (mode, layers)
            def set_prompt_shield_mask(self, mask): pass
            def set_router_context_mask(self, mask): pass
            def __call__(self, **kwargs): return None
            def intervention_diagnostics(self):
                return {
                    "realized_k_distribution": {0: 0.90, 1: 0.10, 2: 0.0},
                    "realized_k_support": 2, "realized_k_entropy": 0.20,
                    "active_layer_count": 1, "layer_selection_entropy": 0.0,
                    "gate": {}, "steering": {},
                }

        row = {"input_ids": torch.ones(4, dtype=torch.long), "attention_mask": torch.ones(4, dtype=torch.long), "labels": torch.tensor([-100, -100, 1, 2])}
        cfg = RSCConfig(allow_inconclusive_small_runs=False, router_preflight_examples=2, continue_on_diagnostic_failure=True)
        report = audit_router_policy_health(FakeRouterModel(), [row, row], cfg, torch.device("cpu"))
        self.assertFalse(report["passed"])
        self.assertTrue(any("K>=2" in failure for failure in report["failures"]))
        self.assertTrue(any("active layers" in failure for failure in report["failures"]))

    def test_per_shot_deployment_does_not_pool_a_winning_zero_shot_policy(self):
        from src.evaluation.deployment import finalize_deployment_by_shot
        import tempfile
        import json
        import os

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ars.pt")
            with open(path + ".manifest.json", "w", encoding="utf-8") as handle:
                json.dump({}, handle)
            # Identical paired improvements make the 0-shot lower CI positive;
            # 2-shot remains tied and therefore baseline.
            modes = finalize_deployment_by_shot(
                {0: ([0] * 20, [1] * 20), 2: ([1] * 20, [1] * 20)}, path, n_boot=100, min_n=20,
            )
            self.assertEqual(modes["0"], "ars")
            self.assertEqual(modes["2"], "baseline")

    def test_phase1b_coadaptation_extracts_safe_layers_from_calibration_ranking(self):
        """Phase 1B must extract safe candidate layers from calibration_ranking."""
        from src.training.trainer import train_phase1b_coadaptation
        from src.configs.config import RSCConfig

        class DummySteerNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)
            def forward(self, h, active_k=None):
                return h

        class DummyWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.target_layers = [11, 13, 15, 22]
                self.steer_nets = nn.ModuleDict({str(l): DummySteerNet() for l in self.target_layers})
                self.router = nn.Linear(4, len(self.target_layers))
                self.base_model = nn.Identity()
            def set_gate_freeze(self, *args, **kwargs): pass
            def set_router_warmup(self, *args, **kwargs): pass
            def set_routing_override(self, *args, **kwargs): pass
            def set_prompt_shield_mask(self, *args, **kwargs): pass
            def set_router_context_mask(self, *args, **kwargs): pass
            def compute_active_steer_norm_loss(self, **kwargs):
                return torch.zeros((), requires_grad=True), {}
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                logits = torch.randn(1, 4, 32, requires_grad=True)
                return type("Out", (), {"logits": logits})()

        model = DummyWrapper()
        cfg = RSCConfig(phase1b_coadapt_epochs=1, phase1b_candidate_layers=3)
        row = {
            "input_ids": torch.ones(4, dtype=torch.long),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor([1, 2, 3, 4]),
        }
        report = {
            "calibration_ranking": [
                {"rank": 1, "layer": 22, "gate_scale": 0.60, "safe": True},
                {"rank": 2, "layer": 11, "gate_scale": 0.80, "safe": True},
                {"rank": 3, "layer": 15, "gate_scale": 0.60, "safe": True},
            ],
            "candidates": [],
        }
        res = train_phase1b_coadaptation(model, [row], report, cfg, torch.device("cpu"))
        self.assertEqual(res["status"], "COMPLETE")
        self.assertEqual(res["candidate_layers"], [11, 15, 22])
        self.assertEqual(res["pair_count"], 2)  # long-range (22, 11), (22, 15)
        self.assertEqual(res["schedule_k1_steps"], res["schedule_k2_steps"])
        self.assertEqual(res["candidate_gate_scales"], {22: 0.60, 11: 0.80, 15: 0.60})

    def test_phase1b_coadaptation_falls_back_to_candidates_if_ranking_missing(self):
        """Phase 1B must fall back to candidate scale_screen when ranking is empty."""
        from src.training.trainer import train_phase1b_coadaptation
        from src.configs.config import RSCConfig

        class DummySteerNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)
            def forward(self, h, active_k=None):
                return h

        class DummyWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.target_layers = [11, 13, 15, 22]
                self.steer_nets = nn.ModuleDict({str(l): DummySteerNet() for l in self.target_layers})
                self.router = nn.Linear(4, len(self.target_layers))
                self.base_model = nn.Identity()
            def set_gate_freeze(self, *args, **kwargs): pass
            def set_router_warmup(self, *args, **kwargs): pass
            def set_routing_override(self, *args, **kwargs): pass
            def set_prompt_shield_mask(self, *args, **kwargs): pass
            def set_router_context_mask(self, *args, **kwargs): pass
            def compute_active_steer_norm_loss(self, **kwargs):
                return torch.zeros((), requires_grad=True), {}
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                logits = torch.randn(1, 4, 32, requires_grad=True)
                return type("Out", (), {"logits": logits})()

        model = DummyWrapper()
        cfg = RSCConfig(phase1b_coadapt_epochs=1, phase1b_candidate_layers=2, phase1b_min_candidate_layers=2, phase1b_min_depth_bands=2)
        row = {
            "input_ids": torch.ones(4, dtype=torch.long),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor([1, 2, 3, 4]),
        }
        report = {
            "calibration_ranking": [],
            "candidates": [
                {"layer": 11, "scale_screen": [{"safe": True, "gate_scale": 0.8}]},
                {"layer": 15, "scale_screen": [{"safe": True, "gate_scale": 0.6}]},
            ],
        }
        res = train_phase1b_coadaptation(model, [row], report, cfg, torch.device("cpu"))
        self.assertEqual(res["status"], "COMPLETE")
        self.assertEqual(res["candidate_layers"], [11, 15])

    def test_phase1b_excludes_layers_that_failed_confirmation(self):
        """A confirmation-failed layer cannot re-enter Phase 1B via calibration."""
        from src.training.trainer import train_phase1b_coadaptation
        from src.configs.config import RSCConfig

        class DummySteerNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def forward(self, h, active_k=None):
                return h

        class DummyWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.target_layers = [11, 15, 22]
                self.steer_nets = nn.ModuleDict({str(layer): DummySteerNet() for layer in self.target_layers})
                self.router = nn.Linear(4, len(self.target_layers))
                self.base_model = nn.Identity()

            def set_gate_freeze(self, *args, **kwargs): pass
            def set_router_warmup(self, *args, **kwargs): pass
            def set_routing_override(self, *args, **kwargs): pass
            def set_prompt_shield_mask(self, *args, **kwargs): pass
            def set_router_context_mask(self, *args, **kwargs): pass

            def compute_active_steer_norm_loss(self, **kwargs):
                return torch.zeros((), requires_grad=True), {}

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                logits = torch.randn(1, 4, 32, requires_grad=True)
                return type("Out", (), {"logits": logits})()

        model = DummyWrapper()
        cfg = RSCConfig(phase1b_coadapt_epochs=1, phase1b_candidate_layers=3, phase1b_min_candidate_layers=2, phase1b_min_depth_bands=2)
        row = {
            "input_ids": torch.ones(4, dtype=torch.long),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor([1, 2, 3, 4]),
        }
        report = {
            "calibration_ranking": [
                {"rank": 1, "layer": 22, "safe": True},
                {"rank": 2, "layer": 11, "safe": True},
                {"rank": 3, "layer": 15, "safe": True},
            ],
            "confirmation_trials": [
                {"layer": 22, "safe": False, "passed": False},
            ],
            "candidates": [],
        }
        res = train_phase1b_coadaptation(model, [row], report, cfg, torch.device("cpu"))
        self.assertEqual(res["status"], "COMPLETE")
        self.assertEqual(res["candidate_layers"], [11, 15])
        self.assertEqual(res["pair_count"], 1)

    def test_resume_phase1_missing_checkpoint_fails_closed(self):
        """An explicit Phase 1 resume request must never retrain silently."""
        from src.configs.config import RSCConfig
        from src.training.trainer import train_rsc
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as directory:
            missing_checkpoint = os.path.join(directory, "missing_phase1.pt")
            cfg = RSCConfig(
                resume_phase1=True,
                lora_save_path=missing_checkpoint,
                deployment_min_examples=0,
            )
            with self.assertRaisesRegex(FileNotFoundError, "Refusing silent full retraining"):
                train_rsc(None, None, [], [{}, {}, {}, {}], cfg, torch.device("cpu"))


class TestEngineV32SafetyAndProfile(unittest.TestCase):
    """Regression coverage for post-cap admission and locked decision runs."""

    def test_post_cap_displacement_not_raw_tail_controls_phase1_safety(self):
        """A soft-capped route remains eligible despite a raw saturation tail."""
        cfg = RSCConfig(phase1_max_effective_generation_p95_norm_ratio=1.0)
        result = {"median_tokens": 100.0, "pre_cap_norm": {"p95": 1.64}}
        safe, _, raw_q = _phase1_pair_is_safe(result, 100.0, cfg, gate_scale=0.50)
        self.assertTrue(safe)
        self.assertAlmostEqual(raw_q, 1.64)
        self.assertLess(0.50 * post_cap_ratio_from_pre_cap_q(raw_q), 1.0)

    def test_confirmation_tie_is_diagnostic_not_a_promotion(self):
        """Calibration strength cannot promote a zero-gain confirmation tie."""
        passed, noise, reasons = _phase1_confirmation_verdict(
            confirmation_gain=0.0,
            calibration_gain=0.20,
            combined_gain=0.10,
            confirmation_n=16,
            safe=True,
            nll_safe=True,
            cfg=RSCConfig(phase1_allow_confirmation_noise_tolerance=True),
        )
        self.assertFalse(passed)
        self.assertTrue(noise)  # Retained as telemetry only; it cannot promote.
        self.assertIn("confirmation_accuracy", reasons)

    def test_priority_reservoir_replaces_early_prefix_with_later_stream_samples(self):
        """p95 telemetry samples the stream rather than retaining its prefix."""
        steer = RSCSteerNet(8, 2, 4.0, telemetry_reservoir_size=7)
        steer._update_pre_cap_reservoir(torch.zeros(7))
        steer._update_pre_cap_reservoir(torch.ones(100))
        values = steer._pre_cap_values[0]
        self.assertEqual(values.numel(), 7)
        self.assertTrue((values == 1.0).any())
        self.assertTrue(values.device.type == "cpu")

    def test_two_band_pool_is_valid_for_coadaptation(self):
        """An early+late pool is sufficient for the long-range curriculum."""
        from src.training.trainer import select_phase1b_candidate_layers

        class PoolModel:
            target_layers = [11, 13, 15, 18, 22, 24]

        report = {
            "calibration_ranking": [
                {"layer": 11, "gate_scale": 0.25, "safe": True},
                {"layer": 24, "gate_scale": 0.50, "safe": True},
            ],
            "confirmation_trials": [],
            "candidates": [],
        }
        layers, _, pool = select_phase1b_candidate_layers(
            PoolModel(), report,
            RSCConfig(phase1b_candidate_layers=2, phase1b_min_candidate_layers=2, phase1b_min_depth_bands=2),
        )
        self.assertEqual(layers, [11, 24])
        self.assertEqual(pool["depth_bands"], ["early", "late"])

    def test_kaggle_profile_lock_tracks_hyperparameter_drift(self):
        """A correct commit fingerprints profile changes and can enforce strict locking."""
        cfg = make_engine_v32_kaggle_config()
        digest = assert_engine_v32_kaggle_profile(cfg)
        self.assertEqual(len(digest), 64)
        cfg.phase1_epochs = 4
        custom_digest = assert_engine_v32_kaggle_profile(cfg, strict=False)
        self.assertEqual(len(custom_digest), 64)
        with self.assertRaisesRegex(RuntimeError, "custom overrides"):
            assert_engine_v32_kaggle_profile(cfg, strict=True)


class TestEngineV33MemoryRecovery(unittest.TestCase):
    """CPU regression coverage for cache-aware safety and resumable Phase 2."""

    @staticmethod
    def _snapshot(*, allocated, peak_allocated, reserved, peak_reserved, free):
        return {
            "available": True,
            "allocated_mb": float(allocated),
            "peak_allocated_mb": float(peak_allocated),
            "reserved_mb": float(reserved),
            "peak_reserved_mb": float(peak_reserved),
            "free_mb": float(free),
            "total_mb": 15360.0,
        }

    def test_reserved_cache_pressure_reclaims_instead_of_failing(self):
        """Cached reservation is flushed before an active-allocation decision."""
        before = self._snapshot(allocated=8095, peak_allocated=8095, reserved=13602, peak_reserved=13602, free=1500)
        after = self._snapshot(allocated=8095, peak_allocated=8095, reserved=8300, peak_reserved=13602, free=6500)
        cfg = RSCConfig(cuda_active_memory_limit_mb=11500, cuda_reserved_memory_limit_mb=12000, cuda_memory_min_free_mb=512)
        with patch("src.training.trainer._cuda_memory_snapshot", side_effect=[before, after]), \
             patch("src.training.trainer.gc.collect"), \
             patch("src.training.trainer.torch.cuda.empty_cache") as empty_cache, \
             patch("src.training.trainer.torch.cuda.reset_peak_memory_stats"):
            status = enforce_gpu_memory_budget("unit", cfg)
        self.assertEqual(status["action"], "reclaimed")
        empty_cache.assert_called_once()

    def test_active_allocation_pressure_requests_resumable_emergency(self):
        """Real allocation pressure becomes a controlled checkpointable pause, never a raw raise."""
        before = self._snapshot(allocated=12010, peak_allocated=12010, reserved=12200, peak_reserved=12200, free=300)
        after = self._snapshot(allocated=12010, peak_allocated=12010, reserved=12100, peak_reserved=12200, free=400)
        cfg = RSCConfig(cuda_active_memory_limit_mb=11500, cuda_reserved_memory_limit_mb=14200, cuda_memory_min_free_mb=512)
        with patch("src.training.trainer._cuda_memory_snapshot", side_effect=[before, after]), \
             patch("src.training.trainer.gc.collect"), \
             patch("src.training.trainer.torch.cuda.empty_cache"), \
             patch("src.training.trainer.torch.cuda.reset_peak_memory_stats"):
            status = enforce_gpu_memory_budget("unit", cfg)
        self.assertEqual(status["action"], "emergency")
        self.assertIn("active peak", status["reason"])

    def test_log_gpu_memory_is_report_only(self):
        """Telemetry does not resurrect the retired reserved-memory fatal guard."""
        cfg = RSCConfig(cuda_reserved_memory_limit_mb=1)
        with patch("src.training.losses.torch.cuda.is_available", return_value=True), \
             patch("src.training.losses.torch.cuda.max_memory_allocated", return_value=8 * 1024 ** 2), \
             patch("src.training.losses.torch.cuda.max_memory_reserved", return_value=16 * 1024 ** 2), \
             patch("src.training.losses.torch.cuda.reset_peak_memory_stats"):
            telemetry = log_gpu_memory("unit", cfg)
        self.assertEqual(telemetry["peak_reserved_mb"], 16.0)

    def test_phase2_state_round_trip_restores_model_optimizer_scheduler_and_cursor(self):
        """Interrupted training retains all state needed for an exact cursor resume."""
        class TinyARS(nn.Module):
            def __init__(self):
                super().__init__()
                self.steer_nets = nn.ModuleDict({"11": nn.Linear(2, 2)})
                self.gates = nn.ModuleDict({"11": nn.Linear(2, 1)})
                self.router = nn.Linear(2, 2)
                self._router_warmup = False
                self._router_exploration = True
                self._router_inference_mode = "sample"

            def set_router_warmup(self, value):
                self._router_warmup = value

            def set_router_exploration(self, value):
                self._router_exploration = value

            def set_router_inference_mode(self, value):
                self._router_inference_mode = value

        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "phase2_state.pt")
            model = TinyARS()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loss = sum(parameter.square().sum() for parameter in model.parameters())
            loss.backward()
            optimizer.step()
            scheduler = LRReducer(optimizer, total_steps=20, warmup_steps=2)
            scheduler.scheduler_step()
            expected_weight = model.steer_nets["11"].weight.detach().clone()
            save_phase2_training_state(
                path, model=model, optimizer=optimizer,
                scaler=torch.amp.GradScaler("cuda", enabled=False), lr_reducer=scheduler,
                stage_name="PHASE 2: Unified", epochs=3, epoch=1, next_batch_start=16,
                indices=[3, 1, 2], global_step=50, best_acc=0.60, best_val=1.044,
                patience_ctr=1, baseline_ema=[0.1], effective_rloo_batch_size=4,
                train_steer=False, train_gate=True, train_router=True, save_path="best.pt",
            )
            with torch.no_grad():
                model.steer_nets["11"].weight.zero_()
            payload = load_phase2_training_state(path, model, torch.device("cpu"))
            self.assertTrue(torch.equal(model.steer_nets["11"].weight, expected_weight))
            self.assertEqual(payload["progress"]["epoch"], 1)
            self.assertEqual(payload["progress"]["next_batch_start"], 16)
            self.assertEqual(payload["progress"]["effective_rloo_batch_size"], 4)
            restored_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            restored_optimizer.load_state_dict(payload["optimizer"])
            restored_scheduler = LRReducer(restored_optimizer, total_steps=1)
            restored_scheduler.load_state_dict(payload["lr_reducer"])
            self.assertEqual(restored_scheduler.current_step, scheduler.current_step)
            self.assertTrue(restored_optimizer.state)

    def test_resume_phase2_missing_state_fails_closed(self):
        """An explicit Phase-2 resume request cannot silently begin a new run."""
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            cfg = RSCConfig(
                resume_phase2=True,
                phase2_training_state_path=os.path.join(directory, "missing_phase2_state.pt"),
            )
            with self.assertRaisesRegex(FileNotFoundError, "Refusing silent full retraining"):
                train_rsc(None, None, [], [], cfg, torch.device("cpu"))


class TestEngineV4(unittest.TestCase):
    """CPU-only contracts for Engine v4 routing, allocation, and evidence."""

    def test_routing_manifest_round_trip(self):
        model = Phi2WithRSC.__new__(Phi2WithRSC)
        model.target_layers = [11, 17, 20]
        model._eligible_routing_layers = {11, 20}
        model._certified_gate_scales = {11: 0.35, 20: 0.70}
        model.deployment_disabled = True
        model.deployment_modes_by_shot = {"0": "ars", "4": "baseline"}
        for name in (
            "_cached_decode_routes", "_cached_decode_k", "_cached_decode_probs", "_cached_decode_logits",
            "_cached_decode_margins", "_cached_decode_active_k", "_cached_decode_allocation",
        ):
            setattr(model, name, None)
        manifest = model.get_routing_manifest()
        restored = Phi2WithRSC.__new__(Phi2WithRSC)
        restored.target_layers = [11, 17, 20]
        restored._eligible_routing_layers = None
        restored._certified_gate_scales = {}
        restored.deployment_disabled = False
        restored.deployment_modes_by_shot = {}
        for name in (
            "_cached_decode_routes", "_cached_decode_k", "_cached_decode_probs", "_cached_decode_logits",
            "_cached_decode_margins", "_cached_decode_active_k", "_cached_decode_allocation",
        ):
            setattr(restored, name, None)
        restored.load_routing_manifest(manifest)
        self.assertEqual(restored._eligible_routing_layers, {11, 20})
        self.assertTrue(restored.deployment_disabled)
        self.assertEqual(restored.deployment_modes_by_shot["4"], "baseline")
        self.assertEqual(restored._certified_gate_scales, {11: 0.35, 20: 0.70})

    def test_effective_k_rejects_silenced_nominal_triad(self):
        model = Phi2WithRSC.__new__(Phi2WithRSC)
        model.target_layers = [11, 17, 20]
        model._steering_telemetry = {
            11: {"count": 10, "energy_sum": 8.0},
            17: {"count": 10, "energy_sum": 0.3},
            20: {"count": 10, "energy_sum": 0.01},
        }
        report = model.effective_k_diagnostics(0.05)
        self.assertEqual(report["effective_layers"], [11])
        self.assertEqual(report["effective_layer_count"], 1)

    def test_route_tuple_entropy_detects_static_triad(self):
        model = Phi2WithRSC.__new__(Phi2WithRSC)
        model._route_tuple_counts = {(11, 17, 20): 100}
        self.assertEqual(model.route_tuple_entropy(), 0.0)
        model._route_tuple_counts[(11, 20)] = 100
        self.assertGreater(model.route_tuple_entropy(), 0.9)

    def test_token_competitive_allocation_has_null_option(self):
        logits = torch.tensor([[[2.0, 1.0, 9.0], [0.0, 0.0, 0.0]]])
        selected = torch.tensor([[True, True, False]])
        allocation = token_competitive_allocation(logits, selected, temperature=0.8)
        self.assertTrue(torch.all(allocation[..., 2] == 0))
        self.assertTrue(torch.all(allocation.sum(dim=-1) <= 1.0 + 1e-6))
        self.assertGreater(allocation[0, 0, 0], allocation[0, 0, 1])

    def test_route_activity_floor_penalizes_starved_selected_layer(self):
        class Gate(nn.Module):
            def __init__(self, alpha):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))
                self._alpha_for_aux = alpha
                self._valid_mask = torch.ones(alpha.shape[:2], dtype=torch.bool)
        class Model:
            def __init__(self):
                self.gates = {"11": Gate(torch.ones(1, 3, 1)), "20": Gate(torch.zeros(1, 3, 1))}
        self.assertGreater(compute_route_activity_floor_loss(Model(), 0.05).item(), 0.0)

    def test_pair_evidence_shrinkage(self):
        table = PairEvidenceTable(shrinkage=8)
        table.update(11, 20, 0.8)
        self.assertAlmostEqual(table.estimate(20, 11), 0.8 / 9.0, places=6)
        self.assertEqual(table.as_dict()["L11+L20"]["observations"], 1)

    def test_k_preference_tiny_temperature_and_k0_are_finite(self):
        loss = counterfactual_k_preference_loss(
            torch.tensor([[0.0, -4.0, -5.0]], requires_grad=True),
            torch.tensor([[1, 2]]), torch.tensor([1.0]), torch.tensor([[0.9, 0.8]]),
            temperature=1e-6,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_v40_profile_enables_decision_contract(self):
        cfg = make_engine_v40_kaggle_config()
        self.assertEqual(cfg.max_active_layers, 3)
        self.assertTrue(cfg.token_competitive_allocation)
        self.assertTrue(cfg.require_adaptive_beats_fixed)
        self.assertGreater(cfg.router_min_eval_route_tuple_entropy, 0.0)

    def test_v40_profile_requires_declared_run_override(self):
        from unittest import mock
        from src.configs.config import assert_engine_v40_kaggle_profile
        cfg = make_engine_v40_kaggle_config()
        cfg.n_samples = 4000
        with mock.patch.dict(os.environ, {"ARS_STRICT_PROFILE": "1"}):
            with self.assertRaisesRegex(RuntimeError, "undeclared overrides"):
                assert_engine_v40_kaggle_profile(cfg, strict=True)
            self.assertIsInstance(
                assert_engine_v40_kaggle_profile(cfg, strict=True, allowed_overrides=("n_samples",)), str,
            )

    def test_adaptive_promotion_requires_positive_ci_over_fixed(self):
        fixed = [1] * 40
        adaptive = [1] * 40
        decision = adaptive_vs_fixed_promotion(fixed, adaptive, require_adaptive_beats_fixed=True, n_boot=100)
        self.assertEqual(decision["mode"], "fixed_steernet")

    def test_k0_can_be_disabled_without_masking_valid_k(self):
        model = Phi2WithRSC.__new__(Phi2WithRSC)
        model.cfg = RSCConfig(max_active_layers=2, router_k0_enabled=False)
        model.target_layers = [11, 20]
        model._eligible_routing_layers = {11, 20}
        model.device = torch.device("cpu")
        masked = model.mask_routing_k_logits(torch.zeros(1, 3))
        self.assertLess(masked[0, 0].item(), -1000)
        self.assertTrue(torch.isfinite(masked[0, 1:]).all())

    def test_pair_evidence_is_neutral_until_observed(self):
        self.assertEqual(PairEvidenceTable().estimate(11, 24), 0.0)

    def test_v40_profile_expands_stratified_pool(self):
        cfg = make_engine_v40_kaggle_config()
        self.assertEqual(cfg.phase1b_candidate_layers, 6)
        self.assertEqual(cfg.phase1b_candidates_per_band, 2)

    def test_token_competitive_k0_is_exactly_zero(self):
        allocation = token_competitive_allocation(torch.randn(1, 2, 3), torch.zeros(1, 3, dtype=torch.bool))
        self.assertEqual(float(allocation.abs().sum()), 0.0)

    def test_k1_tariff_free(self):
        from src.routing.rloo import capped_route_tariff
        cost = torch.tensor([0.05])
        k_actions = torch.tensor([[0, 1, 2]])
        raw_delta = torch.tensor([[0.0, 0.5, 0.5]])
        # With k1_tariff_free=True, K=0 and K=1 have 0 tariff, K=2 has positive tariff
        tariff_free = capped_route_tariff(cost, k_actions, raw_delta, tariff_scale=1.0, max_positive_gain_fraction=0.5, k1_tariff_free=True)
        self.assertEqual(float(tariff_free[0, 0]), 0.0)
        self.assertEqual(float(tariff_free[0, 1]), 0.0)
        self.assertGreater(float(tariff_free[0, 2]), 0.0)

    def test_layer_coverage_loss(self):
        from src.routing.rloo import layer_coverage_loss
        # u_policy with large negative logits for some layers (starvation)
        u_starved = torch.tensor([[-5.0, -5.0, 5.0, 5.0]])
        loss_starved = layer_coverage_loss(u_starved, min_prob=0.10)
        self.assertGreater(float(loss_starved), 0.0)

        # u_policy with balanced logits
        u_balanced = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
        loss_balanced = layer_coverage_loss(u_balanced, min_prob=0.10)
        self.assertEqual(float(loss_balanced), 0.0)

    def test_hierarchical_deployment_arbiter_3tier(self):
        from src.evaluation.deployment import finalize_deployment_by_shot
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "ckpt.pt")
            # Case 1: Adaptive doesn't beat fixed (tie), fixed beats base -> fixed_steernet
            base = [0] * 50
            fixed = [1] * 50
            adaptive = [1] * 50
            decisions = finalize_deployment_by_shot(
                outcomes_by_shot={0: (base, adaptive)},
                checkpoint_path=ckpt,
                fixed_outcomes_by_shot={0: fixed},
                require_adaptive_beats_fixed=True,
                n_boot=100,
                min_n=10,
            )
            self.assertEqual(decisions["0"], "fixed_steernet")

            # Case 2: Adaptive beats fixed and baseline -> ars
            base = [0] * 50
            fixed = [0] * 25 + [1] * 25
            adaptive = [1] * 50
            decisions = finalize_deployment_by_shot(
                outcomes_by_shot={0: (base, adaptive)},
                checkpoint_path=ckpt,
                fixed_outcomes_by_shot={0: fixed},
                require_adaptive_beats_fixed=True,
                n_boot=100,
                min_n=10,
            )
            self.assertEqual(decisions["0"], "ars")

            # Case 3: Neither beats baseline -> baseline
            base = [1] * 50
            fixed = [0] * 50
            adaptive = [0] * 50
            decisions = finalize_deployment_by_shot(
                outcomes_by_shot={0: (base, adaptive)},
                checkpoint_path=ckpt,
                fixed_outcomes_by_shot={0: fixed},
                require_adaptive_beats_fixed=True,
                n_boot=100,
                min_n=10,
            )
            self.assertEqual(decisions["0"], "baseline")

    def test_phase2_state_preserves_certified_gate_scales(self):
        from src.training.trainer import _phase2_runtime_state, _restore_phase2_runtime_state
        class DummyModel:
            def __init__(self):
                self._certified_gate_scales = {11: 0.35, 18: 0.50}
                self.target_layers = [11, 18]
                self._restored_scales = None
                self.device = torch.device("cpu")
            def routing_eligibility_mask(self, device):
                return torch.tensor([True, True])
            def set_certified_gate_scales(self, scales):
                self._restored_scales = scales

        m = DummyModel()
        state = _phase2_runtime_state(m)
        self.assertEqual(state["certified_gate_scales"], {11: 0.35, 18: 0.50})
        m2 = DummyModel()
        _restore_phase2_runtime_state(m2, state)
        self.assertEqual(m2._restored_scales, {11: 0.35, 18: 0.50})

    def test_nonregressing_confirmation_promotion(self):
        from src.training.trainer import _phase1_confirmation_verdict
        # With phase1_allow_nonregressing_confirmation=True, a 0.0pp confirmation gain with nonregressing combined gain passes
        cfg = RSCConfig(phase1_allow_nonregressing_confirmation=True)
        passed, noise, reasons = _phase1_confirmation_verdict(
            confirmation_gain=0.0,
            calibration_gain=0.0,
            combined_gain=0.0,
            confirmation_n=28,
            safe=True,
            nll_safe=True,
            cfg=cfg,
        )
        self.assertTrue(passed)
        self.assertEqual(len(reasons), 0)

    def test_noise_tolerated_confirmation_promotion(self):
        from src.training.trainer import _phase1_confirmation_verdict
        # With phase1_allow_nonregressing_confirmation=True, small noise (-1 question on N=28) passes when calibration is positive and combined is nonregressing
        cfg = RSCConfig(phase1_allow_nonregressing_confirmation=True)
        passed, noise, reasons = _phase1_confirmation_verdict(
            confirmation_gain=-1.0 / 28.0,
            calibration_gain=0.08,
            combined_gain=0.01,
            confirmation_n=28,
            safe=True,
            nll_safe=True,
            cfg=cfg,
        )
        self.assertTrue(passed)
        self.assertTrue(noise)
        self.assertEqual(len(reasons), 0)
