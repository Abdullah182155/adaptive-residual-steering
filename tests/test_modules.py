import sys
import os
import unittest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.configs.config import RSCConfig
from src.steering.steernet import RSCSteerNet
from src.gating.gate import RSCUsefulnessGate
from src.models.wrapper import Phi2WithRSC
from src.routing.router import JointLayerRouter
from src.routing.rloo import sample_k, sample_subset_plackett_luce, policy_entropy_bonus, mutual_information_diversity_loss, counterfactual_k_preference_loss
from src.training import (
    train_rsc,
    train_phase1_steernet,
    train_router_bootstrap,
    run_staged_phase,
    save_steer_weights,
    load_steer_weights,
    run_pg_loss,
)
from src.training.trainer import _small_run_waiver

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
        k_a = torch.tensor([2, 3])
        k_b = torch.tensor([4, 1])
        loss = counterfactual_k_preference_loss(
            logits, k_a, k_b,
            reward_0=torch.tensor([0.0, 0.0]),
            reward_a=torch.tensor([-1.0, -1.0]),
            reward_b=torch.tensor([-2.0, -2.0]),
        )
        loss.backward()
        self.assertLess(logits.grad[:, 0].mean().item(), 0.0)

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
        ]
        for attr in required_attrs:
            self.assertTrue(hasattr(cfg, attr), f"Missing config attribute: {attr}")

    def test_tiny_holdout_marks_audit_inconclusive_not_passed(self):
        cfg = RSCConfig(min_reliable_validation_examples=16, allow_inconclusive_small_runs=True)
        self.assertTrue(_small_run_waiver(4, cfg))
        self.assertFalse(_small_run_waiver(16, cfg))
        cfg.allow_inconclusive_small_runs = False
        self.assertFalse(_small_run_waiver(4, cfg))

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
        wrapper.set_routing_override("all")
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
        wrapper.set_routing_override("all")
        wrapper.set_gate_freeze(True, 1.0)

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
        self.assertTrue((syn_mat >= -1.0).all() and (syn_mat <= 1.0).all())
        # Diagonal must be 0
        diag = torch.diagonal(syn_mat, dim1=-2, dim2=-1)
        self.assertTrue((diag == 0.0).all())

        analysis = router.get_cooperation_analysis()
        self.assertIn("top_synergy_pairs", analysis)
        self.assertIn("top_antagonistic_pairs", analysis)
        self.assertIn("mean_synergy", analysis)
        self.assertIsInstance(analysis["top_synergy_pairs"], list)

    def test_anchor_grounded_rloo_loss(self):
        """Ensure 3-arm RLOO loss runs with baseline anchor and synergy alignment."""
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


