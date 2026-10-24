import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from transformers import PhiConfig, PhiForCausalLM
    HAS_PHI = True
except (ImportError, ModuleNotFoundError):
    from transformers import LlamaConfig as PhiConfig, LlamaForCausalLM as PhiForCausalLM
    HAS_PHI = False

from src.configs.config import RSCConfig
from src.gating.gate import RSCUsefulnessGate
from src.models.wrapper import Phi2WithRSC
from src.routing.catalog import Catalog
from src.training.losses import (
    dominance_hinge,
    counterfactual_token_labels,
    TokenLabelStore,
    gate_total_loss,
    gate_bce_tokens,
)
from src.training.trainer import train_s3_gate

VOCAB = 64


def make_tiny_model(gate_task_gradient: bool = True) -> Phi2WithRSC:
    torch.manual_seed(0)
    if HAS_PHI:
        pcfg = PhiConfig(
            vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=8,
            num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=1,
            partial_rotary_factor=0.5, resid_pdrop=0.0, embd_pdrop=0.0, attention_dropout=0.0,
        )
        pcfg._attn_implementation = "eager"
    else:
        pcfg = PhiConfig(
            vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=8,
            num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=1,
            attention_dropout=0.0,
        )
    base = PhiForCausalLM(pcfg).eval()
    cfg = RSCConfig()
    cfg.router_policy = "catalog"
    cfg.gate_loss_mode = "v5"
    cfg.gate_init_logit = 2.0
    cfg.gate_task_gradient = gate_task_gradient
    model = Phi2WithRSC(base, cfg, device=torch.device("cpu")).eval()
    for sn in model.steer_nets.values():
        nn.init.normal_(sn.down1.weight, std=0.3)
        nn.init.normal_(sn.mid.weight, std=0.3)
        nn.init.normal_(sn.up.weight, std=0.3)
    model.set_catalog(Catalog.build(model.target_layers, 3))
    return model


class TestGateStage(unittest.TestCase):
    def test_gate_init(self):
        cfg = RSCConfig()
        cfg.gate_loss_mode = "v5"
        cfg.gate_init_logit = 2.0
        hidden_dim = 32
        gate = RSCUsefulnessGate(hidden_dim, cfg)

        # 1. Bias sigmoid == 0.8808 ± 1e-3
        bias_prob = torch.sigmoid(gate.fc2.bias).item()
        self.assertAlmostEqual(bias_prob, 0.880797, delta=1e-3)

        # 2. fc2 weight is zero in v5 mode
        self.assertTrue(torch.all(gate.fc2.weight == 0.0))

        # 3. Random inputs give alpha ≈ 0.8808
        h = torch.randn(2, 5, hidden_dim)
        delta = torch.randn(2, 5, hidden_dim)
        alpha = gate(h, delta)
        self.assertTrue(torch.allclose(alpha, torch.full_like(alpha, 0.880797), atol=1e-3))

    def test_task_gradient_flow(self):
        model = make_tiny_model(gate_task_gradient=True)
        # Freeze backbone and steernets
        for p in model.parameters():
            p.requires_grad = False
        for g in model.gates.values():
            for p in g.parameters():
                p.requires_grad = True

        L = model.target_layers[1]
        model.set_routing_override("only", [L])

        ids = torch.randint(3, VOCAB, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.long)
        lbls = ids.clone()

        out = model(input_ids=ids, attention_mask=mask, labels=lbls)
        out.loss.backward()

        gate = model.gates[str(L)]
        self.assertIsNotNone(gate.fc2.weight.grad)
        self.assertGreater(float(gate.fc2.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(gate.fc2.bias.grad)

        # SteerNet params must have no grad
        sn = model.steer_nets[str(L)]
        self.assertIsNone(sn.up.weight.grad)
        self.assertIsNone(sn.down1.weight.grad)

    def test_dominance_hinge_math(self):
        # Case 1: gated <= ungated -> hinge is 0.0
        gated = torch.tensor([1.2, 0.8], requires_grad=True)
        ungated = torch.tensor([1.5, 1.0], requires_grad=True)
        h = dominance_hinge(gated, ungated, margin=0.0)
        self.assertAlmostEqual(float(h.item()), 0.0, places=5)

        # Case 2: gated > ungated -> hinge > 0.0
        gated2 = torch.tensor([2.0, 1.5], requires_grad=True)
        ungated2 = torch.tensor([1.0, 1.0], requires_grad=True)
        h2 = dominance_hinge(gated2, ungated2, margin=0.0)
        expected = ((2.0 - 1.0) + (1.5 - 1.0)) / 2.0
        self.assertAlmostEqual(float(h2.item()), expected, places=5)

        # Gradient flows only through gated
        h2.backward()
        self.assertIsNotNone(gated2.grad)
        self.assertIsNone(ungated2.grad)

    def test_counterfactual_token_labels(self):
        class DummyTable:
            qid = np.array(["q1", "q2", "q3"])
            split = np.array(["train", "train", "train"])
            n_shot = np.array([0, 0, 0])
            token_mask = np.array([
                [True, True, True, False],
                [True, True, False, False],
                [True, True, True, True],
            ])
            # (N, C, L)
            # class 0: empty
            # class 1: non-empty
            token_nll = np.array([
                # q1: class 0 has higher nll than class 1 on token 0 and 2
                [[1.0, 0.5, 1.0, 0.0], [0.1, 0.5, 0.2, 0.0]],
                # q2: class 0 is picked as router choice -> should be skipped!
                [[0.5, 0.5, 0.0, 0.0], [0.8, 0.8, 0.0, 0.0]],
                # q3: class 1
                [[0.5, 0.5, 0.5, 0.5], [0.4, 0.4, 0.6, 0.6]],
            ])

        router_choices = np.array([1, 0, 1])
        store = counterfactual_token_labels(DummyTable(), router_choices, tau_u=0.05)

        # q2 had router choice 0 -> absent
        self.assertNotIn(("train", "q2", 0), store)

        # q1 had u = [0.9, 0.0, 0.8] on valid tokens
        # u > 0.05 -> [True, False, True]
        item1 = store.get(("train", "q1", 0))
        self.assertIsNotNone(item1)
        lbls1, w1 = item1
        self.assertEqual(lbls1.tolist(), [True, False, True])
        self.assertEqual(len(w1), 3)

        # q3 had u = [0.1, 0.1, -0.1, -0.1]
        # u > 0.05 -> [True, True, False, False]
        item3 = store.get(("train", "q3", 0))
        self.assertIsNotNone(item3)
        lbls3, w3 = item3
        self.assertEqual(lbls3.tolist(), [True, True, False, False])

    def test_gate_total_loss_composition(self):
        cfg = RSCConfig()
        cfg.lambda_dom = 1.0
        cfg.lambda_tok = 0.5
        cfg.lambda_l1_gate_v5 = 0.1
        cfg.gate_dom_margin = 0.0

        nll_gated = torch.tensor([1.5], requires_grad=True)
        nll_ungated = torch.tensor([1.0])
        bce_tok = torch.tensor(0.4, requires_grad=True)
        mean_alpha = torch.tensor(0.8, requires_grad=True)

        tot, metrics = gate_total_loss(nll_gated, nll_ungated, bce_tok, mean_alpha, cfg)
        expected = 1.5 + 1.0 * (1.5 - 1.0) + 0.5 * 0.4 + 0.1 * 0.8
        self.assertAlmostEqual(float(tot.item()), expected, places=5)
        self.assertIn("loss_total", metrics)
        self.assertIn("loss_dom", metrics)
        self.assertIn("loss_bce_tok", metrics)
        self.assertIn("mean_alpha", metrics)

    def test_tiny_model_gate_training_smoke(self):
        model = make_tiny_model(gate_task_gradient=True)
        # Bias catalog router so a candidate layer is selected during training
        L = model.target_layers[1]
        cls_idx = model._catalog.subsets.index((L,))
        with torch.no_grad():
            model.router.catalog_head.bias.fill_(-10.0)
            model.router.catalog_head.bias[cls_idx] = 10.0

        cfg = RSCConfig()
        cfg.gate_epochs = 2
        cfg.batch_size = 2
        cfg.gate_lr = 1e-2
        cfg.gate_v5_save_path = "./checkpoints/test_gate_smoke.pt"

        class DummyTokenizer:
            pad_token_id = 0

        tokenizer = DummyTokenizer()

        # Record initial parameter hashes for SteerNets and Router
        sn_init = {k: v.clone() for k, v in model.steer_nets.state_dict().items()}
        r_init = {k: v.clone() for k, v in model.router.state_dict().items()}
        g_init = {k: v.clone() for k, v in model.gates.state_dict().items()}

        rows = []
        for i in range(4):
            ids = torch.randint(3, VOCAB, (8,))
            lbls = ids.clone()
            lbls[:3] = -100  # prompt
            rows.append({
                "qid": f"q{i}",
                "split": "train",
                "n_shot": 0,
                "input_ids": ids,
                "labels": lbls,
            })

        res = train_s3_gate(
            model, tokenizer, rows, rows, cfg, device=torch.device("cpu"), table=None
        )

        self.assertIn("train_loss_history", res)
        self.assertTrue(all(np.isfinite(x) for x in res["train_loss_history"]))

        # Verify ONLY gate parameters changed
        for k, v in model.steer_nets.state_dict().items():
            self.assertTrue(torch.equal(v, sn_init[k]), f"SteerNet param {k} modified!")
        for k, v in model.router.state_dict().items():
            self.assertTrue(torch.equal(v, r_init[k]), f"Router param {k} modified!")

        # Gate weights must have been updated
        changed = any(not torch.equal(v, g_init[k]) for k, v in model.gates.state_dict().items())
        self.assertTrue(changed, "Gate weights did not update during training!")

        # Check saved checkpoint
        self.assertTrue(os.path.exists(cfg.gate_v5_save_path))
        if os.path.exists(cfg.gate_v5_save_path):
            os.remove(cfg.gate_v5_save_path)


if __name__ == "__main__":
    unittest.main()
