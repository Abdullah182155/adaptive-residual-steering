import hashlib
import os
import random
import sys
import tempfile
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.configs.config import RSCConfig
from src.evaluation.paired_generation import generate_batch
from src.routing.catalog import Catalog
from src.training import contrastive
from src.training.losses import (
    compute_calc_token_weights,
    unlikelihood_on_divergence,
)
from src.training.trainer import (
    evaluate_subset_nll,
    sample_training_subset,
    train_s1_steernet,
)

DEVICE = torch.device("cpu")
LAYERS = (2, 4, 6, 8, 10)


class CharTokenizer:
    """Minimal char-level tokenizer: 0=pad, 1=eos, 2=unk, printable ASCII -> 3..97."""

    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2
    pad_token = "<pad>"
    eos_token = "<eos>"
    unk_token = "<unk>"

    def __init__(self):
        self.padding_side = "right"
        self._vocab = {chr(c): 3 + c - 32 for c in range(32, 127)}
        self._inv = {v: k for k, v in self._vocab.items()}

    def __len__(self):
        return 100

    def get_vocab(self):
        v = dict(self._vocab)
        v.update({"<pad>": 0, "<eos>": 1, "<unk>": 2})
        return v

    def encode(self, text, add_special_tokens=True):
        return [self._vocab.get(ch, 2) for ch in text]

    def decode(self, ids, skip_special_tokens=False):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        out = []
        for i in ids:
            if i in (0, 1):
                if not skip_special_tokens:
                    out.append("<pad>" if i == 0 else "<eos>")
                continue
            out.append(self._inv.get(int(i), "?"))
        return "".join(out)

    def batch_decode(self, seqs, skip_special_tokens=False):
        return [self.decode(s, skip_special_tokens=skip_special_tokens) for s in seqs]

    def __call__(self, texts, return_tensors=None, padding=False, truncation=False,
                 max_length=None, add_special_tokens=True):
        lst = [texts] if isinstance(texts, str) else list(texts)
        enc = [self.encode(t) for t in lst]
        if truncation and max_length:
            enc = [e[:max_length] for e in enc]
        t_max = max(len(e) for e in enc)
        ids, masks = [], []
        for e in enc:
            n_pad = t_max - len(e)
            if self.padding_side == "left":
                ids.append([0] * n_pad + e)
                masks.append([0] * n_pad + [1] * len(e))
            else:
                ids.append(e + [0] * n_pad)
                masks.append([1] * len(e) + [0] * n_pad)
        return BatchEncoding({"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)})


class TinyModel(nn.Module):
    """CPU-only stand-in exposing the Phi2WithRSC surface used by the S1' trainer."""

    def __init__(self, layers=LAYERS, vocab=16, d=32):
        super().__init__()
        self.target_layers = list(layers)
        self.base_model = nn.ModuleDict(
            {"embed": nn.Embedding(vocab, d), "head": nn.Linear(d, vocab, bias=False)}
        )
        nn.init.normal_(self.base_model["head"].weight, std=0.5)
        self.steer_nets = nn.ModuleDict(
            {
                str(l): nn.Sequential(nn.Linear(d, 16, bias=False), nn.GELU(), nn.Linear(16, d, bias=False))
                for l in layers
            }
        )
        for sn in self.steer_nets.values():
            for m in sn:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
        self.router = nn.Linear(d, len(layers))
        for p in self.base_model.parameters():
            p.requires_grad = False
        self._routing_override = None
        self._scale = 1.0

    def set_steer_scale(self, s):
        self._scale = float(s)

    def set_gate_freeze(self, freeze, val=1.0):
        return

    def set_routing_override(self, mode=None, layers=None):
        self._routing_override = None if mode is None else (mode, layers)

    def subset_matrix(self, subsets):
        return torch.tensor(
            [[l in s for l in self.target_layers] for s in subsets], dtype=torch.bool
        )

    @contextmanager
    def counterfactual_mode(self):
        was_training = self.training
        self.eval()
        try:
            yield
        finally:
            self.train(was_training)

    def _selected(self, bsz):
        n = len(self.target_layers)
        if self._routing_override is None:
            return torch.zeros(bsz, n, dtype=torch.bool)
        mode, layers = self._routing_override
        if mode == "matrix":
            return layers
        if mode == "only":
            row = torch.tensor([l in set(layers) for l in self.target_layers], dtype=torch.bool)
            return row.unsqueeze(0).expand(bsz, -1)
        return torch.zeros(bsz, n, dtype=torch.bool)

    def forward(self, input_ids, attention_mask=None, labels=None):
        h = self.base_model["embed"](input_ids)
        sel = self._selected(input_ids.size(0)).to(h.device)
        delta = torch.zeros_like(h)
        for j, l in enumerate(self.target_layers):
            delta = delta + sel[:, j].view(-1, 1, 1).to(h.dtype) * self.steer_nets[str(l)](h)
        logits = self.base_model["head"](h + self._scale * delta)
        return SimpleNamespace(logits=logits)


def _param_hash(module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(module.named_parameters(), key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class TestS1Trainer(unittest.TestCase):
    def setUp(self):
        self._saved_ids = getattr(contrastive, "_CONTRAST_MATH_IDS", None)

    def tearDown(self):
        contrastive._CONTRAST_MATH_IDS = self._saved_ids

    def test_subset_sampler_distribution(self):
        catalog = Catalog.build(LAYERS, 3)
        rng = random.Random(0)
        n = 3000
        counts = {1: 0, 2: 0, 3: 0}
        for _ in range(n):
            s = sample_training_subset(catalog, rng, (1, 2, 3))
            self.assertGreater(len(s), 0)
            counts[len(s)] += 1
        for k in (1, 2, 3):
            self.assertLess(abs(counts[k] / n - 1.0 / 3.0), 0.05, msg=f"K={k} freq={counts[k] / n:.3f}")

    def test_tiny_model_overfit(self):
        torch.manual_seed(0)
        contrastive._CONTRAST_MATH_IDS = {5, 6, 7}
        vocab, seq_len, n_prompt = 16, 10, 3
        rows = []
        for i in range(8):
            ids = torch.tensor([(i * 3 + t) % vocab for t in range(seq_len)], dtype=torch.long)
            labels = ids.clone()
            labels[:n_prompt] = -100
            rows.append({"input_ids": ids, "labels": labels, "is_onpolicy": False})

        model = TinyModel(vocab=vocab)
        tok = CharTokenizer()
        tmp = tempfile.mkdtemp()
        cfg = RSCConfig()
        cfg.catalog_layers = LAYERS
        cfg.catalog_max_k = 3
        cfg.steer_scale = 1.0
        cfg.batch_size = 1
        cfg.grad_accum = 1
        cfg.s1_epochs = 5  # 8 rows x 5 epochs = 40 optimizer steps
        cfg.s1_lr = 3e-2
        cfg.s1_onpolicy_fraction = 0.0
        cfg.s1_k_choices = (1, 2, 3)
        cfg.s1_final_gen_n = 0
        cfg.s1_save_path = os.path.join(tmp, "s1.pt")
        cfg.checkpoint_dir = tmp
        cfg.phase1_patience = 100
        cfg.lr_reduce_patience = 100
        catalog = Catalog.build(LAYERS, 3)

        base_before = _param_hash(model.base_model)
        router_before = _param_hash(model.router)
        steer_before = _param_hash(model.steer_nets)
        nll_before = evaluate_subset_nll(model, rows, catalog, cfg, DEVICE)[:, 1:].mean()

        train_s1_steernet(model, tok, rows, [], rows, cfg, DEVICE)

        nll_after = evaluate_subset_nll(model, rows, catalog, cfg, DEVICE)[:, 1:].mean()
        self.assertLess(float(nll_after), 0.8 * float(nll_before))
        self.assertEqual(base_before, _param_hash(model.base_model))
        self.assertEqual(router_before, _param_hash(model.router))
        self.assertNotEqual(steer_before, _param_hash(model.steer_nets))

    def test_calc_token_weights(self):
        contrastive._CONTRAST_MATH_IDS = {3, 4, 5}
        ids = torch.tensor([[9, 9, 3, 7, 4, 8, 5, 0]])
        labels = torch.tensor([[-100, -100, 3, 7, 4, 8, 5, -100]])
        w = compute_calc_token_weights(None, labels, ids, 1.75)
        expected = torch.tensor([[0.0, 0.0, 1.75, 1.0, 1.75, 1.0, 1.75, 0.0]])
        self.assertTrue(torch.equal(w, expected))

        contrastive._CONTRAST_MATH_IDS = None
        w_uniform = compute_calc_token_weights(None, labels, ids, 1.75)
        self.assertTrue(torch.equal(w_uniform, labels.ne(-100).float()))

    def test_unlikelihood_non_negative_and_zero_at_zero_prob(self):
        torch.manual_seed(0)
        vocab, t_len = 12, 6
        neg_labels = torch.randint(0, vocab, (1, t_len))
        neg_labels[:, :2] = -100
        div_mask = torch.zeros(1, t_len, dtype=torch.bool)
        div_mask[:, 3:5] = True

        logits = torch.randn(1, t_len, vocab)
        loss = unlikelihood_on_divergence(logits, neg_labels, div_mask)
        self.assertGreaterEqual(float(loss), 0.0)

        # logits[:, t-1] predicts token t; push divergent-token probability to ~0
        zero_logits = torch.zeros(1, t_len, vocab)
        for t in (3, 4):
            zero_logits[0, t - 1, neg_labels[0, t]] = -60.0
        loss0 = unlikelihood_on_divergence(zero_logits, neg_labels, div_mask)
        self.assertLess(float(loss0), 1e-6)
        self.assertGreaterEqual(float(loss0), 0.0)

        empty = unlikelihood_on_divergence(logits, neg_labels, torch.zeros_like(div_mask))
        self.assertEqual(float(empty), 0.0)

    def test_generate_batch_batch2_equals_batch1_x2(self):
        torch.manual_seed(0)
        cfg = LlamaConfig(
            vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256,
            pad_token_id=0, eos_token_id=1,
        )
        cfg._attn_implementation = "eager"
        model = LlamaForCausalLM(cfg).float().eval()
        tok = CharTokenizer()
        p1 = "Question: 2+3?\nAnswer:"
        p2 = "Q: what is 12*7 given extra words here?\nAnswer:"

        batched = generate_batch(model, tok, [p1, p2], DEVICE, max_new_tokens=10)
        single = generate_batch(model, tok, [p1], DEVICE, max_new_tokens=10) + generate_batch(
            model, tok, [p2], DEVICE, max_new_tokens=10
        )
        self.assertEqual(len(batched), 2)
        self.assertEqual(batched, single)


if __name__ == "__main__":
    unittest.main()
