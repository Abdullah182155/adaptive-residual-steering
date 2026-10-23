import contextlib
import csv
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.onpolicy import (
    BatchAnswerStopping,
    OnPolicyChain,
    OnPolicyStore,
    sample_chains,
)
from src.evaluation.verifier import build_prompt

CPU = torch.device("cpu")


# ------------------------------------------------------------------ synthetic fixtures

class CharTokenizer:
    """ASCII char-level tokenizer. id 0 = pad, id 1 = eos (both skipped on decode)."""
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text if 2 <= ord(c) < 128]

    def decode(self, ids, skip_special_tokens=True):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return "".join(chr(i) for i in ids if i >= 2)

    def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=False, **kw):
        if isinstance(texts, str):
            texts = [texts]
        seqs = [self.encode(t) for t in texts]
        width = max(len(s) for s in seqs)
        ids, masks = [], []
        for s in seqs:
            pad = [self.pad_token_id] * (width - len(s))
            m_pad = [0] * (width - len(s))
            if self.padding_side == "left":
                ids.append(pad + s)
                masks.append(m_pad + [1] * len(s))
            else:
                ids.append(s + pad)
                masks.append([1] * len(s) + m_pad)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


class TinyBase(nn.Module):
    def __init__(self, vocab=128):
        super().__init__()
        torch.manual_seed(1234)
        self.emb = nn.Embedding(vocab, 16)
        self.out = nn.Linear(16, vocab)
        self.steer_on = True
        self.register_buffer("steer_bias", torch.randn(vocab) * 5)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, do_sample=True, temperature=1.0,
                 max_new_tokens=8, pad_token_id=0, eos_token_id=None,
                 stopping_criteria=None, **kw):
        ids = input_ids.clone()
        finished = torch.zeros(ids.shape[0], dtype=torch.bool)
        for _ in range(max_new_tokens):
            logits = self.out(self.emb(ids[:, -1]))
            if self.steer_on:
                logits = logits + self.steer_bias
            probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(1)
            nxt = torch.where(finished, torch.full_like(nxt, pad_token_id), nxt)
            ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
            if eos_token_id is not None:
                finished |= nxt == eos_token_id
            if stopping_criteria is not None and stopping_criteria(ids, None):
                break
            if bool(finished.all()):
                break
        return ids


class TinyWrapper(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base_model = base

    @contextlib.contextmanager
    def hooks_disabled(self):
        prev = self.base_model.steer_on
        self.base_model.steer_on = False
        try:
            yield
        finally:
            self.base_model.steer_on = prev

    def generate(self, *a, **kw):
        return self.base_model.generate(*a, **kw)


class ExplodingWrapper(TinyWrapper):
    def generate(self, *a, **kw):
        raise RuntimeError("boom")


def _cfg(**over):
    d = dict(
        onpolicy_n_questions=1500, onpolicy_samples=(2, 2), onpolicy_temperature=0.7,
        onpolicy_max_new_tokens=6, onpolicy_shots_mix=(0, 4), onpolicy_batch_size=4,
        onpolicy_max_tok_len=4000, onpolicy_gold_mix=0.4,
    )
    d.update(over)
    return SimpleNamespace(**d)


def _questions():
    return [
        {"question": "What is 1 + 1?", "answer": "1 + 1 = 2\n#### 2"},
        {"question": "What is 2 + 3?", "answer": "2 + 3 = 5\n#### 5"},
        {"question": "What is 4 * 3?", "answer": "4 * 3 = 12\n#### 12"},
    ]


def _chain(qid, idx, shot, text, correct, prompt="Question: q\nAnswer: Let's think step by step.\n"):
    return OnPolicyChain(qid=qid, sample_idx=idx, shot=shot, prompt=prompt, text=text,
                         pred_strict=None, pred_lenient=None, gold=10.0, correct=correct,
                         n_tokens=len(text), div_line=None)


# ------------------------------------------------------------------ tests

def _padded(tok, texts, prompt_len, width):
    rows = []
    for t in texts:
        ids = [65] * prompt_len + tok.encode(t)
        ids += [tok.pad_token_id] * (prompt_len + width - len(ids))
        rows.append(ids)
    return torch.tensor(rows)


def test_batch_answer_stopping_independent_rows():
    tok, pl, width = CharTokenizer(), 3, 21
    texts = ["x #### 5", "Final answer: 7", "nothing to see here"]
    full = _padded(tok, texts, pl, width)
    stop = BatchAnswerStopping(tok, pl, grace_tokens=2)
    results = {}
    for t in range(1, width + 1):
        results[t] = (stop(full[:, : pl + t], None), list(stop.row_done))
    assert results[7][1] == [False, False, False]
    assert results[8][1][0] is False          # marker seen, still in grace period
    assert results[10][1][0] is True          # grace elapsed
    assert results[10][1][1] is False         # rows are independent
    assert results[width] == (False, [True, True, False])

    eos_col = torch.tensor([[tok.pad_token_id], [tok.pad_token_id], [tok.eos_token_id]])
    assert stop(torch.cat([full, eos_col], dim=1), None) is True

    # early stop when every row hits a marker (incl. runaway "\nQuestion:")
    texts2 = ["x #### 5", "Final answer: 7", "ab\nQuestion: c"]
    full2 = _padded(tok, texts2, pl, width)
    stop2 = BatchAnswerStopping(tok, pl, grace_tokens=2)
    flags = [stop2(full2[:, : pl + t], None) for t in range(1, width + 1)]
    assert flags[7] is False and flags[-1] is True


def test_sampling_deterministic_and_restores_padding():
    tok = CharTokenizer()
    tok.padding_side = "right"
    cfg = _cfg()

    def run():
        model = TinyWrapper(TinyBase())
        store = sample_chains(model, tok, _questions(), cfg, CPU, seed=7)
        assert tok.padding_side == "right"
        return [(c.qid, c.sample_idx, c.text) for c in store.chains]

    a, b = run(), run()
    assert a == b and len(a) >= 6

    with pytest.raises(RuntimeError):
        sample_chains(ExplodingWrapper(TinyBase()), tok, _questions(), cfg, CPU, seed=7)
    assert tok.padding_side == "right"


def test_hooks_disabled_equals_base_generation():
    base = TinyBase()
    wrapper = TinyWrapper(base)
    ids = torch.tensor([[65, 66, 67, 68]])
    kw = dict(do_sample=True, temperature=0.7, max_new_tokens=10, pad_token_id=0, eos_token_id=None)
    torch.manual_seed(0)
    ref = base.generate(input_ids=ids.clone(), **kw)
    base.steer_on = True
    torch.manual_seed(0)
    with wrapper.hooks_disabled():
        assert base.steer_on is False
        # unwrapped base == generation with hooks off
        base_plain = TinyBase()
        torch.manual_seed(0)
        out = wrapper.generate(input_ids=ids.clone(), **kw)
    assert base.steer_on is True  # restored
    base.steer_on = False
    torch.manual_seed(0)
    expected = base.generate(input_ids=ids.clone(), **kw)
    assert torch.equal(out, expected)
    assert ref.shape == out.shape


def test_build_training_rows_prompt_masking():
    tok = CharTokenizer()
    prompt = build_prompt("What is 2 + 3 * 2?", n_shot=4)
    good_text = "Step 1: 2 + 3 = 5.\nStep 2: 5 * 2 = 10.\n#### 10"
    bad_text = "Step 1: 2 + 3 = 5.\nStep 2: 5 * 2 = 12.\n#### 12"
    chains = [
        _chain(0, 0, 4, good_text, True, prompt),
        _chain(0, 1, 4, bad_text, False, prompt),
        _chain(1, 0, 4, "Step 1: 1 + 1 = 2.\n#### 2", True, prompt),
    ]
    store = OnPolicyStore(chains)
    gold = []
    for i in range(3):
        ids = tok.encode(f"Question: g{i}\nAnswer: x\n#### {i}")
        lbl = [-100] * 10 + ids[10:]
        pad = 5
        gold.append({
            "input_ids": torch.tensor(ids + [0] * pad),
            "labels": torch.tensor(lbl + [-100] * pad),
            "attention_mask": torch.tensor([1] * len(ids) + [0] * pad),
        })

    train, val = store.build_training_rows(tok, _cfg(onpolicy_gold_mix=0.4), gold, val_fraction=0.0)
    assert val == []
    on = [r for r in train if r["is_onpolicy"]]
    gd = [r for r in train if not r["is_onpolicy"]]
    assert len(on) == 2 and len(gd) == 1  # 1 / 3 ~= 0.4 share after rounding

    p_len = len(tok.encode(prompt))
    assert p_len > 1000  # exemplars are part of the prompt
    for r in on:
        ids, lbl = r["input_ids"], r["labels"]
        assert (lbl[:p_len] == -100).all()
        assert torch.equal(lbl[p_len:], ids[p_len:])
        assert (lbl != -100).sum().item() == len(ids) - p_len

    paired = next(r for r in on if "neg_input_ids" in r)
    n_ids, n_lbl, mask = paired["neg_input_ids"], paired["neg_labels"], paired["div_token_mask"]
    assert mask.dtype == torch.bool and len(mask) == len(n_ids)
    assert (n_lbl[:p_len] == -100).all()
    assert mask.any() and not (mask & (n_lbl == -100)).any()
    assert tok.decode(n_ids[mask]) == "Step 2: 5 * 2 = 12."


def test_store_save_load_roundtrip_and_stats(tmp_path):
    chains = [
        _chain(0, 0, 0, "Step 1: 2 + 3 = 5.\nStep 2: 5 * 2 = 10.\n#### 10", True),
        _chain(0, 1, 0, "Step 1: 2 + 3 = 5.\nStep 2: 5 * 2 = 12.\n#### 12", False),
        _chain(1, 0, 4, "wrong a", False),
        _chain(1, 1, 4, "wrong b", False),
        _chain(2, 0, 4, "right", True),
    ]
    chains[0].pred_strict = 10.0
    store = OnPolicyStore(chains)
    path = str(tmp_path / "sub" / "store.jsonl")
    store.save(path)
    loaded = OnPolicyStore.load(path)
    assert loaded.chains == store.chains

    s = loaded.stats()
    assert s["n_total"] == 5 and s["n_correct"] == 2 and s["n_questions"] == 3
    assert s["n_questions_with_pair"] == 1
    assert s["pass_rate_per_shot"][0] == pytest.approx(0.5)
    assert s["pass_rate_per_shot"][4] == pytest.approx(1 / 3)
    assert len(loaded.correct_chains()) == 2

    pairs = loaded.pairs()
    assert len(pairs) == 1
    good, bad = pairs[0]
    assert good.correct and not bad.correct and bad.div_line == 1

    csv_path = str(tmp_path / "spot" / "spot.csv")
    loaded.export_spotcheck(10, csv_path, seed=3)
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    assert list(rows[0].keys()) == ["qid", "sample_idx", "shot", "prompt", "text", "pred_strict",
                                    "pred_lenient", "gold", "correct", "human_pred"]
    assert all(r["human_pred"] == "" for r in rows)
    for r in rows:
        r["human_pred"] = r["correct"]
    rows[0]["human_pred"] = "False" if rows[0]["correct"] == "True" else "True"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    assert OnPolicyStore.spotcheck_agreement(csv_path) == pytest.approx(4 / 5)
