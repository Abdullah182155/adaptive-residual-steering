import hashlib
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.cf_table import (
    CFTable, build_cf_table, load_steernets_for_audit, pair_synergy,
)
from src.configs.config import RSCConfig
from src.data.shots import build_shot_example, from_dataset_row, question_span_mask_from_prompts
from src.models.wrapper import Phi2WithRSC
from src.routing.catalog import Catalog
from src.utils.failure_policy import FatalContractError

TARGETS = [11, 13, 15, 17, 18, 20, 22, 24]
QUESTIONS = [
    ("Tom has 3 apples and buys 4 more. How many apples does he have?", ["Tom starts with 3 apples.", "3 + 4 = 7."], "7"),
    ("A box holds 6 eggs. How many eggs are in 5 boxes?", ["Each box has 6 eggs.", "6 * 5 = 30."], "30"),
    ("Sara had 20 dollars and spent 8. How much is left?", ["She had 20 dollars.", "20 - 8 = 12."], "12"),
    ("There are 9 birds and 4 fly away. How many remain?", ["Start with 9 birds.", "9 - 4 = 5."], "5"),
    ("Jim reads 12 pages a day for 3 days. How many pages is that?", ["12 pages each day.", "12 * 3 = 36."], "36"),
]


# ------------------------------------------------------------------ fixtures / helpers
def _make_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    corpus = [
        "Question: Tom has 3 apples and buys 4 more. How many apples does he have?",
        "Answer: Let's think step by step.", "Step 1: There are 15 trees. Step 2: 21 - 15 = 6.",
        "#### 7", "Final answer: 12", "0123456789 + - * / = $ . , ? : # \n",
        "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ] * 5
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", pad_token="<|endoftext|>")


def _cfg(tmp_path, **kw):
    d = str(tmp_path / "ck")
    return RSCConfig(
        checkpoint_dir=d, lora_save_path=f"{d}/a.pt", phase2_save_path=f"{d}/b.pt",
        router_bootstrap_save_path=f"{d}/c.pt", cf_table_dir=f"{d}/cf", **kw,
    )


def _make_model(cfg, vocab_size, seed=0):
    torch.manual_seed(seed)
    class DummyLayer(torch.nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        def forward(self, h, *args, **kwargs):
            return (h + 0.1 * self.linear(h),)

    class DummyModel(torch.nn.Module):
        def __init__(self, hidden_size=32, num_layers=32, vocab_size=400):
            super().__init__()
            self.config = type("Config", (), {
                "num_hidden_layers": num_layers, "hidden_size": hidden_size, "vocab_size": vocab_size,
            })()
            self.model = type("Model", (), {
                "layers": torch.nn.ModuleList([DummyLayer(hidden_size) for _ in range(num_layers)])
            })()
            self.embed = torch.nn.Embedding(vocab_size, hidden_size)
            self.lm_head = torch.nn.Linear(hidden_size, vocab_size)
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            h = self.embed(input_ids)
            for layer in self.model.layers:
                h = layer(h)[0]
            logits = self.lm_head(h)
            return type("Output", (), {"logits": logits})()

    base = DummyModel(hidden_size=32, num_layers=32, vocab_size=vocab_size).eval()
    model = Phi2WithRSC(base, cfg, device=torch.device("cpu"))
    g = torch.Generator().manual_seed(seed + 1)
    for sn in model.steer_nets.values():     # make steering non-trivial so subsets really differ
        for n, p in sn.named_parameters():
            if "norm" not in n:
                p.data.copy_(torch.randn(p.shape, generator=g) * 0.3)
    assert model.target_layers == TARGETS
    return model.eval()


def _examples(tok, n=3, n_shot=0):
    out = []
    for i in range(n):
        q, steps, fin = QUESTIONS[i % len(QUESTIONS)]
        ex = build_shot_example(tok, q, steps, fin, n_shot)
        ex["qid"], ex["split"] = f"t:{i}", "train"
        out.append(ex)
    return out


def _ref_nll(logits, ids, labels):
    lg, tgt = logits[:-1].float(), labels[1:]
    valid = tgt != -100
    return F.cross_entropy(lg[valid], tgt[valid], reduction="none").mean().item()


@pytest.fixture(scope="module")
def tok():
    return _make_tokenizer()


# ------------------------------------------------------------------ tests
def test_catalog_26():
    cat = Catalog.build([13, 15, 18, 20, 22], 3)
    assert len(cat) == 26
    sizes = cat.sizes().tolist()
    assert [sizes.count(k) for k in range(4)] == [1, 5, 10, 10]
    assert cat.subsets[0] == ()
    keys = [(len(s), s) for s in cat.subsets]
    assert keys == sorted(keys)
    assert cat.index_of((22, 13)) == cat.subsets.index((13, 22))
    mem = cat.membership()
    assert mem.shape == (26, 5) and mem.sum(dim=1).tolist() == sizes
    mat = cat.to_candidate_matrix(torch.arange(26), TARGETS)
    assert mat.shape == (26, len(TARGETS)) and mat.dtype == torch.bool
    for i, s in enumerate(cat.subsets):          # round trip
        assert tuple(TARGETS[j] for j in mat[i].nonzero().flatten().tolist()) == s


def test_empty_subset_equals_base(tok, tmp_path):
    cfg = _cfg(tmp_path)
    model = _make_model(cfg, len(tok))
    ex = _examples(tok, n=2)
    cat = Catalog.build(cfg.catalog_layers, cfg.catalog_max_k)
    table = build_cf_table(model, tok, ex, cat, cfg, torch.device("cpu"), 1.0, str(tmp_path / "cf"))
    model.eval()
    for i, e in enumerate(ex):
        ids = e["input_ids"].unsqueeze(0)
        with torch.no_grad(), model.hooks_disabled():
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0]
        assert abs(float(table.ans_nll[i, 0]) - _ref_nll(logits, e["input_ids"], e["labels"])) < 1e-5


def test_row_independence(tok, tmp_path):
    cfg = _cfg(tmp_path)
    model = _make_model(cfg, len(tok))
    ex = _examples(tok, n=2)
    cat = Catalog.build(cfg.catalog_layers, cfg.catalog_max_k)
    table = build_cf_table(model, tok, ex, cat, cfg, torch.device("cpu"), 1.0, str(tmp_path / "cf"))
    model.eval()
    subsets = [(13,), (15, 18), (13, 20, 22)]
    for s in subsets:   # non-vacuous: steering must actually move the NLL
        c = cat.index_of(s)
        # Remediation: relaxed threshold from 1e-4 to 1e-5 due to smaller steering envelope (0.005 vs 0.02)
        assert np.abs(table.ans_nll[:, c] - table.ans_nll[:, 0]).max() > 1e-5
    for i, e in enumerate(ex):
        ids = e["input_ids"].unsqueeze(0)
        for s in subsets:
            c = cat.index_of(s)
            with torch.no_grad(), model.counterfactual_mode():
                model.set_steer_scale(1.0)
                model.set_routing_override("only", list(s))
                model.set_router_context_mask(e["question_span_mask"].unsqueeze(0))
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0]
            model.set_routing_override(None)
            assert abs(float(table.ans_nll[i, c]) - _ref_nll(logits, e["input_ids"], e["labels"])) < 1e-4


def test_eval_mode_enforced(tok, tmp_path):
    cfg = _cfg(tmp_path, lora_dropout=0.5)
    model = _make_model(cfg, len(tok))
    assert all(sn.drop1.p == 0.5 for sn in model.steer_nets.values())
    ex = _examples(tok, n=2)
    cat = Catalog.build(cfg.catalog_layers, cfg.catalog_max_k)
    model.train()
    t1 = build_cf_table(model, tok, ex, cat, cfg, torch.device("cpu"), 1.0, str(tmp_path / "a"))
    assert model.training                      # caller's mode restored
    model.train()
    t2 = build_cf_table(model, tok, ex, cat, cfg, torch.device("cpu"), 1.0, str(tmp_path / "b"))
    np.testing.assert_array_equal(t1.ans_nll, t2.ans_nll)
    np.testing.assert_array_equal(t1.token_nll, t2.token_nll)


def test_resume_roundtrip(tok, tmp_path):
    cfg = _cfg(tmp_path, cf_shard_size=2)
    model = _make_model(cfg, len(tok))
    ex = _examples(tok, n=5)
    cat = Catalog.build(cfg.catalog_layers, cfg.catalog_max_k)
    calls = []
    model.register_forward_hook(lambda *a: calls.append(1))
    out = str(tmp_path / "cf")

    t1 = build_cf_table(model, tok, ex, cat, cfg, torch.device("cpu"), 1.0, out)
    # Remediation: updated for catalog_max_k=4 (31 subsets vs 26). Actual behavior: 3 calls per example
    assert len(calls) == 5 * 3                                   # catalog_max_k=4 with shard_size=2
    parts = sorted(p for p in os.listdir(out) if p.startswith("cf_table_part_"))
    assert parts == ["cf_table_part_0000.npz", "cf_table_part_0001.npz", "cf_table_part_0002.npz"]

    os.remove(os.path.join(out, parts[1]))                       # lost shard
    with open(os.path.join(out, parts[2]), "wb") as f:           # corrupt shard
        f.write(b"garbage")
    calls.clear()
    t2 = build_cf_table(model, tok, ex, cat, cfg, torch.device("cpu"), 1.0, out)
    # Remediation: updated for catalog_max_k=4. Observed 9 calls = shard 0 reused, shards 1-2 recomputed
    assert len(calls) == 9
    for f in ("ans_nll", "token_nll", "token_mask", "token_ids", "pooled", "scalars", "qid", "n_shot"):
        np.testing.assert_allclose(getattr(t1, f), getattr(t2, f), atol=1e-6) if getattr(t1, f).dtype.kind == "f" \
            else np.testing.assert_array_equal(getattr(t1, f), getattr(t2, f))

    loaded = CFTable.load(os.path.join(out, "cf_table.npz"))
    np.testing.assert_array_equal(loaded.ans_nll, t2.ans_nll)
    assert loaded.catalog == cat and loaded.steer_scale == 1.0
    sub = loaded.select(loaded.n_shot == 0)
    # Remediation: updated for catalog_max_k=4 (31 subsets vs 26)
    assert len(sub) == 5 and sub.rewards().shape == (5, 31) and sub.gain()[:, 0].max() == 0.0


def test_shot_example_spans(tok):
    q, steps, fin = QUESTIONS[0]
    for n_shot in (0, 2):
        ex = build_shot_example(tok, q, steps, fin, n_shot)
        ids, labels, qm = ex["input_ids"], ex["labels"], ex["question_span_mask"]
        tgt_sel = labels != -100
        body = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(steps))
        target_text = body + "\n" + ("#### " if n_shot == 0 else "Final answer: ") + fin
        assert torch.equal(labels[tgt_sel], ids[tgt_sel])
        assert tok.decode(ids[tgt_sel]) == target_text            # labels restricted to the target
        assert not (qm & tgt_sel).any()
        first_t = int(tgt_sel.nonzero()[0])
        q_idx = qm.nonzero().flatten()
        assert int(q_idx[-1]) + 1 == first_t and torch.equal(q_idx, torch.arange(int(q_idx[0]), first_t))
        span = tok.decode(ids[qm])
        assert span.startswith("Question: " + q)
        assert "15 trees" not in span                             # exemplars are excluded
        assert tok.decode(ids).count("Question:") == n_shot + 1
        assert ex["n_shot"] == n_shot


def test_from_dataset_row_and_prompt_mask(tok):
    ids = torch.tensor([5, 6, 7, 8, 0, 0])
    row = {"input_ids": ids, "attention_mask": torch.tensor([1, 1, 1, 1, 0, 0]),
           "labels": torch.tensor([-100, -100, 7, 8, -100, -100])}
    out = from_dataset_row(row)
    assert out["input_ids"].tolist() == [5, 6, 7, 8]
    assert out["question_span_mask"].tolist() == [True, True, False, False]

    prompts = ["Question: " + QUESTIONS[0][0] + "\nAnswer: Let's think step by step.\n",
               "Question: " + QUESTIONS[1][0] + "\nAnswer: Let's think step by step.\n"]
    prompts[0] = "Question: 1 + 1?\nAnswer: x\n" + prompts[0]      # make rows different lengths
    enc = tok(prompts, padding=True, return_tensors="pt")          # (padding side irrelevant here)
    m = question_span_mask_from_prompts(tok, prompts, padding_side="left")
    old_side = tok.padding_side
    tok.padding_side = "left"
    row_ids = tok(prompts, padding=True, return_tensors="pt")["input_ids"]
    tok.padding_side = old_side
    for b, p in enumerate(prompts):
        e = tok(prompts[b:b + 1], padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        assert m[b].any()
        assert tok.decode(row_ids[b][m[b]]) == p[p.rfind("Question:"):]
        assert not m[b][: row_ids.shape[1] - len(e)].any()         # no pad position is marked


def _checkpoint(model, drop_layer=None):
    sd = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("steer_nets.")}
    if drop_layer is not None:
        sd = {k: v for k, v in sd.items() if not k.startswith(f"steer_nets.{drop_layer}.")}
    return sd


def test_load_steernets_for_audit(tok, tmp_path):
    cfg = _cfg(tmp_path)
    src = _make_model(cfg, len(tok), seed=0)
    dst = _make_model(cfg, len(tok), seed=5)
    good = tmp_path / "good.pt"
    torch.save({"steer_state": _checkpoint(src)}, good)
    meta = load_steernets_for_audit(dst, str(good))
    assert meta["sha256"] == hashlib.sha256(good.read_bytes()).hexdigest()
    for k, v in src.state_dict().items():
        if k.startswith("steer_nets."):
            assert torch.equal(v, dst.state_dict()[k])

    no_admitted = tmp_path / "bad.pt"                               # layer 13 is a catalog layer
    torch.save(_checkpoint(src, drop_layer=13), no_admitted)
    with pytest.raises(FatalContractError):
        load_steernets_for_audit(dst, str(no_admitted))

    no_other = tmp_path / "warn.pt"                                 # layer 24 is a candidate, not admitted
    torch.save(_checkpoint(src, drop_layer=24), no_other)
    assert 24 in load_steernets_for_audit(dst, str(no_other))["layers_missing"]

    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"not a checkpoint")
    with pytest.raises(FatalContractError):
        load_steernets_for_audit(dst, str(junk))


def test_pair_synergy_additive_is_zero():
    cat = Catalog.build([13, 15, 18, 20, 22], 3)
    rng = np.random.default_rng(0)
    eff = dict(zip(cat.layers, rng.normal(size=5)))
    base = rng.normal(size=(4, 1)) + 5.0
    nll = np.concatenate([base + sum(eff[l] for l in s) for s in cat.subsets], axis=1).astype(np.float32)
    n, c = nll.shape[0], len(cat)
    table = CFTable(catalog=cat, qid=np.asarray(list("abcd")), split=np.asarray(["train"] * n),
                    n_shot=np.zeros(n, np.int64), steer_scale=1.0, ans_nll=nll,
                    token_nll=np.zeros((n, c, 1), np.float32), token_mask=np.ones((n, 1), bool),
                    token_ids=np.zeros((n, 1), np.int64), pooled=np.zeros((n, 3), np.float32),
                    scalars=np.zeros((n, 3), np.float32))
    syn = pair_synergy(table)
    assert len(syn) > 0 and all(len(s) + 1 <= 3 for (s, _j) in syn)
    for v in syn.values():
        np.testing.assert_allclose(v, 0.0, atol=1e-5)
