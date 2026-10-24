import os
import sys
import unittest

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
from src.models.wrapper import Phi2WithRSC
from src.routing.catalog import Catalog

VOCAB = 64
GEN_KW = dict(max_new_tokens=5, min_new_tokens=5, do_sample=False, pad_token_id=0, use_cache=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_model(gate_task_gradient: bool = False, policy: str = "catalog") -> Phi2WithRSC:
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
    cfg.router_policy = policy
    cfg.gate_task_gradient = gate_task_gradient
    model = Phi2WithRSC(base, cfg, device=torch.device("cpu")).eval()
    # Make steering visibly non-trivial.
    for sn in model.steer_nets.values():
        nn.init.normal_(sn.down1.weight, std=0.3)
        nn.init.normal_(sn.mid.weight, std=0.3)
        nn.init.normal_(sn.up.weight, std=0.3)
    model.set_catalog(Catalog.build(model.target_layers, 3))
    return model


def make_batch(T: int = 6, B: int = 2, pad: int = 2, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, VOCAB, (B, T), generator=g)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, :pad] = 0
    ids[0, :pad] = 0  # left padding on row 0
    return ids, mask


def class_of(model, route):
    return model._catalog.subsets.index(tuple(sorted(route)))


def bias_router(model, classes):
    """Force catalog_logits to pick `classes` (int for all rows, or a per-row list)."""
    n_cat = len(model._catalog.subsets)

    def fake(h, valid_mask, shot_bucket=None):
        B = h.shape[0]
        cls = [classes] * B if isinstance(classes, int) else list(classes)
        logits = torch.full((B, n_cat), -10.0, device=h.device)
        logits[torch.arange(B), torch.tensor(cls)] = 10.0
        return logits

    model.router.catalog_logits = fake


def fixed_probs_router(model, probs):
    logits = torch.log(torch.as_tensor(probs, dtype=torch.float32))

    def fake(h, valid_mask, shot_bucket=None):
        return logits.unsqueeze(0).expand(h.shape[0], -1).to(h.device).clone()

    model.router.catalog_logits = fake


def record_step_routes(model):
    rec = []
    handle = model.base_model.register_forward_hook(
        lambda m, a, o: rec.append([tuple(r) for r in model.current_route_tuples()])
    )
    return rec, handle


# ---------------------------------------------------------------------------
# Contract checks (local, self-contained equivalents)
# ---------------------------------------------------------------------------
def check_baseline_identity(tc, model, ids, mask):
    prev = model._routing_override
    model.set_routing_override("none")
    with torch.no_grad():
        off = model(input_ids=ids, attention_mask=mask).logits
        with model.hooks_disabled():
            ref = model(input_ids=ids, attention_mask=mask).logits
    tc.assertTrue(torch.allclose(off, ref, atol=1e-6))
    if prev is None:
        model.set_routing_override(None)
    else:
        model.set_routing_override(prev[0], prev[1])


def check_hook_noop(tc, model, ids, mask):
    bias_router(model, 0)  # empty class
    with torch.no_grad():
        got = model(input_ids=ids, attention_mask=mask).logits
        gen_got = model.generate(input_ids=ids, attention_mask=mask, **GEN_KW)
        with model.hooks_disabled():
            ref = model(input_ids=ids, attention_mask=mask).logits
            gen_ref = model.generate(input_ids=ids, attention_mask=mask, **GEN_KW)
    tc.assertTrue(torch.allclose(got, ref, atol=1e-6))
    tc.assertTrue(torch.equal(gen_got, gen_ref))


def check_route_plumbing(tc, model, ids, mask):
    L = model.target_layers[1]
    bias_router(model, class_of(model, (L,)))
    rec, handle = record_step_routes(model)
    try:
        with torch.no_grad():
            model.generate(input_ids=ids, attention_mask=mask, **GEN_KW)
    finally:
        handle.remove()
    tc.assertGreaterEqual(len(rec), 5)  # prefill + 4 decode steps
    for step_routes in rec:
        tc.assertEqual(step_routes, [(L,)] * ids.shape[0])


class TestWrapperCatalog(unittest.TestCase):
    def test_contracts_pass_with_catalog_policy(self):
        model = make_model()
        self.assertEqual(model._policy(), "catalog")
        ids, mask = make_batch(B=2, pad=2)
        check_baseline_identity(self, model, ids, mask)
        check_hook_noop(self, model, ids, mask)
        check_route_plumbing(self, model, ids, mask)

    def test_degenerate_router_equals_override(self):
        model = make_model()
        L = model.target_layers[1]
        ids, mask = make_batch()
        bias_router(model, class_of(model, (L,)))
        with torch.no_grad():
            logits_cat = model(input_ids=ids, attention_mask=mask).logits
            gen_cat = model.generate(input_ids=ids, attention_mask=mask, **GEN_KW)
            with model.hooks_disabled():
                logits_base = model(input_ids=ids, attention_mask=mask).logits

            model.set_routing_override("only", [L])
            logits_ov = model(input_ids=ids, attention_mask=mask).logits
            gen_ov = model.generate(input_ids=ids, attention_mask=mask, **GEN_KW)
        model.set_routing_override(None)

        self.assertTrue(torch.allclose(logits_cat, logits_ov, atol=1e-5))
        self.assertTrue(torch.equal(gen_cat, gen_ov))
        # sanity: steering actually changed something
        self.assertFalse(torch.allclose(logits_cat, logits_base, atol=1e-6))

    def test_predict_routes_matches_generate(self):
        model = make_model()
        L0, L1 = model.target_layers[0], model.target_layers[2]
        classes = [class_of(model, (L0,)), class_of(model, (L0, L1))]
        bias_router(model, classes)
        ids, mask = make_batch()

        seen_before = model.router.seen_forwards.clone()
        sel_before = model.router.selected_forwards.clone()
        preds = model.predict_routes(ids, mask)

        self.assertEqual([p["route"] for p in preds], [(L0,), tuple(sorted((L0, L1)))])
        self.assertEqual([p["class_idx"] for p in preds], classes)
        for p in preds:
            self.assertLess(p["p_empty"], 1e-3)
        # diagnostics untouched, caches cleared
        self.assertTrue(torch.equal(model.router.seen_forwards, seen_before))
        self.assertTrue(torch.equal(model.router.selected_forwards, sel_before))
        self.assertIsNone(model._cached_decode_routes)
        self.assertIsNone(model._cached_decode_active_k)

        rec, handle = record_step_routes(model)
        try:
            with torch.no_grad():
                model.generate(input_ids=ids, attention_mask=mask, **GEN_KW)
        finally:
            handle.remove()
        expected = [p["route"] for p in preds]
        self.assertGreaterEqual(len(rec), 5)
        for step_routes in rec:
            self.assertEqual(step_routes, expected)

    def test_abstain_threshold_behavior(self):
        model = make_model()
        cat = model._catalog
        n_cat = len(cat.subsets)
        # NOTE: p0 > 0.5 forces argmax == 0, so "p0 = 0.6 with argmax != 0" is impossible.
        # Use p0 = 0.4 < p1 = 0.5 so argmax = class 1 while the empty class is still sizeable.
        probs = torch.zeros(n_cat)
        probs[0], probs[1] = 0.4, 0.5
        probs[2:] = 0.1 / (n_cat - 2)
        fixed_probs_router(model, probs)
        ids, mask = make_batch()
        argmax_route = tuple(sorted(cat.subsets[1]))

        model.set_abstain_threshold(1.01)  # disabled
        r = model.predict_routes(ids, mask)
        self.assertTrue(all(x["route"] == argmax_route and x["class_idx"] == 1 for x in r))
        self.assertTrue(all(abs(x["p_empty"] - 0.4) < 1e-5 for x in r))

        model.set_abstain_threshold(0.35)  # p0 >= theta -> abstain
        r = model.predict_routes(ids, mask)
        self.assertTrue(all(x["route"] == () and x["class_idx"] == 0 for x in r))

        model.set_abstain_threshold(0.45)  # p0 < theta -> keep argmax
        r = model.predict_routes(ids, mask)
        self.assertTrue(all(x["route"] == argmax_route for x in r))

    def test_gate_task_gradient_switch(self):
        ids, mask = make_batch(pad=0)
        labels = ids.clone()
        labels[mask == 0] = -100

        for flag in (True, False):
            model = make_model(gate_task_gradient=flag)
            L = model.target_layers[1]
            model.set_routing_override("only", [L])
            model.zero_grad(set_to_none=True)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            out.loss.backward()
            gate = model.gates[str(L)]
            if flag:
                self.assertIsNotNone(gate.fc2.weight.grad)
                self.assertGreater(float(gate.fc2.weight.grad.abs().sum()), 0.0)
            else:
                self.assertIsNone(gate.fc2.weight.grad)
            # SteerNet always receives task-loss gradient
            self.assertIsNotNone(model.steer_nets[str(L)].up.weight.grad)

    def test_guard_state_tripped(self):
        model = make_model()
        n = len(model.target_layers)
        L_pos = 1
        ids, mask = make_batch()

        # Row 0 steered with one layer, row 1 unsteered (K = 0).
        matrix = torch.zeros(2, n, dtype=torch.bool)
        matrix[0, L_pos] = True
        model.set_routing_override("matrix", matrix)
        model.set_guards(True, rel_norm_cap=1e-5, k_max=3)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
        tripped = model.guard_state["tripped"]
        self.assertTrue(bool(tripped[0]))
        self.assertFalse(bool(tripped[1]))

        # Huge cap: nothing trips, and nothing raises.
        model.set_guards(True, rel_norm_cap=1e9, k_max=3)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
        self.assertFalse(bool(model.guard_state["tripped"].any()))

        # K > k_max trips; guards disabled still records without raising.
        matrix2 = torch.zeros(2, n, dtype=torch.bool)
        matrix2[0, 0] = matrix2[0, 1] = True
        model.set_routing_override("matrix", matrix2)
        model.set_guards(False, rel_norm_cap=1e9, k_max=1)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
        tripped = model.guard_state["tripped"]
        self.assertTrue(bool(tripped[0]))
        self.assertFalse(bool(tripped[1]))

    def test_question_span_plumbing(self):
        model = make_model()
        bias_router(model, 0)
        q_len = 4
        captured, shots = [], []
        n_cat = len(model._catalog.subsets)

        def fake(h, valid_mask, shot_bucket=None):
            captured.append(valid_mask.clone())
            shots.append(None if shot_bucket is None else shot_bucket.clone())
            logits = torch.full((h.shape[0], n_cat), -10.0)
            logits[:, 0] = 10.0
            return logits

        model.router.catalog_logits = fake
        model.set_shot_bucket(2)

        for exemplar_len in (2, 10):
            T = exemplar_len + q_len
            ids = torch.randint(3, VOCAB, (2, T))
            mask = torch.ones(2, T, dtype=torch.long)
            span = torch.zeros(2, T, dtype=torch.bool)
            span[:, -q_len:] = True

            model.set_question_span_mask(span)
            self.assertIs(model._router_context_mask_pending, span)
            with torch.no_grad():
                model(input_ids=ids, attention_mask=mask)
            self.assertIsNone(model._router_context_mask_pending)
            self.assertIs(model._router_context_mask_active, span)

            seen = captured[-1]
            self.assertTrue(torch.equal(seen, span))
            # question-token count is independent of few-shot exemplar length
            self.assertTrue(torch.equal(seen.sum(dim=1), torch.full((2,), q_len)))
            self.assertTrue(torch.equal(shots[-1], torch.full((2,), 2, dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
