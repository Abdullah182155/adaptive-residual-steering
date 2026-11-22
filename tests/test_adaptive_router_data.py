import json
from types import SimpleNamespace

import torch


def test_weighted_shot_objective_prioritizes_zero_shot_and_reports_active_shots():
    from src.training.adaptive_data import weighted_shot_average

    value, expression = weighted_shot_average({0: 0.5, 2: 1.0, 4: 0.0}, "accuracy")
    assert value == 0.55
    assert expression == "0.7*accuracy_0shot + 0.2*accuracy_2shot + 0.1*accuracy_4shot"

    pilot_value, pilot_expression = weighted_shot_average({0: 0.5, 2: 1.0}, "accuracy")
    assert abs(pilot_value - (0.7 * 0.5 + 0.2 * 1.0) / 0.9) < 1e-12
    assert pilot_expression == "0.777778*accuracy_0shot + 0.222222*accuracy_2shot"

    single_value, single_expression = weighted_shot_average({0: 0.5}, "accuracy")
    assert single_value == 0.5
    assert single_expression == "1*accuracy_0shot"


def test_weighted_shot_objective_rejects_unconfigured_shot_counts():
    from src.training.adaptive_data import weighted_shot_average

    try:
        weighted_shot_average({1: 0.5}, "accuracy")
    except ValueError as exc:
        assert "no task-selection weight" in str(exc)
    else:
        raise AssertionError("an unconfigured shot count must not silently receive a default weight")


def test_preference_pair_shrinks_to_one_shared_prompt_when_lengths_differ(tmp_path, monkeypatch):
    import src.training.adaptive_data as adaptive_data

    rollout = tmp_path / "shard.jsonl"
    rollout.write_text(
        json.dumps({
            "question_index": 3,
            "question": "What is one plus one?",
            "final_num": "2",
            "samples": [
                {"text": "verified positive", "strict_ok": True},
                {"text": "well-formed wrong", "strict_ok": False},
            ],
        }) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        adaptive_data,
        "score_tiers",
        lambda text, _gold: (
            {"format_ok": True, "lenient": True}
            if text == "verified positive"
            else {"format_ok": True, "lenient": False}
        ),
    )
    class FixedRandom:
        def __init__(self, *_args):
            pass

        def choices(self, *_args, **_kwargs):
            return [4]

        def shuffle(self, _items):
            pass

    monkeypatch.setattr(adaptive_data.random, "Random", FixedRandom)
    tokenized_shots = []

    def tokenize(_tokenizer, question, completion, _cfg, *, question_index,
                 completion_index, n_shot, template_id, max_length):
        tokenized_shots.append((completion, n_shot))
        effective = 2 if completion == "well-formed wrong" and n_shot == 4 else n_shot
        return {
            "question_index": question_index,
            "completion_index": completion_index,
            "effective_shots": effective,
            "template_id": template_id,
            "prompt_signature": (question_index, effective, template_id),
            "question": question,
            "gold_answer": "2",
        }

    monkeypatch.setattr(adaptive_data, "tokenize_rft_example", tokenize)
    pairs, stats = adaptive_data.build_adaptive_preference_pairs(
        [rollout], object(), SimpleNamespace(max_tok_len=64), [3], seed=1
    )

    assert len(pairs) == 1
    assert pairs[0]["positive"]["effective_shots"] == pairs[0]["negative"]["effective_shots"]
    assert pairs[0]["positive"]["prompt_signature"] == pairs[0]["negative"]["prompt_signature"]
    assert stats["pairs_retokenized_to_common_shots"] == 1
    assert stats["pairs_skipped_prompt_mismatch"] == 0
    assert ("verified positive", 2) in tokenized_shots
    assert ("well-formed wrong", 2) in tokenized_shots


def test_oracle_teacher_keeps_questions_without_preference_pairs(monkeypatch):
    import src.training.adaptive_router_training as training

    class RouterStub:
        candidate_layers = (0, 1, 2, 3)
        action_catalog = (((), 1.0), ((0,), 1.0))

        @staticmethod
        def raw_prompt_pool(hidden, prompt_mask):
            return hidden * prompt_mask.unsqueeze(-1)

    class ModelStub:
        def set_routing_override(self, *_args):
            pass

        def set_route_gains(self, *_args):
            pass

        def set_prompt_shield_mask(self, *_args):
            pass

    rows = [
        {"question_index": question_id, "prompt_signature": (question_id, 4),
         "input_ids": torch.tensor([1, 2, 3, 4])}
        for question_id in (10, 20)
    ]
    pair = {
        "positive": {**rows[0], "completion_index": 0},
        "negative": {**rows[0], "completion_index": 1},
    }
    monkeypatch.setattr(training, "_prepare", lambda items, _device: (items[0], {}))
    monkeypatch.setattr(
        training,
        "_capture_prompt_hidden",
        lambda *_args: (torch.ones(1, 2, 4), torch.ones(1, 2, dtype=torch.bool)),
    )
    monkeypatch.setattr(
        training,
        "_score_action",
        lambda _model, _row, route, alpha, _device, _shield, **_kwargs: route.float().sum(dim=1) + alpha,
    )

    targets, stats = training._build_oracle_targets(
        ModelStub(), rows, [[], [0]], RouterStub(), object(), torch.device("cpu"),
        route_cost=0.01, row_limit=2, preference_pairs=[pair],
    )

    assert stats["questions"] == 2
    assert stats["preference_target_rows"] == 1
    assert stats["positive_nll_target_rows"] == 1
    assert {target["question_index"] for target in targets} == {10, 20}


def test_router_task_validation_resets_baseline_after_prior_router_decision(monkeypatch):
    import src.training.adaptive_router_training as training

    class ModelStub:
        route_state = "none"

        def set_routing_override(self, mode, _layers=None):
            self.route_state = mode

        def set_route_gains(self, _gains):
            pass

        def set_prompt_shield_mask(self, _mask):
            pass

    class RouterStub:
        candidate_layers = (0, 1, 2, 3)

    model = ModelStub()
    route_sequence = []
    monkeypatch.setattr(
        training,
        "build_prompts",
        lambda _tokenizer, questions, *_args, **_kwargs: (list(questions), [None] * len(questions)),
    )
    monkeypatch.setattr(
        training,
        "build_question_feature_masks",
        lambda _tokenizer, prompts, _questions: [torch.ones(1, dtype=torch.bool) for _ in prompts],
    )

    def prefill_factory(_model, _router, _probe):
        def prefill(*_args):
            model.route_state = "router"
        return prefill

    monkeypatch.setattr(training, "_build_router_prefill", prefill_factory)

    def generate(_model, _tokenizer, prompts, _device, **kwargs):
        if kwargs.get("route_prefill_fn") is not None:
            kwargs["route_prefill_fn"](None, None, None, None)
        route_sequence.append(model.route_state)
        return [SimpleNamespace(text="answer") for _ in prompts]

    monkeypatch.setattr(training, "generate_batch", generate)
    monkeypatch.setattr(training, "score_tiers", lambda *_args: {"lenient": True})
    rows = [
        {"question_index": 1, "question": "Q1", "gold_answer": "1"},
        {"question_index": 2, "question": "Q2", "gold_answer": "2"},
    ]
    result = training._evaluate_router_task_accuracy(
        model,
        RouterStub(),
        rows,
        object(),
        object(),
        SimpleNamespace(shield_policy="prompt", generation_max_new_tokens=2,
                        generation_temperature=0.0, generation_repetition_penalty=1.0,
                        generation_no_repeat_ngram=0, max_tok_len=8),
        torch.device("cpu"),
        limit=2,
        shots=(0,),
        static_action={"route": [1], "alpha": 1.5},
    )

    assert route_sequence == ["none", "matrix", "router", "none", "matrix", "router"]
    assert result["validation_task_selection_reference"] == "static_bank"
    assert result["validation_task_accuracy_by_shot"]["0"]["static_bank_accuracy"] == 1.0
    assert result["validation_task_selection_method"] == "1*router_vs_static_bank_0shot"
