import json
import copy
from types import SimpleNamespace

import torch
from torch import nn

from src.steering.steernet import SteerNetV2
from src.training.adaptive_subset_training import train_adaptive_steering_bank


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(5)])
        self.lm_head = nn.Linear(8, 16)
        self.config = SimpleNamespace(hidden_size=8)

    def forward(self, input_ids):
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
        return self.lm_head(hidden)


class _TinySteeringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = _TinyBackbone()
        self.target_layers = (0, 1, 2, 3)
        self._layers = self.base_model.layers
        self.steer_nets = nn.ModuleDict(
            {str(i): SteerNetV2(8, rank=2, layer_norm_ref=2.0) for i in self.target_layers}
        )
        self.route = None
        self.route_gains = None
        self.shield = None
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

    def set_routing_override(self, mode, layers):
        self.route = torch.as_tensor(layers, dtype=torch.bool) if mode == "matrix" else None

    def set_route_gains(self, gains):
        self.route_gains = gains

    def set_prompt_shield_mask(self, mask):
        self.shield = mask

    def forward(self, input_ids, attention_mask, use_cache=False):
        hidden = self.base_model.embedding(input_ids)
        for index, layer in enumerate(self.base_model.layers):
            hidden = torch.tanh(layer(hidden))
            if index < len(self.target_layers) and self.route is not None and self.route[0, index]:
                delta = self.steer_nets[str(index)](
                    hidden,
                    is_decode=True,
                    gain=self.route_gains,
                    injection_mask=~self.shield if self.shield is not None else None,
                )
                hidden = hidden + delta
        return SimpleNamespace(logits=self.base_model.lm_head(hidden))


def test_adaptive_bank_trains_variable_subsets_and_saves_route_metadata(tmp_path):
    model = _TinySteeringModel()
    cfg = SimpleNamespace(phase1_lr=1.5e-4, weight_decay=0.0, clip_grad=1.0, rho_max=0.10,
                          lora_rank=2, steernet_norm_stat="p99_norm")
    rows = []
    for offset in range(4):
        rows.append({
            "input_ids": torch.tensor([1, 2, 3, 4 + offset]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor([-100, -100, 3, 4 + offset]),
            "prompt_shield_mask": torch.tensor([True, True, False, False]),
        })

    checkpoint = tmp_path / "adaptive_bank.pt"
    result = train_adaptive_steering_bank(
        model, rows, rows[:2], cfg, torch.device("cpu"), epochs=1, save_path=checkpoint,
        seed=3, grad_accum=2, route_examples_per_layer=2,
        screen_validation_rows=2, max_pair_routes=2, max_triple_routes=1,
        max_quad_routes=1,
    )

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert result["candidate_layers"] == [0, 1, 2, 3]
    assert saved["format"] == "ars_adaptive_steering_v1"
    assert saved["frozen_backbone"] is True
    assert saved["curriculum"]["catalog"][0] == []
    assert "singletons" in saved["curriculum"]["stages"]
    assert 0.0 < saved["curriculum"]["k0_validation_nll"] < 10.0
    assert saved["curriculum"]["stages"]["pairs"]["candidates"]
    assert model.base_model.embedding.weight.grad is None
    assert saved["steer_nets"]["0.up_proj.weight"].abs().sum() > 0


def test_memory_safe_preference_backward_matches_joint_pairwise_gradient():
    from src.training.adaptive_subset_training import (
        _backward_pairwise_preference,
        _nll_sum,
    )

    positive = {
        "question_index": 7,
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "attention_mask": torch.ones(4, dtype=torch.long),
        "labels": torch.tensor([-100, -100, 3, 4]),
        "prompt_shield_mask": torch.tensor([True, True, False, False]),
    }
    negative = {
        **positive,
        "input_ids": torch.tensor([1, 2, 5, 6]),
        "labels": torch.tensor([-100, -100, 5, 6]),
    }
    pair = {"positive": positive, "negative": negative}
    initial = _TinySteeringModel()

    def configure(model):
        model.set_routing_override("matrix", [[True, False, False, False]])
        model.set_route_gains(torch.ones(1))
        model.eval()

    joint_model = copy.deepcopy(initial)
    split_model = copy.deepcopy(initial)
    configure(joint_model)
    configure(split_model)

    positive_row = {
        key: value.unsqueeze(0) for key, value in positive.items()
        if isinstance(value, torch.Tensor) and key != "question_index"
    }
    negative_row = {
        key: value.unsqueeze(0) for key, value in negative.items()
        if isinstance(value, torch.Tensor) and key != "question_index"
    }
    joint_model.set_prompt_shield_mask(positive_row["prompt_shield_mask"])
    positive_logits = joint_model(
        positive_row["input_ids"], positive_row["attention_mask"]
    ).logits
    positive_nll = _nll_sum(positive_logits, positive_row["labels"]) / 2
    joint_model.set_prompt_shield_mask(negative_row["prompt_shield_mask"])
    negative_logits = joint_model(
        negative_row["input_ids"], negative_row["attention_mask"]
    ).logits
    negative_nll = _nll_sum(negative_logits, negative_row["labels"]) / 2
    (0.125 * torch.nn.functional.softplus(0.2 + positive_nll - negative_nll)).backward()

    scaler = torch.amp.GradScaler("cuda", enabled=False)
    _backward_pairwise_preference(
        split_model, pair, torch.device("cpu"), scaler, weight=0.125
    )

    joint_parameters = dict(joint_model.named_parameters())
    split_parameters = dict(split_model.named_parameters())
    assert joint_parameters.keys() == split_parameters.keys()
    for name in joint_parameters:
        expected = joint_parameters[name].grad
        actual = split_parameters[name].grad
        assert (expected is None) == (actual is None), name
        if expected is not None:
            assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5), name


def test_adaptive_router_policy_trains_and_saves_probe_contract(tmp_path):
    from src.training.adaptive_router_training import train_adaptive_router

    model = _TinySteeringModel()
    row = {
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "attention_mask": torch.ones(4, dtype=torch.long),
        "labels": torch.tensor([-100, -100, 3, 4]),
        "prompt_shield_mask": torch.tensor([True, True, False, False]),
    }
    checkpoint = tmp_path / "adaptive_router.pt"

    result = train_adaptive_router(
        model,
        [row, row],
        [row],
        candidate_layers=[0, 1, 2, 3],
        device=torch.device("cpu"),
        epochs=1,
        save_path=checkpoint,
        probe_layer=4,
        max_train_rows=2,
        validation_limit=1,
    )

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert result["history"][0]["optimizer_steps"] > 0
    assert saved["format"] == "ars_adaptive_router_v1"
    assert saved["candidate_layers"] == [0, 1, 2, 3]
    assert saved["probe_layer"] == 4


def test_adaptive_router_can_learn_from_route_catalog_oracle_labels(tmp_path):
    from src.training.adaptive_router_training import train_adaptive_router

    model = _TinySteeringModel()
    row = {
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "attention_mask": torch.ones(4, dtype=torch.long),
        "labels": torch.tensor([-100, -100, 3, 4]),
        "prompt_shield_mask": torch.tensor([True, True, False, False]),
    }
    checkpoint = tmp_path / "catalog_router.pt"
    result = train_adaptive_router(
        model,
        [row, row],
        [row],
        candidate_layers=[0, 1, 2, 3],
        device=torch.device("cpu"),
        epochs=1,
        save_path=checkpoint,
        probe_layer=4,
        max_train_rows=2,
        validation_limit=1,
        route_catalog=[[], [0], [1], [0, 1]],
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    report_path = tmp_path / "session3_router_training.json"
    assert report_path.is_file()
    training_report = json.loads(report_path.read_text(encoding="utf-8"))
    teacher_targets = torch.load(
        tmp_path / "session3_route_teacher_targets.pt", map_location="cpu", weights_only=True
    )
    assert result["teacher_labels"]["rows"] == 2
    assert saved["route_catalog"][0] == []
    assert saved["router_type"] == "catalog_value"
    assert saved["format"] == "ars_catalog_router_v2"
    assert saved["alpha_values"] == [1.0, 1.5, 2.0]
    assert teacher_targets["action_scores"].shape == (2, len(saved["action_catalog"]))
    assert training_report["teacher_data"]["rows"] == 2
    assert training_report["route_cost"] == 0.01
    assert training_report["action_count"] == len(saved["action_catalog"])
    assert result["history"][0]["optimizer_steps"] == 1

    from src.training.adaptive_router_training import load_adaptive_router

    loaded, prefill, probe = load_adaptive_router(model, checkpoint, torch.device("cpu"))
    assert loaded.action_catalog == tuple(
        (tuple(action["route"]), action["alpha"]) for action in saved["action_catalog"]
    )
    prefill(row["input_ids"].unsqueeze(0), row["attention_mask"].unsqueeze(0))
    assert model.route is not None and model.route.shape == (1, len(model.target_layers))
    probe.close()


def test_router_context_uses_question_token_count_not_full_prompt_length():
    from src.training.adaptive_router_training import _router_context_features

    features = _router_context_features(
        {
            "effective_shots": 2,
            "prompt_signature": (1, 2, 3, 4, 5, 6),
            "router_feature_mask": (False, True, True, False, True, False),
        },
        "cpu",
    )
    assert torch.allclose(features, torch.tensor([[0.5, 3.0 / 256.0]]))


def test_static_bank_route_mask_has_one_entry_per_candidate_layer(monkeypatch):
    import src.training.adaptive_router_training as training

    model = _TinySteeringModel()
    observed_masks = []

    monkeypatch.setattr(
        training,
        "_prepare",
        lambda items, device: ({"placeholder": torch.tensor([1])}, {"prompt_shield_mask": None}),
    )

    def capture_score(_model, _row, route_mask, _alpha, _device, _prompt_shield):
        observed_masks.append(route_mask.detach().cpu().clone())
        return torch.tensor([1.0])

    monkeypatch.setattr(training, "_score_action", capture_score)
    training.select_static_bank_action(
        model,
        [{"question_index": 1}],
        [0, 1, 2, 3],
        [[], [2]],
        torch.device("cpu"),
        max_questions=1,
    )
    assert any(torch.equal(mask, torch.tensor([[False, False, True, False]])) for mask in observed_masks)


def test_static_action_uses_disjoint_questions_for_screening_and_greedy_selection(monkeypatch):
    import src.training.adaptive_router_training as training

    model = _TinySteeringModel()
    screened_questions = []
    greedy_prompts = []

    monkeypatch.setattr(
        training,
        "_prepare",
        lambda items, _device: (
            {"question_id": items[0]["question_index"]},
            {"prompt_shield_mask": None},
        ),
    )

    def score_action(_model, row, _route, _alpha, _device, _shield):
        screened_questions.append(row["question_id"])
        return torch.tensor([1.0])

    monkeypatch.setattr(training, "_score_action", score_action)
    monkeypatch.setattr(
        training,
        "build_prompts",
        lambda _tokenizer, questions, _ids, shots, **_kwargs: (
            [f"{question}|{shots}" for question in questions],
            [None] * len(questions),
        ),
    )

    def generate(_model, _tokenizer, prompts, _device, **_kwargs):
        greedy_prompts.extend(prompts)
        return [SimpleNamespace(text=prompt) for prompt in prompts]

    monkeypatch.setattr(training, "generate_batch", generate)
    monkeypatch.setattr(training, "score_tiers", lambda *_args: {"lenient": True})
    rows = [
        {"question_index": index, "question": f"Q{index}", "gold_answer": "1"}
        for index in range(6)
    ]
    result = training.select_static_bank_action(
        model,
        rows,
        [0, 1, 2, 3],
        [[], [0], [1]],
        torch.device("cpu"),
        alpha_values=(1.0,),
        max_questions=6,
        tokenizer=object(),
        cfg=SimpleNamespace(shield_policy="prompt"),
        generation_shortlist=2,
    )
    screen_ids = set(result["screen_validation_question_ids"])
    greedy_ids = {int(prompt.split("|", 1)[0][1:]) for prompt in greedy_prompts}
    assert screen_ids.isdisjoint(greedy_ids)
    assert result["independent_static_validation_sets"] is True
    assert result["greedy_validation_questions"] == 3
    assert set(screened_questions) == screen_ids


def test_static_action_broadcasts_fixed_route_for_greedy_generation_batch(monkeypatch):
    import src.training.adaptive_router_training as training
    from src.models.wrapper import Phi2WithRSC

    class RouteMatrixProbe:
        target_layers = (0, 1, 2, 3)

        def __init__(self):
            self._routing_override = ("none", None)
            self._route_matrix = None
            self._route_gains = None

        def set_routing_override(self, mode, layers):
            if mode == "matrix":
                self._route_matrix = torch.as_tensor(layers, dtype=torch.bool)
                self._routing_override = (mode, None)
            elif mode == "only":
                self._route_matrix = None
                self._routing_override = (mode, set(layers or []))
            else:
                self._route_matrix = None
                self._routing_override = (mode, set(layers or []))

        def set_route_gains(self, gains):
            self._route_gains = gains

        def set_prompt_shield_mask(self, _mask):
            pass

    model = RouteMatrixProbe()
    monkeypatch.setattr(
        training,
        "_prepare",
        lambda items, _device: (
            {"question_id": items[0]["question_index"]},
            {"prompt_shield_mask": None},
        ),
    )
    monkeypatch.setattr(training, "_score_action", lambda *_args: torch.tensor([1.0]))
    monkeypatch.setattr(
        training,
        "build_prompts",
        lambda _tokenizer, questions, _ids, shots, **_kwargs: (
            [f"{question}|{shots}" for question in questions],
            [None] * len(questions),
        ),
    )
    generation_batch_sizes = []

    def generate(_model, _tokenizer, prompts, _device, **_kwargs):
        batch_route = Phi2WithRSC._select_route(
            _model, len(prompts), torch.device("cpu")
        )
        assert batch_route.shape == (len(prompts), len(_model.target_layers))
        assert torch.equal(batch_route, batch_route[:1].expand_as(batch_route))
        batch_gains = Phi2WithRSC._route_gain_for(
            _model, torch.zeros(len(prompts), 2, 4)
        )
        assert batch_gains is None or batch_gains.ndim == 0
        generation_batch_sizes.append(len(prompts))
        return [SimpleNamespace(text=prompt) for prompt in prompts]

    monkeypatch.setattr(training, "generate_batch", generate)
    monkeypatch.setattr(training, "score_tiers", lambda *_args: {"lenient": True})
    rows = [
        {"question_index": index, "question": f"Q{index}", "gold_answer": "1"}
        for index in range(6)
    ]

    training.select_static_bank_action(
        model,
        rows,
        [0, 1, 2, 3],
        [[], [0], [1]],
        torch.device("cpu"),
        alpha_values=(1.0,),
        max_questions=6,
        tokenizer=object(),
        cfg=SimpleNamespace(shield_policy="prompt"),
        generation_shortlist=2,
    )

    assert generation_batch_sizes
    assert all(size == 3 for size in generation_batch_sizes)


def test_evenly_spread_sample_covers_sequence_endpoints():
    from src.training.adaptive_data import spread_items

    assert spread_items(list(range(10)), 4) == [0, 3, 6, 9]


def test_catalog_router_checkpoint_selection_uses_task_accuracy_when_available(tmp_path, monkeypatch):
    import src.training.adaptive_router_training as training

    model = _TinySteeringModel()
    positive = {
        "question_index": 0,
        "completion_index": 0,
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "attention_mask": torch.ones(4, dtype=torch.long),
        "labels": torch.tensor([-100, -100, 3, 4]),
        "prompt_shield_mask": torch.tensor([True, True, False, False]),
    }
    negative = {
        **positive,
        "completion_index": 1,
        "input_ids": torch.tensor([1, 2, 4, 5]),
        "labels": torch.tensor([-100, -100, 4, 5]),
    }
    pairs = [{"positive": positive, "negative": negative}]
    observed_validation_ids = []
    observed_static_action = []

    def evaluate_task(_model, _router, rows, *_args, **kwargs):
        observed_validation_ids.extend(int(row["question_index"]) for row in rows)
        observed_static_action.append(kwargs.get("static_action"))
        return {
            "validation_task_accuracy_by_shot": {
                "0": {"paired_delta_pp": 4.0}, "2": {"paired_delta_pp": 2.0}
            },
            "validation_task_selection_delta_pp": 2.6,
            "validation_task_selection_reference": "static_bank",
            "validation_task_selection_method": "0.3*router_vs_static_bank_0shot + 0.7*router_vs_static_bank_2shot",
            "validation_task_questions": len(rows),
        }

    monkeypatch.setattr(training, "_evaluate_router_task_accuracy", evaluate_task)
    checkpoint = tmp_path / "task_selected_router.pt"
    other_validation = {**positive, "question_index": 1}
    result = training.train_adaptive_router(
        model,
        [positive],
        [positive, other_validation],
        candidate_layers=[0, 1, 2, 3],
        device=torch.device("cpu"),
        epochs=1,
        save_path=checkpoint,
        probe_layer=4,
        max_train_rows=1,
        validation_limit=1,
        route_catalog=[[], [0], [1], [0, 1]],
        preference_pairs=pairs,
        validation_preference_pairs=pairs,
        tokenizer=object(),
        cfg=object(),
        static_action={"route": [0], "alpha": 1.5},
    )
    assert result["best_checkpoint_selection_metric"] == (
        "validation_task_accuracy_delta: 0.3*router_vs_static_bank_0shot + 0.7*router_vs_static_bank_2shot"
    )
    assert result["best_checkpoint_selection_value"] == 2.6
    assert observed_validation_ids == [0, 1]
    assert observed_static_action == [{"route": [0], "alpha": 1.5}]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["checkpoint_selection_metric"] == (
        "validation_task_accuracy_delta: 0.3*router_vs_static_bank_0shot + 0.7*router_vs_static_bank_2shot"
    )
    assert saved["checkpoint_selection_value"] == 2.6
