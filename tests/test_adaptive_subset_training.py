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
    assert model.base_model.embedding.weight.grad is None
    assert saved["steer_nets"]["0.up_proj.weight"].abs().sum() > 0


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
    assert result["teacher_labels"]["rows"] == 2
    assert saved["route_catalog"][0] == []
    assert saved["alpha_values"] == [1.0, 1.5, 2.0]
    assert result["history"][0]["optimizer_steps"] == 2
