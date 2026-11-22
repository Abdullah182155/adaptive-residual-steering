"""Training utilities for a frozen-backbone, variable-subset steering bank."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.training.route_curriculum import (
    build_router_route_catalog,
    candidate_routes,
    select_routes_by_depth,
)
from src.training.adaptive_data import spread_items
from src.training.rft_trainer import _enable_checkpointing_without_dropout
from src.training.batching import prepare_batch as _prepare
from src.steering.route_bank import route_key


def _nll_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
        reduction="sum",
    )


def _backward_pairwise_preference(model, pair, device, scaler, weight: float) -> None:
    """Backpropagate both preference sides without retaining two graphs at once.

    The detached negative score gives the positive-side derivative. The negative
    side is then recomputed and backpropagated with the exact detached coefficient
    ``-sigmoid(margin)``. This is algebraically the same gradient as a joint
    ``softplus(margin)`` graph, while keeping only one backward graph alive.
    """
    positive_row, positive_extras = _prepare([pair["positive"]], device)
    negative_row, negative_extras = _prepare([pair["negative"]], device)
    model.set_prompt_shield_mask(negative_extras.get("prompt_shield_mask"))
    with torch.no_grad(), torch.autocast(
        device_type=device.type, enabled=device.type == "cuda"
    ):
        negative_logits = model(
            input_ids=negative_row["input_ids"],
            attention_mask=negative_row["attention_mask"],
            use_cache=False,
        ).logits
        negative_nll_sum = _nll_sum(negative_logits, negative_row["labels"])
        negative_token_count = negative_row["labels"][:, 1:].ne(-100).sum().clamp_min(1)
        negative_nll = negative_nll_sum / negative_token_count
    del negative_logits, negative_nll_sum

    model.set_prompt_shield_mask(positive_extras.get("prompt_shield_mask"))
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        positive_logits = model(
            input_ids=positive_row["input_ids"],
            attention_mask=positive_row["attention_mask"],
            use_cache=False,
        ).logits
        positive_nll_sum = _nll_sum(positive_logits, positive_row["labels"])
        positive_token_count = positive_row["labels"][:, 1:].ne(-100).sum().clamp_min(1)
        positive_nll = positive_nll_sum / positive_token_count
        positive_loss = weight * F.softplus(0.2 + positive_nll - negative_nll)
    scaler.scale(positive_loss).backward()
    negative_coefficient = torch.sigmoid(
        0.2 + positive_nll.detach() - negative_nll
    ).detach()
    del positive_logits, positive_nll_sum, positive_nll, positive_loss
    del negative_nll, positive_row, positive_extras

    model.set_prompt_shield_mask(negative_extras.get("prompt_shield_mask"))
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        negative_logits = model(
            input_ids=negative_row["input_ids"],
            attention_mask=negative_row["attention_mask"],
            use_cache=False,
        ).logits
        negative_nll_sum = _nll_sum(negative_logits, negative_row["labels"])
        negative_token_count = negative_row["labels"][:, 1:].ne(-100).sum().clamp_min(1)
        negative_nll = negative_nll_sum / negative_token_count
        negative_loss = -weight * negative_coefficient * negative_nll
    scaler.scale(negative_loss).backward()
    del negative_logits, negative_nll_sum, negative_nll, negative_loss
    del negative_row, negative_extras, negative_coefficient


def _write_bank_checkpoint(
    model, cfg, candidate_layers, route_catalog, path: str | Path, epoch: int,
    curriculum: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    normalized_catalog = [list(route) for route in route_catalog]
    route_parameters = [
        parameter
        for route in normalized_catalog if route
        for parameter in model.bank_experts[route_key(route)].parameters()
    ]
    parameter_ids = [id(parameter) for parameter in route_parameters]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("route experts share trainable parameter objects")
    expert_states = {
        route_key(route): model.route_expert_state(route)
        for route in normalized_catalog if route
    }
    torch.save(
        {
            "format": "ars.independent_route_expert_bank",
            "format_version": 1,
            "training_complete": True,
            "architecture_id": "ars.independent_route_low_rank_residual_bank",
            "architecture_version": "1.0.0",
            "expert_architecture_id": "ars.fixed_route_low_rank_residual",
            "expert_architecture_version": "1.0.0",
            "candidate_layers": list(candidate_layers),
            "route_catalog": normalized_catalog,
            "route_experts": expert_states,
            "rank": int(cfg.lora_rank),
            "rho_max": float(cfg.rho_max),
            "norm_stat": str(cfg.steernet_norm_stat),
            "norm_refs": {
                str(layer): float(model.steer_nets[str(layer)].layer_norm_ref)
                for layer in candidate_layers
            },
            "epoch": int(epoch),
            "adapter_learning_rate": float(cfg.phase1_lr),
            "frozen_backbone": True,
            "adapter_initialization": "trained_singleton_expert_copy_then_route_specific_finetuning",
            "route_experts_independent": True,
            "route_expert_parameter_tensors": len(parameter_ids),
            "route_expert_parameter_count": sum(parameter.numel() for parameter in route_parameters),
            "curriculum": curriculum or {},
        },
        temporary,
    )
    temporary.replace(destination)


def train_adaptive_steering_bank(
    model,
    train_rows: Sequence[dict[str, torch.Tensor]],
    validation_rows: Sequence[dict[str, torch.Tensor]],
    cfg,
    device: torch.device | str,
    *,
    epochs: int,
    save_path: str | Path,
    seed: int = 42,
    grad_accum: int = 16,
    route_examples_per_expert: int = 256,
    screen_validation_rows: int = 24,
    max_pair_routes: int = 8,
    max_triple_routes: int = 6,
    max_quad_routes: int = 4,
    preference_pairs: Sequence[dict[str, Any]] = (),
    validation_preference_pairs: Sequence[dict[str, Any]] = (),
    preference_weight: float = 0.25,
    min_layer_gap: int = 3,
    control_gap: int = 2,
    spacing_control_limit: int = 8,
) -> dict[str, Any]:
    """Train independent route experts and create a compact diverse router bank.

    Singleton experts are trained first. Candidate pairs, triples, and quads
    are initialized from their strongest trained proper-subset experts, then
    receive route-local optimization. The frozen backbone and all other experts
    remain unchanged while one route is trained. The no-steering action is a
    control and has no trainable parameters.
    """
    if not train_rows or not validation_rows or epochs < 1 or grad_accum < 1:
        raise ValueError("training/validation rows, epochs, and grad_accum must be positive")
    if route_examples_per_expert < 1 or screen_validation_rows < 1:
        raise ValueError("route training and screening sample counts must be positive")
    if preference_weight < 0:
        raise ValueError("preference_weight must be nonnegative")
    if any(parameter.requires_grad for parameter in model.base_model.parameters()):
        raise ValueError("Phi-2 must remain frozen while training the steering bank")
    candidates = tuple(model.target_layers)
    if len(candidates) < 4:
        raise ValueError("adaptive steering requires at least four candidate layers")
    if abs(float(cfg.phase1_lr) - 1.5e-4) > 1e-12:
        raise ValueError("phase1_lr is locked at 1.5e-4")

    target_device = torch.device(device)
    preference_by_question = {
        int(pair["positive"]["question_index"]): pair for pair in preference_pairs
    }
    if target_device.type == "cuda" and hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    for parameter in model.steer_nets.parameters():
        parameter.requires_grad_(False)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=target_device.type == "cuda"
    )
    checkpointed_modules = _enable_checkpointing_without_dropout(model.base_model)
    if target_device.type == "cuda" and checkpointed_modules == 0:
        raise RuntimeError("CUDA adaptive training requires active gradient checkpointing")

    screen_by_question: dict[int, list[dict]] = {}
    for index, item in enumerate(validation_rows):
        question_id = int(item.get("question_index", index))
        screen_by_question.setdefault(question_id, []).append(item)
    screen_questions = spread_items(list(screen_by_question.items()), screen_validation_rows)
    screen_row_count = sum(len(question_rows) for _, question_rows in screen_questions)
    if not screen_questions:
        raise ValueError("route screening needs at least one validation row")
    curriculum: dict[str, Any] = {
        "depth_bands": {
            "early": [layer for layer in candidates if layer < 14],
            "middle": [layer for layer in candidates if 14 <= layer < 22],
            "late": [layer for layer in candidates if layer >= 22],
        },
        "stages": {},
        "training_orders": {},
        "k0_validation_nll": None,
        "route_spacing": {
            "minimum_layer_gap": int(min_layer_gap),
            "control_layer_gap": int(control_gap),
            "bounded_spacing_controls_per_stage": int(spacing_control_limit),
        },
        "preference_training": {
            "training_pairs": len(preference_pairs),
            "validation_pairs": len(validation_preference_pairs),
            "weight": float(preference_weight),
            "objective": "positive_token_nll + weighted softplus margin on matched correct/wrong completions",
            "adapter_initialization": "independent_singletons_then_best_proper_subset_copy",
        },
    }
    total_examples = 0
    route_selection_scores: dict[tuple[int, ...], float] = {}

    def initialize_route_experts(
        routes: Sequence[tuple[int, ...]], parent_routes: Sequence[tuple[int, ...]] = ()
    ) -> None:
        """Copy a route's best measured proper-subset experts into fresh modules."""
        parents = sorted(
            (tuple(parent) for parent in parent_routes if parent),
            key=lambda parent: route_selection_scores.get(parent, float("inf")),
        )
        for route in routes:
            route = tuple(route)
            if not route or route_key(route) in model.bank_experts:
                continue
            layer_states: dict[int, dict[str, torch.Tensor]] = {}
            compatible_parents = [
                parent for parent in parents
                if len(parent) < len(route) and set(parent).issubset(route)
                and route_key(parent) in model.bank_experts
            ]
            for layer in route:
                source_route = next(
                    (parent for parent in compatible_parents
                     if layer in parent and str(layer) in model.bank_experts[route_key(parent)]),
                    None,
                )
                if source_route is None:
                    singleton = (layer,)
                    if route_key(singleton) in model.bank_experts:
                        source_route = singleton
                if source_route is not None:
                    layer_states[layer] = {
                        name: value.detach().cpu().clone()
                        for name, value in model.bank_experts[route_key(source_route)][str(layer)].state_dict().items()
                    }
            model.add_route_expert(route, layer_states)

    def retain_route_experts(routes: Sequence[tuple[int, ...]]) -> None:
        keep = {route_key(route) for route in routes if route}
        for key in list(model.bank_experts.keys()):
            if key not in keep:
                del model.bank_experts[key]

    def train_routes(routes: Sequence[tuple[int, ...]], stage: str, examples_per_route: int) -> None:
        nonlocal total_examples
        routes = [tuple(route) for route in routes if route]
        if not routes:
            return
        model.eval()
        checkpointed_modules = _enable_checkpointing_without_dropout(model.base_model)
        if target_device.type == "cuda" and checkpointed_modules == 0:
            raise RuntimeError("CUDA adaptive training requires active gradient checkpointing")
        stage_seed = seed + sum(ord(char) for char in stage)
        seed_indices = list(range(len(train_rows)))
        random.Random(stage_seed).shuffle(seed_indices)
        selected_indices = seed_indices[: min(examples_per_route, len(train_rows))]
        route_order = list(routes)
        random.Random(stage_seed + 1).shuffle(route_order)
        curriculum["training_orders"][stage] = [list(route) for route in route_order]
        initialize_route_experts(route_order)
        progress = tqdm(
            total=len(route_order) * len(selected_indices) * epochs,
            desc=f"Adaptive bank {stage}", unit="example", file=sys.stdout, mininterval=0.5,
        )
        for route in route_order:
            expert_key = route_key(route)
            expert = model.bank_experts[expert_key]
            parameters = list(expert.parameters())
            for parameter in model.bank_experts.parameters():
                parameter.requires_grad_(False)
            for parameter in parameters:
                parameter.requires_grad_(True)
            optimizer = torch.optim.AdamW(
                parameters, lr=cfg.phase1_lr, weight_decay=cfg.weight_decay
            )
            mask = torch.tensor(
                [[layer in route for layer in candidates]], dtype=torch.bool, device=target_device
            )
            model.set_routing_override("matrix", mask)
            model.set_route_experts([route])
            model.set_route_gains(torch.ones(1, device=target_device))
            for epoch in range(epochs):
                order = list(selected_indices)
                random.Random(stage_seed + epoch).shuffle(order)
                optimizer.zero_grad(set_to_none=True)
                preference_questions_seen: set[int] = set()
                for start in range(0, len(order), grad_accum):
                    window = order[start : start + grad_accum]
                    tokens_in_window = sum(
                        int(train_rows[index]["labels"][1:].ne(-100).sum()) for index in window
                    )
                    if tokens_in_window <= 0:
                        progress.update(len(window))
                        continue
                    for index in window:
                        row, extras = _prepare([train_rows[index]], target_device)
                        model.set_prompt_shield_mask(extras.get("prompt_shield_mask"))
                        with torch.autocast(
                            device_type=target_device.type,
                            enabled=target_device.type == "cuda",
                        ):
                            logits = model(
                                input_ids=row["input_ids"],
                                attention_mask=row["attention_mask"],
                                use_cache=False,
                            ).logits
                            nll_sum = _nll_sum(logits, row["labels"])
                            loss = nll_sum / tokens_in_window
                        if loss.requires_grad:
                            scaler.scale(loss).backward()
                        del loss, nll_sum, logits, row, extras
                        question_id = int(train_rows[index].get("question_index", index))
                        pair = preference_by_question.get(question_id)
                        if pair is not None and question_id not in preference_questions_seen and preference_weight:
                            preference_questions_seen.add(question_id)
                            _backward_pairwise_preference(
                                model,
                                pair,
                                target_device,
                                scaler,
                                preference_weight / len(window),
                            )
                        progress.update(1)
                    if any(parameter.grad is not None for parameter in parameters):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(parameters, cfg.clip_grad)
                        scaler.step(optimizer)
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    model.set_prompt_shield_mask(None)
            total_examples += len(selected_indices) * epochs
        progress.close()
        for parameter in model.bank_experts.parameters():
            parameter.requires_grad_(False)
        model.set_route_experts(None)
        model.set_routing_override("none", candidates)
        model.set_route_gains(None)
        model.set_prompt_shield_mask(None)

    @torch.no_grad()
    def score_routes(routes: Sequence[tuple[int, ...]], stage: str) -> list[tuple[tuple[int, ...], float]]:
        model.eval()
        scored: list[tuple[tuple[int, ...], float]] = []
        route_sets = [tuple(route) for route in routes]
        if () not in route_sets:
            route_sets = [(), *route_sets]
        progress = tqdm(
            total=len(route_sets) * screen_row_count, desc=f"Screen routes {stage}",
            unit="route-example", file=sys.stdout, mininterval=0.5,
        )
        for route in route_sets:
            model.set_route_experts([route] if route else None)
            mask = torch.tensor(
                [[layer in route for layer in candidates]], dtype=torch.bool, device=target_device
            )
            model.set_routing_override("matrix", mask)
            model.set_route_gains(torch.ones(1, device=target_device))
            question_scores = []
            for _, question_rows in screen_questions:
                completion_scores = []
                for item in question_rows:
                    row, extras = _prepare([item], target_device)
                    model.set_prompt_shield_mask(extras.get("prompt_shield_mask"))
                    with torch.autocast(
                        device_type=target_device.type,
                        enabled=target_device.type == "cuda",
                    ):
                        logits = model(
                            input_ids=row["input_ids"],
                            attention_mask=row["attention_mask"],
                            use_cache=False,
                        ).logits
                        nll = _nll_sum(logits, row["labels"])
                    token_count = int(row["labels"][:, 1:].ne(-100).sum())
                    completion_scores.append(float(nll.float()) / max(token_count, 1))
                    del nll, logits, row, extras
                    progress.update(1)
                question_scores.append(sum(completion_scores) / max(len(completion_scores), 1))
            scored.append((route, sum(question_scores) / max(len(question_scores), 1)))
        progress.close()
        model.set_routing_override("none", candidates)
        model.set_route_experts(None)
        model.set_route_gains(None)
        model.set_prompt_shield_mask(None)
        return scored

    @torch.no_grad()
    def score_preference_routes(
        routes: Sequence[tuple[int, ...]], stage: str
    ) -> tuple[list[tuple[tuple[int, ...], float]], dict[str, Any]]:
        """Score route synergy on held-out, same-prompt correct/wrong pairs."""
        if not validation_preference_pairs:
            return [], {"pairs": 0, "accuracy": None}
        pairs = spread_items(
            validation_preference_pairs,
            min(max(screen_validation_rows, 24), len(validation_preference_pairs)),
        )
        route_scores: list[tuple[tuple[int, ...], float]] = []
        per_route: dict[str, dict[str, float]] = {}
        model.eval()
        progress = tqdm(
            total=len(routes) * len(pairs) * 2,
            desc=f"Preference-screen routes {stage}", unit="pair-forward",
            file=sys.stdout, mininterval=0.5,
        )
        for route in routes:
            model.set_route_experts([route] if route else None)
            mask = torch.tensor(
                [[layer in route for layer in candidates]], dtype=torch.bool, device=target_device
            )
            model.set_routing_override("matrix", mask)
            model.set_route_gains(torch.ones(1, device=target_device))
            pair_losses = []
            margins = []
            positive_nlls = []
            for pair in pairs:
                scores = {}
                for key in ("positive", "negative"):
                    row, extras = _prepare([pair[key]], target_device)
                    model.set_prompt_shield_mask(extras.get("prompt_shield_mask"))
                    with torch.autocast(
                        device_type=target_device.type,
                        enabled=target_device.type == "cuda",
                    ):
                        logits = model(
                            input_ids=row["input_ids"],
                            attention_mask=row["attention_mask"],
                            use_cache=False,
                        ).logits
                        nll_sum = _nll_sum(logits, row["labels"])
                    token_count = int(row["labels"][:, 1:].ne(-100).sum())
                    scores[key] = float(nll_sum.float()) / max(token_count, 1)
                    progress.update(1)
                    del nll_sum, logits, row, extras
                margin = scores["negative"] - scores["positive"]
                pair_losses.append(float(F.softplus(torch.tensor(0.2 - margin))))
                margins.append(margin)
                positive_nlls.append(scores["positive"])
            route_score = sum(pair_losses) / len(pair_losses) + 0.1 * sum(positive_nlls) / len(positive_nlls)
            route_scores.append((tuple(route), route_score))
            key = "+".join(map(str, route)) if route else "k0"
            per_route[key] = {
                "preference_loss": sum(pair_losses) / len(pair_losses),
                "preference_accuracy": sum(margin > 0 for margin in margins) / len(margins),
                "mean_positive_minus_negative_nll": -sum(margins) / len(margins),
            }
        progress.close()
        model.set_routing_override("none", candidates)
        model.set_route_experts(None)
        model.set_route_gains(None)
        model.set_prompt_shield_mask(None)
        return route_scores, {
            "pairs": len(pairs),
            "selection_metric": "pairwise_correct_vs_wrong_completion_loss + 0.1*positive_nll",
            "per_route": per_route,
        }

    def screen_and_select(
        stage: str,
        routes: Sequence[tuple[int, ...]],
        *,
        k: int,
        limit: int,
        parent_routes: Sequence[tuple[int, ...]] = (),
    ) -> list[tuple[int, ...]]:
        initialize_route_experts(routes, parent_routes)
        scored = score_routes(routes, stage)
        provisional = select_routes_by_depth(
            [(route, score) for route, score in scored if route],
            k=k,
            per_bucket=2,
            limit=min(len(routes), max(limit, 2 * limit)),
        )
        preference_scored, preference_summary = score_preference_routes(provisional, stage)
        selection_scores = preference_scored if preference_scored else scored
        nonempty_selection = [(route, value) for route, value in selection_scores if route]
        selected = select_routes_by_depth(
            nonempty_selection, k=k, per_bucket=1, limit=limit
        )
        route_selection_scores.update({tuple(route): float(value) for route, value in selection_scores})
        empty_score = next((value for route, value in scored if not route), None)
        if empty_score is not None:
            curriculum["k0_validation_nll"] = empty_score
        curriculum["stages"][stage] = {
            "candidates": [list(route) for route in routes],
            "preference_shortlist": [list(route) for route in provisional],
            "selected": [list(route) for route in selected],
            "screen_questions": len(screen_questions),
            "screen_rows": screen_row_count,
            "validation_token_mean_nll": {"+".join(map(str, route)): value for route, value in scored},
            "validation_pairwise": preference_summary,
            "selection_metric": (
                preference_summary["selection_metric"] if preference_scored
                else "positive_completion_token_mean_nll"
            ),
        }
        return selected

    model.train()
    singles = candidate_routes(candidates, 1)
    train_routes(singles, "singletons", route_examples_per_expert)
    selected_singles = screen_and_select("singletons", singles, k=1, limit=len(singles))

    pair_candidates = candidate_routes(
        candidates, 2, min_layer_gap=min_layer_gap, control_gap=control_gap,
        spacing_control_limit=spacing_control_limit,
    )
    pairs = screen_and_select("pair_screen", pair_candidates, k=2, limit=max_pair_routes)
    trained_pairs = list(pairs)
    train_routes(pairs, "pairs", max(64, route_examples_per_expert * 2 // 3))
    pair_scores = score_routes(pairs, "pairs_after_joint_training")
    pair_preference_scores, pair_preference_summary = score_preference_routes(pairs, "pairs_after_joint_training")
    pair_selection_scores = pair_preference_scores or pair_scores
    route_selection_scores.update({tuple(route): float(value) for route, value in pair_selection_scores})
    pairs = select_routes_by_depth(
        [(route, score) for route, score in pair_selection_scores if route],
        k=2, per_bucket=1, limit=max_pair_routes,
    )
    curriculum["stages"]["pairs"] = {
        "candidates": [list(route) for route in trained_pairs],
        "selected": [list(route) for route in pairs],
        "validation_token_mean_nll": {"+".join(map(str, route)): value for route, value in pair_scores},
        "validation_pairwise": pair_preference_summary,
        "selection_metric": "pairwise_preference" if pair_preference_scores else "positive_completion_token_mean_nll",
    }
    retain_route_experts([*[(layer,) for layer in candidates], *pairs])

    triple_candidates = candidate_routes(
        candidates, 3, selected_routes=pairs, min_layer_gap=min_layer_gap,
        control_gap=control_gap, spacing_control_limit=spacing_control_limit,
    )
    triples = screen_and_select(
        "triple_screen", triple_candidates, k=3, limit=max_triple_routes,
        parent_routes=pairs,
    )
    trained_triples = list(triples)
    train_routes(triples, "triples", max(48, route_examples_per_expert // 2))
    triple_scores = score_routes(triples, "triples_after_joint_training")
    triple_preference_scores, triple_preference_summary = score_preference_routes(triples, "triples_after_joint_training")
    triple_selection_scores = triple_preference_scores or triple_scores
    route_selection_scores.update({tuple(route): float(value) for route, value in triple_selection_scores})
    triples = select_routes_by_depth(
        [(route, score) for route, score in triple_selection_scores if route],
        k=3, per_bucket=1, limit=max_triple_routes,
    )
    curriculum["stages"]["triples"] = {
        "candidates": [list(route) for route in trained_triples],
        "selected": [list(route) for route in triples],
        "validation_token_mean_nll": {"+".join(map(str, route)): value for route, value in triple_scores},
        "validation_pairwise": triple_preference_summary,
        "selection_metric": "pairwise_preference" if triple_preference_scores else "positive_completion_token_mean_nll",
    }
    retain_route_experts([*[(layer,) for layer in candidates], *pairs, *triples])

    quad_candidates = candidate_routes(
        candidates, 4, selected_routes=triples, min_layer_gap=min_layer_gap,
        control_gap=control_gap, spacing_control_limit=spacing_control_limit,
    )
    quads = screen_and_select(
        "quad_screen", quad_candidates, k=4, limit=max_quad_routes,
        parent_routes=[*triples, *pairs],
    )
    trained_quads = list(quads)
    train_routes(quads, "quads", max(32, route_examples_per_expert // 3))
    quad_scores = score_routes(quads, "quads_after_joint_training")
    quad_preference_scores, quad_preference_summary = score_preference_routes(quads, "quads_after_joint_training")
    quad_selection_scores = quad_preference_scores or quad_scores
    route_selection_scores.update({tuple(route): float(value) for route, value in quad_selection_scores})
    quads = select_routes_by_depth(
        [(route, score) for route, score in quad_selection_scores if route],
        k=4, per_bucket=1, limit=max_quad_routes,
    )
    curriculum["stages"]["quads"] = {
        "candidates": [list(route) for route in trained_quads],
        "selected": [list(route) for route in quads],
        "validation_token_mean_nll": {"+".join(map(str, route)): value for route, value in quad_scores},
        "validation_pairwise": quad_preference_summary,
        "selection_metric": "pairwise_preference" if quad_preference_scores else "positive_completion_token_mean_nll",
    }
    catalog = [[], *[list(route) for route in selected_singles],
               *[list(route) for route in pairs],
               *[list(route) for route in triples],
               *[list(route) for route in quads]]
    curriculum["catalog"] = catalog
    family_routes = {1: selected_singles, 2: pairs, 3: triples, 4: quads}
    family_scores = {
        k: [
            (tuple(route), route_selection_scores.get(tuple(route), float("inf")))
            for route in routes
        ]
        for k, routes in family_routes.items()
    }
    curriculum["router_catalog"] = build_router_route_catalog(
        {1: selected_singles, 2: pairs, 3: triples, 4: quads},
        family_scores,
        singleton_limit=4,
        routes_per_k=2,
    )
    final_scores = score_routes(
        [tuple(route) for route in catalog], "final_catalog_after_all_training"
    )
    curriculum["final_catalog_validation_token_mean_nll"] = {
        "+".join(map(str, route)) if route else "k0": value
        for route, value in final_scores
    }
    retain_route_experts([*[(layer,) for layer in candidates], *pairs, *triples, *quads])
    curriculum["route_expert_count"] = len(model.bank_experts)
    curriculum["router_action_route_count"] = len(curriculum["router_catalog"])
    curriculum["router_action_counts_by_k"] = {
        str(k): sum(len(route) == k for route in curriculum["router_catalog"])
        for k in range(5)
    }
    curriculum["route_layer_usage"] = {
        str(layer): sum(layer in route for route in catalog)
        for layer in candidates
    }
    _write_bank_checkpoint(
        model, cfg, candidates, catalog, save_path, epochs, curriculum
    )
    model.set_route_experts(None)
    model.set_routing_override("none", candidates)
    model.set_route_gains(None)
    return {
        "checkpoint": str(save_path),
        "candidate_layers": list(candidates),
        "route_catalog": curriculum["catalog"],
        "router_catalog": curriculum["router_catalog"],
        "curriculum": curriculum,
        "trained_examples": total_examples,
        "preference_training": {
            "pairs_available": len(preference_pairs),
            "weight": float(preference_weight),
            "objective": "positive_token_nll + weighted softplus pairwise margin on same-prompt correct/wrong completions",
        },
    }
