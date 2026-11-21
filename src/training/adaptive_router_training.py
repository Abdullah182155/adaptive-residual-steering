"""Offline contextual-bandit training for the adaptive subset router."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from src.evaluation.answers import score_tiers
from src.evaluation.generation import (
    build_prompts,
    build_question_feature_masks,
    generate_batch,
)
from src.routing.adaptive_subset_router import (
    AdaptiveSubsetRouter,
    RouteValueRouter,
    sample_route_actions,
)
from src.training.adaptive_data import spread_items, weighted_shot_average
from src.training.batching import prepare_batch as _prepare


class _Probe:
    """Capture one unsteered backbone layer for prompt-only router features."""

    def __init__(self, layer: nn.Module) -> None:
        self.hidden: torch.Tensor | None = None
        self.handle = layer.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output) -> None:
        value = output[0] if isinstance(output, tuple) else output
        self.hidden = value.detach()

    def close(self) -> None:
        self.handle.remove()


def _completion_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_labels = labels[:, 1:]
    losses = F.cross_entropy(
        logits[:, :-1].float().transpose(1, 2),
        shifted_labels,
        ignore_index=-100,
        reduction="none",
    )
    valid = shifted_labels.ne(-100)
    return (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def _save_router_checkpoint(
    router, path: str | Path, probe_layer: int, best_val_nll: float,
    route_catalog: Sequence[Sequence[int]] = (),
    best_val_objective: float | None = None,
    bank_sha256: str | None = None,
    selection_metric: str | None = None,
    selection_value: float | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "format": (
                "ars_catalog_router_v2"
                if isinstance(router, RouteValueRouter)
                else "ars_adaptive_router_v1"
            ),
            "router_type": (
                "catalog_value"
                if isinstance(router, RouteValueRouter)
                else "factorized_subset"
            ),
            "candidate_layers": list(router.candidate_layers),
            "probe_layer": int(probe_layer),
            "max_active_layers": router.max_active_layers,
            "alpha_values": list(router.alpha_values),
            "router": router.state_dict(),
            "best_validation_nll": float(best_val_nll),
            "best_validation_objective": (
                float(best_val_objective)
                if best_val_objective is not None
                else float(best_val_nll)
            ),
            "route_catalog": [list(route) for route in route_catalog],
            "bank_sha256": bank_sha256,
            "checkpoint_selection_metric": selection_metric,
            "checkpoint_selection_value": (
                float(selection_value) if selection_value is not None else None
            ),
            "action_catalog": [
                {"route": list(route), "alpha": float(alpha)}
                for route, alpha in router.action_catalog
            ] if isinstance(router, RouteValueRouter) else [],
        },
        temporary,
    )
    temporary.replace(destination)


def _score_action(
    model, row, route, alpha: torch.Tensor, device, prompt_shield_mask=None
) -> torch.Tensor:
    model.set_routing_override("matrix", route)
    model.set_route_gains(alpha)
    model.set_prompt_shield_mask(prompt_shield_mask)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits = model(
            input_ids=row["input_ids"],
            attention_mask=row["attention_mask"],
            use_cache=False,
        ).logits
        nll = _completion_nll(logits, row["labels"])
    del logits
    return nll


def _distinct_question_rows(rows, limit: int):
    """Select at most one deterministic completion row per source question."""
    selected = []
    seen = set()
    for index, item in enumerate(rows):
        question_id = item.get("question_index", index)
        if question_id in seen:
            continue
        seen.add(question_id)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _group_prompt_rows(rows: Sequence[dict]) -> list[list[dict]]:
    """Group completion rows that share the exact tokenized prompt and question."""
    groups: dict[tuple[int, tuple[int, ...]], list[dict]] = {}
    for index, item in enumerate(rows):
        question_id = int(item.get("question_index", index))
        signature = tuple(int(token) for token in item.get("prompt_signature", ()))
        groups.setdefault((question_id, signature), []).append(item)
    return list(groups.values())


def _router_context_features(item: dict, device: torch.device | str) -> torch.Tensor:
    feature_mask = item.get("router_feature_mask", ())
    question_tokens = sum(bool(value) for value in feature_mask)
    if not question_tokens:
        question_tokens = len(item.get("prompt_signature", ()))
    shots = float(item.get("effective_shots", 0))
    return torch.tensor(
        [[min(max(shots, 0.0), 4.0) / 4.0, min(question_tokens, 256) / 256.0]],
        dtype=torch.float32,
        device=device,
    )


@torch.no_grad()
def _capture_prompt_hidden(model, item: dict, row, probe: _Probe, candidates, device):
    """Capture router features from the exact prompt-only prefill used at inference."""
    signature = tuple(int(token) for token in item.get("prompt_signature", ()))
    if signature:
        input_ids = torch.tensor([signature], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        feature_mask = torch.as_tensor(
            item.get("router_feature_mask", ()), dtype=torch.bool, device=device
        ).flatten()
        prompt_mask = (
            feature_mask.unsqueeze(0)
            if feature_mask.numel() == input_ids.shape[1] and bool(feature_mask.any())
            else attention_mask.bool()
        )
    else:
        input_ids = row["input_ids"]
        attention_mask = row["attention_mask"]
        prompt_mask = row["labels"].eq(-100) & attention_mask.bool()
    model.set_routing_override("none", candidates)
    model.set_route_gains(None)
    model.set_prompt_shield_mask(None)
    probe.hidden = None
    model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    if probe.hidden is None:
        return None, prompt_mask
    return probe.hidden, prompt_mask


@torch.no_grad()
def _evaluate_router_nll(
    model,
    router: AdaptiveSubsetRouter,
    rows: Sequence[dict],
    probe: _Probe,
    device: torch.device,
    limit: int,
    route_cost: float = 0.0,
) -> dict[str, Any]:
    router.eval()
    nll_values: list[float] = []
    k0_values: list[float] = []
    candidates = router.candidate_layers
    groups_by_question: dict[int, list[list[dict]]] = {}
    for group_index, group in enumerate(_group_prompt_rows(rows)):
        question_id = int(group[0].get("question_index", group_index))
        groups_by_question.setdefault(question_id, []).append(group)
    selected_questions = list(groups_by_question.items())[:limit]
    selected_groups = [group for _, groups in selected_questions for group in groups]
    route_counts: Counter[str] = Counter()
    k_counts: Counter[str] = Counter()
    alpha_counts: Counter[str] = Counter()
    per_shot: dict[str, dict[str, Any]] = {}
    question_nlls: dict[int, list[float]] = {}
    question_objectives: dict[int, list[float]] = {}
    question_k0_values: dict[int, list[float]] = {}
    question_k_values: dict[int, list[int]] = {}
    scored_rows = 0
    for group in selected_groups:
        first_row, extras = _prepare([group[0]], device)
        model.set_routing_override("none", candidates)
        model.set_route_gains(None)
        model.set_prompt_shield_mask(None)
        base_logits = model(
            input_ids=first_row["input_ids"],
            attention_mask=first_row["attention_mask"],
            use_cache=False,
        ).logits
        k0_nll = _completion_nll(base_logits, first_row["labels"])
        k0_total = float(k0_nll.mean())
        del base_logits, k0_nll
        prompt_hidden, prompt_mask = _capture_prompt_hidden(
            model, group[0], first_row, probe, candidates, device
        )
        if prompt_hidden is None:
            del first_row, extras
            continue
        if isinstance(router, RouteValueRouter):
            context_features = _router_context_features(group[0], device)
            decision = router.choose(prompt_hidden, prompt_mask, context_features)
        else:
            decision = router.choose(prompt_hidden, prompt_mask)
        route = decision.mask
        group_nlls = []
        for item_index, item in enumerate(group):
            row, item_extras = _prepare([item], device)
            if item_index:
                model.set_routing_override("none", candidates)
                model.set_route_gains(None)
                model.set_prompt_shield_mask(None)
                base_logits = model(
                    input_ids=row["input_ids"],
                    attention_mask=row["attention_mask"],
                    use_cache=False,
                ).logits
                baseline_nll = _completion_nll(base_logits, row["labels"])
                k0_total += float(baseline_nll.mean())
                del base_logits, baseline_nll
            nll = _score_action(
                model, row, route, decision.alpha, device,
                item_extras.get("prompt_shield_mask"),
            )
            group_nlls.append(float(nll.mean()))
            del row, item_extras, nll
        route_nll = sum(group_nlls) / len(group_nlls)
        k0_values.append(k0_total / len(group))
        active_layers = int(decision.k.reshape(-1)[0])
        nll_values.append(route_nll)
        route_objective = route_nll + route_cost * active_layers
        selected_route = tuple(
            layer for layer, active in zip(candidates, decision.mask[0].tolist()) if active
        )
        route_key = f"{'+'.join(map(str, selected_route)) or 'k0'}@{float(decision.alpha[0]):g}"
        route_counts[route_key] += 1
        question_id = int(group[0].get("question_index", len(nll_values) - 1))
        question_nlls.setdefault(question_id, []).append(route_nll)
        question_objectives.setdefault(question_id, []).append(route_objective)
        question_k0_values.setdefault(question_id, []).append(k0_values[-1])
        question_k_values.setdefault(question_id, []).append(active_layers)
        k_counts[str(active_layers)] += 1
        alpha_counts[str(float(decision.alpha[0]))] += 1
        shot_key = str(int(group[0].get("effective_shots", 0)))
        shot_metrics = per_shot.setdefault(
            shot_key,
            {
                "nll_values": [],
                "objective_values": [],
                "k0_values": [],
                "k_values": [],
                "action_counts": Counter(),
            },
        )
        shot_metrics["nll_values"].append(route_nll)
        shot_metrics["objective_values"].append(route_nll + route_cost * active_layers)
        shot_metrics["k0_values"].append(k0_values[-1])
        shot_metrics["k_values"].append(active_layers)
        shot_metrics["action_counts"][route_key] += 1
        scored_rows += len(group)
        del first_row, extras
    model.set_routing_override("none", candidates)
    model.set_route_gains(None)
    model.set_prompt_shield_mask(None)
    if not question_nlls:
        raise ValueError("router validation produced no evaluable question rows")
    n_actions = sum(route_counts.values())
    probabilities = [count / max(n_actions, 1) for count in route_counts.values()]
    route_entropy = -sum(probability * math.log(probability) for probability in probabilities)
    per_shot_summary = {}
    for shot, metrics in sorted(per_shot.items(), key=lambda item: int(item[0])):
        action_count = sum(metrics["action_counts"].values())
        action_probabilities = [
            count / max(action_count, 1) for count in metrics["action_counts"].values()
        ]
        per_shot_summary[shot] = {
            "prompt_groups": len(metrics["nll_values"]),
            "validation_nll": sum(metrics["nll_values"]) / max(len(metrics["nll_values"]), 1),
            "validation_objective": sum(metrics["objective_values"]) / max(len(metrics["objective_values"]), 1),
            "k0_validation_nll": sum(metrics["k0_values"]) / max(len(metrics["k0_values"]), 1),
            "mean_k": sum(metrics["k_values"]) / max(len(metrics["k_values"]), 1),
            "unique_actions": len(metrics["action_counts"]),
            "action_entropy": -sum(p * math.log(p) for p in action_probabilities),
            "most_used_action_share": max(metrics["action_counts"].values()) / max(action_count, 1),
            "action_counts": dict(metrics["action_counts"]),
        }
    def question_mean(values):
        return sum(
            sum(question_values) / len(question_values) for question_values in values.values()
        ) / len(values)

    return {
        "validation_nll": question_mean(question_nlls),
        "validation_objective": question_mean(question_objectives),
        "k0_validation_nll": question_mean(question_k0_values),
        "validation_questions": len(question_nlls),
        "validation_prompt_groups": len(nll_values),
        "validation_completion_rows": scored_rows,
        "validation_mean_k": question_mean(question_k_values),
        "validation_unique_actions": len(route_counts),
        "validation_action_entropy": route_entropy,
        "validation_most_used_action_share": max(route_counts.values()) / max(n_actions, 1),
        "validation_action_counts": dict(route_counts),
        "validation_k_counts": dict(k_counts),
        "validation_alpha_counts": dict(alpha_counts),
        "validation_by_shot": per_shot_summary,
    }


@torch.no_grad()
def _evaluate_router_preference(
    model,
    router: RouteValueRouter,
    preference_pairs: Sequence[dict[str, Any]],
    probe: _Probe,
    device: torch.device,
    limit: int,
    route_cost: float,
) -> dict[str, Any]:
    """Evaluate the selected route on matched correct/wrong validation rollouts."""
    selected = spread_items(preference_pairs, limit)
    if not selected:
        raise ValueError("preference validation received no matched completion pairs")
    router.eval()
    candidates = router.candidate_layers
    costs: list[float] = []
    margins: list[float] = []
    hits = 0
    route_counts: Counter[str] = Counter()
    k_counts: Counter[str] = Counter()
    for pair in selected:
        positive_item = pair["positive"]
        positive_row, positive_extras = _prepare([positive_item], device)
        prompt_hidden, prompt_mask = _capture_prompt_hidden(
            model, positive_item, positive_row, probe, candidates, device
        )
        if prompt_hidden is None:
            continue
        context_features = _router_context_features(positive_item, device)
        decision = router.choose(prompt_hidden, prompt_mask, context_features)
        positive_nll = _score_action(
            model, positive_row, decision.mask, decision.alpha, device,
            positive_extras.get("prompt_shield_mask"),
        )
        negative_item = pair["negative"]
        negative_row, negative_extras = _prepare([negative_item], device)
        negative_nll = _score_action(
            model, negative_row, decision.mask, decision.alpha, device,
            negative_extras.get("prompt_shield_mask"),
        )
        pos_value = float(positive_nll.mean())
        neg_value = float(negative_nll.mean())
        margin = neg_value - pos_value
        active = int(decision.k[0])
        costs.append(
            math.log1p(math.exp(min(30.0, 0.2 - margin)))
            + 0.1 * pos_value
            + route_cost * active
        )
        margins.append(margin)
        hits += int(margin > 0.0)
        route = tuple(
            layer for layer, enabled in zip(candidates, decision.mask[0].tolist()) if enabled
        )
        route_counts["+".join(map(str, route)) if route else "k0"] += 1
        k_counts[str(active)] += 1
        del positive_row, positive_extras, negative_row, negative_extras
        del positive_nll, negative_nll
    if not costs:
        raise ValueError("router preference validation could not score any pairs")
    model.set_routing_override("none", candidates)
    model.set_route_gains(None)
    model.set_prompt_shield_mask(None)
    return {
        "validation_preference_loss": sum(costs) / len(costs),
        "validation_preference_accuracy": hits / len(costs),
        "validation_preference_margin": sum(margins) / len(margins),
        "validation_preference_pairs": len(costs),
        "validation_preference_route_counts": dict(route_counts),
        "validation_preference_k_counts": dict(k_counts),
    }


def _build_router_prefill(model, router: RouteValueRouter, probe: _Probe):
    """Build the same question-only route callback used by deployment inference."""
    candidates = router.candidate_layers

    @torch.no_grad()
    def prefill(input_ids, attention_mask, route_feature_mask=None, route_context_features=None):
        model.set_routing_override("none", candidates)
        model.set_route_gains(None)
        probe.hidden = None
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        if probe.hidden is None:
            raise RuntimeError("router probe did not capture prompt hidden states")
        feature_mask = (
            route_feature_mask.bool()
            if route_feature_mask is not None
            else attention_mask.bool()
        )
        if isinstance(router, RouteValueRouter):
            decision = router.choose(probe.hidden, feature_mask, route_context_features)
        else:
            decision = router.choose(probe.hidden, feature_mask)
        model.set_routing_override("matrix", decision.mask)
        model.set_route_gains(decision.alpha)

    prefill.uses_question_features = isinstance(router, RouteValueRouter)
    return prefill


@torch.no_grad()
def _evaluate_router_task_accuracy(
    model,
    router: RouteValueRouter,
    validation_rows: Sequence[dict[str, Any]],
    probe: _Probe,
    tokenizer,
    cfg,
    device: torch.device,
    *,
    limit: int = 32,
    shots: Sequence[int] = (0, 2),
    static_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate greedy task correctness on held-out questions for checkpoint selection."""
    unique: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(validation_rows):
        question_id = int(item.get("question_index", index))
        if "question" not in item or "gold_answer" not in item:
            continue
        unique.setdefault(question_id, item)
    selected = spread_items(list(unique.values()), limit)
    if not selected:
        raise ValueError("task-accuracy validation received no held-out questions")
    if any("question" not in item or "gold_answer" not in item for item in selected):
        raise ValueError("task-accuracy validation rows must include question and gold_answer")

    questions = [str(item["question"]) for item in selected]
    question_ids = [int(item["question_index"]) for item in selected]
    templates = [int(item.get("template_id", 0)) for item in selected]
    golds = [
        float(str(item["gold_answer"]).replace(",", "").replace("$", "").strip())
        for item in selected
    ]
    router_prefill = _build_router_prefill(model, router, probe)
    results: dict[str, Any] = {}
    for n_shot in shots:
        prompts, shields = build_prompts(
            tokenizer,
            questions,
            question_ids,
            int(n_shot),
            policy=getattr(cfg, "shield_policy", "prompt"),
            template_ids=templates,
        )
        feature_masks = build_question_feature_masks(tokenizer, prompts, questions)
        question_token_counts = [int(mask.sum()) for mask in feature_masks]
        context_features = torch.tensor(
            [
                [
                    min(max(float(n_shot), 0.0), 4.0) / 4.0,
                    min(token_count, 256) / 256.0,
                ]
                for token_count in question_token_counts
            ],
            dtype=torch.float32,
            device=device,
        )
        model.set_routing_override("none", router.candidate_layers)
        model.set_route_gains(None)
        baseline_hits = []
        static_bank_hits = []
        routed_hits = []
        for index, (prompt, shield, gold) in enumerate(zip(prompts, shields, golds)):
            # Clear the previous example's selected route before measuring a true baseline.
            model.set_routing_override("none", router.candidate_layers)
            model.set_route_gains(None)
            model.set_prompt_shield_mask(None)
            generation_options = {
                "prompt_shields": [shield],
                "max_new_tokens": int(getattr(cfg, "generation_max_new_tokens", 256)),
                "temperature": float(getattr(cfg, "generation_temperature", 0.0)),
                "repetition_penalty": float(getattr(cfg, "generation_repetition_penalty", 1.1)),
                "no_repeat_ngram_size": int(getattr(cfg, "generation_no_repeat_ngram", 6)),
                "max_prompt_len": max(int(getattr(cfg, "max_tok_len", 768)), 1024),
            }
            baseline_output = generate_batch(
                model, tokenizer, [prompt], device, **generation_options
            )[0]
            baseline_hits.append(bool(score_tiers(baseline_output.text, gold)["lenient"]))
            if static_action is not None:
                static_route = tuple(int(layer) for layer in static_action.get("route", ()))
                if static_route:
                    route_mask = torch.tensor(
                        [[layer in static_route for layer in router.candidate_layers]],
                        dtype=torch.bool,
                        device=device,
                    )
                    model.set_routing_override("matrix", route_mask)
                else:
                    model.set_routing_override("none", router.candidate_layers)
                model.set_route_gains(torch.tensor(
                    [float(static_action.get("alpha", 1.0))], dtype=torch.float32, device=device
                ))
                static_output = generate_batch(
                    model, tokenizer, [prompt], device, **generation_options
                )[0]
                static_bank_hits.append(bool(score_tiers(static_output.text, gold)["lenient"]))
            routed_output = generate_batch(
                model,
                tokenizer,
                [prompt],
                device,
                route_prefill_fn=router_prefill,
                route_feature_masks=[feature_masks[index]],
                route_context_features=context_features[index : index + 1],
                **generation_options,
            )[0]
            routed_hits.append(bool(score_tiers(routed_output.text, gold)["lenient"]))
        wins = sum(route and not baseline for route, baseline in zip(routed_hits, baseline_hits))
        losses = sum(baseline and not route for route, baseline in zip(routed_hits, baseline_hits))
        results[str(int(n_shot))] = {
            "n": len(selected),
            "baseline_accuracy": sum(baseline_hits) / len(baseline_hits),
            "router_accuracy": sum(routed_hits) / len(routed_hits),
            "paired_delta_pp": 100.0 * (sum(routed_hits) - sum(baseline_hits)) / len(selected),
            "wins": wins,
            "losses": losses,
            "ties": len(selected) - wins - losses,
        }
        if static_bank_hits:
            results[str(int(n_shot))].update({
                "static_bank_accuracy": sum(static_bank_hits) / len(static_bank_hits),
                "router_delta_vs_static_bank_pp": 100.0 * (
                    sum(routed_hits) - sum(static_bank_hits)
                ) / len(selected),
            })
        model.set_routing_override("none", router.candidate_layers)
        model.set_route_gains(None)
        model.set_prompt_shield_mask(None)

    active_shots = {
        int(shot): metrics for shot, metrics in
        ((int(key), value) for key, value in results.items())
    }
    weighted_delta_vs_baseline_pp, baseline_selection_method = weighted_shot_average(
        {shot: metrics["paired_delta_pp"] for shot, metrics in active_shots.items()},
        "router_vs_baseline",
    )
    has_static_reference = all(
        "router_delta_vs_static_bank_pp" in metrics for metrics in active_shots.values()
    )
    if has_static_reference:
        weighted_delta_vs_static_pp, static_selection_method = weighted_shot_average(
            {
                shot: metrics["router_delta_vs_static_bank_pp"]
                for shot, metrics in active_shots.items()
            },
            "router_vs_static_bank",
        )
    else:
        weighted_delta_vs_static_pp, static_selection_method = None, None
    return {
        "validation_task_accuracy_by_shot": results,
        "validation_task_weighted_delta_vs_baseline_pp": weighted_delta_vs_baseline_pp,
        "validation_task_weighted_delta_vs_static_bank_pp": weighted_delta_vs_static_pp,
        "validation_task_selection_delta_pp": (
            weighted_delta_vs_static_pp if has_static_reference else weighted_delta_vs_baseline_pp
        ),
        "validation_task_selection_reference": "static_bank" if has_static_reference else "baseline",
        "validation_task_selection_method": (
            static_selection_method if has_static_reference else baseline_selection_method
        ),
        "validation_task_questions": len(selected),
    }


@torch.no_grad()
def select_static_bank_action(
    model,
    rows: Sequence[dict],
    candidate_layers: Sequence[int],
    route_catalog: Sequence[Sequence[int]],
    device: torch.device | str,
    *,
    route_cost: float = 0.01,
    alpha_values: Sequence[float] = (1.0,),
    max_questions: int = 16,
    tokenizer=None,
    cfg=None,
    shots: Sequence[int] = (0, 2),
    generation_shortlist: int = 6,
    preference_pairs: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Select a static action using greedy answer accuracy on held-out questions.

    Completion NLL is used only to shortlist candidates. The winner is selected by
    a predeclared 0/2-shot accuracy blend and is never selected on benchmark rows.
    """
    if max_questions < 1 or route_cost < 0:
        raise ValueError("static-action screening needs positive questions and nonnegative route cost")
    target_device = torch.device(device)
    candidates = tuple(int(layer) for layer in candidate_layers)
    routes = list(dict.fromkeys(tuple(sorted(set(int(layer) for layer in route))) for route in route_catalog))
    if () not in routes:
        routes.insert(0, ())
    if any(len(route) > 4 or not set(route).issubset(candidates) for route in routes):
        raise ValueError("static-action route catalog contains an invalid candidate-layer subset")
    if not alpha_values or any(not 1.0 <= float(alpha) <= 2.0 for alpha in alpha_values):
        raise ValueError("static-action alpha values must lie in [1, 2]")
    actions = [
        (route, 1.0) if not route else (route, float(alpha))
        for route in routes
        for alpha in ((1.0,) if not route else alpha_values)
    ]
    if not actions:
        raise ValueError("static-action screening requires at least one route action")
    groups: dict[int, list[dict]] = {}
    for index, item in enumerate(rows):
        question_id = int(item.get("question_index", index))
        groups.setdefault(question_id, []).append(item)
    selected_groups = spread_items(list(groups.items()), max_questions)
    if not selected_groups:
        raise ValueError("static-action screening received no validation questions")
    generation_metadata_available = tokenizer is not None and cfg is not None and all(
        "question" in rows_by_id[0] and "gold_answer" in rows_by_id[0]
        for _, rows_by_id in selected_groups
    )
    if generation_metadata_available and len(selected_groups) >= 2:
        screen_groups = selected_groups[::2]
        generation_groups = selected_groups[1::2]
    else:
        screen_groups = selected_groups
        generation_groups = selected_groups
    screen_question_ids = {question_id for question_id, _ in screen_groups}

    score_totals = [0.0 for _ in actions]
    score_questions = 0
    score_rows = 0
    for _, question_rows in screen_groups:
        question_scores = [0.0 for _ in actions]
        for item in question_rows:
            row, extras = _prepare([item], target_device)
            prompt_shield = extras.get("prompt_shield_mask")
            for action_index, (route, alpha) in enumerate(actions):
                route_mask = torch.tensor(
                    [[layer in route for layer in candidates]],
                    dtype=torch.bool,
                    device=target_device,
                )
                alpha_tensor = torch.tensor([alpha], dtype=torch.float32, device=target_device)
                nll = _score_action(
                    model, row, route_mask, alpha_tensor, target_device, prompt_shield
                )
                question_scores[action_index] += (
                    float(nll.mean()) + route_cost * len(route)
                )
                del nll
            score_rows += 1
            del row, extras
        for action_index, score in enumerate(question_scores):
            score_totals[action_index] += score / max(len(question_rows), 1)
        score_questions += 1
    model.set_routing_override("none", candidates)
    model.set_route_gains(None)
    model.set_prompt_shield_mask(None)

    action_scores = [score / score_questions for score in score_totals]
    preference_scores: dict[int, float] = {}
    selection_scores = list(action_scores)
    if preference_pairs:
        screen_pairs = [
            pair for pair in preference_pairs
            if int(pair["positive"].get("question_index", -1)) in screen_question_ids
        ]
        pair_subset = spread_items(screen_pairs, max_questions)
        for action_index, (route, alpha) in enumerate(actions):
            route_mask = torch.tensor(
                [[layer in route for layer in candidates]], dtype=torch.bool, device=target_device
            )
            alpha_tensor = torch.tensor([alpha], dtype=torch.float32, device=target_device)
            pair_losses = []
            for pair in pair_subset:
                pair_nll = {}
                for key in ("positive", "negative"):
                    row, extras = _prepare([pair[key]], target_device)
                    nll = _score_action(
                        model, row, route_mask, alpha_tensor, target_device,
                        extras.get("prompt_shield_mask"),
                    )
                    pair_nll[key] = float(nll.mean())
                    del row, extras, nll
                pair_losses.append(
                    math.log1p(math.exp(min(30.0, 0.2 + pair_nll["positive"] - pair_nll["negative"])))
                    + 0.1 * pair_nll["positive"]
                )
            if pair_losses:
                preference_scores[action_index] = sum(pair_losses) / len(pair_losses) + route_cost * len(route)
                selection_scores[action_index] = preference_scores[action_index]
        model.set_routing_override("none", candidates)
        model.set_route_gains(None)
        model.set_prompt_shield_mask(None)
    coarse_ranked = sorted(range(len(actions)), key=selection_scores.__getitem__)
    generation_stats: dict[int, dict[str, float]] = {}
    if generation_metadata_available:
        candidate_indices = coarse_ranked[:max(1, generation_shortlist)]
        selected_examples = [question_rows[0] for _, question_rows in generation_groups]
        prompts_by_shot: dict[int, tuple[list[str], list[Any], list[float]]] = {}
        for n_shot in shots:
            questions = [str(item["question"]) for item in selected_examples]
            question_ids = [int(item.get("question_index", index)) for index, item in enumerate(selected_examples)]
            templates = [int(item.get("template_id", 0)) for item in selected_examples]
            prompts, shields = build_prompts(
                tokenizer, questions, question_ids, int(n_shot),
                policy=getattr(cfg, "shield_policy", "prompt"),
                template_ids=templates,
            )
            golds = [float(str(item["gold_answer"]).replace(",", "").replace("$", "").strip())
                     for item in selected_examples]
            prompts_by_shot[int(n_shot)] = (prompts, shields, golds)

        for action_index in candidate_indices:
            route, alpha = actions[action_index]
            if route:
                route_mask = torch.tensor(
                    [[layer in route for layer in candidates]], dtype=torch.bool, device=target_device
                )
                model.set_routing_override("matrix", route_mask)
            else:
                model.set_routing_override("none", candidates)
            model.set_route_gains(torch.tensor([alpha], dtype=torch.float32, device=target_device))
            per_shot = {}
            for n_shot, (prompts, shields, golds) in prompts_by_shot.items():
                outputs = generate_batch(
                    model, tokenizer, prompts, target_device, prompt_shields=shields,
                    max_new_tokens=int(getattr(cfg, "generation_max_new_tokens", 256)),
                    repetition_penalty=float(getattr(cfg, "generation_repetition_penalty", 1.0)),
                    no_repeat_ngram_size=int(getattr(cfg, "generation_no_repeat_ngram", 0)),
                    max_prompt_len=int(getattr(cfg, "max_tok_len", 768)),
                )
                per_shot[n_shot] = sum(
                    bool(score_tiers(output.text, gold)["lenient"])
                    for output, gold in zip(outputs, golds)
                ) / max(len(golds), 1)
            generation_stats[action_index] = per_shot
        model.set_routing_override("none", candidates)
        model.set_route_gains(None)
        model.set_prompt_shield_mask(None)

    if generation_stats:
        def generation_score(index: int) -> float:
            return weighted_shot_average(
                generation_stats[index], "greedy_lenient_accuracy"
            )[0]
        best_index = max(generation_stats, key=generation_score)
        ranking = sorted(generation_stats, key=generation_score, reverse=True)
        _, shot_objective = weighted_shot_average(
            generation_stats[best_index], "greedy_lenient_accuracy"
        )
        shortlist_source = "pairwise preference/NLL" if preference_scores else "completion NLL"
        scoring_method = f"{shot_objective} (greedy accuracy selects; {shortlist_source} shortlists)"
    else:
        best_index = coarse_ranked[0]
        ranking = coarse_ranked
        scoring_method = "macro mean of per-question completion-token-mean NLL plus route_cost*k (fallback; generation metadata unavailable)"
    best_route, best_alpha = actions[best_index]
    return {
        "route": list(best_route),
        "alpha": best_alpha,
        "mean_cost_adjusted_score": action_scores[best_index],
        "scoring": scoring_method,
        "generation_selected": bool(generation_stats),
        "greedy_accuracy_by_shot": {
            str(shot): value for shot, value in generation_stats.get(best_index, {}).items()
        },
        "selection_objective": (
            generation_score(best_index) if generation_stats else action_scores[best_index]
        ),
        "selection_preference_score": preference_scores.get(best_index),
        "generation_shortlist_size": len(generation_stats),
        "route_cost": float(route_cost),
        "validation_questions": score_questions,
        "screen_validation_question_ids": sorted(screen_question_ids),
        "greedy_validation_questions": len(generation_groups) if generation_stats else 0,
        "greedy_validation_question_ids": (
            [int(question_id) for question_id, _ in generation_groups]
            if generation_stats else []
        ),
        "independent_static_validation_sets": bool(
            generation_stats and screen_question_ids.isdisjoint(
                int(question_id) for question_id, _ in generation_groups
            )
        ),
        "validation_rows": score_rows,
        "action_count": len(actions),
        "top_actions": [
            {
                "route": list(actions[index][0]),
                "alpha": actions[index][1],
                "mean_cost_adjusted_score": action_scores[index],
                "preference_score": preference_scores.get(index),
                "greedy_accuracy_by_shot": {
                    str(shot): value for shot, value in generation_stats.get(index, {}).items()
                },
            }
            for index in ranking[:10]
        ],
    }


@torch.no_grad()
def _build_oracle_targets(
    model,
    rows: Sequence[dict[str, torch.Tensor]],
    route_catalog: Sequence[Sequence[int]],
    router: AdaptiveSubsetRouter,
    probe: _Probe,
    device: torch.device,
    *,
    route_cost: float,
    row_limit: int,
    preference_pairs: Sequence[dict[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score route actions on positives and matched wrong-answer rollouts."""
    candidates = router.candidate_layers
    routes = [tuple(int(layer) for layer in route) for route in route_catalog]
    if () not in routes:
        routes.insert(0, ())
    if not routes or any(not set(route).issubset(candidates) or len(route) > 4 for route in routes):
        raise ValueError("route catalog contains an invalid candidate-layer subset")
    actions = tuple(router.action_catalog)
    preference_by_question = {
        int(pair["positive"]["question_index"]): pair for pair in preference_pairs
    }
    # Use every selected question. Matched positive/wrong-answer pairs provide
    # stronger labels where available; all remaining questions still contribute
    # positive-completion NLL labels instead of disappearing from router training.
    prompt_groups = _group_prompt_rows(rows)[:row_limit]
    targets: list[dict[str, Any]] = []
    k_counts: Counter[int] = Counter()
    route_counts: Counter[str] = Counter()
    layer_counts: Counter[int] = Counter()
    alpha_counts: Counter[str] = Counter()
    action_score_spreads: list[float] = []
    low_signal_rows = 0

    forward_count = sum(
        1 + len(actions) * (2 if int(group[0].get("question_index", -1)) in preference_by_question else len(group))
        for group in prompt_groups
    )
    progress = tqdm(
        total=forward_count,
        desc="Score route teacher",
        unit="forward",
        file=sys.stdout,
        mininterval=0.5,
    )
    for group in prompt_groups:
        question_id = int(group[0].get("question_index", len(targets)))
        preference_pair = preference_by_question.get(question_id)
        forwards_per_action = 2 if preference_pair is not None else len(group)
        feature_item = preference_pair["positive"] if preference_pair else group[0]
        first_row, extras = _prepare([feature_item], device)
        prompt_hidden, prompt_mask = _capture_prompt_hidden(
            model, feature_item, first_row, probe, candidates, device
        )
        progress.update(1)
        if prompt_hidden is None:
            progress.update(forwards_per_action * len(actions))
            del first_row, extras
            continue
        # Router inputs must contain prompt tokens only; positive completion
        # tokens are used to score routes, never as router features.
        pooled = router.raw_prompt_pool(prompt_hidden, prompt_mask).squeeze(0).float().cpu()
        best_score = float("inf")
        best_route: tuple[int, ...] = ()
        best_alpha = 1.0
        action_scores: list[float] = []
        for route, alpha in actions:
            route_mask = torch.tensor(
                [[layer in route for layer in candidates]], dtype=torch.bool, device=device
            )
            alpha_tensor = torch.tensor([alpha], dtype=torch.float32, device=device)
            completion_nlls = []
            if preference_pair is not None:
                pair_nlls = {}
                for pair_key in ("positive", "negative"):
                    item = preference_pair[pair_key]
                    row, item_extras = _prepare([item], device)
                    nll = _score_action(
                        model, row, route_mask, alpha_tensor, device,
                        item_extras.get("prompt_shield_mask"),
                    )
                    pair_nlls[pair_key] = float(nll.mean())
                    progress.update(1)
                    del row, item_extras, nll
                preference_loss = math.log1p(math.exp(min(
                    30.0, 0.2 + pair_nlls["positive"] - pair_nlls["negative"]
                )))
                score = preference_loss + 0.1 * pair_nlls["positive"] + route_cost * len(route)
                completion_nlls.append(pair_nlls["positive"])
            else:
                for item in group:
                    row, item_extras = _prepare([item], device)
                    nll = _score_action(
                        model,
                        row,
                        route_mask,
                        alpha_tensor,
                        device,
                        item_extras.get("prompt_shield_mask"),
                    )
                    completion_nlls.append(float(nll.mean()))
                    progress.update(1)
                    del row, item_extras, nll
                score = sum(completion_nlls) / len(completion_nlls) + route_cost * len(route)
            action_scores.append(score)
            if score < best_score:
                best_score, best_route, best_alpha = score, route, alpha
        score_tensor = torch.tensor(action_scores, dtype=torch.float32)
        score_spread = float(score_tensor.std(unbiased=False))
        action_score_spreads.append(score_spread)
        low_signal_rows += int(score_spread < 0.005)
        target = {
            "pooled_prompt": pooled,
            "context_features": _router_context_features(feature_item, "cpu").squeeze(0),
            "action_scores": score_tensor,
            "route": best_route,
            "k": len(best_route),
            "alpha": best_alpha,
            "question_index": question_id,
            "completion_indices": (
                [int(preference_pair[key].get("completion_index", 0)) for key in ("positive", "negative")]
                if preference_pair else [int(item.get("completion_index", 0)) for item in group]
            ),
            "completions_scored": 2 if preference_pair else len(group),
            "target_objective": "matched_positive_vs_wrong_answer_preference" if preference_pair else "positive_completion_nll",
            "effective_shots": int(feature_item.get("effective_shots", 0)),
            "template_id": int(feature_item.get("template_id", 0)),
            "action_score_spread": score_spread,
            "k0_score": float(action_scores[0]),
            "teacher_tie_count": sum(abs(score - best_score) <= 1e-6 for score in action_scores),
        }
        targets.append(target)
        k_counts[len(best_route)] += 1
        route_counts["+".join(map(str, best_route)) if best_route else "k0"] += 1
        layer_counts.update(best_route)
        alpha_counts[str(best_alpha)] += 1
        model.set_prompt_shield_mask(None)
        del first_row, extras
    progress.close()

    model.set_routing_override("none", candidates)
    model.set_route_gains(None)
    model.set_prompt_shield_mask(None)
    if not targets:
        raise ValueError("no route teacher labels could be produced")
    return targets, {
        "rows": len(targets),
        "questions": len({int(target["question_index"]) for target in targets}),
        "prompt_groups_with_multiple_completions": sum(
            int(target["completions_scored"] > 1) for target in targets
        ),
        "preference_target_rows": sum(
            int(target.get("target_objective") == "matched_positive_vs_wrong_answer_preference")
            for target in targets
        ),
        "positive_nll_target_rows": sum(
            int(target.get("target_objective") == "positive_completion_nll")
            for target in targets
        ),
        "completion_rows_scored": sum(int(target["completions_scored"]) for target in targets),
        "k_counts": {str(k): k_counts[k] for k in range(5)},
        "route_counts": dict(route_counts),
        "layer_counts": {str(layer): layer_counts[layer] for layer in candidates},
        "alpha_counts": dict(alpha_counts),
        "low_signal_rows": low_signal_rows,
        "action_score_spread_mean": sum(action_score_spreads) / max(len(action_score_spreads), 1),
        "action_score_spread_min": min(action_score_spreads, default=0.0),
        "action_score_spread_max": max(action_score_spreads, default=0.0),
        "teacher_tie_rows": sum(int(target["teacher_tie_count"] > 1) for target in targets),
        "effective_shot_counts": dict(Counter(str(target["effective_shots"]) for target in targets)),
        "catalog_routes": [list(route) for route in routes],
        "catalog_actions": [
            {"route": list(route), "alpha": float(alpha)} for route, alpha in actions
        ],
    }


def _train_router_from_oracle(
    router: RouteValueRouter,
    targets: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, torch.Tensor]],
    model,
    probe: _Probe,
    device: torch.device,
    *,
    epochs: int,
    seed: int,
    save_path: str | Path,
    probe_layer: int,
    route_catalog: Sequence[Sequence[int]],
    validation_limit: int,
    route_cost: float,
    validation_preference_pairs: Sequence[dict[str, Any]] = (),
    bank_sha256: str | None = None,
    tokenizer=None,
    cfg=None,
    task_validation_limit: int = 32,
    static_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(router.parameters(), lr=5e-5, weight_decay=0.01)
    target_device = torch.device(device)
    prompt_features = torch.stack([item["pooled_prompt"] for item in targets]).to(target_device)
    context_features = torch.stack([item["context_features"] for item in targets]).to(target_device)
    teacher_scores = torch.stack([item["action_scores"] for item in targets]).to(target_device)
    if teacher_scores.shape[1] != len(router.action_catalog):
        raise ValueError("oracle action scores do not match the route action catalog")
    teacher_center = teacher_scores.mean(dim=1, keepdim=True)
    teacher_scale = teacher_scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(0.01)
    teacher_targets = (teacher_scores - teacher_center) / teacher_scale
    history: list[dict[str, Any]] = []
    best_state = None
    best_val_nll = float("inf")
    best_val_objective = float("inf")
    best_selection_value = float("-inf")
    k_counts: Counter[int] = Counter()
    route_counts: Counter[str] = Counter()
    layer_counts: Counter[int] = Counter()
    alpha_counts: Counter[str] = Counter()
    for target in targets:
        route = tuple(target["route"])
        k_counts[len(route)] += 1
        route_counts["+".join(map(str, route)) if route else "k0"] += 1
        layer_counts.update(route)
        alpha_counts[str(target["alpha"])] += 1

    for epoch in range(epochs):
        router.train()
        order = list(range(len(targets)))
        random.Random(seed + epoch).shuffle(order)
        batches = [order[start : start + 16] for start in range(0, len(order), 16)]
        progress = tqdm(
            batches,
            desc=f"Route-value router epoch {epoch + 1}/{epochs}",
            unit="batch",
            file=sys.stdout,
            mininterval=0.5,
        )
        losses = []
        for indices in progress:
            index_tensor = torch.tensor(indices, dtype=torch.long, device=target_device)
            pooled = prompt_features.index_select(0, index_tensor)
            context = context_features.index_select(0, index_tensor)
            target = teacher_targets.index_select(0, index_tensor)
            optimizer.zero_grad(set_to_none=True)
            predicted = router.score_pooled(pooled, context)
            regression_loss = F.smooth_l1_loss(predicted, target)
            teacher_distribution = F.softmax(-target / 0.5, dim=-1)
            student_log_distribution = F.log_softmax(-predicted / 0.5, dim=-1)
            preference_loss = F.kl_div(
                student_log_distribution, teacher_distribution, reduction="batchmean"
            )
            loss = regression_loss + 0.5 * preference_loss
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            progress.set_postfix(loss=f"{losses[-1]:.4f}")
        progress.close()
        validation = _evaluate_router_nll(
            model, router, validation_rows, probe, target_device, validation_limit,
            route_cost=route_cost,
        )
        if validation_preference_pairs:
            preference_validation = _evaluate_router_preference(
                model, router, validation_preference_pairs, probe, target_device,
                validation_limit, route_cost,
            )
            validation.update(preference_validation)
        if tokenizer is not None and cfg is not None:
            task_validation = _evaluate_router_task_accuracy(
                model,
                router,
                validation_rows,
                probe,
                tokenizer,
                cfg,
                target_device,
                limit=task_validation_limit,
                static_action=static_action,
            )
            validation.update(task_validation)
            selection_value = float(task_validation["validation_task_selection_delta_pp"])
            selection_metric = (
                "validation_task_accuracy_delta: "
                + task_validation["validation_task_selection_method"]
            )
            selection_is_better = selection_value > best_selection_value
        elif validation_preference_pairs:
            selection_value = -float(validation["validation_preference_loss"])
            selection_metric = "negative_validation_preference_loss"
            selection_is_better = selection_value > best_selection_value
        else:
            selection_value = -float(validation["validation_objective"])
            selection_metric = "validation_objective_nll_fallback"
            selection_is_better = selection_value > best_selection_value
        record = {
            "epoch": epoch + 1,
            "supervised_loss": sum(losses) / max(len(losses), 1),
            **validation,
            "optimizer_steps": len(losses),
            "training_rows": len(targets),
            "scored_route_actions": len(router.action_catalog),
            "selection_metric": selection_metric,
            "teacher_k_counts": {str(k): k_counts[k] for k in range(5)},
            "teacher_route_counts": dict(route_counts),
            "teacher_layer_counts": {str(layer): layer_counts[layer] for layer in router.candidate_layers},
            "teacher_alpha_counts": dict(alpha_counts),
        }
        history.append(record)
        print(f"[adaptive-router] {json.dumps(record)}", flush=True)
        if selection_is_better:
            best_val_nll = float(validation["validation_nll"])
            best_val_objective = float(validation["validation_objective"])
            best_selection_value = selection_value
            best_selection_metric = selection_metric
            best_state = {name: value.detach().cpu().clone() for name, value in router.state_dict().items()}
            _save_router_checkpoint(
                router, save_path, probe_layer, best_val_nll, route_catalog,
                best_val_objective=best_val_objective,
                bank_sha256=bank_sha256,
                selection_metric=selection_metric,
                selection_value=best_selection_value,
            )
    if best_state is not None:
        router.load_state_dict(best_state)
        _save_router_checkpoint(
            router, save_path, probe_layer, best_val_nll, route_catalog,
            best_val_objective=best_val_objective,
            bank_sha256=bank_sha256,
            selection_metric=selection_metric,
            selection_value=best_selection_value,
        )
    return {
        "router": router,
        "history": history,
        "best_validation_nll": best_val_nll,
        "best_validation_objective": best_val_objective,
        "best_checkpoint_selection_value": best_selection_value,
        "best_checkpoint_selection_metric": best_selection_metric,
        "checkpoint": str(save_path),
        "teacher_labels": {"rows": len(targets), "k_counts": {str(k): k_counts[k] for k in range(5)},
                           "route_counts": dict(route_counts), "layer_counts": dict(layer_counts),
                           "alpha_counts": dict(alpha_counts)},
    }


def train_adaptive_router(
    model,
    train_rows: Sequence[dict[str, torch.Tensor]],
    validation_rows: Sequence[dict[str, torch.Tensor]],
    candidate_layers: Sequence[int],
    device: torch.device | str,
    *,
    epochs: int,
    save_path: str | Path,
    probe_layer: int,
    seed: int = 42,
    max_train_rows: int = 512,
    validation_limit: int = 64,
    route_cost: float = 0.01,
    entropy_weight: float = 0.01,
    alpha_values: Sequence[float] = (1.0, 1.5, 2.0),
    route_catalog: Sequence[Sequence[int]] | None = None,
    preference_pairs: Sequence[dict[str, Any]] = (),
    validation_preference_pairs: Sequence[dict[str, Any]] = (),
    bank_sha256: str | None = None,
    tokenizer=None,
    cfg=None,
    task_validation_limit: int = 32,
    static_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Learn routes from measured catalog rewards, with pairwise fallback support."""
    if not train_rows or not validation_rows or epochs < 1:
        raise ValueError("training/validation rows and positive epochs are required")
    if len(candidate_layers) < 4:
        raise ValueError("adaptive routing needs at least four candidate layers")
    if probe_layer in candidate_layers:
        raise ValueError("the router probe layer must remain outside the steering candidates")

    target_device = torch.device(device)
    oracle_alphas = tuple(float(alpha) for alpha in alpha_values)
    if not oracle_alphas or any(not 1.0 <= alpha <= 2.0 for alpha in oracle_alphas):
        raise ValueError("router alpha values must be nonempty and lie in [1, 2]")
    if route_catalog is not None:
        router = RouteValueRouter(
            candidate_layers=candidate_layers,
            route_catalog=route_catalog,
            hidden_dim=int(model.base_model.config.hidden_size),
            max_active_layers=4,
            alpha_values=oracle_alphas,
        ).to(target_device)
    else:
        router = AdaptiveSubsetRouter(
            candidate_layers=candidate_layers,
            hidden_dim=int(model.base_model.config.hidden_size),
            n_backbone_layers=len(model._layers),
            max_active_layers=4,
            alpha_values=(1.0, 1.25, 1.5, 1.75, 2.0),
        ).to(target_device)
    for parameter in model.base_model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.steer_nets.parameters():
        parameter.requires_grad_(False)
    model.eval()
    probe = _Probe(model._layers[probe_layer])
    if route_catalog is not None:
        if max_train_rows < 1 or route_cost < 0.0:
            probe.close()
            raise ValueError("oracle router needs positive max_train_rows and non-negative route_cost")
        try:
            rows_by_question: dict[int, list[dict]] = {}
            for index, item in enumerate(train_rows):
                question_id = int(item.get("question_index", index))
                rows_by_question.setdefault(question_id, []).append(item)
            question_ids = list(rows_by_question)
            chooser = random.Random(seed)
            chooser.shuffle(question_ids)
            selected_question_ids = question_ids[:max_train_rows]
            selected_rows = [
                item
                for question_id in selected_question_ids
                for item in rows_by_question[question_id]
            ]
            selected_prompt_groups = _group_prompt_rows(selected_rows)
            targets, teacher_stats = _build_oracle_targets(
                model,
                selected_rows,
                route_catalog,
                router,
                probe,
                target_device,
                route_cost=route_cost,
                row_limit=len(selected_prompt_groups),
                preference_pairs=preference_pairs,
            )
            target_path = Path(save_path).with_name("session3_route_teacher_targets.pt")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_payload = {
                "format": "ars_route_value_targets_v1",
                "candidate_layers": list(candidate_layers),
                "action_catalog": [
                    {"route": list(route), "alpha": float(alpha)}
                    for route, alpha in router.action_catalog
                ],
                "route_cost": float(route_cost),
                "pooled_prompts": torch.stack([item["pooled_prompt"] for item in targets]),
                "context_features": torch.stack([item["context_features"] for item in targets]),
                "action_scores": torch.stack([item["action_scores"] for item in targets]),
                "question_indices": [item["question_index"] for item in targets],
                "completion_indices": [item["completion_indices"] for item in targets],
                "completions_scored": [item["completions_scored"] for item in targets],
                "effective_shots": [item["effective_shots"] for item in targets],
                "template_ids": [item["template_id"] for item in targets],
                "action_score_spreads": [item["action_score_spread"] for item in targets],
                "k0_scores": [item["k0_score"] for item in targets],
            }
            target_temporary = target_path.with_suffix(target_path.suffix + ".tmp")
            torch.save(target_payload, target_temporary)
            target_temporary.replace(target_path)
            result = _train_router_from_oracle(
                router,
                targets,
                validation_rows,
                model,
                probe,
                target_device,
                epochs=epochs,
                seed=seed,
                save_path=save_path,
                probe_layer=probe_layer,
                route_catalog=route_catalog,
                validation_limit=validation_limit,
                route_cost=route_cost,
                validation_preference_pairs=validation_preference_pairs,
                bank_sha256=bank_sha256,
                tokenizer=tokenizer,
                cfg=cfg,
                task_validation_limit=task_validation_limit,
                static_action=static_action,
            )
            result["teacher_data"] = teacher_stats
            result["teacher_unique_questions"] = len(selected_question_ids)
            report_path = Path(save_path).with_name("session3_router_training.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report = {
                key: value for key, value in result.items() if key != "router"
            }
            report.update({
                "route_cost": float(route_cost),
                "alpha_values": list(router.alpha_values),
                "route_catalog": [list(route) for route in route_catalog],
                "action_count": len(router.action_catalog),
                "teacher_target_file": str(target_path),
            })
            temporary = report_path.with_suffix(report_path.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            temporary.replace(report_path)
            result["training_report"] = str(report_path)
            return result
        finally:
            probe.close()
            model.set_prompt_shield_mask(None)
            model.set_route_gains(None)
            model.set_routing_override("none", candidate_layers)
    optimizer = torch.optim.AdamW(router.parameters(), lr=5e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=target_device.type == "cuda")
    rows = _distinct_question_rows(train_rows, max_train_rows)
    history: list[dict[str, float]] = []
    best_state = None
    best_val_nll = float("inf")
    best_val_objective = float("inf")

    try:
        for epoch in range(epochs):
            router.train()
            order = list(range(len(rows)))
            torch_generator = torch.Generator().manual_seed(seed + epoch)
            order = torch.randperm(len(order), generator=torch_generator).tolist()
            progress = tqdm(
                order,
                desc=f"Adaptive router epoch {epoch + 1}/{epochs}",
                unit="example",
                file=sys.stdout,
                mininterval=0.5,
            )
            losses: list[float] = []
            k_counts: Counter[int] = Counter()
            layer_counts: Counter[int] = Counter()
            alpha_counts: Counter[float] = Counter()
            for index in progress:
                item = rows[index]
                row, _extras = _prepare([item], target_device)
                model.set_routing_override("none", candidate_layers)
                model.set_route_gains(None)
                model.set_prompt_shield_mask(None)
                probe.hidden = None
                with torch.no_grad(), torch.autocast(
                    device_type=target_device.type,
                    enabled=target_device.type == "cuda",
                ):
                    model(
                        input_ids=row["input_ids"],
                        attention_mask=row["attention_mask"],
                        use_cache=False,
                    )
                if probe.hidden is None:
                    continue

                prompt_mask = row["labels"].eq(-100) & row["attention_mask"].bool()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=target_device.type,
                    enabled=target_device.type == "cuda",
                ):
                    action_a, log_prob_a, entropy_a = sample_route_actions(
                        router, probe.hidden, prompt_mask
                    )
                    action_b, log_prob_b, entropy_b = sample_route_actions(
                        router, probe.hidden, prompt_mask
                    )
                for action in (action_a, action_b):
                    k_counts[int(action.k.item())] += 1
                    layer_counts.update(
                        layer
                        for layer, selected in zip(candidate_layers, action.mask[0].tolist())
                        if selected
                    )
                    alpha_counts[float(action.alpha.item())] += 1

                prompt_shield = _extras.get("prompt_shield_mask")
                nll_a = _score_action(
                    model, row, action_a.mask, action_a.alpha, target_device, prompt_shield
                )
                nll_b = _score_action(
                    model, row, action_b.mask, action_b.alpha, target_device, prompt_shield
                )
                reward_a = -nll_a - route_cost * action_a.k.float()
                reward_b = -nll_b - route_cost * action_b.k.float()
                advantage = (reward_a - reward_b).detach()
                policy_loss = -((log_prob_a - log_prob_b) * advantage).mean()
                entropy = 0.5 * (entropy_a.mean() + entropy_b.mean())
                loss = policy_loss - entropy_weight * entropy
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
                progress.set_postfix(loss=f"{losses[-1]:.4f}", mean_k=f"{(action_a.k.float().mean() + action_b.k.float().mean()) / 2:.2f}")
                model.set_prompt_shield_mask(None)
                del row, _extras, nll_a, nll_b, loss
            progress.close()

            validation = _evaluate_router_nll(
                model, router, validation_rows, probe, target_device, validation_limit,
                route_cost=route_cost,
            )
            record = {
                "epoch": float(epoch + 1),
                "policy_loss": sum(losses) / max(len(losses), 1),
                **validation,
                "optimizer_steps": float(len(losses)),
                "k_counts": {str(k): k_counts[k] for k in range(5)},
                "layer_selection_counts": {
                    str(layer): layer_counts[layer] for layer in candidate_layers
                },
                "alpha_counts": {str(alpha): alpha_counts[alpha] for alpha in router.alpha_values},
            }
            history.append(record)
            print(f"[adaptive-router] {json.dumps(record)}", flush=True)
            if validation["validation_objective"] < best_val_objective:
                best_val_nll = float(validation["validation_nll"])
                best_val_objective = float(validation["validation_objective"])
                best_state = {name: value.detach().cpu().clone() for name, value in router.state_dict().items()}
                _save_router_checkpoint(
                    router, save_path, probe_layer, best_val_nll,
                    best_val_objective=best_val_objective,
                )

        if best_state is not None:
            router.load_state_dict(best_state)
    finally:
        probe.close()
        model.set_prompt_shield_mask(None)
        model.set_route_gains(None)
        model.set_routing_override("none", candidate_layers)

    if best_state is not None:
        _save_router_checkpoint(
            router, save_path, probe_layer, best_val_nll,
            best_val_objective=best_val_objective,
        )
    destination = Path(save_path)
    return {
        "router": router,
        "history": history,
        "best_validation_nll": best_val_nll,
        "best_validation_objective": best_val_objective,
        "checkpoint": str(destination),
    }


def load_adaptive_router(
    model, checkpoint_path: str | Path, device, *, expected_bank_sha256: str | None = None
):
    """Load router weights and return a prompt-prefill callback for generation."""
    target_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=True)
    if expected_bank_sha256 is not None and checkpoint.get("bank_sha256") != expected_bank_sha256:
        raise ValueError("router checkpoint was trained against a different steering-bank checkpoint")
    candidates = tuple(int(value) for value in checkpoint["candidate_layers"])
    if candidates != tuple(model.target_layers):
        raise ValueError("router candidate layers do not match the model's steering bank")
    if checkpoint.get("router_type") == "catalog_value":
        action_catalog = checkpoint.get("action_catalog", [])
        route_catalog = list(dict.fromkeys(
            tuple(int(layer) for layer in action["route"]) for action in action_catalog
        ))
        router = RouteValueRouter(
            candidate_layers=candidates,
            route_catalog=route_catalog,
            hidden_dim=int(model.base_model.config.hidden_size),
            max_active_layers=int(checkpoint["max_active_layers"]),
            alpha_values=checkpoint["alpha_values"],
        ).to(target_device)
        expected_actions = [
            {"route": list(route), "alpha": float(alpha)} for route, alpha in router.action_catalog
        ]
        if action_catalog != expected_actions:
            raise ValueError("router checkpoint action catalog ordering is inconsistent")
    else:
        router = AdaptiveSubsetRouter(
            candidate_layers=candidates,
            hidden_dim=int(model.base_model.config.hidden_size),
            n_backbone_layers=len(model._layers),
            max_active_layers=int(checkpoint["max_active_layers"]),
            alpha_values=checkpoint["alpha_values"],
        ).to(target_device)
    router.load_state_dict(checkpoint["router"])
    router.eval()
    probe = _Probe(model._layers[int(checkpoint["probe_layer"])])

    return router, _build_router_prefill(model, router, probe), probe
