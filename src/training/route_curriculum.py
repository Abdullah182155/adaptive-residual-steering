"""Depth-aware candidate route construction for adaptive steering training."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence


def depth_band(layer: int) -> str:
    """Map Phi-2 layer indices to coarse early, middle, and late bands."""
    if layer < 14:
        return "early"
    if layer < 22:
        return "middle"
    return "late"


def band_signature(route: Sequence[int]) -> tuple[str, ...]:
    """Return the ordered unique depth bands represented by a route."""
    return tuple(dict.fromkeys(depth_band(layer) for layer in sorted(route)))


def candidate_routes(
    candidate_layers: Sequence[int],
    k: int,
    *,
    selected_routes: Sequence[Sequence[int]] = (),
    min_layer_gap: int = 3,
    control_gap: int = 2,
    spacing_control_limit: int = 8,
) -> list[tuple[int, ...]]:
    """Build separated routes plus a small, deterministic spacing-control set.

    Every route in the primary set has at least ``min_layer_gap`` between any
    two selected layers. Controls satisfy ``control_gap`` but violate the primary
    gap; only a bounded, evenly spread sample is retained. Higher-cardinality
    routes are grown from the previous stage's selected routes when provided.
    """
    layers = tuple(sorted(set(int(layer) for layer in candidate_layers)))
    if not layers or k < 1 or k > 4 or k > len(layers):
        raise ValueError("candidate layers and k must define a valid route stage")
    if min_layer_gap < 1 or control_gap < 1 or control_gap >= min_layer_gap:
        raise ValueError("spacing requires 1 <= control_gap < min_layer_gap")
    if spacing_control_limit < 0:
        raise ValueError("spacing_control_limit must be nonnegative")
    all_routes = list(combinations(layers, k))
    if k == 1:
        return all_routes

    previous = [tuple(sorted(int(layer) for layer in route)) for route in selected_routes]
    if previous:
        generated = set()
        for route in previous:
            if len(route) != k - 1:
                continue
            for layer in layers:
                if layer in route:
                    continue
                grown = tuple(sorted((*route, layer)))
                if len(set(grown)) != k:
                    continue
                generated.add(grown)
        all_routes = sorted(generated)

    def has_minimum_gap(route: tuple[int, ...], gap: int) -> bool:
        return all(right - left >= gap for left, right in combinations(route, 2))

    primary = [route for route in all_routes if has_minimum_gap(route, min_layer_gap)]
    controls = [
        route for route in all_routes
        if not has_minimum_gap(route, min_layer_gap) and has_minimum_gap(route, control_gap)
    ]
    if controls and spacing_control_limit:
        if len(controls) > spacing_control_limit:
            indices = [
                round(index * (len(controls) - 1) / max(spacing_control_limit - 1, 1))
                for index in range(spacing_control_limit)
            ]
            controls = [controls[index] for index in sorted(set(indices))]
    else:
        controls = []
    return sorted(set(primary + controls))


def select_routes_by_depth(
    scored_routes: Sequence[tuple[Sequence[int], float]],
    *,
    k: int,
    per_bucket: int,
    limit: int,
) -> list[tuple[int, ...]]:
    """Keep strong routes while balancing depth patterns and layer coverage."""
    if k < 1 or k > 4 or per_bucket < 1 or limit < 0:
        raise ValueError("k, per_bucket, and limit must define a valid selection")
    if limit == 0:
        return []
    buckets: dict[tuple[str, ...], list[tuple[tuple[int, ...], float]]] = {}
    normalized_scores = []
    for route, score in scored_routes:
        normalized = tuple(int(layer) for layer in route)
        if len(normalized) != k or len(set(normalized)) != k:
            raise ValueError(f"expected unique routes containing exactly k={k} layers")
        normalized_scores.append((normalized, float(score)))
    for route, score in sorted(normalized_scores, key=lambda item: item[1]):
        key = band_signature(route)
        buckets.setdefault(key, []).append((route, score))
    selected: list[tuple[int, ...]] = []
    selected_set: set[tuple[int, ...]] = set()
    bucket_order = sorted(buckets, key=lambda key: buckets[key][0][1])
    for key in bucket_order:
        for route, _score in buckets[key][:per_bucket]:
            if route not in selected_set:
                selected.append(route)
                selected_set.add(route)
                if len(selected) >= limit:
                    return selected

    remaining = sorted(
        ((route, score) for route, score in normalized_scores if route not in selected_set),
        key=lambda item: item[1],
    )
    layer_counts: dict[int, int] = {}
    for route in selected:
        for layer in route:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
    while remaining and len(selected) < limit:
        route, _score = min(
            remaining,
            key=lambda item: (
                sum(layer_counts.get(layer, 0) for layer in item[0]) / max(k, 1),
                item[1],
            ),
        )
        selected.append(route)
        for layer in route:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        remaining = [item for item in remaining if item[0] != route]
    return selected


def build_router_route_catalog(
    selected_by_k: dict[int, Sequence[Sequence[int]]],
    scored_by_k: dict[int, Sequence[tuple[Sequence[int], float]]],
    *,
    singleton_limit: int = 4,
    routes_per_k: int = 2,
) -> list[list[int]]:
    """Build a compact diverse action catalog, preserving k=0 as a control."""
    if singleton_limit < 0 or routes_per_k < 1:
        raise ValueError("router catalog limits must be nonnegative/positive")
    catalog: list[list[int]] = [[]]
    singleton_scores = sorted(scored_by_k.get(1, ()), key=lambda item: float(item[1]))
    singles = select_routes_by_depth(
        singleton_scores,
        k=1,
        per_bucket=1,
        limit=min(singleton_limit, len(singleton_scores)),
    ) if singleton_scores and singleton_limit else []
    catalog.extend([list(route) for route in singles])
    for k in (2, 3, 4):
        scored = scored_by_k.get(k, ())
        if not scored:
            scored = [(route, float(index)) for index, route in enumerate(selected_by_k.get(k, ()))]
        routes = select_routes_by_depth(
            scored,
            k=k,
            per_bucket=1,
            limit=min(routes_per_k, len(scored)),
        ) if scored else []
        catalog.extend([list(route) for route in routes])
    return catalog
