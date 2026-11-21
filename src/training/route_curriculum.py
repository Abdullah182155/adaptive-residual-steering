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
    adjacent_probe_limit: int = 3,
) -> list[tuple[int, ...]]:
    """Build a pruned route set for one curriculum stage.

    Singles include every layer. Pairs prioritize cross-band, non-adjacent
    layers; a small set of adjacent pairs remains as a synergy control. Triples
    must span at least two bands. Quads must span all three bands. Higher stages
    are grown only from routes selected at the previous stage.
    """
    layers = tuple(sorted(set(int(layer) for layer in candidate_layers)))
    if not layers or k < 1 or k > 4 or k > len(layers):
        raise ValueError("candidate layers and k must define a valid route stage")
    all_routes = list(combinations(layers, k))
    if k == 1:
        return all_routes

    if k == 2:
        cross_band = [
            route for route in all_routes
            if depth_band(route[0]) != depth_band(route[1])
            and route[1] - route[0] >= 2
        ]
        adjacent = [route for route in all_routes if route[1] - route[0] == 1]
        grouped_adjacent: dict[str, list[tuple[int, ...]]] = {}
        for route in adjacent:
            grouped_adjacent.setdefault(depth_band(route[0]), []).append(route)
        controls = []
        for band in ("early", "middle", "late"):
            group = grouped_adjacent.get(band, [])
            if group and len(controls) < adjacent_probe_limit:
                controls.append(group[len(group) // 2])
        if len(controls) < adjacent_probe_limit:
            remaining = [route for route in adjacent if route not in controls]
            slots = adjacent_probe_limit - len(controls)
            if remaining and slots:
                positions = [round(index * (len(remaining) - 1) / max(slots - 1, 1)) for index in range(slots)]
                controls.extend(remaining[position] for position in positions)
        return sorted(set(cross_band + controls))

    required_bands = 3 if k == 4 else 2
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

    routes = []
    for route in all_routes:
        if len(set(band_signature(route))) < required_bands:
            continue
        # Adjacent layers are excluded at higher cardinalities unless they are
        # part of a previously measured and retained synergistic route.
        retained_adjacent_pairs = {
            frozenset(pair)
            for prior in previous
            for pair in combinations(prior, 2)
            if pair[1] - pair[0] == 1
        }
        if any(
            right - left == 1 and frozenset((left, right)) not in retained_adjacent_pairs
            for left, right in zip(route, route[1:])
        ):
            continue
        routes.append(route)
    return routes


def select_routes_by_depth(
    scored_routes: Sequence[tuple[Sequence[int], float]],
    *,
    k: int,
    per_bucket: int,
    limit: int,
) -> list[tuple[int, ...]]:
    """Keep the lowest-loss routes while preserving depth-pattern coverage."""
    if k < 1 or k > 4 or per_bucket < 1 or limit < 0:
        raise ValueError("k, per_bucket, and limit must define a valid selection")
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
    bucket_order = sorted(buckets, key=lambda key: buckets[key][0][1])
    for key in bucket_order:
        selected.extend(route for route, _ in buckets[key][:per_bucket])
    if len(selected) < limit:
        selected_set = set(selected)
        for route, _score in sorted(normalized_scores, key=lambda item: item[1]):
            normalized = tuple(route)
            if normalized not in selected_set:
                selected.append(normalized)
                selected_set.add(normalized)
                if len(selected) >= limit:
                    break
    return selected[:limit]
