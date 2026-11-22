"""Shared helpers and validation for independent route-specific adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def route_key(route: Sequence[int]) -> str:
    """Return a stable module/checkpoint key for a layer route."""
    normalized = tuple(sorted(set(int(layer) for layer in route)))
    if len(normalized) != len(route):
        raise ValueError("route layers must be unique")
    return "k0" if not normalized else "route_" + "_".join(map(str, normalized))


def normalize_route_catalog(
    route_catalog: Sequence[Sequence[int]], candidate_layers: Sequence[int], *, max_k: int = 4
) -> tuple[tuple[int, ...], ...]:
    """Validate and canonicalize a route catalog, keeping k=0 at index zero."""
    candidates = tuple(int(layer) for layer in candidate_layers)
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate layers must be unique")
    candidate_set = set(candidates)
    normalized = []
    for route in route_catalog:
        raw_route = tuple(int(layer) for layer in route)
        if len(set(raw_route)) != len(raw_route):
            raise ValueError("route layers must be unique")
        normalized.append(tuple(sorted(raw_route)))
    if len(set(normalized)) != len(normalized):
        raise ValueError("route catalog contains duplicate routes")
    if any(
        len(route) > max_k or not set(route).issubset(candidate_set)
        for route in normalized
    ):
        raise ValueError("route catalog contains a route outside the candidate-layer set")
    if () not in normalized:
        normalized.insert(0, ())
    else:
        normalized.remove(())
        normalized.insert(0, ())
    return tuple(normalized)


def validate_route_expert_states(
    route_catalog: Sequence[Sequence[int]],
    expert_states: Mapping[str, Mapping[str, torch.Tensor]],
) -> None:
    """Validate catalog-to-expert coverage and each expert's exact layer scope."""
    expected = {route_key(route): tuple(route) for route in route_catalog if route}
    missing = sorted(set(expected) - set(expert_states))
    unexpected = sorted(set(expert_states) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"route expert keys do not match the catalog: missing={missing}, unexpected={unexpected}"
        )
    for key, route in expected.items():
        state = expert_states[key]
        if not isinstance(state, Mapping) or not state:
            raise ValueError(f"route expert {key} has no state dictionary")
        layer_ids = set()
        for name, value in state.items():
            prefix, separator, suffix = str(name).partition(".")
            if not separator or not suffix or not prefix.isdigit():
                raise ValueError(f"route expert {key} has an invalid state key: {name!r}")
            layer_ids.add(int(prefix))
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"route expert {key} state {name!r} is not a tensor")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"route expert {key} state {name!r} contains non-finite values")
        if layer_ids != set(route):
            raise ValueError(
                f"route expert {key} layer scope does not match its route: "
                f"expected={list(route)}, found={sorted(layer_ids)}"
            )
