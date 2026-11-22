import pytest
import torch

from src.steering.route_bank import (
    normalize_route_catalog,
    route_key,
    validate_route_expert_states,
)


def _expert_state(route):
    return {
        f"{layer}.projection.weight": torch.ones(2, 2)
        for layer in route
    }


def test_route_catalog_canonicalizes_routes_and_keeps_noop_first():
    catalog = normalize_route_catalog([[4, 2], [], [3]], [2, 3, 4])

    assert catalog == ((), (2, 4), (3,))
    assert route_key(()) == "k0"
    assert route_key((4, 2)) == "route_2_4"


def test_route_catalog_rejects_duplicate_layers_and_routes():
    with pytest.raises(ValueError, match="layers must be unique"):
        normalize_route_catalog([[2, 2]], [2, 3])
    with pytest.raises(ValueError, match="duplicate routes"):
        normalize_route_catalog([[2, 3], [3, 2]], [2, 3])


def test_expert_state_keys_and_layer_scope_must_match_catalog():
    catalog = ((), (2, 4))
    state = {"route_2_4": _expert_state((2, 4))}
    validate_route_expert_states(catalog, state)

    with pytest.raises(ValueError, match="unexpected"):
        validate_route_expert_states(catalog, {**state, "route_3": _expert_state((3,))})
    with pytest.raises(ValueError, match="layer scope"):
        validate_route_expert_states(catalog, {"route_2_4": _expert_state((2, 3, 4))})


def test_expert_state_rejects_nonfinite_parameters():
    state = _expert_state((2,))
    state["2.projection.weight"][0, 0] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        validate_route_expert_states(((), (2,)), {"route_2": state})
