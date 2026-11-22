from itertools import combinations

from src.training.route_curriculum import candidate_routes, depth_band, select_routes_by_depth


LAYERS = [8, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24, 26]


def test_depth_bands_and_pair_pruning_keep_bounded_spacing_controls():
    assert [depth_band(layer) for layer in (8, 15, 24)] == ["early", "middle", "late"]
    pairs = candidate_routes(LAYERS, 2)
    assert (8, 15) in pairs
    spacing_controls = [route for route in pairs if route[1] - route[0] == 2]
    assert len(spacing_controls) == 8
    assert (11, 13) in spacing_controls
    assert (11, 12) not in pairs
    assert (8, 11) in pairs
    assert len(pairs) < len(LAYERS) * (len(LAYERS) - 1) // 2


def test_triples_grow_from_pairs_and_quads_cover_all_depth_bands():
    triples = candidate_routes(LAYERS, 3, selected_routes=[(8, 15), (15, 22)])
    assert triples
    assert all(
        all(right - left >= 2 for left, right in combinations(route, 2))
        for route in triples
    )

    quads = candidate_routes(LAYERS, 4, selected_routes=[(8, 15, 22), (11, 18, 24)])
    assert quads
    assert all(
        all(right - left >= 2 for left, right in combinations(route, 2))
        for route in quads
    )


def test_depth_selection_preserves_band_patterns_before_global_fill():
    scores = [
        ((8, 15), 0.20),
        ((11, 22), 0.21),
        ((15, 22), 0.22),
        ((12, 17), 0.23),
    ]
    selected = select_routes_by_depth(scores, k=2, per_bucket=1, limit=3)
    assert (8, 15) in selected
    assert (11, 22) in selected
    assert (15, 22) in selected
