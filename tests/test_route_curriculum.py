from src.training.route_curriculum import candidate_routes, depth_band, select_routes_by_depth


LAYERS = [8, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24, 26]


def test_depth_bands_and_pair_pruning_keep_small_adjacent_control_set():
    assert [depth_band(layer) for layer in (8, 15, 24)] == ["early", "middle", "late"]
    pairs = candidate_routes(LAYERS, 2)
    assert (8, 15) in pairs
    assert (11, 12) in pairs
    assert (8, 11) not in pairs
    assert len(pairs) < len(LAYERS) * (len(LAYERS) - 1) // 2


def test_triples_grow_from_pairs_and_quads_cover_all_depth_bands():
    triples = candidate_routes(LAYERS, 3, selected_routes=[(8, 15), (15, 22)])
    assert triples
    assert all(len({depth_band(layer) for layer in route}) >= 2 for route in triples)
    assert all(all(right - left > 1 for left, right in zip(route, route[1:])) for route in triples)

    quads = candidate_routes(LAYERS, 4, selected_routes=[(8, 15, 22), (11, 18, 24)])
    assert quads
    assert all({depth_band(layer) for layer in route} == {"early", "middle", "late"} for route in quads)


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
