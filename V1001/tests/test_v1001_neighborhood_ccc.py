import numpy as np
from phenoniche.simulation.structured_ccc import decode_feature, feature_index
from phenoniche.v1001.neighborhoods import build_neighborhoods
from phenoniche.v1001.spatial_ccc import aggregate_spatial_ccc, brute_force_spatial_ccc


def _toy():
    anchors = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    cells = np.array([[-0.2, 0.0], [0.2, 0.0], [0.8, 0.0], [1.2, 0.0], [1.8, 0.0], [2.2, 0.0],
                      [1.0, 0.3], [1.0, -0.3]])
    cell_types = np.array([0, 1, 0, 2, 1, 2, 1, 0])
    parent = np.array([0, 0, 1, 1, 2, 2, 1, 1])
    niche = np.array([[0.8, 0.2], [0.7, 0.3], [0.4, 0.6], [0.3, 0.7],
                      [0.2, 0.8], [0.1, 0.9], [0.5, 0.5], [0.6, 0.4]])
    hi = np.arange(2 * 3 * 3 * 2, dtype=float).reshape(2, 3, 3, 2) / 40
    neighborhoods = build_neighborhoods(anchors, cells, radius=1.05, sigma=0.7)
    return neighborhoods, cell_types, parent, niche, hi


def test_neighborhood_membership():
    neighborhoods, *_ = _toy()
    first, weights, distances = neighborhoods.members(0)
    assert set(first.tolist()) == {0, 1, 2, 6, 7}
    assert np.all(weights > 0) and np.all(distances <= 1.05)


def test_composition_uses_same_neighborhood():
    neighborhoods, cell_types, parent, niche, hi = _toy()
    result = aggregate_spatial_ccc(neighborhoods, cell_types, parent, niche, hi)
    cells, weights, _ = neighborhoods.members(1)
    expected = np.bincount(cell_types[cells], weights=weights, minlength=3) / weights.sum()
    assert np.allclose(result.composition[1], expected)
    assert result.audit["mean_cells_per_neighborhood"] == np.diff(neighborhoods.offsets).mean()


def test_directed_pairs():
    neighborhoods, cell_types, parent, niche, hi = _toy()
    result = aggregate_spatial_ccc(neighborhoods, cell_types, parent, niche, hi)
    programs = hi.shape[-1]
    forward = feature_index(0, 1, 0, 3, programs)
    reverse = feature_index(1, 0, 0, 3, programs)
    assert (result.opportunity[:, forward] > 0).any()
    assert (result.opportunity[:, reverse] > 0).any()
    assert not np.allclose(result.raw[:, forward], result.raw[:, reverse])


def test_no_self_pairs():
    anchors = np.array([[0.0, 0.0]])
    cells = np.array([[-0.1, 0.0], [0.1, 0.0]])
    neighborhoods = build_neighborhoods(anchors, cells, radius=0.5, sigma=1.0)
    cell_types = np.array([0, 0])
    parent = np.array([0, 0])
    niche = np.ones((2, 1))
    hi = np.ones((1, 1, 1, 1))
    result = aggregate_spatial_ccc(neighborhoods, cell_types, parent, niche, hi, baseline=0, pair_sigma=1.0)
    _, weights, _ = neighborhoods.members(0)
    pair_kernel = np.exp(-(0.2 ** 2) / 2)
    assert np.allclose(result.opportunity[0, 0], 2 * weights[0] * weights[1] * pair_kernel)
    assert result.audit["local_directed_pair_instances"] == 2


def test_overlapping_neighborhood_pair_counting():
    neighborhoods, *_ = _toy()
    first = neighborhoods.members(0)[0]
    second = neighborhoods.members(1)[0]
    assert 1 in first and 2 in first and 1 in second and 2 in second
    assert len(first) == len(np.unique(first))
    assert len(second) == len(np.unique(second))


def test_pair_opportunity():
    neighborhoods, cell_types, parent, niche, hi = _toy()
    result = aggregate_spatial_ccc(neighborhoods, cell_types, parent, niche, hi)
    assert np.all(result.opportunity >= 0)
    assert np.allclose(result.opportunity[:, 0::2], result.opportunity[:, 1::2])


def test_bruteforce_ccc_equivalence():
    arguments = _toy()
    optimized = aggregate_spatial_ccc(*arguments)
    reference = brute_force_spatial_ccc(*arguments)
    for name in ("composition", "raw", "opportunity", "normalized", "peripheral_raw"):
        assert np.allclose(getattr(optimized, name), getattr(reference, name), atol=1e-12)


def test_peripheral_peripheral_edges_present():
    neighborhoods, cell_types, parent, niche, hi = _toy()
    result = aggregate_spatial_ccc(neighborhoods, cell_types, parent, niche, hi)
    assert result.audit["peripheral_pair_instances"] > 0
    assert result.audit["peripheral_signal_fraction"] > 0
    assert np.count_nonzero(result.peripheral_raw) > 0


def test_feature_mapping_sender_receiver_program():
    for sender in range(3):
        for receiver in range(3):
            for program in range(2):
                index = feature_index(sender, receiver, program, 3, 2)
                assert decode_feature(index, 3, 2) == (sender, receiver, program)


def test_opportunity_normalization():
    neighborhoods, cell_types, parent, niche, hi = _toy()
    result = aggregate_spatial_ccc(neighborhoods, cell_types, parent, niche, hi, tau=0.01)
    assert np.allclose(result.normalized, result.raw / (result.opportunity + 0.01))
