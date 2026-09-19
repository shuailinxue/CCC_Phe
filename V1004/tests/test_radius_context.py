import numpy as np

from phenoniche.v1002.simulation_cells import aggregate_pairwise_ccc
from phenoniche.v1004.radius_context import (
    aggregate_radius_pairwise_ccc, build_radius_context, radius_composition,
)


def test_radius_includes_cells_beyond_thirtieth_neighbor():
    xy = np.column_stack((np.arange(40, dtype=np.float32), np.zeros(40)))
    context = build_radius_context(xy, radius_um=200, anchor_sigma_um=20,
                                   query_batch_size=7)
    assert context.counts.tolist() == [40] * 40
    assert 39 in context.members(0)[0]
    composition = radius_composition(context, np.arange(40) % 2, 2)
    np.testing.assert_allclose(composition.sum(1), 1, atol=1e-6)


def test_exact_radius_pairwise_matches_existing_physical_pair_implementation():
    xy = np.array([[0., 0.], [8., 0.], [20., 4.], [75., 0.], [130., 0.]],
                  dtype=np.float32)
    types = np.array([0, 0, 1, 1, 0], dtype=np.int16)
    ligand = np.array([[1., 2.], [4., 1.], [0., 3.], [2., 5.], [6., 2.]],
                      dtype=np.float32)
    receptor = np.array([[2., 3.], [1., 1.], [5., 2.], [3., 4.], [1., 6.]],
                        dtype=np.float32)
    context = build_radius_context(xy, 200, 20, query_batch_size=2)
    members = np.tile(np.arange(5), (5, 1))
    weights = context.weights.reshape(5, 5)
    c_ref, i_ref, opp_ref = aggregate_pairwise_ccc(
        xy, types, ligand, receptor, 2, sigma=20,
        members=members, anchor_weights=weights, device='cpu', batch_size=2)
    i, opp = aggregate_radius_pairwise_ccc(
        xy, types, ligand, receptor, 2, context, sigma_um=20,
        pair_cutoff_um=None, device='cpu', batch_size=2, lr_batch_size=1)
    np.testing.assert_allclose(radius_composition(context, types, 2), c_ref,
                               rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(opp, opp_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(i, i_ref, rtol=1e-5, atol=1e-6)
    assert opp[0, 0] > 0  # same-cell-type physical pairs survive
    i_reused, opp_reused = aggregate_radius_pairwise_ccc(
        xy, types, ligand, receptor, 2, context, sigma_um=20,
        pair_cutoff_um=None, opportunity_input=opp, device='cpu', batch_size=2)
    assert opp_reused is opp
    np.testing.assert_allclose(i_reused, i, rtol=1e-6, atol=1e-6)


def test_pair_cutoff_applies_to_opportunity_and_signal_together():
    xy = np.array([[0., 0.], [5., 0.], [180., 0.]], dtype=np.float32)
    types = np.array([0, 0, 1], dtype=np.int16)
    l = np.ones((3, 1), dtype=np.float32)
    context = build_radius_context(xy, 200, 20)
    i, opp = aggregate_radius_pairwise_ccc(xy, types, l, l, 2, context,
                                            pair_cutoff_um=100, batch_size=2)
    assert opp[0, 0] > 0
    assert opp[0, 1] == 0 and opp[0, 2] == 0
    assert i[0, 1] == 0 and i[0, 2] == 0
