import numpy as np
import torch

from phenoniche.v1002.final_model import canonicalize_factors, fit_balanced
from phenoniche.v1002.simulation_cells import aggregate_pairwise_ccc


def _brute_force(xyz, types, ligand, receptor, weights, sigma, tau, n_types):
    opportunity = np.zeros((n_types, n_types), dtype=np.float64)
    numerator = np.zeros((n_types, n_types, ligand.shape[1]), dtype=np.float64)
    for u in range(len(types)):
        for v in range(len(types)):
            if u == v:
                continue
            kernel = np.exp(-np.sum((xyz[u] - xyz[v]) ** 2) / (2 * sigma ** 2))
            weight = weights[u] * weights[v] * kernel
            a, b = types[u], types[v]
            opportunity[a, b] += weight
            numerator[a, b] += weight * np.sqrt(ligand[u] * receptor[v])
    return numerator / (opportunity[:, :, None] + tau), opportunity


def test_pairwise_ccc_matches_brute_force_and_opportunity_normalization():
    xyz = np.array([[[0., 0.], [1., 0.], [2., 0.]]], dtype=np.float32)
    types = np.array([[0, 0, 1]], dtype=np.int8)
    ligand = np.array([[[1., 4.], [9., 1.], [4., 9.]]], dtype=np.float32)
    receptor = np.array([[[4., 1.], [1., 9.], [16., 4.]]], dtype=np.float32)
    weights = np.array([[1., 2., 3.]], dtype=np.float32)
    expected_i, expected_opp = _brute_force(xyz[0], types[0], ligand[0], receptor[0], weights[0], .8, 1e-4, 2)
    c, i, opp = aggregate_pairwise_ccc(xyz, types, ligand, receptor, 2, .8,
                                        anchor_weights=weights, batch_size=1, lr_batch_size=1)
    np.testing.assert_allclose(c[0], [3 / 6, 3 / 6], rtol=1e-6)
    np.testing.assert_allclose(opp[0].reshape(2, 2), expected_opp, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(i[0].reshape(2, 2, 2), expected_i, rtol=1e-6, atol=1e-6)
    assert opp[0, 0] > 0  # two distinct cells of the same type remain eligible
    assert opp[0, 3] == 0  # a lone type-1 cell cannot pair with itself


def test_duplicate_physical_cell_is_excluded_even_at_different_neighbor_positions():
    xyz = np.array([[0., 0.], [1., 0.]], dtype=np.float32)
    types = np.array([0, 1], dtype=np.int8)
    ligand = np.ones((2, 1), dtype=np.float32)
    receptor = np.ones_like(ligand)
    _, signal, opp = aggregate_pairwise_ccc(xyz, types, ligand, receptor, 2, 1.,
                                             members=np.array([[0, 0, 1]]), batch_size=1)
    assert opp[0, 0] == 0
    assert signal[0, 0] == 0
    assert opp[0, 1] > 0 and opp[0, 2] > 0


def test_canonicalization_preserves_every_reconstruction_and_single_views_run():
    w = torch.tensor([[.2, 3.], [2., .1]], dtype=torch.float64)
    h = {"HC": torch.tensor([[1., 4.], [2., 8.]], dtype=torch.float64),
         "HI": torch.tensor([[9., 1., 2.], [4., 1., 1.]], dtype=torch.float64)}
    canonical_w, canonical_h = canonicalize_factors(w, h)
    for name in h:
        torch.testing.assert_close(w @ h[name], canonical_w @ canonical_h[name], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        torch.sqrt(sum(value.square().mean(1) for value in canonical_h.values())),
        torch.ones(2, dtype=torch.float64), rtol=1e-12, atol=1e-12,
    )
    rng = np.random.default_rng(31)
    blocks = {"HC": rng.random((18, 3), dtype=np.float32),
              "HI": rng.random((18, 5), dtype=np.float32)}
    for name in ("HC", "HI"):
        fit = fit_balanced({name: blocks[name]}, seed=31, iterations=3, device="cpu", niches=2)
        activity = fit.W.numpy()
        assignment = (activity / np.maximum(activity.sum(1, keepdims=True), 1e-12)).argmax(1)
        assert assignment.shape == (18,) and np.isfinite(activity).all()
