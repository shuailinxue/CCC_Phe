import numpy as np
import torch
from scipy import sparse

from phenoniche.ccc.coarse_conditional import ConditionalCoherence, empirical_max_test, within_class_graph


def test_within_class_statistic_removes_between_class_mean():
    labels = np.array([0, 0, 1, 1])
    graph = sparse.csr_matrix(np.ones((4, 4), dtype=np.float32) - np.eye(4, dtype=np.float32))
    graph = within_class_graph(graph, labels)
    scorer = ConditionalCoherence(graph, labels, torch.ones((4, 2)), device='cpu')
    values = torch.tensor([[1., 2.], [1., 2.], [10., -4.], [10., -4.]])
    np.testing.assert_allclose(scorer.score(values).numpy(), 0, atol=1e-7)
    assert np.all(graph.tocoo().row // 2 == graph.tocoo().col // 2)


def test_max_statistic_calibration_on_independent_global_nulls():
    rng = np.random.default_rng(12)
    reference = rng.normal(size=(100, 3, 8))
    calibration = rng.normal(size=(500, 3, 8))
    valid = np.ones((3, 8), dtype=bool)
    heldout = rng.normal(size=(500, 3, 8))
    rejections = []
    for obs in heldout:
        _, p, _, _, _ = empirical_max_test(obs, reference, calibration, valid)
        rejections.append(np.any(p <= 0.05))
    # Monte Carlo guard against an uncorrected per-feature test.
    assert 0.005 < np.mean(rejections) < 0.12
