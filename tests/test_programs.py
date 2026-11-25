import numpy as np
import pytest
import torch

from phenoniche.spatial import Settings
from phenoniche.spatial.programs import fit_programs, masked_squared_error, orthogonality


def test_masked_loss_matches_dense_reference():
    a = torch.tensor([[1., 2.], [3., 4.]])
    b = torch.tensor([[.5, 1.], [1., .5]])
    c = torch.tensor([[1., 2.], [2., 1.]])
    rows, neighbors = torch.tensor([0, 1]), torch.tensor([0, 1])
    observed = torch.tensor([[4., 0.], [0., 8.]])
    expected = ((a[rows] * b[neighbors]) @ c.T - observed).square().sum()
    # Observed zeros contribute, but neither absent (cell, type) pair contributes.
    got = masked_squared_error(
        a, b, c, rows, neighbors, observed.to_sparse_csr(), observed.square().sum()
    )
    torch.testing.assert_close(got, expected.double())


def test_orthogonality_is_scale_invariant_and_joint():
    a = torch.tensor([[1., 1.], [0., 0.]], requires_grad=True)
    b = torch.tensor([[2., 2.], [0., 0.]])
    c = torch.eye(2)
    loss_a, loss_bc = orthogonality(a, b, c)
    torch.testing.assert_close(loss_a, torch.tensor(2.))
    torch.testing.assert_close(loss_bc, torch.tensor(0.))
    for original, scaled in zip(
        (loss_a, loss_bc), orthogonality(a * 8, b * .25, c * torch.tensor([3., 4.])),
        strict=True,
    ):
        torch.testing.assert_close(original, scaled)
    # Sharing a neighbor type is allowed if the directed LR patterns differ.
    _, duplicate = orthogonality(a, b, b)
    torch.testing.assert_close(duplicate, torch.tensor(2.))


def _fit(batch_size, device="cpu", alpha=.02, beta=.02, iterations=18, tol=1e-4):
    data = [
        ([0], [[4., 0., 2.]]),
        ([0, 1], [[0., 3., 3.], [1., 1., 0.]]),
        ([1], [[0., 0., 0.]]),
        ([0, 1], [[3., 0., 0.], [0., 2., 2.]]),
        ([1], [[1., 3., 0.]]),
        ([], []),
        ([0], [[0., 1., 4.]]),
    ]
    visits = []

    def fiber(i):
        visits.append(i)
        categories, values = data[i]
        return np.array(categories, dtype=int), np.array(values, np.float32).reshape(-1, 3), None

    settings = Settings(
        cp_device=device, cp_batch_size=batch_size, cp_alpha=alpha,
        cp_beta=beta, rank=2, iterations=iterations, cp_tol=tol,
    )
    fit = fit_programs(np.arange(len(data)), fiber, 2, 3, settings, np.random.default_rng(73))
    assert fit is not None
    assert visits == list(range(len(data))) * (len(fit.history) + 1)
    assert fit.metrics["observed_entries"] == 24
    assert fit.metrics["centers"] == 7
    assert fit.metrics["centers_with_observations"] == 6
    assert np.array_equal(fit.factors[0][5], [0, 0])
    assert all(np.isfinite(f).all() and (f >= 0).all() for f in fit.factors)
    return fit


def test_full_batch_and_chunked_cp_match_and_reduce_loss():
    full, chunked = _fit(64), _fit(3)
    for a, b in zip(full.factors, chunked.factors, strict=True):
        np.testing.assert_allclose(a, b, atol=3e-6, rtol=3e-5)
    for fit in (full, chunked):
        assert fit.metrics["total_loss"] < fit.history[0]["total_loss"]
        assert fit.metrics["calibrated_alpha"] > 0
        assert fit.metrics["calibrated_beta"] > 0


def test_orthogonality_coefficients_can_be_disabled_independently_and_stop_early():
    for alpha, beta in ((0., 0.), (0., .02), (.02, 0.)):
        fit = _fit(4, alpha=alpha, beta=beta, tol=2.)
        assert (fit.metrics["calibrated_alpha"] > 0) == (alpha > 0)
        assert (fit.metrics["calibrated_beta"] > 0) == (beta > 0)
        assert fit.metrics["stopped_early"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_cp_agree():
    cpu, cuda = _fit(3), _fit(3, "cuda:0")
    for a, b in zip(cpu.factors, cuda.factors, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-4, rtol=5e-4)
    assert cuda.metrics["total_loss"] == pytest.approx(cpu.metrics["total_loss"], abs=1e-6)
