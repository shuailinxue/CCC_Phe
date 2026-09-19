import inspect
import pytest
import torch
from phenoniche.v1002.exact_loss import exact_chunked_frobenius_mse, guard_dense_materialization
from phenoniche.v1002.lr_atlas import load_lr_atlas
from phenoniche.v1002.bulk_bridge import directional_communication_potential
from phenoniche.v1002.large_lr import SparseExpandedMatrix


def test_exact_large_frobenius_loss():
    torch.manual_seed(4)
    observed = torch.rand(11, 37, dtype=torch.float64)
    activity = torch.rand(11, 6, dtype=torch.float64)
    dictionary = torch.rand(6, 37, dtype=torch.float64)
    dense = (observed - activity @ dictionary).square().mean()
    exact = exact_chunked_frobenius_mse(observed, activity, dictionary, feature_chunk_size=7)
    assert torch.allclose(dense, exact, atol=1e-12, rtol=1e-12)


def test_exact_loss_gradient_equivalence():
    torch.manual_seed(8)
    observed = torch.rand(9, 23, dtype=torch.float64)
    activity = torch.rand(9, 4, dtype=torch.float64, requires_grad=True)
    dictionary = torch.rand(4, 23, dtype=torch.float64, requires_grad=True)
    dense = (observed - activity @ dictionary).square().mean()
    dense_gradients = torch.autograd.grad(dense, (activity, dictionary), retain_graph=True)
    exact = exact_chunked_frobenius_mse(observed, activity, dictionary, feature_chunk_size=5)
    exact_gradients = torch.autograd.grad(exact, (activity, dictionary))
    assert all(torch.allclose(left, right, atol=1e-11, rtol=1e-11)
               for left, right in zip(dense_gradients, exact_gradients))


def test_full_atlas_memory_guard():
    full_features = 8 * 8 * 9311
    with pytest.raises(MemoryError):
        guard_dense_materialization(1200, full_features, copies=1, limit_bytes=1_000_000_000)
    assert guard_dense_materialization(6, full_features, copies=1, limit_bytes=100_000_000) < 100_000_000


def test_no_phenotype_in_st():
    source = inspect.getsource(exact_chunked_frobenius_mse)
    assert all(term not in source for term in ("survival", "event", "phenotype", "cox"))


def test_implicit_large_matrix_equivalence():
    base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    indices = torch.tensor([1, 4])
    delta = torch.tensor([[0.5, 0.2], [0.7, 0.3]])
    matrix = SparseExpandedMatrix(base, 3, indices, delta)
    dense = base[:, :, None].expand(-1, -1, 3).clone().reshape(2, 6)
    dense.index_add_(1, indices, delta)
    activity = torch.rand(2, 2)
    dictionary = torch.rand(2, 6)
    assert torch.allclose(matrix.sum(), dense.sum())
    assert torch.allclose(matrix.square_sum(), dense.square().sum())
    assert torch.allclose(matrix.wt_x(activity), activity.T @ dense)
    assert torch.allclose(matrix.x_h_t(dictionary), dense @ dictionary.T)
    assert torch.allclose(matrix.dense_chunk(1, 3).reshape(2, -1), dense.reshape(2, 2, 3)[:, :, 1:3].reshape(2, -1))
