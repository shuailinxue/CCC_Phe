import pytest
import torch
from phenoniche.v1002.large_lr import SparseExpandedMatrix
from phenoniche.v1002.tensor_experiment import numerical_audit
from phenoniche.v1002.tensor_lr import exact_tensor_mse, fit_tensor_st, normalize_lr_programs


def test_tensor_exact_loss_and_gradients():
    audit = numerical_audit()
    assert audit["value_allclose"]
    assert audit["gradient_allclose"]


def test_lr_program_normalization_preserves_dictionary():
    torch.manual_seed(2)
    g = torch.rand(3, 4, 5)
    v = torch.rand(7, 5)
    before = torch.einsum("kpr,lr->kpl", g, v)
    normalized_g, normalized_v = normalize_lr_programs(g, v)
    after = torch.einsum("kpr,lr->kpl", normalized_g, normalized_v)
    assert torch.allclose(normalized_v.sum(0), torch.ones(5))
    assert torch.allclose(before, after, rtol=1e-6, atol=1e-6)


def test_tensor_training_fixes_rank_and_avoids_dense_reconstruction():
    base = torch.rand(8, 4)
    indices = torch.tensor([0, 7, 12])
    delta = torch.rand(8, 3)
    communication = SparseExpandedMatrix(base, 5, indices, delta)
    cs = torch.rand(8, 3)
    os = torch.rand(8, 2)
    with pytest.raises(ValueError, match="R=64"):
        fit_tensor_st(cs, communication, os, niches=2, rank=4, iterations=1, device="cpu")
    w = torch.rand(8, 2)
    g = torch.rand(2, 4, 3)
    v = torch.rand(5, 3)
    loss = exact_tensor_mse(communication, w, g, v)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
