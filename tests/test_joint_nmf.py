import numpy as np
import pytest
import torch

from phenoniche.controls.joint_nmf import JointNMF, JointNMFConfig, fit_joint_nmf, reconstruction_metrics


def test_both_decoders_update_the_same_activity_parameters():
    model = JointNMF(np.array([[1., 3.], [4., 2.]]), np.eye(2),
                     np.array([[.8, .2], [.3, .7]]), np.ones(2))
    result = model()
    grad_x = torch.autograd.grad(result['reconstruction'][0, 0], model.raw_activity, retain_graph=True)[0]
    grad_c = torch.autograd.grad(result['composition'][0, 0], model.raw_activity)[0]
    assert grad_x[0].abs().sum() > 0
    assert grad_c[0].abs().sum() > 0
    assert (grad_x[1] == 0).all() and (grad_c[1] == 0).all()
    torch.testing.assert_close(result['P'], result['Z']/result['Z'].sum(1, keepdim=True))
    torch.testing.assert_close(result['H'].sum(1), torch.ones(2))
    torch.testing.assert_close(result['B'].sum(1), torch.ones(2))


def test_dual_factorization_learns_composition_and_preserves_nonnegativity():
    rng = np.random.default_rng(17)
    p = np.c_[np.linspace(.05, .95, 32), np.linspace(.95, .05, 32)]
    x = (p @ np.array([[4., .2, 2.], [.2, 5., 1.]])).astype(np.float32)
    c = p.astype(np.float32)
    before_x, before_c = x.copy(), c.copy()
    config = JointNMFConfig(programs=2, epochs=100, threads=1, regularization_weight=0.)
    result = fit_joint_nmf(x, c, config)
    assert result['history'][-1]['composition'] < result['history'][0]['composition']*.7
    assert result['history'][-1]['total'] < result['history'][0]['total']
    for key in ('Z', 'P', 'H', 'B'):
        assert np.isfinite(result[key]).all() and (result[key] >= 0).all()
    np.testing.assert_array_equal(x, before_x)
    np.testing.assert_array_equal(c, before_c)
    metrics = reconstruction_metrics(x, c, result, np.sqrt((x*x).mean(0)))
    assert metrics['fraction_mse_removed'] > .8


def test_rejects_unaligned_or_invalid_composition():
    x = np.ones((5, 4), dtype=np.float32)
    with pytest.raises(ValueError, match='row-normalized'):
        fit_joint_nmf(x, np.ones((5, 2)), JointNMFConfig(programs=2, epochs=1))
