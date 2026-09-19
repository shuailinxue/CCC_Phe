from dataclasses import replace
import torch
from phenoniche.model.niche_factorization import NicheFactorization
from phenoniche.losses.survival import cox_breslow_loss
from phenoniche.training.config import TrainingConfig, LossWeights
from phenoniche.training.trainer import train, joint_loss


def test_cox_and_joint_gradients(tiny_data):
    model = NicheFactorization(24, 20, 4, 5, 6, 2, composition=tiny_data.CB, communication=tiny_data.IB)
    with torch.no_grad():
        model.gamma.copy_(torch.tensor([0.7, -0.6]))
    wb = model.bulk_factors()
    loss = cox_breslow_loss(wb @ model.gamma, tiny_data.time, tiny_data.event)
    gradients = torch.autograd.grad(loss, (wb, model.gamma, model.raw_HC, model.raw_HI), allow_unused=True)
    for gradient in gradients:
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    joint_loss(model, tiny_data, LossWeights())["total_loss"].backward()
    for parameter in (model.raw_HC, model.raw_HI, model.raw_WS):
        assert parameter.grad.abs().sum() > 0


def test_supervision_changes_shared_dictionaries_and_spatial_projection(tiny_data):
    config = TrainingConfig(number_of_niches=2, warmup_epochs=50, joint_epochs=150)
    weights = LossWeights.per_entry(tiny_data, ph=0.1)
    enabled = train(tiny_data, config, weights).model.factors()
    disabled = train(tiny_data, config, replace(weights, ph=0)).model.factors()
    for name in ("HC", "HI", "WS"):
        assert torch.linalg.vector_norm(enabled[name] - disabled[name]) > 1e-3


def test_disabled_supervision_ignores_outcomes(tiny_data):
    config = TrainingConfig(number_of_niches=2, warmup_epochs=5, joint_epochs=10)
    weights = LossWeights.per_entry(tiny_data, ph=0)
    original = train(tiny_data, config, weights).model.factors()
    changed = train(replace(tiny_data, time=tiny_data.time.flip(0), event=tiny_data.event.flip(0)), config, weights).model.factors()
    for name in original:
        torch.testing.assert_close(original[name], changed[name], atol=0, rtol=0)
    assert torch.equal(original["gamma"], torch.zeros(2))
