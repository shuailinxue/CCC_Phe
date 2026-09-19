import torch
import pytest
from phenoniche.training.config import LossWeights
from phenoniche.training.dual_bank_trainer import DualBankTrainingConfig, background_loss, train_background, train_residual_from_background


def test_residual_stage_freezes_background_and_disables_collapse(tiny_data):
    config = DualBankTrainingConfig(1, 1, background_epochs=2, phenotype_epochs=3,
                                    jointft_epochs=0, scratch_epochs=3, inner_steps=5)
    weights = LossWeights.per_entry(tiny_data, ph=0.1, reg=0.0001, lambda_collapse=0)
    background = train_background(tiny_data, config, weights)
    state = {name: value.clone() for name, value in background.model.state_dict().items()}
    frozen, joint = train_residual_from_background(tiny_data, config, weights, state, jointft=False)
    assert joint is None
    for name in ("raw_HC0", "raw_HI0", "raw_HO0", "raw_WS0"):
        assert torch.equal(state[name], frozen.model.state_dict()[name])
        assert getattr(frozen.model, name).grad is None
    bad = LossWeights.per_entry(tiny_data, ph=0.1, lambda_collapse=0.01)
    with pytest.raises(ValueError, match="lambda_collapse=0"):
        train_background(tiny_data, config, bad)


def test_background_loss_never_reads_survival(tiny_data):
    config = DualBankTrainingConfig(1, 1, background_epochs=1, phenotype_epochs=1,
                                    jointft_epochs=0, scratch_epochs=1, inner_steps=3)
    weights = LossWeights.per_entry(tiny_data, ph=0.1)
    trained = train_background(tiny_data, config, weights)

    class BackgroundOnlyData:
        def __getattr__(self, name):
            if name in ("time", "event"):
                raise AssertionError("Background warmup read survival")
            return getattr(tiny_data, name)

    loss = background_loss(trained.model, BackgroundOnlyData(), weights)
    assert torch.isfinite(loss["total_loss"])
