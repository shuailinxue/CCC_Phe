import torch
from phenoniche.model.dual_bank_factorization import DualBankFactorization
from phenoniche.training.dual_bank_trainer import DualBankTrainingConfig


def test_dual_bank_shapes_and_parameter_contract(tiny_data):
    model = DualBankFactorization(24, 20, 4, 5, 6, 2, 3,
                                  composition=tiny_data.CB, communication=tiny_data.IB)
    output = model()
    parameters = dict(model.named_parameters())
    assert not any("WB0" in name or "WBp" in name for name in parameters)
    assert model.gamma.shape == (2,)
    assert output["factors"]["WB0"].shape == (24, 3)
    assert output["factors"]["WBp"].shape == (24, 2)
    assert output["CB0"].shape == output["CBp"].shape == tiny_data.CB.shape
    assert output["IS0"].shape == output["ISp"].shape == tiny_data.IS.shape
    assert all(torch.isfinite(value).all() for value in output.values() if isinstance(value, torch.Tensor))


def test_total_factor_count_is_explicit():
    same = DualBankTrainingConfig(phenotype_niches=2, background_niches=4)
    capacity = DualBankTrainingConfig(phenotype_niches=2, background_niches=2)
    assert same.total_niches == 6
    assert capacity.total_niches == 4
