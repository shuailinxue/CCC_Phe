from dataclasses import replace
import pytest
import torch
from phenoniche.losses.spatial import spatial_loss
from phenoniche.losses.reconstruction import squared_frobenius
from phenoniche.training.config import TrainingConfig, LossWeights
from phenoniche.training.trainer import train, joint_loss
from phenoniche.model.niche_factorization import NicheFactorization
from phenoniche.evaluation.metrics import concordance_index


@pytest.mark.parametrize("sparse", [False, True])
def test_laplacian_energy_and_gradients(sparse):
    laplacian = torch.tensor([[1., -1., 0.], [-1., 3., -2.], [0., -2., 2.]])
    activity = torch.tensor([[1., 2.], [3., 1.], [2., 4.]], requires_grad=True)
    expected = (activity[0] - activity[1]).square().sum() + 2 * (activity[1] - activity[2]).square().sum()
    loss = spatial_loss(activity, laplacian.to_sparse() if sparse else laplacian)
    torch.testing.assert_close(loss, expected)
    gradient = torch.autograd.grad(loss, activity)[0]
    torch.testing.assert_close(gradient, 2 * laplacian @ activity)


def test_invalid_laplacian():
    with pytest.raises(ValueError, match="Laplacian"):
        spatial_loss(torch.ones(3, 2), torch.eye(3))


def test_exact_frobenius_sum():
    predicted = torch.tensor([[1., 2.], [3., 4.]])
    assert squared_frobenius(predicted, torch.zeros_like(predicted)) == 30


def test_graph_required(tiny_data):
    model = NicheFactorization(24, 20, 4, 5, 6, 2)
    with pytest.raises(ValueError, match="graph"):
        joint_loss(model, tiny_data, LossWeights(sp=1))


def test_early_stopping_resets_for_joint_stage(tiny_data):
    config = TrainingConfig(number_of_niches=2, warmup_epochs=30, joint_epochs=30,
                            early_stopping=2, min_delta=1e9)
    result = train(tiny_data, config, LossWeights.per_entry(tiny_data))
    assert len(result.history) == 6
    assert {item["stage"] for item in result.history} == {"warmup", "joint"}
    assert result.model.gamma.abs().sum() > 0


def test_reproducibility(tiny_data):
    config = TrainingConfig(number_of_niches=2, warmup_epochs=2, joint_epochs=3)
    a = train(tiny_data, config, LossWeights.per_entry(tiny_data))
    b = train(tiny_data, config, LossWeights.per_entry(tiny_data))
    for name in a.model.state_dict():
        assert torch.equal(a.model.state_dict()[name], b.model.state_dict()[name])


def test_concordance_orientation_and_censor_ties():
    assert concordance_index([1., 2., 3.], [1, 1, 0], [3., 2., 1.]) == 1
    assert concordance_index([1., 2., 3.], [1, 1, 0], [1., 2., 3.]) == 0
    assert concordance_index([1., 1.], [1, 0], [2., 1.]) == 1


def test_weighted_joint_objective_matches_definition(tiny_data):
    model = NicheFactorization(24, 20, 4, 5, 6, 2)
    weights = LossWeights(bc=0.2, bi=0.3, sc=0.4, so=0.5, si=0.6, ph=0.7, reg=0.01)
    output = model(tiny_data.CB, tiny_data.IB)
    from phenoniche.losses.survival import cox_breslow_loss
    expected = sum(weight * (output[name] - getattr(tiny_data, name)).square().sum()
                   for name, weight in (("CB", 0.2), ("IB", 0.3), ("CS", 0.4), ("OS", 0.5), ("IS", 0.6)))
    expected += 0.7 * cox_breslow_loss(output["risk"], tiny_data.time, tiny_data.event)
    expected += 0.01 * sum(value.square().sum() for value in output["factors"].values())
    torch.testing.assert_close(joint_loss(model, tiny_data, weights)["total_loss"], expected)


def test_spatial_penalty_training(tiny_data):
    n = tiny_data.CS.shape[0]
    adjacency = torch.zeros(n, n)
    index = torch.arange(n - 1)
    adjacency[index, index + 1] = 1
    adjacency[index + 1, index] = 1
    laplacian = torch.diag(adjacency.sum(1)) - adjacency
    data = replace(tiny_data, laplacian=laplacian.to_sparse())
    result = train(data, TrainingConfig(number_of_niches=2, warmup_epochs=2, joint_epochs=3),
                   LossWeights.per_entry(data, sp=0.01))
    assert all(row["spatial_loss"] > 0 for row in result.history)
