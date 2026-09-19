from dataclasses import replace
from pathlib import Path
import json
import pytest
import torch
from phenoniche.losses.collapse import cross_view_collapse_loss, pairwise_view_similarities
from phenoniche.data.synthetic import generate_synthetic
from phenoniche.model.niche_factorization import NicheFactorization
from phenoniche.training.config import TrainingConfig, LossWeights
from phenoniche.training.trainer import joint_loss, train


def test_identical_views_have_high_penalty_and_pair_mean():
    for k in (2, 5, 9):
        hc = torch.ones(k, 3, dtype=torch.float64)
        ho = torch.ones(k, 4, dtype=torch.float64)
        hi = torch.ones(k, 7, dtype=torch.float64)
        assert cross_view_collapse_loss(hc, ho, hi).item() == pytest.approx(1.0, abs=1e-7)


@pytest.mark.parametrize("distinct_view", [0, 1, 2])
def test_one_distinct_view_is_sufficient(distinct_view):
    views = [torch.ones(2, 2, dtype=torch.float64) for _ in range(3)]
    views[distinct_view] = torch.eye(2, dtype=torch.float64)
    assert cross_view_collapse_loss(*views) == 0


def test_loss_is_product_mean_not_sum_of_view_similarities():
    hc = torch.tensor([[1., 1., 0.], [0., 1., 1.], [1., 0., 1.]], dtype=torch.float64)
    ho = torch.tensor([[1., 2.], [2., 1.], [1., 1.]], dtype=torch.float64)
    hi = torch.tensor([[3., 1.], [1., 3.], [2., 2.]], dtype=torch.float64)
    values = []
    for i in range(3):
        for j in range(i + 1, 3):
            score = torch.tensor(1., dtype=torch.float64)
            for view in (hc, ho, hi):
                score *= (view[i] @ view[j]) / ((view[i].norm() + 1e-8) * (view[j].norm() + 1e-8))
            values.append(score)
    torch.testing.assert_close(cross_view_collapse_loss(hc, ho, hi), torch.stack(values).mean())


def test_permutation_and_positive_row_scaling():
    torch.manual_seed(97)
    views = [torch.rand(4, columns, dtype=torch.float64) + 0.2 for columns in (3, 5, 7)]
    expected = cross_view_collapse_loss(*views)
    permutation = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(cross_view_collapse_loss(*(view[permutation] for view in views)), expected)
    scales = [torch.tensor([0.3, 2., 4., 0.7], dtype=torch.float64)[:, None] for _ in views]
    torch.testing.assert_close(cross_view_collapse_loss(*(view * scale for view, scale in zip(views, scales))), expected, atol=1e-7, rtol=1e-7)


def test_finite_nonzero_gradients_and_gradcheck():
    torch.manual_seed(19)
    views = [(torch.rand(3, columns, dtype=torch.float64) + 0.2).requires_grad_() for columns in (3, 4, 5)]
    gradients = torch.autograd.grad(cross_view_collapse_loss(*views), views)
    assert all(torch.isfinite(gradient).all() and gradient.norm() > 1e-5 for gradient in gradients)
    assert torch.autograd.gradcheck(cross_view_collapse_loss, tuple(views))


def test_single_niche_and_zero_rows_are_finite():
    views = [torch.zeros(1, columns, requires_grad=True) for columns in (2, 3, 4)]
    loss = cross_view_collapse_loss(*views)
    assert loss == 0
    gradients = torch.autograd.grad(loss, views)
    assert all(torch.equal(value, torch.zeros_like(value)) for value in gradients)
    zero_loss = cross_view_collapse_loss(torch.zeros(2, 2), torch.zeros(2, 3), torch.zeros(2, 4))
    assert torch.isfinite(zero_loss) and zero_loss == 0


def test_shape_and_parameter_validation():
    with pytest.raises(ValueError, match="niche count"):
        cross_view_collapse_loss(torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 5))
    with pytest.raises(ValueError, match="nonnegative"):
        cross_view_collapse_loss(-torch.ones(2, 3), torch.ones(2, 4), torch.ones(2, 5))
    with pytest.raises(ValueError, match="eps"):
        pairwise_view_similarities(torch.ones(2, 3), torch.ones(2, 4), torch.ones(2, 5), eps=0)
    with pytest.raises(ValueError):
        LossWeights(lambda_collapse=-0.1)


def test_zero_weight_reproduces_prechange_training_fixture():
    reference = json.loads((Path(__file__).parent / "fixtures/inferred_wb_zero_collapse.json").read_text())
    data = generate_synthetic(patients=24, anchors=20, cell_types=4, contacts=5, communications=6, number_of_niches=2).data
    weights = LossWeights(**reference["weights"])
    assert weights.lambda_collapse == 0
    result = train(data, TrainingConfig(**reference["config"]), weights)
    for name, expected in reference["factors"].items():
        torch.testing.assert_close(result.model.factors()[name], torch.tensor(expected), atol=0, rtol=0)
    for name, expected in reference["final_losses"].items():
        assert result.diagnostics["final_losses"][name] == expected


def test_weighted_loss_adds_only_requested_term(tiny_data):
    torch.manual_seed(47)
    model = NicheFactorization(24, 20, 4, 5, 6, 2, composition=tiny_data.CB, communication=tiny_data.IB)
    weights = LossWeights()
    original = joint_loss(model, tiny_data, weights)
    updated = joint_loss(model, tiny_data, replace(weights, lambda_collapse=0.1))
    torch.testing.assert_close(updated["total_loss"], original["total_loss"] + 0.1 * original["collapse_loss"], atol=0, rtol=0)
    for name in original:
        if name != "total_loss":
            torch.testing.assert_close(original[name], updated[name], atol=0, rtol=0)


def test_training_records_gradient_clipping(tiny_data):
    result = train(tiny_data, TrainingConfig(number_of_niches=2, warmup_epochs=1, joint_epochs=2, gradient_clip=1e-7),
                   LossWeights.per_entry(tiny_data, lambda_collapse=0.1))
    assert result.diagnostics["gradient_clip_count"] == 3
    assert result.diagnostics["max_gradient_norm"] > 1e-7
    assert result.diagnostics["final_losses"]["collapse_loss"] > 0
    assert all(row["gradient_clipped"] for row in result.history)
