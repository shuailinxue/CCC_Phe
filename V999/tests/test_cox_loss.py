import math
import pytest
import torch
from phenoniche.losses.survival import cox_breslow_loss


def test_hand_computed_ties():
    risk = torch.log(torch.tensor([1., 2., 3., 4.], dtype=torch.float64))
    time = torch.tensor([1., 2., 2., 3.], dtype=torch.float64)
    event = torch.tensor([1., 1., 1., 0.], dtype=torch.float64)
    expected = (math.log(10) + math.log(9 / 2) + math.log(9 / 3)) / 3
    assert float(cox_breslow_loss(risk, time, event)) == pytest.approx(expected)


def test_censor_at_event_time_included_in_risk_set():
    risk = torch.log(torch.tensor([2., 3., 5.]))
    assert float(cox_breslow_loss(risk, torch.tensor([1., 1., 2.]), torch.tensor([1., 0., 0.]))) == pytest.approx(math.log(5))


def test_stability_permutation_and_shift():
    risk = torch.tensor([10000., 10001., 9999., 10002.], dtype=torch.float64, requires_grad=True)
    time = torch.tensor([1., 2., 2., 3.], dtype=torch.float64)
    event = torch.tensor([1., 1., 0., 1.], dtype=torch.float64)
    loss = cox_breslow_loss(risk, time, event)
    permutation = torch.tensor([3, 2, 0, 1])
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, cox_breslow_loss(risk - 10000, time, event))
    torch.testing.assert_close(loss, cox_breslow_loss(risk[permutation], time[permutation], event[permutation]))
    loss.backward()
    assert torch.isfinite(risk.grad).all()


def test_cox_gradcheck():
    risk = torch.tensor([0.2, -0.4, 0.7, 0.1], dtype=torch.float64, requires_grad=True)
    time = torch.tensor([1., 2., 2., 3.], dtype=torch.float64)
    event = torch.tensor([1., 1., 0., 1.], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x: cox_breslow_loss(x, time, event), (risk,))


def test_eventless_zero_differentiable():
    risk = torch.randn(3, requires_grad=True)
    loss = cox_breslow_loss(risk, torch.ones(3), torch.zeros(3))
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(risk.grad, torch.zeros_like(risk))


def test_invalid_shape():
    with pytest.raises(ValueError, match="shapes"):
        cox_breslow_loss(torch.zeros(3), torch.ones(2), torch.ones(3))


def test_against_explicit_risk_sets():
    risk = torch.tensor([-0.2, 0.9, 0.7, -1., 2.], dtype=torch.float64)
    time = torch.tensor([3., 2., 1., 2., 5.], dtype=torch.float64)
    event = torch.tensor([0., 1., 1., 1., 1.], dtype=torch.float64)
    expected = torch.stack([torch.logsumexp(risk[time >= time[i]], 0) - risk[i]
                            for i in range(5) if event[i] == 1]).mean()
    torch.testing.assert_close(cox_breslow_loss(risk, time, event), expected)
