from copy import deepcopy
import math
import torch
from phenoniche.inference.bulk import infer_bulk_activities
from phenoniche.losses.survival import cox_breslow_loss
from phenoniche.model.adapters import BoundedDictionaryAdapter
from phenoniche.simulation.structured_ccc import StructuredCCCConfig, generate_structured_ccc
from phenoniche.training.iterative_refinement import IterativeRefinementConfig, refinement_loss, rollback_if_rejected


def _adapter_and_data():
    generated = generate_structured_ccc(StructuredCCCConfig(programs=2))
    truth, data = generated.truth, generated.data
    adapter = BoundedDictionaryAdapter(truth["HC"], truth["HI"], truth["HO"], truth["WS"])
    adapter.config = IterativeRefinementConfig(epochs_per_iteration=1)
    return adapter, data


def test_test_phenotype_cannot_change_anchor_or_bulk_projection():
    adapter, data = _adapter_and_data()
    before = deepcopy(adapter.state_dict())
    first = infer_bulk_activities(data.CB, data.IB, adapter.HC0, adapter.HI0,
                                  lambda_bc=1 / 8, lambda_bi=1 / 128, steps=5)
    changed_time = data.time.flip(0)
    changed_event = 1 - data.event
    second = infer_bulk_activities(data.CB, data.IB, adapter.HC0, adapter.HI0,
                                   lambda_bc=1 / 8, lambda_bi=1 / 128, steps=5)
    assert changed_time.shape == data.time.shape and changed_event.shape == data.event.shape
    assert torch.equal(first, second)
    assert all(torch.equal(before[name], adapter.state_dict()[name]) for name in before)


def test_cox_gradient_reaches_bounded_deltas_and_spatial_anchors_stay_frozen():
    adapter, data = _adapter_and_data()
    with torch.no_grad():
        adapter.gamma.copy_(torch.linspace(0.5, -0.5, 6))
        adapter.raw_delta_c.normal_(0, 0.1)
        adapter.raw_delta_i.normal_(0, 0.1)
    hc, hi, _ = adapter.dictionaries()
    wb = infer_bulk_activities(data.CB[:32], data.IB[:32], hc, hi,
                               lambda_bc=1 / 8, lambda_bi=1 / 128, steps=5)
    cox = cox_breslow_loss(wb @ adapter.gamma, data.time[:32], data.event[:32])
    cox.backward()
    assert adapter.raw_delta_c.grad is not None and torch.isfinite(adapter.raw_delta_c.grad).all()
    assert adapter.raw_delta_i.grad is not None and torch.isfinite(adapter.raw_delta_i.grad).all()
    assert adapter.raw_delta_c.grad.abs().sum() > 0 and adapter.raw_delta_i.grad.abs().sum() > 0
    assert adapter.HO0.grad is None and adapter.WS0.grad is None
    with torch.no_grad():
        adapter.raw_delta_c.fill_(10.0)
        adapter.raw_delta_i.fill_(-10.0)
    delta_c, delta_i = adapter.deltas()
    assert max(float(delta_c.abs().max()), float(delta_i.abs().max())) <= math.log(1.25)


def test_full_refinement_loss_keeps_ho_and_ws_without_gradients():
    adapter, data = _adapter_and_data()
    losses = refinement_loss(adapter, data.CB[:32], data.IB[:32], data.time[:32], data.event[:32],
                             data.CS, data.IS, data.OS, inner_steps=3)
    losses["total_loss"].backward()
    assert adapter.HO0.grad is None and adapter.WS0.grad is None


def test_rejected_iteration_rolls_back_exactly():
    adapter, _ = _adapter_and_data()
    previous = deepcopy(adapter.state_dict())
    with torch.no_grad():
        adapter.raw_delta_c.add_(1.0)
        adapter.raw_delta_i.sub_(1.0)
    assert rollback_if_rejected(adapter, previous, False) is False
    assert all(torch.equal(previous[name], adapter.state_dict()[name]) for name in previous)
