from dataclasses import replace
import inspect
import numpy as np
import pytest
import torch
from phenoniche.inference.bulk import infer_bulk_activities
from phenoniche.data.synthetic import generate_synthetic
from phenoniche.data.scaling import BlockScaler
from phenoniche.evaluation.patient_protocol import split_patients, subset_bulk, fit_frozen_cox
from phenoniche.evaluation.challenging_benchmark import fit_candidate
from phenoniche.training.config import TrainingConfig, LossWeights


def test_exact_inference_and_phenotype_exclusion():
    synthetic = generate_synthetic(patients=32, anchors=20, noise=0)
    data, truth = synthetic.data, synthetic.truth
    hc, hi = truth["HC"].clone().requires_grad_(), truth["HI"].clone().requires_grad_()
    before = (hc.clone(), hi.clone())
    inferred = infer_bulk_activities(data.CB, data.IB, hc, hi)
    assert inferred.shape == truth["WB"].shape and (inferred >= 0).all()
    torch.testing.assert_close(inferred, truth["WB"], atol=2e-6, rtol=2e-6)
    changed = replace(data, time=data.time.flip(0) * 10, event=1 - data.event)
    again = infer_bulk_activities(changed.CB, changed.IB, hc, hi)
    torch.testing.assert_close(inferred, again, atol=0, rtol=0)
    assert inferred.requires_grad and hc.grad is None and hi.grad is None
    assert torch.equal(hc, before[0]) and torch.equal(hi, before[1])
    assert not {"time", "event", "phenotype", "gamma"} & set(inspect.signature(infer_bulk_activities).parameters)


def test_inference_optimality_noisy_weighted_problem():
    sample = generate_synthetic(patients=40, anchors=20, noise=0.04)
    data, truth = sample.data, sample.truth
    hc, hi = truth["HC"].double(), truth["HI"].double()
    cb, ib = data.CB.double(), data.IB.double()
    fitted = infer_bulk_activities(cb, ib, hc, hi, lambda_bc=0.3, lambda_bi=1.7)
    gradient = 2 * (0.3 * (fitted @ hc - cb) @ hc.T + 1.7 * (fitted @ hi - ib) @ hi.T)
    assert (gradient >= -1e-8).all()
    assert (gradient * fitted).abs().max() < 1e-8
    objective = lambda w: 0.3 * (w @ hc - cb).square().sum() + 1.7 * (w @ hi - ib).square().sum()
    assert objective(fitted) < objective(torch.full_like(fitted, 0.5))
    assert objective(fitted) <= objective(truth["WB"].double()) + 1e-8


def test_inference_row_independence():
    sample = generate_synthetic(patients=12, anchors=12)
    args = (sample.truth["HC"], sample.truth["HI"])
    whole = infer_bulk_activities(sample.data.CB, sample.data.IB, *args)
    one = infer_bulk_activities(sample.data.CB[3:4], sample.data.IB[3:4], *args)
    torch.testing.assert_close(whole[3:4], one, atol=0, rtol=0)


def test_invalid_inference_shapes_and_weights(tiny_data):
    with pytest.raises(ValueError, match="align"):
        infer_bulk_activities(tiny_data.CB, tiny_data.IB, torch.ones(2, 3), torch.ones(2, 6))
    with pytest.raises(ValueError, match="weights"):
        infer_bulk_activities(tiny_data.CB, tiny_data.IB, torch.ones(2, 4), torch.ones(2, 6), 0, 0)


def test_test_labels_cannot_change_fitting_or_scaling():
    data = generate_synthetic(patients=30, anchors=20).data
    split = split_patients(30)
    altered_time, altered_event = data.time.clone(), data.event.clone()
    altered_time[split.test] *= 37
    altered_event[split.test] = 1 - altered_event[split.test]
    altered = replace(data, time=altered_time, event=altered_event)
    trained = []
    for cohort in (data, altered):
        training = subset_bulk(cohort, split.train)
        scaler = BlockScaler.fit(training)
        scaled = scaler.transform(training)
        validation = scaler.transform(subset_bulk(cohort, split.validation))
        trained.append(fit_candidate(scaled, validation, TrainingConfig(warmup_epochs=3, joint_epochs=4), LossWeights.per_entry(scaled)))
    for key in trained[0].factors:
        torch.testing.assert_close(trained[0].factors[key], trained[1].factors[key], atol=0, rtol=0)
    assert trained[0].validation_c_index == trained[1].validation_c_index


def test_frozen_cox_head_does_not_change_activities():
    sample = generate_synthetic()
    activity = sample.truth["WB"].clone().requires_grad_()
    before = activity.detach().clone()
    gamma = fit_frozen_cox(activity, sample.data.time, sample.data.event)
    assert gamma[0] > 0 and gamma[1] < 0
    assert torch.equal(before, activity) and activity.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_inference_cuda_return_device():
    sample = generate_synthetic(patients=12, anchors=12)
    values = [sample.data.CB, sample.data.IB, sample.truth["HC"], sample.truth["HI"]]
    result = infer_bulk_activities(*(v.cuda() for v in values))
    assert result.is_cuda and result.dtype == torch.float32


def test_scaler_excludes_heldout_features():
    data = generate_synthetic(patients=30, anchors=20).data
    split = split_patients(30)
    cb, ib = data.CB.clone(), data.IB.clone()
    cb[split.test] *= 1000
    ib[split.validation] *= 500
    changed = replace(data, CB=cb, IB=ib)
    original_scale = BlockScaler.fit(subset_bulk(data, split.train))
    changed_scale = BlockScaler.fit(subset_bulk(changed, split.train))
    assert original_scale == changed_scale
