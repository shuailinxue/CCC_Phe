from dataclasses import replace
import numpy as np
import pytest
import torch
from phenoniche.inference import bulk
from phenoniche.inference.nnls_reference import nnls_reference
from phenoniche.model.niche_factorization import NicheFactorization
from phenoniche.losses.survival import cox_breslow_loss
from phenoniche.training.config import TrainingConfig, LossWeights
from phenoniche.training.trainer import train, joint_loss
from phenoniche.data.challenging_synthetic import generate_challenging, ChallengingConfig
from phenoniche.evaluation.inferred_benchmark import diagnostic_weights, summarize_pairs
from phenoniche.evaluation.pair_diagnostics import pair_separation, truth_activity_correlations


def test_wb_is_not_parameter(tiny_data):
    result = train(tiny_data, TrainingConfig(number_of_niches=2, warmup_epochs=1, joint_epochs=1), LossWeights.per_entry(tiny_data))
    assert all("WB" not in name for name, _ in result.model.named_parameters())
    assert all("WB" not in name for name in result.model.state_dict())
    assert not hasattr(result.model, "raw_WB")
    assert not isinstance(result.model.bulk_factors(), torch.nn.Parameter)


def test_bulk_inference_nonnegative_and_reconstruction_improves(tiny_data):
    torch.manual_seed(31)
    hc, hi = torch.rand(2, 4), torch.rand(2, 6)
    initial = tiny_data.CB.square().sum() + tiny_data.IB.square().sum()
    w = bulk.infer_bulk_activities(tiny_data.CB, tiny_data.IB, hc, hi)
    assert (w >= 0).all() and w.shape == (24, 2)
    final = (tiny_data.CB - w @ hc).square().sum() + (tiny_data.IB - w @ hi).square().sum()
    assert final < initial * 0.8
    reference = nnls_reference(tiny_data.CB, tiny_data.IB, hc, hi)
    optimum = (tiny_data.CB - reference @ hc).square().sum() + (tiny_data.IB - reference @ hi).square().sum()
    assert final <= optimum + initial * 1e-4


def test_cox_gradient_reaches_hc_hi_and_labels_change_gradient(tiny_data):
    model = NicheFactorization(24, 20, 4, 5, 6, 2, composition=tiny_data.CB, communication=tiny_data.IB)
    with torch.no_grad():
        model.gamma.copy_(torch.tensor([0.7, -0.6]))
    gradients = []
    for time, event in ((tiny_data.time, tiny_data.event), (tiny_data.time.flip(0), tiny_data.event.flip(0))):
        output = model()
        assert output["factors"]["WB"].grad_fn is not None
        loss = cox_breslow_loss(output["risk"], time, event)
        gradient = torch.autograd.grad(loss, (model.raw_HC, model.raw_HI, model.gamma))
        assert all(torch.isfinite(v).all() and v.norm() > 1e-6 for v in gradient)
        gradients.append(gradient)
    assert all(not torch.allclose(a, b) for a, b in zip(*gradients))


def test_train_test_same_inference_path(tiny_data, monkeypatch):
    original = bulk.infer_bulk_activities
    calls = []

    def traced(*args, **kwargs):
        calls.append((args[0].shape[0], kwargs.copy()))
        return original(*args, **kwargs)

    monkeypatch.setattr(bulk, "infer_bulk_activities", traced)
    model = NicheFactorization(24, 20, 4, 5, 6, 2, composition=tiny_data.CB, communication=tiny_data.IB)
    training = model.bulk_factors()
    with torch.no_grad():
        same = model.bulk_factors(tiny_data.CB, tiny_data.IB)
        subset = model.bulk_factors(tiny_data.CB[:5], tiny_data.IB[:5])
    torch.testing.assert_close(training, same, atol=0, rtol=0)
    torch.testing.assert_close(training[:5], subset, atol=2e-6, rtol=2e-6)
    assert len(calls) == 3 and calls[0][1] == calls[1][1] == calls[2][1]


def test_unrolled_inference_dictionary_gradcheck():
    torch.manual_seed(31)
    cb, ib = torch.rand(3, 4, dtype=torch.float64), torch.rand(3, 5, dtype=torch.float64)
    hc = (torch.rand(2, 4, dtype=torch.float64) + 0.2).requires_grad_()
    hi = (torch.rand(2, 5, dtype=torch.float64) + 0.2).requires_grad_()
    assert torch.autograd.gradcheck(lambda a, b: bulk.infer_bulk_activities(cb, ib, a, b, steps=10), (hc, hi), atol=1e-4)


def test_same_composition_capacity_diagnostic():
    sample = generate_challenging(ChallengingConfig(scenario="same_composition", patients=30, anchors=20))
    for k in (4, 6, 8):
        result = train(sample.data, TrainingConfig(number_of_niches=k, warmup_epochs=2, joint_epochs=2), diagnostic_weights(sample.data, 0.1))
        report = pair_separation(result.model.factors(), sample.truth)
        assert 0 <= report["A_best_factor"] < k and 0 <= report["B_best_factor"] < k
        assert report["collision"] == (report["A_best_factor"] == report["B_best_factor"])
        assert np.isfinite(report["A_HI_cosine"])
        assert np.isfinite(report["B_WS_correlation"])


def test_ccc_only_diagnostic_excludes_composition_and_topology(tiny_data):
    weights = diagnostic_weights(tiny_data, 0.1, ccc_only=True)
    config = TrainingConfig(number_of_niches=2, warmup_epochs=2, joint_epochs=3)
    result = train(tiny_data, config, weights)
    model = result.model
    changed = replace(tiny_data, CB=tiny_data.CB * 29, CS=tiny_data.CS * 17, OS=tiny_data.OS * 43)
    a, b = joint_loss(model, tiny_data, weights), joint_loss(model, changed, weights)
    torch.testing.assert_close(a["total_loss"], b["total_loss"], atol=0, rtol=0)
    gradients = torch.autograd.grad(a["total_loss"], (model.raw_HC, model.raw_HO, model.raw_HI))
    assert gradients[0].abs().sum() == 0 and gradients[1].abs().sum() == 0
    assert gradients[2].abs().sum() > 0
    w1, w2 = model.bulk_factors(), model.bulk_factors(changed.CB, changed.IB)
    torch.testing.assert_close(w1, w2, atol=0, rtol=0)


def test_truth_activity_correlation_report():
    sample = generate_challenging(ChallengingConfig(scenario="same_composition"))
    report = truth_activity_correlations(sample.truth, sample.data.time, sample.data.event)
    wb = sample.truth["WB"].numpy()
    assert report["WB_A_B"] == pytest.approx(np.corrcoef(wb[:, 0], wb[:, 1])[0, 1])
    assert report["WB_A_eta"] == pytest.approx(1.0)
    assert abs(report["WB_B_eta"]) < 0.15
    assert report["true_gamma_B"] == 0


def test_pair_collision_explicit_and_distinct_ground_truth():
    sample = generate_challenging(ChallengingConfig(scenario="same_composition"))
    truth = sample.truth
    exact = pair_separation(truth, truth)
    assert not exact["collision"] and exact["A_HI_cosine"] > 0.99999
    merged = {name: value[:1] if name in ("HC", "HI", "HO", "gamma") else value[:, :1]
              for name, value in truth.items() if name in ("HC", "HI", "HO", "gamma", "WS", "WB")}
    report = pair_separation(merged, truth)
    assert report["collision"] and report["learned_pair_HI_cosine"] == pytest.approx(1.0)
