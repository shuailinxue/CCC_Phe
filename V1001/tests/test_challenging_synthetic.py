from dataclasses import replace
import numpy as np
import pytest
import torch
from phenoniche.data.challenging_synthetic import ChallengingConfig, generate_challenging
from phenoniche.evaluation.patient_protocol import split_patients, select_positive_lambda
from phenoniche.evaluation.phenotype_recovery import phenotype_recovery
from phenoniche.inference.bulk import infer_bulk_activities


def test_truth_generates_survival_and_capacity_constraint():
    config = ChallengingConfig(noise_fraction=0)
    sample = generate_challenging(config)
    truth = sample.truth
    assert config.true_niches > config.fit_niches
    torch.testing.assert_close(truth["eta"], truth["WB"] @ truth["gamma"])
    torch.testing.assert_close(truth["failure_time"], truth["survival_draw"] / (0.04 * truth["eta"].exp()), atol=1e-4, rtol=2e-6)
    torch.testing.assert_close(sample.data.time, torch.minimum(truth["failure_time"], truth["censoring_time"]))
    assert torch.equal(sample.data.event.bool(), truth["failure_time"] <= truth["censoring_time"])
    assert truth["gamma"][0] > 0 and truth["gamma"][1] < 0
    assert torch.equal(truth["gamma"][2:], torch.zeros(config.true_niches - 2))
    for name, w, h in (("CB", "WB", "HC"), ("IB", "WB", "HI"), ("CS", "WS", "HC"), ("OS", "WS", "HO"), ("IS", "WS", "HI")):
        torch.testing.assert_close(getattr(sample.data, name), truth[w] @ truth[h])


def test_changing_effects_does_not_change_input_features():
    config = ChallengingConfig()
    a = generate_challenging(config)
    b = generate_challenging(replace(config, beta_risk=0.4, beta_protective=3.0))
    for name, block in a.data.blocks().items():
        assert torch.equal(block, b.data.blocks()[name])
    for name in ("WB", "WS", "HC", "HI", "HO", "censoring_time", "survival_draw"):
        assert torch.equal(a.truth[name], b.truth[name])
    assert not torch.equal(a.data.time, b.data.time)


def test_nuisance_independence_and_structural_dominance():
    sample = generate_challenging(ChallengingConfig(patients=20000, anchors=20))
    wb = sample.truth["WB"].numpy()
    eta = sample.truth["eta"].numpy()
    for niche in range(2, 6):
        assert abs(np.corrcoef(wb[:, niche], eta)[0, 1]) < 0.05
    assert wb[:, 2:4].var(axis=0).min() > wb[:, :2].var(axis=0).max() * 3
    assert sample.truth["HI"][2:4].sum(1).min() > sample.truth["HI"][:2].sum(1).max() * 2
    modified = sample.truth["WB"].clone()
    modified[:, 2:] *= 100
    torch.testing.assert_close(modified @ sample.truth["gamma"], sample.truth["eta"])


def test_same_composition_distinct_ccc_risk_only():
    sample = generate_challenging(ChallengingConfig(scenario="same_composition"))
    truth = sample.truth
    assert torch.equal(truth["HC"][0], truth["HC"][1])
    assert torch.nn.functional.cosine_similarity(truth["HI"][0], truth["HI"][1], dim=0) < 0.1
    assert truth["gamma"][0] > 0 and (truth["gamma"][1:] == 0).all()
    assert sample.metadata["directional_features"][0]["sender"] == "Tumor"
    assert sample.metadata["directional_features"][1]["sender"] == "T_cell"


@pytest.mark.parametrize("scenario", ["capacity", "same_composition"])
def test_phenotype_signal_observable_with_true_dictionary(scenario):
    sample = generate_challenging(ChallengingConfig(scenario=scenario))
    inferred = infer_bulk_activities(sample.data.CB, sample.data.IB, sample.truth["HC"], sample.truth["HI"])
    for niche in (0, 1):
        assert np.corrcoef(inferred[:, niche].numpy(), sample.truth["WB"][:, niche].numpy())[0, 1] > 0.9


def test_disjoint_reproducible_label_blind_split():
    a, b = split_patients(720), split_patients(720)
    a.validate(720)
    assert (len(a.train), len(a.validation), len(a.test)) == (432, 144, 144)
    assert np.array_equal(a.train, b.train)
    assert not set(a.train) & set(a.test)
    assert select_positive_lambda({0: [0.9], 0.001: [0.6, 0.7], 0.01: [0.8, 0.7]}) == 0.01


def test_capacity_matching_reports_unmatched_and_collision():
    sample = generate_challenging()
    truth = sample.truth
    learned = {name: value[:4] if name.startswith("H") else value[:, :4]
               for name, value in truth.items() if name in ("HC", "HI", "HO", "WB", "WS")}
    learned["gamma"] = truth["gamma"][:4]
    report = phenotype_recovery(learned, truth, sample.metadata["roles"], learned["WB"], truth["WB"])
    assert report["hungarian_coverage"] == pytest.approx(4 / 6)
    assert len(report["unmatched_true_niches"]) == 2
    assert report["best_match_collisions"] >= 1
    assert report["overall_dictionary_recovery"] == pytest.approx(4 / 6, abs=1e-6)
    assert report["risk_niche_recovery"]["HI_cosine"] > 0.9999
