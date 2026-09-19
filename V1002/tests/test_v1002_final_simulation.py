import numpy as np
import pytest
import torch
from phenoniche.v1002.lr_atlas import load_lr_atlas
from phenoniche.v1002.final_simulation import (
    build_final_spec, simulate_final_bulk, simulate_final_spatial, composition_prototypes,
    NICHE_PROTOTYPES, BACKGROUND, bulk_potential, alr, TRUE_BETA,
)
from phenoniche.v1002.final_model import fit_balanced, primary_pass
from phenoniche.v1002.final_experiment import collinearity


def test_two_hard_case_truth_is_fixed_and_complementary():
    spec = build_final_spec(load_lr_atlas("data/commuspace_human_lr_atlas.tsv"))
    hc = composition_prototypes(.5)
    assert np.allclose(NICHE_PROTOTYPES.sum(1), 1)
    assert np.allclose(hc.sum(1), 1)
    assert np.allclose(BACKGROUND.sum(), 1)
    assert not np.array_equal(hc[0], hc[1])
    assert not np.array_equal(hc[2], hc[3])
    assert not np.array_equal(spec.expression_delta[2], spec.expression_delta[3])
    active = [set(feature for niche, feature, *_ in spec.active_edges if niche == k) for k in range(5)]
    assert len(active[0] & active[1]) == 25
    assert len(active[2] & active[3]) == 80
    assert not np.array_equal(spec.expression_delta[0], spec.expression_delta[1])


def test_local_microenvironment_and_outcome_blind_filter_smoke():
    spec = build_final_spec(load_lr_atlas("data/commuspace_human_lr_atlas.tsv"))
    sim = simulate_final_spatial(spec, n_anchors=24, seed=40700, device="cpu")
    assert sim.cell_types.shape == (24, 48)
    assert sim.relative_positions.shape == (24, 48, 2)
    assert sim.expression.shape == (24, 48, len(spec.genes))
    assert np.allclose(sim.theta.sum(1), 1)
    assert np.allclose(sim.cs.sum(1), 1)
    assert sim.communication.shape == (24, 64 * len(spec.atlas))
    assert not hasattr(sim, "os")
    assert sim.final_mask.shape == (64, len(spec.atlas))
    assert np.array_equal(sim.final_mask, sim.coverage_mask & (sim.pair_support[:, None] >= 20))
    pi, _, expression, *_ = simulate_final_bulk(spec, sim.hc_truth, .05, 40710, patients=4)
    assert pi.shape == (4, 6)
    assert bulk_potential(expression, spec).shape[1] == sim.communication.shape[1]
    assert np.isfinite(alr(pi)).all()


def test_balancing_equalizes_initial_weighted_ws_gradients_without_outcomes():
    rng = np.random.default_rng(42)
    blocks = {"HC": rng.uniform(size=(30, 8)).astype("float32"),
              "HI": rng.uniform(size=(30, 25)).astype("float32")}
    fit = fit_balanced(blocks, seed=31, iterations=2, device="cpu")
    gradients = list(fit.audit["weighted_initial_gradient_norm"].values())
    assert np.allclose(gradients, gradients[0], rtol=1e-6)
    assert np.isclose(np.mean(list(fit.audit["alpha"].values())), 1)
    assert all(torch.isfinite(value).all() for value in fit.dictionaries.values())
    fit8 = fit_balanced(blocks, seed=31, iterations=2, device="cpu", niches=8)
    assert fit8.W.shape == (30, 8)
    with pytest.raises(ValueError, match="only HC"):
        fit_balanced({**blocks, "unsupported": rng.uniform(size=(30, 12))}, seed=31, iterations=2, device="cpu")


def test_background_reference_alr_and_collinearity():
    pi = np.array([[.4, .2, .1, .1, .1, .1], [.2, .3, .1, .2, .1, .1]], dtype=float)
    assert np.allclose(alr(pi)[:, 0], np.log((pi[:, 1] + 1e-6) / (pi[:, 0] + 1e-6)))
    assert TRUE_BETA.shape == (5,)
    assert TRUE_BETA[0] > 0 and TRUE_BETA[2] < 0
    rng = np.random.default_rng(7)
    x = rng.dirichlet(np.ones(6), size=80)
    raw = collinearity(x)
    transformed = collinearity(alr(x))
    assert raw["condition_number_with_intercept"] > 1e12
    assert transformed["condition_number_with_intercept"] < 1e4


def test_bulk_survival_truth_uses_exact_background_reference_alr():
    spec = build_final_spec(load_lr_atlas("data/commuspace_human_lr_atlas.tsv"))
    pi, cb, expression, time, event, beta, eta = simulate_final_bulk(spec, composition_prototypes(.5), .05, 40700, patients=80)
    assert pi.shape == (80, 6) and np.all(pi > 0)
    assert np.allclose(pi.sum(1), 1)
    assert cb.shape == (80, 8)
    assert expression.shape == (80, 8, len(spec.genes))
    assert np.array_equal(beta, TRUE_BETA)
    assert np.allclose(eta, alr(pi) @ TRUE_BETA)
    assert np.all(time > 0) and set(np.unique(event)).issubset({0, 1})


def test_primary_gate_requires_each_seed_to_pass():
    good = {"per_factor": [{"sensitivity": .9}] * 6, "overall_ws": .8,
            "collision_12": False, "collision_34": False}
    assert primary_pass([good] * 3)
    bad = dict(good, collision_34=True)
    assert not primary_pass([good, bad, good])
