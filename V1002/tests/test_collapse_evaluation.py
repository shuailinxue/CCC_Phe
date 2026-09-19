import pytest
import torch
from phenoniche.evaluation.collapse_diagnostics import redundancy_diagnostics, select_collapse_lambda
from phenoniche.losses.collapse import cross_view_collapse_loss


def test_validation_rule_excludes_low_validation_and_ignores_unavailable_outcomes():
    selected = select_collapse_lambda({0.0: 0.72, 0.001: 0.73, 0.01: 0.71, 0.1: 0.74},
                                      {0.0: 0.8, 0.001: 0.2, 0.01: 0.01, 0.1: 0.3})
    assert selected["lambda_collapse"] == 0.001
    assert 0.01 not in selected["eligible_lambdas"]
    assert selected["test_or_truth_used"] is False
    with pytest.raises(ValueError):
        select_collapse_lambda({0.0: 0.6}, {0.1: 0.1})


def test_validation_tie_break_is_deterministic_and_zero_is_eligible():
    selected = select_collapse_lambda({0.0: 0.72, 0.001: 0.72}, {0.0: 0.2, 0.001: 0.2})
    assert selected["lambda_collapse"] == 0.0


def test_redundancy_diagnostics_match_loss_and_include_all_views():
    torch.manual_seed(23)
    factors = {name: torch.rand(4, features) + 0.1 for name, features in (("HC", 3), ("HO", 5), ("HI", 7))}
    report = redundancy_diagnostics(factors, (1, 1))
    loss = cross_view_collapse_loss(factors["HC"], factors["HO"], factors["HI"])
    assert report["summary"]["mean_pair_collapse"] == pytest.approx(float(loss), abs=1e-7)
    assert len(report["pairs"]) == 6
    assert all(row["first"] < row["second"] for row in report["pairs"])
    assert report["matched_pair"]["matched_A_B_joint_collapse_score"] == pytest.approx(1.0, abs=1e-6)
    for name in ("HC", "HO", "HI"):
        summary = report["summary"]
        assert summary[f"min_{name}_cosine"] <= summary[f"median_{name}_cosine"] <= summary[f"max_{name}_cosine"]
        assert summary[f"min_{name}_cosine"] <= summary[f"mean_{name}_cosine"] <= summary[f"max_{name}_cosine"]
