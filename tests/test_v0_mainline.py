from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from cccphe.ccc_tensor import construct_patient_ccc, ligand_receptor_support
from cccphe.core import cox_breslow_loss, feature_batch, feature_mask, structural_feature_indices
from cccphe.network_cox import _group_smoothness, _safe_concordance_index
from cccphe.merfish_anchor import _entity_cpm
from cccphe.merfish_st_niche import discover_programs, robust_standardize_profiles
from cccphe.st_niche import (
    call_spatial_hotspots,
    discover_nmf_programs,
    load_protective_cccs,
    robust_nonnegative_standardize,
)


def test_patient_ccc_uses_current_st_anchor_ratio() -> None:
    w_st = np.array([[[2.0, 4.0]]])
    m_st = np.array([[[1.0, 2.0]]])
    m_rna = np.array([[[[3.0, 1.0]]], [[[1.0, 4.0]]]])
    observed = construct_patient_ccc(w_st, m_st, m_rna, epsilon=1e-12)
    expected = np.array([[[[6.0, 2.0]]], [[[2.0, 8.0]]]], dtype=np.float32)
    np.testing.assert_allclose(observed, expected, rtol=1e-6)


def test_api_has_no_separate_reference_or_validation_st() -> None:
    parameters = inspect.signature(construct_patient_ccc).parameters
    assert not any("reference" in name or "validation" in name for name in parameters)


def test_geometric_mean_molecular_support() -> None:
    observed = ligand_receptor_support(np.array([4.0, 9.0]), np.array([1.0, 4.0]))
    np.testing.assert_allclose(observed, np.array([2.0, 6.0]))


def test_merfish_complex_requires_every_subunit_to_be_observed() -> None:
    pseudobulk = np.array([[4.0, 9.0], [4.0, 0.0]])
    observed = _entity_cpm(pseudobulk, {"A": 0, "B": 1}, "A_B")
    np.testing.assert_allclose(observed[0], 6.0)
    assert observed[1] == 0.0


def test_exact_ccc_flattening_and_mask() -> None:
    tensor = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    mask = np.array(
        [[[True, False], [False, False]], [[False, False], [False, True]]]
    )
    features = structural_feature_indices(mask)
    np.testing.assert_array_equal(features, np.array([0, 7]))
    observed = feature_batch(tensor, np.array([0, 1]), features)
    np.testing.assert_array_equal(observed, np.array([[0, 7], [8, 15]]))
    observed_mask = feature_mask(mask, np.array([0, 1]), features)
    assert observed_mask.shape == (2, 2)
    assert observed_mask.all()


def test_cox_loss_is_finite_and_differentiable() -> None:
    risk = torch.tensor([0.1, -0.2, 0.3], requires_grad=True)
    time = torch.tensor([3.0, 2.0, 1.0])
    event = torch.tensor([1.0, 0.0, 1.0])
    loss = cox_breslow_loss(risk, time, event)
    loss.backward()
    assert torch.isfinite(loss)
    assert risk.grad is not None
    assert torch.isfinite(risk.grad).all()


def test_cindex_is_neutral_when_no_pairs_are_admissible() -> None:
    observed = _safe_concordance_index(
        np.array([1.0, 1.0]), np.array([0.2, -0.2]), np.array([0, 0])
    )
    assert observed == 0.5


def test_graph_penalty_uses_laplacian_sum_scale() -> None:
    coefficient = torch.tensor([1.0, 3.0])
    group = torch.tensor([0, 0])
    observed = _group_smoothness(coefficient, group, 1)
    # Group mean is 2, so w^T L w in this representation is 1^2 + 1^2.
    torch.testing.assert_close(observed, torch.tensor(2.0))


def test_protective_ccc_filter_uses_only_stability_and_coefficient(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "sender": ["B cells", "CAFs", "T-cells"],
            "receiver": ["T-cells", "Myeloid", "B cells"],
            "lr_id": ["L1--R1|A", "L2--R2|B", "L3--R3|C"],
            "ligand": ["L1", "L2", "L3"],
            "receptor": ["R1", "R2", "R3"],
            "full_coefficient": [-0.2, -0.1, 0.3],
            "sign_selection_probability": [0.71, 0.69, 0.9],
            "stability_importance": [0.2, 0.1, 0.3],
        }
    )
    source = tmp_path / "stable.tsv"
    frame.to_csv(source, sep="\t", index=False)
    observed = load_protective_cccs(source, stability_threshold=0.70)
    assert observed.sender.tolist() == ["B cells"]


def test_nmf_input_standardization_is_nonnegative_and_slice_balanced() -> None:
    raw = pd.DataFrame(
        {"a": [0, 1, 2, 0, 100, 200], "b": [0, 2, 4, 0, 20, 40]},
        index=[f"x{i}" for i in range(6)],
    )
    normalized, audit = robust_nonnegative_standardize(
        raw, ["s1"] * 3 + ["s2"] * 3, minimum_positive_spots=2
    )
    assert normalized.shape == raw.shape
    assert np.isfinite(normalized.to_numpy()).all()
    assert (normalized.to_numpy() >= 0).all()
    assert audit.used_globally.all()


def test_repeat_nmf_returns_label_free_diagnostics() -> None:
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.gamma(1.5, 1, size=(80, 6)), columns=list("abcdef"))
    diagnostics, loadings, activities, chosen = discover_nmf_programs(
        x, k_values=[2, 3], n_restarts=3, random_seed=11, max_iter=500
    )
    assert chosen in {2, 3}
    assert diagnostics.selected.sum() == 1
    assert loadings.shape == (6, chosen)
    assert activities.shape == (80, chosen)
    assert "tls" not in " ".join(diagnostics.columns).lower()


def test_hotspots_leave_background_unassigned() -> None:
    graph = sp.diags([1, 1], [-1, 1], shape=(20, 20), format="csr")
    spots = pd.DataFrame(
        {"sample_id": "s", "spot_id": [str(i) for i in range(20)],
         "x": np.arange(20), "y": 0},
        index=[f"s|{i}" for i in range(20)],
    )
    activity = pd.DataFrame({"P1": [0] * 12 + [5] * 8}, index=spots.index)
    membership, summary, _ = call_spatial_hotspots(
        activity, spots, {"s": graph}, hotspot_quantile=0.55, minimum_niche_spots=3
    )
    assert not summary.empty
    assert 0 < len(membership) < len(spots)


def test_merfish_rare_observed_cccs_use_cohort_scale_instead_of_being_erased(tmp_path) -> None:
    protective = pd.DataFrame({"ccc_id": ["a", "b"]})
    profile = pd.DataFrame(
        {
            "location_id": [f"x{i}" for i in range(8)],
            "sample_id": ["s1"] * 4 + ["s2"] * 4,
            "a": [1, 0, 0, 0, 2, 0, 0, 0],
            "b": [0, 3, 0, 0, 0, 4, 0, 0],
        }
    )
    normalized, usable = robust_standardize_profiles(
        profile, protective, tmp_path, minimum_positive_cells=100
    )
    assert usable == ["a", "b"]
    assert (normalized.sum(axis=0) > 0).all()
    audit = pd.read_csv(tmp_path / "normalization_audit.tsv", sep="\t")
    assert set(audit.scale_source) == {"cohort_q95"}


def test_merfish_nmf_never_tests_more_programs_than_ccc_features(tmp_path) -> None:
    rng = np.random.default_rng(19)
    normalized = pd.DataFrame(
        rng.gamma(1.5, 1.0, size=(1200, 3)),
        index=[f"x{i}" for i in range(1200)],
        columns=["a", "b", "c"],
    )
    profile = pd.DataFrame(
        {"location_id": normalized.index, "sample_id": ["s1"] * 1200}
    )
    _, _, chosen = discover_programs(
        normalized, profile, tmp_path, k_values=[2, 4], n_restarts=2,
        fit_cells_per_sample=2000, random_seed=23, max_iter=300,
    )
    assert chosen == 2
    diagnostics = pd.read_csv(tmp_path / "nmf_k_diagnostics.tsv", sep="\t")
    assert diagnostics.k.tolist() == [2]
