import numpy as np
import pandas as pd
import pytest

from phenoniche.comparative import (
    ComparativeSettings, _StoredTensor, _reference_analysis, _weighted_quantile, fit_control_case,
)
from phenoniche.spatial import Settings


def test_matching_calibration_normal_heterogeneity_and_novel_composition():
    # Each healthy donor has two ordinary communication states at the same composition.
    composition = np.array([[1., 0.]] * 7 + [[0., 1.]])
    activity = np.array([[0., 0.], [10., 0.], [0., 0.], [10., 0.],
                         [0., 0.], [10., 0.], [20., 0.], [3., 0.]])
    conditions = np.array(["control"] * 4 + ["case"] * 4)
    donors = np.array(["h1", "h1", "h2", "h2", "d1", "d1", "d1", "d1"])
    cfg = ComparativeSettings(k=2)
    spatial = Settings(cp_device="cpu", repeats=2, rank=1)
    out = _reference_analysis(np.arange(8), composition, activity, conditions, donors, cfg, spatial, 73)
    assert out["calibrated"] and out["abnormality_calibrated"]
    assert list(out["abnormality"][4:7]) == ["within_reference", "within_reference", "outside_reference"]
    assert out["reference_status"][7] == "no_reference"
    assert np.isnan(out["delta"][7]).all() and out["labels"][7].startswith("U")
    np.testing.assert_allclose(out["delta"][4:7, 0], [-5, 5, 15])
    assert all(donors[q] != donors[r] for q, r, _, _ in out["matches"] if conditions[q] == "control")


def test_unavailable_donor_calibration_retains_residual_and_niche():
    composition = np.array([[1., 0.], [1., 0.], [1., 0.]])
    activity = np.array([[1.], [3.], [7.]])
    result = _reference_analysis(np.arange(3), composition, activity,
             np.array(["control", "control", "case"]), np.array(["h", "h", "d"]),
             ComparativeSettings(k=2), Settings(cp_device="cpu", repeats=2), 7)
    assert not result["calibrated"]
    assert result["reference_status"][2] == result["abnormality"][2] == "uncalibrated"
    assert result["delta"][2, 0] == pytest.approx(5)
    assert result["labels"][2].startswith("R")


def test_weighted_quantile_gives_equal_total_weight_to_each_donor():
    assert _weighted_quantile([0] * 100 + [10], ["large"] * 100 + ["small"], .75) == 10


def test_stored_tensor_preserves_type_specific_mask_and_observed_zeros():
    from phenoniche.spatial.channels import Channels
    from phenoniche.spatial.cell_programs import CellTensor
    from phenoniche.spatial.geometry import physical_graph
    xyz = np.array([[0., 0.], [10., 0.], [40., 0.]])
    graph = physical_graph(xyz, np.array(["s"] * 3), 100)
    genes = pd.Index(["L", "R1", "R2"])
    counts = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
    lr = pd.DataFrame({"ligand": ["L", "L"], "receptor": ["R1", "R2"],
                       "signaling_type": ["Contact", "Secreted"]})
    channels = Channels(counts, genes, lr)
    cfg = Settings()
    original = CellTensor(graph, np.array([0, 1, 2]), channels, cfg.sigma_um, settings=cfg)
    stored = _StoredTensor(np.arange(3), original, ["a", "b", "c"])
    for cell in range(3):
        categories, values, support = original.fiber(cell)
        scategories, svalues, smask = stored.fiber(cell)
        np.testing.assert_array_equal(scategories, categories)
        np.testing.assert_allclose(svalues, values)
        np.testing.assert_array_equal(smask, support > 0)
    assert stored.mask[0, 2, 0] == 0 and stored.mask[0, 2, 1] == 1


def test_end_to_end_shared_factors_and_cache(tmp_path, monkeypatch):
    import phenoniche.comparative as module
    n = 12
    xyz = np.tile([[0., 0.], [1., 0.], [2., 0.], [3., 0.]], (3, 1))
    labels = np.tile(["T", "B", "T", "B"], 3)
    samples = np.repeat(["s1", "s2", "s3"], 4)
    metadata = pd.DataFrame({"sample_id": samples, "donor_id": np.repeat(["h1", "h2", "case"], 4),
                             "condition": np.repeat(["control", "control", "case"], 4), "batch": "b"})
    counts = np.tile([[2, 5], [4, 2], [3, 4], [5, 1]], (3, 1))
    lr = pd.DataFrame({"ligand": ["L"], "receptor": ["R"], "signaling_type": ["Secreted"]})
    spatial = Settings(window_um=3, secreted_radius_um=3, contact_um=3,
                       cp_device="cpu", rank=1, iterations=2, repeats=2)
    comp = ComparativeSettings(k=1, balance="donor")
    # A single coarse class makes the expected shared fit and donor comparison explicit.
    monkeypatch.setattr(module, "composition_partition", lambda graph, types, cfg:
                        (np.tile([.5, .5], (n, 1)), np.zeros(n, dtype=int)))
    args = (counts, ["L", "R"], [f"c{i}" for i in range(n)], xyz, labels, metadata, lr)
    result = fit_control_case(*args, cache_dir=tmp_path, settings=spatial, comparative=comp)
    assert len(result.cells) == n and len(result.contexts[0]["fits"]) == 2
    assert result.contexts[0]["fits"][0].metrics["balance_groups"] == 3
    assert result.contexts[0]["features"].shape[0] == n
    np.testing.assert_allclose(result.donor_proportions.groupby(["donor_id", "coarse"]).fraction.sum(), 1)
    assert (result.cache_path / "tensor_0.pkl").exists()
    assert (result.cache_path / "composition.pkl").exists()
    assert (result.cache_path / "result.pkl").exists()
    monkeypatch.setattr(module, "physical_graph", lambda *args: pytest.fail("cache was not reused"))
    again = fit_control_case(*args, cache_dir=tmp_path, settings=spatial, comparative=comp)
    pd.testing.assert_frame_equal(result.cells, again.cells)
