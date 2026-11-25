import numpy as np
import pandas as pd

from phenoniche.spatial import Settings, SpatialNiches
from phenoniche.spatial.channels import Channels
from phenoniche.spatial.geometry import physical_graph


def test_blind_local_partition(tmp_path):
    x, y = np.meshgrid(np.arange(24), np.arange(6))
    xyz = np.c_[x.ravel(), y.ravel()].astype(float)
    patch = (xyz[:, 0] >= 10) & (xyz[:, 0] <= 13)
    counts = np.c_[
        np.where(patch, 80, 1), np.ones(len(xyz)) * 5, np.ones(len(xyz)) * 80
    ]
    cfg = Settings(
        window_um=1.5,
        contact_um=1.5,
        rank=1,
        iterations=12,
        fit_windows=100,
        repeats=4,
        block_um=3,
        min_cells=5,
    )
    model = SpatialNiches.fit(
        counts,
        ["L", "R", "B"],
        np.arange(len(xyz)),
        xyz,
        ["T"] * len(xyz),
        ["s"] * len(xyz),
        pd.DataFrame({"ligand": ["L"], "receptor": ["R"]}),
        cfg,
    )
    labels = model.cells.niche
    assert labels.nunique() >= 2
    dominant = labels[patch].mode().iloc[0]
    detected = labels.eq(dominant).to_numpy()
    assert detected[patch].mean() > 0.7 and detected[~patch].mean() < 0.2
    assert model.evidence.focal_type.eq("T").all() and len(model.evidence)
    model.save(tmp_path / "out")
    assert len(pd.read_csv(tmp_path / "out/cells.tsv", sep="\t")) == len(xyz)


def test_no_signal_keeps_full_coarse_partition():
    n = 18
    result = SpatialNiches.fit(
        np.zeros((n, 2)),
        ["L", "R"],
        np.arange(n),
        np.c_[np.arange(n), np.zeros(n)],
        ["T"] * n,
        ["s"] * n,
        pd.DataFrame({"ligand": ["L"], "receptor": ["R"]}),
        Settings(window_um=2, contact_um=2, min_cells=2, block_um=2, repeats=2, rank=1),
    )
    assert result.cells.niche.tolist() == ["C0.1"] * n
    assert result.evidence.empty


def test_interface_contexts_partition_both_sides_once():
    from phenoniche.spatial.contexts import partition_contexts

    xy = np.array([(x, y) for x in range(10) for y in range(8)])
    coarse = (xy[:, 0] >= 5).astype(int)
    graph = physical_graph(xy, np.repeat("s", len(xy)), 1.5)
    groups, summary = partition_contexts(graph, coarse, Settings(min_cells=4))
    assert summary[0][-1]
    assert set(coarse[groups[(0, 1)]]) == {0, 1}
    assert set(groups) == {(0, 0), (0, 1), (1, 1)}
    np.testing.assert_array_equal(
        np.sort(np.concatenate(list(groups.values()))), np.arange(len(xy))
    )


def test_focal_sender_receiver_are_directional():
    from phenoniche.spatial.cell_programs import CellTensor

    graph = physical_graph(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array(["s", "s"]), 2)
    atlas = pd.DataFrame({"ligand": ["L"], "receptor": ["R"]})
    channels = Channels([[5, 0], [0, 5]], pd.Index(["L", "R"]), atlas)
    tensor = CellTensor(graph, np.array([0, 1]), channels, 20)
    allowed = np.ones(2, dtype=bool)
    out = tensor.fiber(0, allowed)
    incoming = tensor.fiber(1, allowed)
    assert out[2][0, 0] > 0 and out[2][0, 1] == 0
    assert incoming[2][0, 1] > 0 and incoming[2][0, 0] == 0
    np.testing.assert_allclose(out[2][0, 0], incoming[2][0, 1])
