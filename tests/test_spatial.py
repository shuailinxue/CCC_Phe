import numpy as np
import pandas as pd
from scipy import sparse

from phenoniche.spatial import Settings, SpatialNiches
from phenoniche.spatial.channels import Channels, WindowTensor
from phenoniche.spatial.geometry import physical_graph
from phenoniche.spatial.refine import assign


def test_interface_direction_mask_and_complete_complex():
    xyz = np.array([[0, 0], [1, 0], [2, 0], [1, 0]])
    g = physical_graph(xyz, np.array([0, 0, 0, 1]), 3)
    atlas = pd.DataFrame({"ligand": ["L", "L", "L"], "receptor": ["A,B", "B_A", "X"]})
    ch = Channels(
        [[4, 0, 0], [0, 2, 2], [0, 0, 4], [4, 2, 2]], pd.Index(["L", "A", "B"]), atlas
    )
    assert ch.names == [("L", "A+B")]
    t = WindowTensor(g, g, np.array([0, 1, 1, 0]), np.zeros(4, dtype=int), ch, 20)
    sender, receiver, value, opportunity = t.fibers(0, (0, 1))
    assert sender.tolist() == [0, 1] and receiver.tolist() == [0, 0]
    assert 0 < value[0, 0] < np.sqrt(10000 * 5000)
    assert value[1, 0] == 0 and np.all(opportunity > 0)
    assert t.fibers(3, (0, 0))[2].shape == (0, 1)


def test_unique_assignment_ties_and_cross_boundary():
    coarse = np.array([0, 0, 1, 1, 1])
    graph = sparse.csr_matrix(np.ones((5, 5)))
    from unittest.mock import Mock

    audit = Mock()
    audit.contrast.return_value = [(0, 0, 0, 1.0, 1.0, 2, 2)]
    first = ((0, 1), np.arange(4), np.ones(4), 0, 1, audit)
    second = ((1, 1), np.array([3, 4]), np.ones(2), 0, 1, audit)
    labels, _, _, _ = assign(coarse, graph, [first, second], 2)
    assert list(labels) == ["F1", "F1", "F1", "C1", "C1"]


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
        effect_mad=2,
        agreement=0.75,
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
    from phenoniche.spatial.subclasses import partition_contexts

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


def test_full_interface_keeps_all_four_flank_directions():
    xyz = np.c_[np.arange(4), np.zeros(4)]
    graph = physical_graph(xyz, np.repeat("s", 4), 4)
    atlas = pd.DataFrame({"ligand": ["L"], "receptor": ["R"]})
    ch = Channels(np.tile([2, 3], (4, 1)), pd.Index(["L", "R"]), atlas)
    tensor = WindowTensor(
        graph,
        graph,
        np.repeat([0, 1], 2),
        np.zeros(4, dtype=int),
        ch,
        20,
        full_interface=True,
    )
    senders, receivers, values, opportunity = tensor.fibers(0, (0, 1))
    np.testing.assert_array_equal(senders, [0, 1, 2, 3])
    np.testing.assert_array_equal(receivers, [0, 0, 0, 0])
    np.testing.assert_allclose(values, np.sqrt(4000 * 6000), rtol=1e-6)
    assert np.all(opportunity > 0)


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
