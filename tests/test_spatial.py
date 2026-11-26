import numpy as np
import pandas as pd
import pytest

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
        iterations=150,
        repeats=4,
        cp_device="cpu",
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
    out = tmp_path / "out"
    with np.load(out / "programs_0_0_repeat0.npz") as factors:
        assert factors["focal_activity"].shape[0] == len(xyz)
        np.testing.assert_array_equal(factors["focal_activity"], model.programs[(0, 0)]["models"][0][0])
        assert {"A", "H1", "H2", "H3"}.issubset(factors.files)
        np.testing.assert_allclose(factors["H3"], factors["H1"])
    from scipy import sparse
    assert sparse.load_npz(out / "tensor_0_0.npz").shape[0] == len(xyz)
    with np.load(out / "tensor_mask_0_0.npz", allow_pickle=False) as saved:
        assert saved["mask"].shape[0] == len(xyz)
        assert len(saved["cell_ids"]) == len(xyz)
    assert (out / "rank_evaluation_0_0.json").is_file()
    history = pd.read_csv(out / "cp_losses_0_0.tsv", sep="\t")
    metrics = pd.read_csv(out / "cp_metrics_0_0.tsv", sep="\t")
    assert history.groupby("repeat").size().size == cfg.repeats
    assert metrics.centers.eq(len(xyz)).all()
    assert (out / "cp_losses_0_0.png").is_file()
    assert not list(out.glob("sampling_*"))


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
        Settings(window_um=2, contact_um=2, cp_device="cpu", repeats=2, rank=1),
    )
    assert result.cells.niche.tolist() == ["C0.1"] * n
    assert result.evidence.empty


def test_focal_sender_receiver_are_directional():
    from phenoniche.spatial.cell_programs import CellTensor

    graph = physical_graph(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array(["s", "s"]), 2)
    atlas = pd.DataFrame({"ligand": ["L"], "receptor": ["R"]})
    channels = Channels([[5, 0], [0, 5]], pd.Index(["L", "R"]), atlas)
    tensor = CellTensor(graph, np.array([0, 1]), channels, Settings())
    out = tensor.fiber(0)
    incoming = tensor.fiber(1)
    assert out[1][0, 0] > 0 and out[1][0, 1] == 0
    assert incoming[1][0, 1] > 0 and incoming[1][0, 0] == 0
    np.testing.assert_allclose(out[1][0, 0], incoming[1][0, 1])


def test_delaunay_filters_long_edges_and_specimens_in_2d_and_3d():
    from phenoniche.spatial.contexts import delaunay_edges

    for points in (
        [[0, 0], [1, 0], [0, 1], [10, 10]],
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [10, 10, 10]],
    ):
        xyz = np.tile(points, (2, 1))
        n = len(points)
        samples = np.repeat(["a", "b"], n)
        edges = delaunay_edges(xyz, samples, 1.1)
        assert set(map(tuple, edges)) == {
            (offset, offset + j) for offset in (0, n) for j in range(1, n - 1)
        }


def test_enrichment_counts_and_specimen_null():
    from phenoniche.spatial.contexts import neighborhood_enrichment

    edges = np.array([[0, 1], [1, 2], [3, 4], [3, 5]])
    coarse = np.array([0, 0, 0, 1, 1, 1])
    samples = np.repeat(["a", "b"], 3)
    null = neighborhood_enrichment(edges, coarse, samples, 100, 7)
    assert null.observed_edges.tolist() == [2, 0, 2]
    assert null.contact_cells_a.tolist() == [3, 0, 3]
    np.testing.assert_array_equal(null.null_mean, null.observed_edges)
    assert null.z_score.isna().all()

    labels = np.array([0, 0, 1, 0, 1, 1])
    mixed = neighborhood_enrichment(edges, labels, samples, 2000, 7).iloc[1]
    assert mixed.observed_edges == 3
    assert (mixed.contact_cells_a, mixed.contact_cells_b) == (2, 3)
    assert abs(mixed.null_mean - 8 / 3) < 0.06
    assert np.isfinite(mixed.z_score)


def test_manual_interface_uses_midpoints_and_does_not_remove_coarse_cells(monkeypatch):
    from phenoniche.spatial import model

    xyz = np.array([[-2, 0], [0, 0], [2, 0], [4, 0], [2, 0], [3, 0]])
    labels = np.array([0, 0, 1, 1, 1, 1])
    samples = ["a"] * 4 + ["b"] * 2
    types = ["T", "T", "B", "B", "B", "B"]
    monkeypatch.setattr(
        model, "composition_partition",
        lambda graph, types, cfg, cache_path: (np.eye(2)[types], labels),
    )
    args = (
        np.ones((6, 2)), ["L", "R"], np.arange(6), xyz, types, samples,
        pd.DataFrame({"ligand": ["L"], "receptor": ["R"]}),
        Settings(cp_device="cpu", contact_um=2, window_um=2, interface_um=1.1, repeats=2, rank=1,
                 iterations=2, enrichment_permutations=10),
    )
    coarse = SpatialNiches.fit(*args)
    interface = SpatialNiches.fit(*args, interface=(1, 0))
    assert coarse.cells.index.tolist() == list(range(6))
    assert interface.cells.index.tolist() == [1, 2]
    assert interface.cells.context.eq("I0_1").all()
    # Center 1 contacts cell 0 even though cell 0 is outside the interface.
    assert interface.neighbor_counts[0].sum() == 2
    np.testing.assert_array_equal(interface.programs[(0, 1)]["pool_weights"] > 0,
                                  [[True, True], [True, True]])


def test_cell_type_mean_normalization_and_structural_absence():
    from phenoniche.spatial.cell_programs import CellTensor, contact_support, pool_programs, program_graph

    xyz = np.array([[0, 0], [1, 0], [2, 0], [0, 1]])
    graph = physical_graph(xyz, np.repeat("s", 4), 2)
    types = np.array([0, 1, 1, 2])
    atlas = pd.DataFrame({"ligand": ["L"], "receptor": ["R"]})
    ch = Channels([[1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 0, 2]],
                  pd.Index(["L", "R", "B"]), atlas, preprocessing="cp10k")
    weights = np.exp(-0.5 * np.array([1, 2]) ** 2)
    present, means, _ = CellTensor(graph, types, ch, Settings()).fiber(0)
    np.testing.assert_array_equal(present, [1, 2])
    np.testing.assert_allclose(means[0], [5000, 5000])
    np.testing.assert_array_equal(means[1], [0, 0])
    counts, support = contact_support(graph, [0], types)
    np.testing.assert_array_equal(counts[0], [0, 2, 1])
    np.testing.assert_allclose(support[0], [0, 2, 1])

    allowed = np.array([True, True, True, False])
    pooling, support = program_graph(np.array([0]), allowed, graph, types, 1)
    scores = np.array([[3], [6], [12], [99]])
    avg = (scores[1:, 0][:2] @ weights) / weights.sum()
    np.testing.assert_allclose(pool_programs(pooling, support, types, scores, True), [[3, avg, 0]])
    np.testing.assert_allclose(pool_programs(pooling, support, types, scores, False),
                               [[3 / support.sum(), avg * weights.sum() / support.sum(), 0]])
    assert not support[0, 2]


def test_signaling_specific_radii_weights_and_masks():
    from phenoniche.spatial.channels import Channels
    from phenoniche.spatial.cell_programs import CellTensor
    from phenoniche.spatial.geometry import physical_graph
    from phenoniche.spatial.programs import _observed_batch

    xyz = np.array([[0., 0.], [10., 0.], [40., 0.], [90., 0.], [10., 0.]])
    graph = physical_graph(xyz, np.array(['a', 'a', 'a', 'a', 'b']), 100)
    types = np.array([0, 1, 1, 2, 2])
    lr = pd.DataFrame({'ligand': ['L', 'L', 'L'], 'receptor': ['R1', 'R2', 'R3'],
                       'signaling_type': ['Contact', 'Secreted', 'ECM']})
    # R1 expression is zero: its contact observation still enters the mask.
    counts = np.array([[1, 0, 1, 1], [1, 0, 1, 1], [1, 0, 2, 1],
                       [1, 0, 1, 1], [1, 0, 10, 1]])
    channels = Channels(counts, pd.Index(['L', 'R1', 'R2', 'R3']), lr)
    cfg = Settings(contact_radius_um=20, secreted_radius_um=100, ecm_radius_um=20)
    tensor = CellTensor(graph, types, channels, cfg)
    cats, values, support = tensor.fiber(0)
    assert cats.tolist() == [1, 2]
    assert support[0, 0] == 1
    assert support[0, 1] == 2
    assert support[1, 0] == 0 and support[1, 1] > 0 and support[1, 2] == 0
    assert values[0, 0] == 0
    expr = channels.expression.toarray()
    expected = expr[0, channels.left[1]] * np.mean(expr[[1, 2], channels.right[1]])
    assert values[0, 1] == pytest.approx(expected)
    batches = _observed_batch(np.array([0]), tensor.fiber)
    assert [len(neighbors) for _, neighbors, _, _ in batches] == [1, 2, 1]
    assert sum(matrix.shape[0] * matrix.shape[1] for _, _, matrix, _ in batches) == 8
    assert 4 not in graph.indices[graph.indptr[0]:graph.indptr[1]]
    from phenoniche.spatial.programs import fit_programs
    fit = fit_programs(np.array([0, 1, 2, 3, 4]), tensor.fiber, 3, 6,
                       Settings(cp_device="cpu", rank=1, iterations=2, cp_batch_size=2),
                       np.random.default_rng(2))
    assert fit is not None and np.isfinite(fit.metrics["total_loss"])
def test_rare_activity_scaling_preserves_zeros_without_amplification():
    from phenoniche.spatial.cell_programs import scale_activity

    activity = np.zeros((100, 3))
    activity[:, 0] = 2
    activity[0, 1] = 5
    scaled = scale_activity(activity)
    np.testing.assert_allclose(scaled[0], np.log1p([1, 1, 0]))
    assert np.all(scaled[1:, 1:] == 0)
    assert np.isfinite(scaled).all()



def test_commuspace_expression_scaling_uses_nonzero_quantile_and_preserves_zeros():
    from scipy import sparse
    counts = np.array([[2, 8, 0], [4, 6, 0], [8, 2, 0], [0, 10, 0]], dtype=float)
    genes = pd.Index(['L', 'R', 'unused'])
    lr = pd.DataFrame({'ligand': ['L'], 'receptor': ['R']})
    cp = Channels(counts, genes, lr, preprocessing='cp10k')
    scaled = Channels(sparse.csr_matrix(counts), genes, lr, preprocessing='commuspace', quantile=.5)
    values = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 10000)
    for col in range(2):
        nz = values[:, col] > 0
        values[nz, col] = np.minimum(values[nz, col] / np.quantile(values[nz, col], .5), 1)
    # Arm expression contains the square root of the preprocessed gene expression.
    np.testing.assert_allclose(scaled.expression.toarray() ** 2, values[:, :2], atol=1e-6)
    np.testing.assert_allclose(cp.expression.toarray() ** 2, counts[:, :2] * 1000, atol=1e-3)
    assert scaled.expression.toarray()[3, 0] == 0


def test_tensor_log1p_and_h3_type_contributions_sum_to_h1():
    from phenoniche.spatial.cell_programs import CellTensor, communication_features, program_graph
    xyz = np.array([[0., 0.], [1., 0.], [2., 0.]])
    graph = physical_graph(xyz, ['s'] * 3, 3)
    types = np.array([0, 1, 1])
    ch = Channels([[2, 1], [1, 3], [3, 1]], pd.Index(['L', 'R']),
                  pd.DataFrame({'ligand': ['L'], 'receptor': ['R']}), preprocessing='cp10k')
    plain = CellTensor(graph, types, ch, Settings())
    logged = CellTensor(graph, types, ch, Settings(tensor_log1p=True))
    cats, values, support = plain.fiber(0)
    cats2, transformed, support2 = logged.fiber(0)
    np.testing.assert_array_equal(cats, cats2)
    np.testing.assert_array_equal(support, support2)
    np.testing.assert_allclose(transformed, np.log1p(values))
    ids = np.arange(3); pool, weights = program_graph(ids, np.ones(3, bool), graph, types, 1)
    scores = np.array([[1., 2.], [2., 1.], [3., 4.]])
    h1 = communication_features(pool, weights, types, scores, ids, 'H1')
    h3 = communication_features(pool, weights, types, scores, ids, 'H3')
    np.testing.assert_allclose(h3.reshape(3, 2, 2).sum(axis=1), h1)
    np.testing.assert_array_equal(communication_features(pool, weights, types, scores, ids, 'A'), scores)
