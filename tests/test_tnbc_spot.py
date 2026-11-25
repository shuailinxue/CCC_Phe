"""Scientific and cache invariants for the separate spot-level experiment."""
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from phenoniche.cohorts.tnbc_spot import (
    SpotConfig, SpotData, center_lr_targets, local_geometry, validate_counts,
    build_targets, deconvolve, _hash, _stamp, _save_json,
)


def test_counts_reject_normalized_expression():
    validate_counts(np.array([[0, 1, 3]], dtype=np.float32))
    with pytest.raises(ValueError, match='raw integer'):
        validate_counts(np.array([[0, 1.2, 3]]))


def test_reference_uses_backed_raw_and_preserves_reference_types():
    import anndata as ad
    from scipy import sparse
    from phenoniche.cohorts.tnbc_spot_deconvolution import read_reference
    with tempfile.TemporaryDirectory() as tmp:
        genes = [f'G{i}' for i in range(120)]
        atlas = ad.AnnData(sparse.csr_matrix(np.full((6, 120), 3, dtype=np.float32)),
                          obs=pd.DataFrame({'celltype_major': ['A', 'A', 'A', 'B', 'B', 'Unknown'],
                                            'donor_id': ['D1']*6}, index=[f'c{i}' for i in range(6)]),
                          var=pd.DataFrame({'feature_name': genes}, index=genes))
        atlas.raw = atlas.copy()
        atlas.X.data = np.log1p(atlas.X.data)
        path = Path(tmp)/'atlas.h5ad'
        atlas.write_h5ad(path)
        reference = read_reference(dict(reference=str(path), reference_label='celltype_major',
                                        reference_batch='donor_id', reference_cells_per_type=2, seed=42))
        assert reference.shape == (4, 120)
        assert set(reference.obs.cell_type) == {'A', 'B'}
        assert (reference.X.data == 3).all()


def test_graph_holes_and_no_self_edges():
    xy = np.array([[0, 0], [1, 0], [0, 1], [10, 10]], dtype=np.float32)
    config = SpotConfig(neighbors=3, max_distance_pitch=1.6)
    index, anchors, weights, pitch, sigma = local_geometry(xy, config)
    assert (index[:, 0] == np.arange(4)).all()
    assert weights[3].sum() == 0  # never bridges the tissue gap
    assert (weights[index[:, :, None] == index[:, None, :]] == 0).all()
    assert np.allclose(weights, weights.transpose(0, 2, 1))
    assert anchors[3].sum() == 1


def test_lr_center_roles_are_distinct_and_exact():
    # Center L=4 R=0, neighbor L=0 R=9; no artificial self communication.
    expression = np.array([[4., 0.], [0., 9.]])
    neighbors = np.array([[0, 1], [1, 0]])
    weights = np.array([[0, 0.3], [0, 0.3]])
    outgoing, incoming = center_lr_targets(expression, neighbors, weights, 0, 1)
    np.testing.assert_allclose(outgoing, [6, 0])
    np.testing.assert_allclose(incoming, [0, 6])


def _synthetic(folder):
    config = SpotConfig(output_root=str(folder / 'output'), cache_root=str(folder / 'cache'),
                        neighbors=4, programs=3, encoder_genes=5, hidden=8,
                        epochs=2, batch_size=12, device='cpu', threads=1,
                        niche_k=2, reference_epochs=1, deconvolution_epochs=1, posterior_samples=2)
    n = 36
    xy = np.indices((6, 6)).reshape(2, -1).T.astype(np.float32)
    rng = np.random.default_rng(17)
    counts = rng.poisson(np.where(xy[:, :1] < 3, [14, 3, 2, 1], [2, 4, 12, 5])).astype(np.float32)
    spots = [f's{i}' for i in range(n)]
    abundance = pd.DataFrame(np.where(xy[:, :1] < 3, [8, 2], [2, 8]), index=spots, columns=['A', 'B'])
    atlas = folder / 'lr.tsv'
    atlas.write_text('ligand\treceptor\nL1\tR1\nL2\tR2\nL1\tR2\n')
    config.lr_atlas = str(atlas)
    data = SpotData(counts, ['L1', 'R1', 'L2', 'R2'], spots, xy, _hash(counts), 'synthetic', abundance)
    return data, config




def test_exact_deconvolution_cache_and_spot_alignment(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr("phenoniche.cohorts.tnbc_spot.WORK", Path(tmp))
        data, config = _synthetic(Path(tmp))
        reference = Path(tmp)/'ref.h5ad'
        reference.write_bytes(b'identity only; cached result bypasses reference reading')
        config.reference = str(reference)
        worker = Path(__import__('phenoniche.cohorts.tnbc_spot', fromlist=['x']).__file__).with_name('tnbc_spot_deconvolution.py')
        keys = ('reference_label', 'reference_batch', 'reference_cells_per_type', 'reference_epochs',
                'deconvolution_epochs', 'posterior_samples', 'cells_per_spot', 'detection_alpha', 'seed')
        signature = _hash(data.signature, {k: getattr(config, k) for k in keys}, _stamp(reference), worker.read_text())
        folder = Path(config.cache_root)/config.sample/'deconvolution'/signature
        folder.mkdir(parents=True)
        data.abundance.iloc[::-1].to_csv(folder/'abundance.csv')
        _save_json(folder/'complete.json', dict(signature=signature))
        expected = data.abundance.copy()
        data.abundance = None
        found = deconvolve(data, config)
        pd.testing.assert_frame_equal(expected, found, check_dtype=False, check_names=False)
