"""Single-section, measured-spot dual-decoder experiment (separate from cell CCC).

No inferred single-cell expression, cell labels or outcomes enter the encoder.
The two targets are spatial LR potential and deconvolved local composition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree


WORK = Path('/home/nas3/biod/xueshuailin/CCC_Phe')
REPO = Path(__file__).resolve().parents[3]
SCHEMA = 'spot-dual-center-directed-v1'


@dataclass
class SpotConfig:
    sample: str = 'TNBC1_CN1_C2'
    input_root: str = '/home/nas3/biod/xueshuailin/New_Projects/data/HBC/TNBC_ST/Processed_Samples'
    reference: str = '/home/nas3/biod/xueshuailin/New_Projects/data/HBC/2021_breast_cancer_atlas.h5ad'
    output_root: str = str(WORK / 'results/tnbc_spot/ours')
    cache_root: str = str(WORK / 'data/tnbc_spot_dual')
    deconvolution_python: str = str(WORK / 'env/envs/tnbc_cell2location/bin/python')
    lr_atlas: str = str(REPO / 'data/commuspace_human_lr_atlas.tsv')
    calibration: str = str(WORK / 'results/tnbc_communication_phenotype/ours/calibration.csv')
    um_per_pixel: float | None = None
    min_counts: int = 500
    min_genes: int = 200
    neighbors: int = 9  # total including the center
    max_distance_pitch: float = 1.6
    sigma_pitch: float = 1.0
    lr_coverage: float = 0.10
    encoder_genes: int = 2000
    reference_label: str = 'celltype_major'
    reference_batch: str = 'donor_id'
    reference_cells_per_type: int = 1000
    reference_epochs: int = 200
    deconvolution_epochs: int = 2000
    posterior_samples: int = 128
    cells_per_spot: float = 100.0  # approximate prior for the original 100-µm ST spots
    detection_alpha: float = 20.0
    programs: int = 32
    hidden: int = 64
    message_layers: int = 2
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 0.001
    composition_weight: float = 1.0
    regularization_weight: float = 0.005
    niche_k: int | None = None
    max_niches: int = 12
    seed: int = 42
    device: str = 'cuda:0'  # logical GPU after CUDA_VISIBLE_DEVICES=2
    threads: int = 8

    def __post_init__(self):
        if self.neighbors < 2 or self.programs < 2 or self.epochs < 1:
            raise ValueError('neighbors/programs/epochs are invalid')
        if not 0 < self.lr_coverage <= 1 or self.composition_weight <= 0:
            raise ValueError('Coverage and dual-decoder weight must be positive')
        if min(self.sigma_pitch, self.max_distance_pitch, self.cells_per_spot) <= 0:
            raise ValueError('Spatial and abundance parameters must be positive')
        if min(self.reference_epochs, self.deconvolution_epochs, self.posterior_samples,
               self.reference_cells_per_type, self.batch_size, self.threads) < 1:
            raise ValueError('Training/sample counts must be positive')
        for name in ('output_root', 'cache_root'):
            p = Path(getattr(self, name)).resolve()
            if not p.is_relative_to(WORK):
                raise ValueError(f'{name} must be on the project NAS workspace')
        if Path(self.sample).name != self.sample:
            raise ValueError('sample must be a directory name')


def _hash(*values):
    h = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            h.update(str((value.shape, value.dtype)).encode())
            h.update(np.ascontiguousarray(value).tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _stamp(path):
    p = Path(path).resolve()
    stat = p.stat()
    return [str(p), stat.st_size, stat.st_mtime_ns]


def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def validate_counts(counts):
    values = counts.data if sparse.issparse(counts) else np.asarray(counts)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Counts must be finite and nonnegative')
    if not np.allclose(values, np.rint(values), atol=1e-5, rtol=0):
        raise ValueError('cell2location requires raw integer counts; normalized data cannot be rounded')


@dataclass
class SpotData:
    counts: np.ndarray
    genes: list[str]
    spots: list[str]
    coordinates: np.ndarray
    signature: str
    calibration_source: str
    abundance: pd.DataFrame | None = None
    neighbors: np.ndarray | None = None
    anchors: np.ndarray | None = None
    pair_weights: np.ndarray | None = None
    X: np.ndarray | None = None
    C: np.ndarray | None = None
    expression: np.ndarray | None = None
    features: pd.DataFrame | None = None
    encoder_gene_names: list[str] = field(default_factory=list)
    sigma: float = 0.0
    pitch: float = 0.0
    target_signature: str = ''


def read_spots(config: SpotConfig) -> SpotData:
    folder = Path(config.input_root) / config.sample
    try:
        import pyarrow
        pyarrow.set_cpu_count(config.threads)
        frame = pd.read_csv(folder / 'cnts.csv', index_col=0, engine='pyarrow')
    except ImportError:
        frame = pd.read_csv(folder / 'cnts.csv', index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.str.upper().str.strip()
    if frame.index.has_duplicates:
        raise ValueError('Duplicate spot IDs in cnts.csv')
    if frame.columns.has_duplicates:
        frame = frame.T.groupby(level=0, sort=False).sum().T
    counts = frame.to_numpy(dtype=np.float32)
    validate_counts(counts)
    locs = pd.read_csv(folder / 'locs.csv', dtype={'spots': str}).set_index('spots')
    if locs.index.has_duplicates or not frame.index.isin(locs.index).all():
        raise ValueError('Spot IDs and spatial coordinates do not align')
    coordinates = locs.loc[frame.index, ['x', 'y']].to_numpy(dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError('Non-finite spatial coordinates')
    keep = (counts.sum(1) >= config.min_counts) & ((counts > 0).sum(1) >= config.min_genes)
    counts, coordinates = counts[keep], coordinates[keep]
    if len(counts) < max(10, config.neighbors):
        raise ValueError('Too few spots after count/gene QC')
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError('Duplicate spot coordinates')
    if config.um_per_pixel is None:
        calibration = pd.read_csv(config.calibration).set_index('sample')
        row = calibration.loc[config.sample]
        factor, source = float(row.um_per_pixel), str(row.source)
    else:
        factor, source = config.um_per_pixel, 'Explicit user-provided µm/pixel'
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('Invalid spatial calibration')
    coordinates *= factor
    spots = frame.index[keep].tolist()
    signature = _hash(SCHEMA, counts, coordinates, spots, frame.columns.tolist())
    data = SpotData(counts, frame.columns.tolist(), spots, coordinates.astype(np.float32), signature, source)
    print(f'{config.sample}: {len(spots):,} spots × {len(data.genes):,} genes; removed {(~keep).sum()} low-quality spots')
    print('Coordinates: µm; published spot-pitch calibration is approximate.' if config.um_per_pixel is None else 'Coordinates: µm (user calibration).')
    return data


def deconvolve(data: SpotData, config: SpotConfig):
    """Run real cell2location in an isolated worker; exact fingerprint cache only."""
    import anndata as ad
    worker = Path(__file__).with_name('tnbc_spot_deconvolution.py')
    keys = ('reference_label', 'reference_batch', 'reference_cells_per_type', 'reference_epochs',
            'deconvolution_epochs', 'posterior_samples', 'cells_per_spot', 'detection_alpha', 'seed')
    options = {k: getattr(config, k) for k in keys}
    signature = _hash(data.signature, options, _stamp(config.reference), worker.read_text())
    folder = Path(config.cache_root) / config.sample / 'deconvolution' / signature
    output = folder / 'abundance.csv'
    complete = folder / 'complete.json'
    if output.exists() and complete.exists() and json.loads(complete.read_text())['signature'] == signature:
        print('cell2location: loading matching cached abundances')
    else:
        if not Path(config.deconvolution_python).is_file():
            raise FileNotFoundError(f'cell2location worker environment missing: {config.deconvolution_python}')
        folder.mkdir(parents=True, exist_ok=True)
        spatial = ad.AnnData(sparse.csr_matrix(data.counts),
                             obs=pd.DataFrame(index=pd.Index(data.spots)),
                             var=pd.DataFrame(index=pd.Index(data.genes)))
        spatial.write_h5ad(folder / 'spots_raw.h5ad')
        request = dict(options, signature=signature, reference=config.reference,
                       cache_root=config.cache_root, folder=str(folder), device=config.device,
                       threads=config.threads, worker_hash=_hash(worker.read_text()))
        _save_json(folder / 'request.json', request)
        command = [config.deconvolution_python, '-u', str(worker), str(folder / 'request.json')]
        print(f'cell2location: fitting one section; progress log: {folder / "run.log"}', flush=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS=str(config.threads))
        # One worker per cache, including notebook restarts. Lock is released on process exit.
        import fcntl
        with (folder / 'worker.lock').open('w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f'A deconvolution worker already owns {folder}') from error
            if not complete.exists():
                with (folder / 'run.log').open('a') as log:
                    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               pass_fds=(lock.fileno(),))
                    try:
                        code = process.wait()
                    except BaseException:
                        process.terminate()
                        process.wait()
                        raise
                if code:
                    tail = (folder / 'run.log').read_text()[-5000:]
                    raise RuntimeError(f'cell2location failed (see run.log):\n{tail}')
    abundance = pd.read_csv(output, index_col=0)
    abundance.index = abundance.index.astype(str)
    if abundance.index.has_duplicates or set(abundance.index) != set(data.spots):
        raise ValueError('Cached deconvolution spot IDs do not match current data')
    abundance = abundance.loc[data.spots]
    values = abundance.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values.sum(1) <= 0).any():
        raise ValueError('Invalid cell2location abundances')
    data.abundance = abundance
    print(f'cell2location: {abundance.shape[0]:,} spots × {abundance.shape[1]} reference cell types')
    return abundance


def local_geometry(coordinates, config):
    """Nine spots at most; pad out-of-radius neighbors with zero-weight center IDs."""
    distance, index = cKDTree(coordinates).query(coordinates, k=config.neighbors)
    pitch = float(np.median(distance[:, 1]))
    sigma = pitch * config.sigma_pitch
    valid = distance <= pitch * config.max_distance_pitch
    index = np.where(valid, index, np.arange(len(index))[:, None])
    anchors = (np.exp(-distance**2 / (2 * sigma**2)) * valid).astype(np.float32)
    local = coordinates[index]
    d2 = ((local[:, :, None] - local[:, None])**2).sum(-1)
    pairs = anchors[:, :, None] * anchors[:, None, :] * np.exp(-d2 / (2 * sigma**2))
    pairs *= (d2 <= (pitch * config.max_distance_pitch)**2)
    pairs *= index[:, :, None] != index[:, None, :]
    return index, anchors, pairs.astype(np.float32), pitch, sigma


def center_lr_targets(expression, neighbors, weights, ligand, receptor):
    """sqrt(L_center R_neighbor) outgoing, sqrt(L_neighbor R_center) incoming.

    Each LR score is normalized by available, distance-weighted center-neighbor
    opportunity. Excludes self edges; these are spot potentials, not cell-pair CCC.
    """
    denominator = weights.sum(1).clip(1e-8)
    l, r = np.sqrt(expression[:, ligand]), np.sqrt(expression[:, receptor])
    outgoing = l * (r[neighbors] * weights).sum(1) / denominator
    incoming = r * (l[neighbors] * weights).sum(1) / denominator
    return outgoing, incoming


def build_targets(data: SpotData, config: SpotConfig):
    """No model fit or labels are used to construct these observed targets."""
    from ..ccc.lr_atlas import load_lr_atlas
    if data.abundance is None:
        raise ValueError('Run deconvolution first')
    if list(data.abundance.index) != data.spots:
        raise ValueError('Abundance order must match expression spot IDs')
    index, anchors, pairs, pitch, sigma = local_geometry(data.coordinates, config)
    if (pairs[:, 0].sum(1) == 0).any():
        raise ValueError('Isolated spots have no communication opportunity; adjust QC/radius explicitly')
    norm = data.counts / data.counts.sum(1, keepdims=True) * 1e4
    genes = {g: i for i, g in enumerate(data.genes)}
    coverage = (data.counts > 0).mean(0)
    lr = load_lr_atlas(config.lr_atlas)
    columns, features = [], []
    center_weights = pairs[:, 0, :]
    # Compound subunits require all genes and use geometric mean expression.
    for row in lr.interactions:
        components = row.ligand_components + row.receptor_components
        if not all(g in genes and coverage[genes[g]] >= config.lr_coverage for g in components):
            continue
        sides = []
        for components in (row.ligand_components, row.receptor_components):
            values = norm[:, [genes[g] for g in components]]
            side = np.exp(np.log(values.clip(1e-30)).mean(1)) * (values > 0).all(1)
            sides.append(side)
        stacked = np.stack(sides, axis=1)
        scores = center_lr_targets(stacked, index, center_weights, 0, 1)
        for direction, score in zip(('center_to_neighbors', 'neighbors_to_center'), scores):
            if np.mean(score > 0) < config.lr_coverage:
                continue
            columns.append(score)
            features.append(dict(lr_id=row.lr_id, ligand=row.ligand, receptor=row.receptor, direction=direction))
    if len(columns) < config.programs:
        raise ValueError(f'Only {len(columns)} supported LR features for {config.programs} programs')
    data.X = np.stack(columns, axis=1).astype(np.float32)
    data.features = pd.DataFrame(features)
    # Same center-neighbor opportunities: center and its partners each contribute
    # their incident edge degree. Sum absolute abundance before normalizing.
    star = np.zeros_like(pairs)
    star[:, 0, :] = center_weights
    star[:, :, 0] = center_weights
    degree = star.sum(1) + star.sum(2)
    abundance = data.abundance.to_numpy(dtype=np.float32)
    mixture = (abundance[index] * degree[..., None]).sum(1)
    data.C = (mixture / mixture.sum(1, keepdims=True)).astype(np.float32)
    lognorm = np.log1p(norm)
    # Expression-only HVGs with LR genes retained; deconvolution never enters input.
    variance = lognorm.var(0)
    allowed = np.array([not g.startswith('MT-') for g in data.genes]) & (coverage > 0)
    hvgs = np.flatnonzero(allowed)[np.argsort(variance[allowed])[-config.encoder_genes:]]
    lr_genes = set()
    for row in features:
        for side in ('ligand', 'receptor'):
            lr_genes.update(genes[g] for g in row[side].split('&'))
    selected = np.unique(np.r_[hvgs, list(lr_genes)]).astype(int)
    data.expression = lognorm[:, selected].astype(np.float32)
    data.encoder_gene_names = [data.genes[i] for i in selected]
    data.neighbors, data.anchors, data.pair_weights = index, anchors, pairs
    data.pitch, data.sigma = pitch, sigma
    data.target_signature = _hash(SCHEMA, data.signature, data.X, data.C, data.expression,
                                  pairs, index, anchors, features)
    folder = Path(config.output_root) / config.sample
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'targets.npz', X=data.X, C=data.C, neighbors=index,
                        anchors=anchors, pair_weights=pairs, spots=np.array(data.spots),
                        cell_types=np.array(data.abundance.columns))
    data.features.to_csv(folder / 'ccc_features.csv', index=False)
    pd.Series(data.encoder_gene_names, name='gene').to_csv(folder / 'encoder_genes.csv', index=False)
    _save_json(folder / 'target_definition.json', dict(schema=SCHEMA, signature=data.target_signature,
               calibration=data.calibration_source, pitch_um=pitch, sigma_um=sigma,
               max_distance_um=pitch*config.max_distance_pitch, neighbors_including_center=config.neighbors,
               X='Two center-role directions per LR; weighted mean sqrt(TP10K_L * TP10K_R); no self edges',
               C='cell2location mean abundance, summed using the same center-edge incident degrees then normalized',
               interpretation='Spot-level LR potential, not assigned sender/receiver cell-type communication',
               isolated_spots=int((pairs[:, 0].sum(1) == 0).sum()),
               actual_spots_per_environment=(anchors > 0).sum(1).tolist()))
    print(f'X (LR potential): {data.X.shape}; C (cell composition): {data.C.shape}; encoder genes: {data.expression.shape[1]}')
    return data


def _batch(data, rows, device):
    import torch
    members = data.neighbors[rows]
    unique, inverse = np.unique(members, return_inverse=True)
    tensor = lambda value, dtype=torch.float32: torch.as_tensor(value, dtype=dtype, device=device)
    return dict(expression=tensor(data.expression[unique]),
                local_index=tensor(inverse.reshape(members.shape), torch.long),
                coordinates=tensor(data.coordinates[members]), anchor_weights=tensor(data.anchors[rows]),
                cell_ids=tensor(members, torch.long), pair_override=tensor(data.pair_weights[rows]))




