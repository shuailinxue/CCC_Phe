"""Isolated cell2location worker; invoke through tnbc_spot.deconvolve.

Requires cell2location 0.1.5 and scvi-tools 1.4.2. The caller's environment
is not upgraded. All model fitting uses raw counts and outcome-blind labels.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from phenoniche.cohorts.tnbc_spot import _hash, _stamp, _save_json, validate_counts


def read_reference(request):
    """Balanced subsampling by cell type; never by disease/response/outcome."""
    source = ad.read_h5ad(request['reference'], backed='r')
    try:
        if source.raw is None:
            raise ValueError('Atlas has no raw counts; normalized X is not a substitute')
        label, batch = request['reference_label'], request['reference_batch']
        if label not in source.obs or batch not in source.obs:
            raise ValueError('Reference label or donor column missing')
        labels = source.obs[label].astype(str)
        known = source.obs[label].notna() & ~labels.str.lower().isin(['unknown', 'unassigned', 'nan'])
        rng = np.random.default_rng(request['seed'])
        rows = []
        for category in sorted(labels[known].unique()):
            indices = np.flatnonzero((labels == category) & known)
            rows.extend(rng.choice(indices, min(len(indices), request['reference_cells_per_type']), replace=False))
        rows = np.sort(rows)
        # raw.X is backed sparse in this atlas; subset cells before materializing.
        counts = sparse.csr_matrix(source.raw.X[rows, :], dtype=np.float32)
        validate_counts(counts)
        var = source.raw.var
        names = var['feature_name'].astype(str).str.upper() if 'feature_name' in var else var.index.astype(str).str.upper()
        names = np.asarray(names)
        unique, inverse = np.unique(names, return_inverse=True)
        mapping = sparse.csr_matrix((np.ones(len(names)), (np.arange(len(names)), inverse)), shape=(len(names), len(unique)))
        counts = (counts @ mapping).astype(np.float32).tocsr()
        obs = source.obs.iloc[rows][[label, batch]].copy()
        obs.columns = ['cell_type', 'donor']
        obs = obs.astype(str)
        obs['donor'] = obs['donor'].fillna('unspecified')
        detected = np.asarray((counts > 0).sum(0)).ravel()
        mean_nonzero = np.asarray(counts.sum(0)).ravel()/np.maximum(detected, 1)
        # Official cell2location expression filter, without its plotting side effect.
        keep = ((detected > len(rows)*0.03) | ((detected > 5) & (mean_nonzero > 1.12)))
        keep &= ~np.char.startswith(unique.astype(str), 'MT-')
        reference = ad.AnnData(counts[:, keep], obs=obs,
                               var=pd.DataFrame(index=pd.Index(unique[keep])))
        if reference.obs.cell_type.nunique() < 2 or reference.n_vars < 100:
            raise ValueError('Insufficient reference types or supported reference genes')
        print(f'[cell2location] reference: {reference.n_obs} raw-count cells, {reference.n_vars} genes, '
              f'{reference.obs.cell_type.nunique()} types', flush=True)
        return reference
    finally:
        source.file.close()


def run(request):
    import fcntl
    import torch
    import scvi
    import cell2location
    from cell2location.models import RegressionModel, Cell2location
    if cell2location.__version__ != '0.1.5' or scvi.__version__ != '1.4.2':
        raise RuntimeError('Use the pinned isolated cell2location 0.1.5 / scvi-tools 1.4.2 environment')
    torch.set_num_threads(request['threads'])
    scvi.settings.seed = request['seed']
    gpu = request['device'].startswith('cuda')
    if gpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in cell2location worker; select CPU explicitly')
    device_index = int(request['device'].split(':')[-1]) if gpu and ':' in request['device'] else 0
    train_device = dict(accelerator='gpu' if gpu else 'cpu', device=[device_index] if gpu else 1)
    sample_device = dict(train_device)
    trainer = dict(train_size=1.0, validation_size=0.0, early_stopping=False, enable_progress_bar=False,
                   enable_model_summary=False, logger=False, **train_device)
    folder = Path(request['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    reference_settings = {k: request[k] for k in ('reference_label', 'reference_batch',
                           'reference_cells_per_type', 'reference_epochs', 'posterior_samples', 'seed', 'worker_hash')}
    reference_key = _hash(_stamp(request['reference']), reference_settings,
                          cell2location.__version__, scvi.__version__)
    ref_folder = Path(request['cache_root']) / 'reference_signatures' / reference_key
    ref_folder.mkdir(parents=True, exist_ok=True)
    with (ref_folder / 'fit.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (ref_folder / 'complete.json').exists():
            signatures = pd.read_csv(ref_folder / 'signatures.csv', index_col=0)
            print('[cell2location] loading matching reference-signature cache', flush=True)
        else:
            reference = read_reference(request)
            RegressionModel.setup_anndata(reference, batch_key='donor', labels_key='cell_type')
            model = RegressionModel(reference)
            started = time.monotonic()
            print(f'[cell2location] fitting reference NB regression: {request["reference_epochs"]} epochs', flush=True)
            model.train(max_epochs=request['reference_epochs'], batch_size=1024, **trainer)
            reference = model.export_posterior(reference, sample_kwargs=dict(
                num_samples=request['posterior_samples'], batch_size=1024, **sample_device), add_to_varm=['means'])
            factors = list(model.factor_names_)
            names = ['means_per_cluster_mu_fg_'+name for name in factors]
            value = reference.varm['means_per_cluster_mu_fg']
            values = value[names].to_numpy() if isinstance(value, pd.DataFrame) else np.asarray(value)
            signatures = pd.DataFrame(values, index=reference.var_names, columns=factors)
            signatures.to_csv(ref_folder / 'signatures.csv')
            model.save(str(ref_folder / 'model'), overwrite=True)
            _save_json(ref_folder / 'complete.json', dict(signature=reference_key, settings=reference_settings,
                       elapsed_seconds=time.monotonic()-started, cell_types=factors,
                       reference_cells=reference.n_obs, reference_genes=reference.n_vars))
            del model, reference
            if gpu:
                torch.cuda.empty_cache()
    spatial = ad.read_h5ad(folder / 'spots_raw.h5ad')
    validate_counts(spatial.X)
    common = spatial.var_names.intersection(signatures.index, sort=False)
    common = common[np.asarray((spatial[:, common].X > 0).sum(0)).ravel() > 0]
    if len(common) < 100:
        raise ValueError('Fewer than 100 shared detected raw-count genes')
    spatial = spatial[:, common].copy()
    signatures = signatures.loc[common].astype(np.float32)
    if not np.isfinite(signatures.to_numpy()).all() or (signatures.to_numpy() < 0).any():
        raise ValueError('Invalid reference signatures')
    spatial.obs['section'] = 'single_section'
    Cell2location.setup_anndata(spatial, batch_key='section')
    model = Cell2location(spatial, cell_state_df=signatures,
                         N_cells_per_location=request['cells_per_spot'], detection_alpha=request['detection_alpha'])
    print(f'[cell2location] spatial fit: {spatial.n_obs} spots, {spatial.n_vars} shared genes; '
          f'{request["deconvolution_epochs"]} epochs', flush=True)
    started = time.monotonic()
    model.train(max_epochs=request['deconvolution_epochs'], batch_size=None, **trainer)
    spatial = model.export_posterior(spatial, sample_kwargs=dict(num_samples=request['posterior_samples'],
             batch_size=256, **sample_device), add_to_obsm=['means', 'q05'])
    values = spatial.obsm['means_cell_abundance_w_sf']
    # export columns carry a prefix; the verified model factor order is authoritative.
    abundance = pd.DataFrame(np.asarray(values), index=spatial.obs_names, columns=list(model.factor_names_))
    if not np.isfinite(abundance.to_numpy()).all() or (abundance.to_numpy() < 0).any():
        raise FloatingPointError('Nonfinite/negative abundance posterior')
    abundance.to_csv(folder / 'abundance.csv')
    pd.DataFrame(np.asarray(spatial.obsm['q05_cell_abundance_w_sf']), index=spatial.obs_names,
                 columns=list(model.factor_names_)).to_csv(folder / 'abundance_q05.csv')
    model.save(str(folder / 'model'), overwrite=True)
    for name, values in model.history.items():
        values.to_csv(folder / f'history_{name}.csv')
    _save_json(folder / 'complete.json', dict(signature=request['signature'], reference_signature=reference_key,
               cell2location=cell2location.__version__, scvi=scvi.__version__,
               elapsed_seconds=time.monotonic()-started, shared_genes=len(common),
               cell_types=list(model.factor_names_), spatial_epochs=request['deconvolution_epochs'],
               note='Quick exploratory fit; convergence is not guaranteed by the configured epoch count'))
    print('[cell2location] done', flush=True)


if __name__ == '__main__':
    run(json.loads(Path(sys.argv[1]).read_text()))
