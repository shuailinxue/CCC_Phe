"""Cached Leiden on the exact exported h, independent of every model backend."""
from importlib.metadata import version
from pathlib import Path
import json
import time
import numpy as np
from scipy import sparse
from .io import output_path, write_json, digest, array_identity, fit_lock


def partition_embedding(exported, output, *, target_k=10, resolution=None, n_neighbors=30,
                        seed=40700, threads=6, max_search=8, report=None):
    import anndata as ad
    import scanpy as sc
    import numba
    from threadpoolctl import threadpool_limits
    from scanpy._utils import get_igraph_from_adjacency
    from scanpy._utils.random import set_igraph_random_state
    exported = Path(exported)
    metadata = json.loads((exported / 'export.json').read_text())
    if not metadata.get('global_training_affine_calibration') or metadata.get('row_normalization'):
        raise ValueError('Leiden requires the unchanged calibrated clustering h')
    if type(target_k) is not int or target_k < 2 or max_search < 1:
        raise ValueError('Invalid target K or resolution search budget')
    if resolution is not None and (not np.isfinite(resolution) or resolution <= 0):
        raise ValueError('Resolution must be finite and positive')
    h = np.load(exported / 'embedding.npy', mmap_mode='r')
    if h.ndim != 2 or not 2 <= n_neighbors < len(h) or not np.isfinite(h).all():
        raise ValueError('Invalid h or neighbor count')
    protocol = dict(embedding=array_identity(h), backend=metadata['backend'],
                    n_neighbors=n_neighbors, metric='euclidean', transformer='pynndescent',
                    seed=seed, extra_PCA=False, row_normalization=False,
                    scanpy_version=version('scanpy'), pynndescent_version=version('pynndescent'))
    root = output_path(output)
    graph = output_path(root / 'graphs' / digest(protocol)[:16])
    with fit_lock(graph):
        if not (graph / 'complete.json').exists():
            print(f'Build Leiden h-neighbor graph: all {len(h):,} cells, '
                  f'{h.shape[1]} dimensions, {n_neighbors} neighbors; no new UMAP.', flush=True)
            if report:
                report(dict(stage='building_h_neighbor_graph', cells=len(h)))
            started = time.monotonic()
            u = ad.AnnData(np.array(h, dtype=np.float32, copy=True, order='C'))
            sc.settings.n_jobs = threads
            numba.set_num_threads(threads)
            with threadpool_limits(limits=threads):
                sc.pp.neighbors(u, n_neighbors=n_neighbors, use_rep='X', metric='euclidean',
                                transformer='pynndescent', random_state=seed)
            adjacency = u.obsp['connectivities'].tocsr()
            sparse.save_npz(graph / 'connectivities.npz', adjacency)
            write_json(graph / 'complete.json', dict(protocol=protocol, cells=len(h),
                       entries=adjacency.nnz, seconds=time.monotonic() - started,
                       graph='weighted symmetrized fuzzy h-similarity graph; not physical adjacency',
                       sampling=False, model_retrained=False, UMAP_recomputed=False))
            del u
        else:
            print('Read cached Leiden h-neighbor graph.', flush=True)
    spec = dict(graph=str(graph), target_k=target_k, resolution=resolution, max_search=max_search,
                seed=seed, flavor='igraph', objective='modularity', directed=False,
                use_weights=True, n_iterations=2, igraph_version=version('igraph'),
                selection='class count only; no biological, spatial, sample labels or UMAP used')
    folder = output_path(root / 'partitions' / digest(spec)[:16])
    with fit_lock(folder):
        if not (folder / 'complete.json').exists():
            adjacency = sparse.load_npz(graph / 'connectivities.npz')
            # Build igraph once and reuse it for all resolutions, matching Scanpy's igraph flavor.
            print('Convert cached similarity graph once for Leiden resolution trials.', flush=True)
            graph_object = get_igraph_from_adjacency(adjacency, directed=False)
            del adjacency
            records, best = [], None
            lo, hi, value = None, None, .1 if resolution is None else float(resolution)
            for index in range(1 if resolution is not None else max_search):
                trial_folder = output_path(folder / f'trial_{index + 1:02d}')
                record_path = trial_folder / 'complete.json'
                if record_path.exists():
                    record = json.loads(record_path.read_text())
                    if record['resolution'] != value:
                        raise ValueError('Saved resolution trial differs from deterministic search')
                    labels = np.load(trial_folder / 'labels.npy')
                else:
                    print(f'Leiden resolution={value:g} ...', flush=True)
                    if report:
                        report(dict(stage='Leiden', resolution=value, trial=index + 1))
                    started = time.monotonic()
                    with threadpool_limits(limits=threads), set_igraph_random_state(seed):
                        part = graph_object.community_leiden(objective_function='modularity',
                            weights='weight', resolution=value, n_iterations=2)
                    labels = np.asarray(part.membership, np.int32)
                    _, labels = np.unique(labels, return_inverse=True)
                    labels = labels.astype(np.int32)
                    record = dict(resolution=value, classes=int(len(np.unique(labels))),
                                  seconds=time.monotonic() - started)
                    np.save(trial_folder / 'labels.npy', labels)
                    write_json(record_path, record)
                records.append(record)
                count = record['classes']
                print(f'Leiden: {count} classes; {record["seconds"]:.1f}s.', flush=True)
                if best is None or abs(count - target_k) < abs(best['classes'] - target_k):
                    best = dict(record, labels_path=str(trial_folder / 'labels.npy'))
                write_json(folder / 'search.json', dict(protocol=spec, trials=records))
                if count == target_k or resolution is not None:
                    break
                if count < target_k:
                    lo = value
                else:
                    hi = value
                value = hi / 2 if lo is None else lo * 2 if hi is None else (lo + hi) / 2
            labels = np.load(best['labels_path'])
            np.save(folder / 'labels.npy', labels)
            write_json(folder / 'complete.json', dict(protocol=spec, classes=best['classes'],
                       resolution=best['resolution'], trials=records,
                       matches_target_count=best['classes'] == target_k,
                       model_retrained=False, UMAP_recomputed=False))
    info = json.loads((folder / 'complete.json').read_text())
    print(f'Final Leiden: {info["classes"]} classes, resolution={info["resolution"]:g}.', flush=True)
    return dict(folder=str(folder), graph=str(graph), **info)
