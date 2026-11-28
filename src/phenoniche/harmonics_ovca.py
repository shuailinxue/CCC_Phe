"""Run the upstream Harmonics baseline on the full Xenium OVCA slice."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

K = 8
PREPROCESS = dict(ct_key="major_celltype", spatial_key="spatial", method="joint",
                  n_step=3, n_neighbors=20, cut_percentage=99, kernel=None)
MODEL = dict(seed=73, refine_k=0, parallel=True, light_mode=True, block_size=100_000)
INITIALIZE = dict(method="kmeans", Qmax=20, dim_reduction=True, standardize=True)
MATCH = dict(assign_metric="jsd", weighted_merge=True, max_iters=100, tol=1e-4, Qmin=K)


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _cached(directory, name, provenance, ids, compute):
    metadata = json.dumps(provenance, sort_keys=True)
    key = hashlib.sha256(metadata.encode()).hexdigest()
    path = directory / f"{name}-{key[:20]}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            if saved["metadata"].item() != metadata or not np.array_equal(saved["ids"], ids):
                raise ValueError(f"Cache provenance mismatch: {path}")
            result = {field: saved[field] for field in saved.files if field not in ("metadata", "ids")}
        print(f"Loaded {path.name}", flush=True)
        return result, path

    result = compute()
    with NamedTemporaryFile(dir=directory, suffix=".partial", delete=False) as temp:
        pending = Path(temp.name)
        try:
            np.savez_compressed(temp, metadata=metadata, ids=ids, **result)
            temp.flush()
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
    print(f"Saved {path.name}", flush=True)
    return result, path


def run_full_slice(source, directory, *, upstream_revision, cpu_threads=16):
    """Return aligned inputs, the fixed eight-niche solution and its cache path."""
    import Harmonics
    from numba import config, set_num_threads

    source = Path(source)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as handle:
        # Read only the required observation fields and coordinates.
        obs_group = handle["obs"]
        ids = np.asarray(read_elem(obs_group[obs_group.attrs["_index"]]), dtype=str)
        cell_types = read_elem(obs_group[PREPROCESS["ct_key"]])
        xy = read_elem(handle["obsm"][PREPROCESS["spatial_key"]])

    obs = pd.DataFrame({PREPROCESS["ct_key"]: cell_types}, index=ids)
    if not obs.index.is_unique or obs.isna().any().any():
        raise ValueError("Harmonics requires unique cell IDs and complete cell-type annotations.")
    if xy.shape != (len(ids), 2) or not np.isfinite(xy).all():
        raise ValueError("Expected finite X/Y coordinates for every cell.")
    values = obs[PREPROCESS["ct_key"]].to_numpy(dtype=str)
    type_names, type_index = np.unique(values, return_inverse=True)
    adata = ad.AnnData(obs=obs, obsm={PREPROCESS["spatial_key"]: xy})
    upstream_dir = Path(Harmonics.__file__).resolve().parent
    provenance = {
        "source": str(source.resolve()),
        "inputs": {"ids": _digest(ids), "coordinates": _digest(xy), "cell_types": _digest(values)},
        "slice": "OVCA",
        "rows": len(ids),
        "schema": 1,
        "upstream_revision": upstream_revision,
        "upstream_sha256": {
            name: hashlib.sha256((upstream_dir / name).read_bytes()).hexdigest()
            for name in ("model.py", "utils.py")
        },
        "versions": {
            name: version(name) for name in
            ("anndata", "h5py", "numpy", "pandas", "scipy", "scikit-learn", "numba", "llvmlite", "threadpoolctl")
        } | {"python": platform.python_version()},
        "preprocess": PREPROCESS,
    }

    def new_model():
        return Harmonics.Harmonics_Model([adata], ["OVCA"], **MODEL)

    def compute_composition():
        model = new_model()
        model.preprocess(**PREPROCESS)
        observed_types = [model.adata_concat.uns["idx2ct"][str(i)] for i in range(model.n_celltypes)]
        np.testing.assert_array_equal(observed_types, type_names)
        return dict(composition=model.micro_dist_basic.toarray(), types=type_names,
                    neighbor_counts=model.cell_count_micro_basic)

    def fit():
        model = new_model()
        model.micro_dist_basic = csr_matrix(composition["composition"])
        model.n_celltypes = len(type_names)
        model.ct_onehot_basic = Harmonics.index2onehot(type_index, n_cols=len(type_names))
        np.testing.assert_array_equal(model.adata_concat.obs_names.to_numpy(dtype=str), ids)
        set_num_threads(min(cpu_threads, config.NUMBA_NUM_THREADS))
        with threadpool_limits(limits=cpu_threads):
            model.initialize_clusters(**INITIALIZE)
            model.hier_dist_match(**MATCH)
            model.select_solution(n_niche=K, niche_key="Harmonics_niche", auto=False, return_adata=False, plot=False)
        labels = adata.obs["Harmonics_niche"].to_numpy(dtype=np.int8)
        np.testing.assert_array_equal(np.unique(labels), np.arange(K))
        niche_counts = np.bincount(labels, minlength=K)
        np.testing.assert_array_equal(model.cell_count_niche_basic.ravel(), niche_counts)
        return dict(labels=labels, counts=niche_counts, niche_composition=model.cn_dist_basic.toarray(),
                    hierarchy_counts=np.asarray(model.cn_count_list_basic), coordinates=xy,
                    cell_types=values)

    composition, prep_path = _cached(directory, "neighborhoods", provenance, ids, compute_composition)
    np.testing.assert_array_equal(composition["types"], type_names)
    if composition["composition"].shape != (len(ids), len(type_names)):
        raise ValueError(f"Unexpected composition shape in {prep_path}")
    np.testing.assert_allclose(composition["composition"].sum(axis=1), 1.0)
    fit_provenance = {
        **provenance, "neighborhoods": prep_path.name,
        "composition_sha256": _digest(composition["composition"]),
        "model": MODEL, "initialize": INITIALIZE, "match": MATCH, "n_niche": K,
        "cpu_threads": cpu_threads, "device": "CPU",
    }
    result, path = _cached(directory, "k8", fit_provenance, ids, fit)
    np.testing.assert_array_equal(np.unique(result["labels"]), np.arange(K))
    np.testing.assert_array_equal(result["counts"], np.bincount(result["labels"], minlength=K))
    np.testing.assert_array_equal(result["coordinates"], xy)
    np.testing.assert_array_equal(result["cell_types"], values)
    return ids, xy, result, path


def summarize_niches(ids, result, fit_path, *, upstream_revision):
    """Cache source-cell composition, similarity and directed co-location statistics."""
    import Harmonics
    from scipy.spatial.distance import cdist
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    cut_percentage = 99
    thresholds = dict(q=0.05, log2fc=1.0, proportion=0.01)
    labels = result["labels"].astype(np.int64)
    xy = result["coordinates"]
    type_names, type_index = np.unique(result["cell_types"], return_inverse=True)
    n_types = len(type_names)
    provenance = {
        "schema": 1, "fit": Path(fit_path).name, "upstream_revision": upstream_revision,
        "labels_sha256": _digest(labels), "coordinates_sha256": _digest(xy),
        "cell_types_sha256": _digest(result["cell_types"]),
        "composition": "fractions of cells assigned to each niche by major_celltype",
        "similarity": "one minus total variation",
        "colocalization": "Harmonics Delaunay, cross-niche only, Fisher two-sided and BY",
        "cut_percentage": cut_percentage, "thresholds": thresholds,
        "versions": {name: version(name) for name in ("numpy", "scipy", "statsmodels")},
    }

    def compute():
        counts = np.bincount(n_types * labels + type_index, minlength=K * n_types).reshape(K, n_types)
        np.testing.assert_array_equal(counts.sum(axis=1), result["counts"])
        fractions = counts / counts.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(fractions, result["niche_composition"])
        similarity = 1 - cdist(fractions, fractions, metric="cityblock") / 2

        print("CPU: computing Harmonics Delaunay co-location.", flush=True)
        adata = ad.AnnData(obs=pd.DataFrame(index=ids), obsm={"spatial": xy})
        Harmonics.Delaunay_adjacency_mtx(adata, cut_percentage=cut_percentage, verbose=False)
        source_cells, target_cells = adata.obsp["delaunay_adj_mtx"].nonzero()
        cross_niche = labels[source_cells] != labels[target_cells]
        encoded = K * labels[source_cells[cross_niche]] + labels[target_cells[cross_niche]]
        edges = np.bincount(encoded, minlength=K * K).reshape(K, K)
        source_totals = edges.sum(axis=1)
        proportions = np.divide(edges, source_totals[:, None],
                                out=np.full((K, K), np.nan), where=source_totals[:, None] > 0)
        rows, columns = np.where(~np.eye(K, dtype=bool))
        observed = edges[rows, columns]
        source_total = source_totals[rows]
        background = edges.sum(axis=0)[columns] - observed
        background_total = edges.sum() - source_total
        p = np.array([
            fisher_exact([[a, r - a], [c, n - c]], alternative="two-sided").pvalue
            for a, r, c, n in zip(observed, source_total, background, background_total, strict=True)
        ])
        q = multipletests(p, method="fdr_by")[1]
        eps = 1e-10
        log2fc = np.log2((observed / (source_total + eps) + eps)
                        / (background / (background_total + eps) + eps))
        enriched = ((q <= thresholds["q"]) & (log2fc >= thresholds["log2fc"])
                    & (proportions[rows, columns] >= thresholds["proportion"]))
        return dict(types=type_names, composition_counts=counts, composition_fractions=fractions,
                    similarity=similarity, edges=edges, edge_proportions=proportions,
                    source_niche=rows, target_niche=columns, p=p, q=q, log2fc=log2fc,
                    enriched=enriched)

    return _cached(Path(fit_path).parent, "niche-summary", provenance, ids, compute)


def composition_umap(ids, fit_path, *, cuda_device="cuda:2"):
    """Embed the exact all-cell Harmonics neighborhood composition matrix."""
    from sklearn.decomposition import PCA

    fit_path = Path(fit_path)
    with np.load(fit_path, allow_pickle=False) as fit:
        metadata = json.loads(fit["metadata"].item())
        np.testing.assert_array_equal(fit["ids"], ids)
    prep_path = fit_path.parent / metadata["neighborhoods"]
    with np.load(prep_path, allow_pickle=False) as composition:
        np.testing.assert_array_equal(composition["ids"], ids)
        features = composition["composition"]
    np.testing.assert_equal(_digest(features), metadata["composition_sha256"])
    features = features.astype(np.float32)
    settings = dict(n_neighbors=20, min_dist=0.2, metric="euclidean", random_state=73, n_jobs=1)
    neighbor_provenance = {
        "schema": 1, "neighborhoods": prep_path.name, "features_sha256": _digest(features),
        "n_neighbors": settings["n_neighbors"],
        "knn": "PyTorch CUDA float64 exhaustive Euclidean, self first",
        "torch": version("torch"),
    }
    provenance = {
        **neighbor_provenance, "umap": settings, "init": "PCA",
        "versions": {name: version(name) for name in
                     ("numpy", "scipy", "scikit-learn", "torch", "umap-learn", "numba", "llvmlite")},
    }

    def compute_neighbors():
        import torch

        k = settings["n_neighbors"]
        print(f"{cuda_device}: finding exact composition neighbors.", flush=True)
        device = torch.device(cuda_device)
        data = torch.as_tensor(features, device=device, dtype=torch.float64)
        indices = np.empty((len(ids), k), dtype=np.int32)
        distances = np.empty((len(ids), k), dtype=np.float32)
        indices[:, 0] = np.arange(len(ids))
        distances[:, 0] = 0
        with torch.inference_mode():
            for start in range(0, len(ids), 512):
                stop = min(start + 512, len(ids))
                distances_block = torch.cdist(data[start:stop], data,
                                              compute_mode="use_mm_for_euclid_dist")
                distances_block[torch.arange(stop - start, device=device),
                                torch.arange(start, stop, device=device)] = torch.inf
                neighbors = torch.topk(distances_block, k - 1, largest=False).indices
                # Evaluate selected pairs directly so identical compositions have zero distance.
                values = torch.linalg.vector_norm(data[start:stop, None, :] - data[neighbors], dim=2)
                values, order = torch.sort(values, dim=1, stable=True)
                neighbors = torch.gather(neighbors, 1, order)
                indices[start:stop, 1:] = neighbors.cpu().numpy()
                distances[start:stop, 1:] = values.cpu().numpy()
        return dict(indices=indices, distances=distances)

    def compute():
        from umap import UMAP

        neighbors, _ = _cached(fit_path.parent, "composition-knn", neighbor_provenance, ids,
                               compute_neighbors)
        print("CPU: optimizing the seeded UMAP layout.", flush=True)
        with threadpool_limits(limits=16):
            initial = PCA(n_components=2, random_state=73).fit_transform(features)
            knn = neighbors["indices"], neighbors["distances"], None
            coordinates = UMAP(init=initial, precomputed_knn=knn,
                               **settings).fit_transform(features)
        if coordinates.shape != (len(ids), 2) or not np.isfinite(coordinates).all():
            raise ValueError("UMAP did not produce two finite coordinates for every cell.")
        return dict(coordinates=coordinates)

    return _cached(fit_path.parent, "composition-umap", provenance, ids, compute)
