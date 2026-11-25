"""Joint control–case communication programs and donor-calibrated comparison."""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import pickle
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from .spatial.cell_programs import CellTensor, consensus, pool_programs, program_graph
from .spatial.channels import Channels
from .spatial.geometry import composition_partition, physical_graph
from .spatial.programs import fit_programs
from .spatial.settings import Settings


@dataclass(frozen=True)
class ComparativeSettings:
    k: int = 10
    matching_metric: str = "euclidean"
    matching_sigma: float = 0.2
    composition_quantile: float = 0.95
    abnormality_quantile: float = 0.95
    balance: str = "none"

    def __post_init__(self):
        if type(self.k) is not int or self.k < 1:
            raise ValueError("k must be a positive integer")
        if self.matching_metric not in {"euclidean", "cosine"}:
            raise ValueError("matching_metric must be euclidean or cosine")
        if not np.isfinite(self.matching_sigma) or self.matching_sigma <= 0:
            raise ValueError("matching_sigma must be positive")
        if any(not 0 < x <= 1 for x in (self.composition_quantile, self.abnormality_quantile)):
            raise ValueError("calibration quantiles must be in (0, 1]")
        if self.balance not in {"none", "sample", "donor"}:
            raise ValueError("balance must be none, sample or donor")


@dataclass
class ComparativeResult:
    cells: pd.DataFrame
    composition: pd.DataFrame
    contexts: dict
    matches: pd.DataFrame
    donor_proportions: pd.DataFrame
    settings: ComparativeSettings
    spatial_settings: Settings
    cache_path: Path


def _cached(path, make):
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    value = make()
    tmp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with tmp.open("wb") as f:
            pickle.dump(value, f, protocol=5)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return value


def _fingerprint(counts, genes, ids, xyz, labels, metadata, lr, cfg, comp):
    digest = hashlib.sha256()
    matrix = sparse.csr_matrix(counts)
    for a in (matrix.data, matrix.indices, matrix.indptr, np.asarray(xyz)):
        a = np.ascontiguousarray(a)
        digest.update(repr((a.shape, a.dtype)).encode())
        digest.update(a.tobytes())
    for value in (matrix.shape, list(genes), list(ids), list(labels), metadata.to_dict("list"),
                  lr.to_dict("list"), asdict(cfg), asdict(comp), sys.version,
                  {name: version(name) for name in ("numpy", "scipy", "torch", "pandas", "scikit-learn", "python-igraph", "leidenalg")}):
        digest.update(repr(value).encode())
    for file in [Path(__file__), *sorted((Path(__file__).parent / "spatial").glob("*.py"))]:
        digest.update(file.read_bytes())
    return digest.hexdigest()


class _StoredTensor:
    """Persist the sparse tensor and its type-specific observation mask."""

    def __init__(self, ids, tensor, cell_ids):
        self.ids = np.asarray(ids)
        self.cell_ids = np.asarray(cell_ids, dtype=str)
        self.lookup = {int(cell): row for row, cell in enumerate(ids)}
        self.groups = tensor.groups
        self.n_types = tensor.n_types
        chunks, columns = [], []
        sizes = np.zeros(len(ids) * tensor.n_types, dtype=np.int64)
        self.mask = np.zeros((len(ids), tensor.n_types, len(self.groups)), dtype=bool)
        for row, cell in enumerate(ids):
            categories, data, support = tensor.fiber(cell)
            self.mask[row, categories] = support > 0
            block = sparse.csr_matrix(data)
            sizes[row * tensor.n_types + categories] = np.diff(block.indptr)
            chunks.append(block.data)
            columns.append(block.indices)
        indptr = np.r_[0, np.cumsum(sizes)]
        self.values = sparse.csr_matrix(
            (np.concatenate(chunks), np.concatenate(columns), indptr),
            shape=(len(ids) * tensor.n_types, 2 * len(tensor.channels.names)),
        )

    def fiber(self, cell):
        row = self.lookup[int(cell)]
        support = self.mask[row]
        categories = np.flatnonzero(support.any(axis=1))
        values = self.values[row * self.n_types + categories].toarray()
        return categories, values, support[categories]


def _weighted_quantile(values, donors, q):
    values = np.asarray(values, dtype=float)
    donors = np.asarray(donors)
    if not len(values):
        return np.nan
    _, inverse, counts = np.unique(donors, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, q * cumulative[-1], side="left")]])


def _coordinates(x, metric):
    if metric == "euclidean":
        return x
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0):
        raise ValueError("Cosine matching requires nonzero composition vectors")
    return x / norms[:, None]


def _match(query, reference, composition, activity, cfg):
    """Return fixed-K matches and original-space distances without silent K changes."""
    if len(reference) < cfg.k:
        return None
    points = _coordinates(composition, cfg.matching_metric)
    distances, indices = cKDTree(points[reference]).query(points[query], k=cfg.k)
    distances = np.asarray(distances).reshape(len(query), cfg.k)
    if cfg.matching_metric == "cosine":
        distances = np.maximum(0, distances ** 2 / 2)
    indices = reference[np.asarray(indices).reshape(len(query), cfg.k)]
    logits = -0.5 * (distances / cfg.matching_sigma) ** 2
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    expected = np.einsum("nk,nkf->nf", weights, activity[indices])
    return indices, distances, weights, expected


def _signed_features(values):
    return np.concatenate((np.maximum(values, 0), np.maximum(-values, 0)), axis=1)


def _reference_analysis(ids, composition, activity, conditions, donors, cfg, spatial, seed):
    n, features = activity.shape
    control = np.flatnonzero(conditions == "control")
    case = np.flatnonzero(conditions == "case")
    expected = np.full((n, features), np.nan)
    delta = np.full((n, features), np.nan)
    distance = np.full(n, np.nan)
    matches = []
    for donor in np.unique(donors[control]):
        query = control[donors[control] == donor]
        reference = control[donors[control] != donor]
        result = _match(query, reference, composition, activity, cfg)
        if result is not None:
            indices, distances, weights, predicted = result
            expected[query] = predicted
            delta[query] = activity[query] - predicted
            distance[query] = distances[:, -1]
            matches.extend((int(ids[q]), int(ids[r]), float(d), float(w))
                           for q, rs, ds, ws in zip(query, indices, distances, weights, strict=True)
                           for r, d, w in zip(rs, ds, ws, strict=True))
    lodo = control[np.isfinite(distance[control])]
    calibrated = len(np.unique(donors[lodo])) >= 2
    distance_limit = _weighted_quantile(distance[lodo], donors[lodo], cfg.composition_quantile) if calibrated else np.nan
    eligible = lodo[distance[lodo] <= distance_limit] if calibrated else np.array([], dtype=int)
    anomaly_calibrated = len(np.unique(donors[eligible])) >= 2
    scores = np.linalg.norm(delta, axis=1)
    abnormality_limit = _weighted_quantile(scores[eligible], donors[eligible], cfg.abnormality_quantile) if anomaly_calibrated else np.nan

    reference_status = np.full(n, "no_reference", dtype=object)
    reference_status[lodo] = "uncalibrated" if not calibrated else "comparable"
    if calibrated:
        reference_status[lodo[distance[lodo] > distance_limit]] = "no_reference"
        bad = lodo[distance[lodo] > distance_limit]
        expected[bad] = np.nan
        delta[bad] = np.nan
    result = _match(case, control, composition, activity, cfg) if len(case) else None
    if result is not None:
        indices, distances, weights, predicted = result
        distance[case] = distances[:, -1]
        comparable = np.ones(len(case), dtype=bool) if not calibrated else distance[case] <= distance_limit
        chosen = case[comparable]
        expected[chosen] = predicted[comparable]
        delta[chosen] = activity[chosen] - predicted[comparable]
        reference_status[chosen] = "comparable" if calibrated else "uncalibrated"
        matches.extend((int(ids[q]), int(ids[r]), float(d), float(w))
                       for q, rs, ds, ws in zip(case, indices, distances, weights, strict=True)
                       for r, d, w in zip(rs, ds, ws, strict=True))
    scores = np.linalg.norm(delta, axis=1)
    abnormality = np.full(n, "uncalibrated", dtype=object)
    abnormality[reference_status == "no_reference"] = "no_reference"
    if anomaly_calibrated:
        valid = reference_status == "comparable"
        abnormality[valid] = np.where(scores[valid] > abnormality_limit, "outside_reference", "within_reference")
    labels = np.full(n, "", dtype=object)
    valid = np.isfinite(delta).all(axis=1)
    if valid.any():
        result, _ = consensus(_signed_features(delta[valid]), spatial, seed)
        labels[valid] = [f"R{x}" for x in result]
    if (~valid).any():
        result, _ = consensus(activity[~valid], spatial, seed + 1)
        labels[~valid] = [f"U{x}" for x in result]
    return dict(expected=expected, delta=delta, distance=distance, score=scores,
                reference_status=reference_status, abnormality=abnormality, labels=labels,
                calibrated=calibrated, abnormality_calibrated=anomaly_calibrated, composition_threshold=distance_limit,
                abnormality_threshold=abnormality_limit, matches=matches)


def fit_control_case(counts, genes, cell_ids, coordinates_um, cell_types, metadata, lr,
                     *, cache_dir, settings=None, comparative=None):
    """Fit shared control–case programs and compare case niches to control donors.

    Metadata is aligned by row to all other inputs and has sample_id, donor_id,
    condition ('control' or 'case') and batch columns. cache_dir is mandatory.
    """
    cfg, comp = settings or Settings(), comparative or ComparativeSettings()
    ids = pd.Index(cell_ids, name="cell_id")
    xyz = np.asarray(coordinates_um, dtype=float)
    if np.shape(cell_types) != (len(ids),) or pd.isna(cell_types).any():
        raise ValueError("One observed cell type is required per cell")
    labels = np.asarray(cell_types, dtype=str)
    if np.any(np.char.str_len(np.char.strip(labels)) == 0):
        raise ValueError("Cell types must not be empty")
    required = ["sample_id", "donor_id", "condition", "batch"]
    if not isinstance(metadata, pd.DataFrame) or any(x not in metadata for x in required):
        raise ValueError("metadata must contain sample_id, donor_id, condition and batch")
    meta = metadata[required].reset_index(drop=True).copy()
    if len(ids) == 0 or not ids.is_unique or ids.hasnans or len(meta) != len(ids):
        raise ValueError("Unique aligned cell IDs and metadata are required")
    if meta.isna().any().any() or any(meta[c].astype(str).str.strip().eq("").any() for c in required):
        raise ValueError("Metadata values must be present")
    meta = meta.astype(str)
    if set(meta.condition.unique()) != {"control", "case"}:
        raise ValueError("condition must contain both control and case")
    if (meta.groupby("sample_id")[["donor_id", "condition", "batch"]].nunique() > 1).any().any():
        raise ValueError("Each sample must have one donor, condition and batch")
    if xyz.shape not in {(len(ids), 2), (len(ids), 3)} or not np.isfinite(xyz).all():
        raise ValueError("Finite aligned 2D or 3D coordinates are required")
    if labels.shape != (len(ids),) or counts.shape != (len(ids), len(genes)):
        raise ValueError("Counts, cell types and gene dimensions must align")
    fingerprint = _fingerprint(counts, genes, ids, xyz, labels, meta, lr, cfg, comp)
    directory = Path(cache_dir) / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"fingerprint": fingerprint, "cell_ids": ids.tolist(), "genes": list(genes),
                "metadata": meta.to_dict("list"), "spatial_settings": asdict(cfg),
                "comparative_settings": asdict(comp), "python": sys.version,
                "versions": {name: version(name) for name in ("numpy", "scipy", "torch", "pandas", "scikit-learn", "python-igraph", "leidenalg")},
                "source_sha256": {str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [Path(__file__), *sorted((Path(__file__).parent / "spatial").glob("*.py"))]}}
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    if (directory / "result.pkl").exists():
        return _cached(directory / "result.pkl", lambda: None)
    samples = meta.sample_id.to_numpy()
    names, types = np.unique(labels, return_inverse=True)
    windows = physical_graph(xyz, samples, cfg.window_um)
    composition, coarse = _cached(directory / "composition.pkl", lambda: composition_partition(windows, types, cfg))
    channels = Channels(counts, pd.Index(genes), lr)
    radius = max(cfg.contact_um, cfg.contact_radius_um, cfg.secreted_radius_um, cfg.ecm_radius_um)
    contacts = windows if radius == cfg.window_um else physical_graph(xyz, samples, radius)
    conditions, donors = meta.condition.to_numpy(), meta.donor_id.to_numpy()
    cells = meta.copy()
    cells.index = ids
    cells["coarse"] = coarse
    cells["niche"] = ""
    cells["reference_status"] = "no_reference"
    cells["abnormality"] = "no_reference"
    cells["match_distance"] = np.nan
    cells["abnormality_score"] = np.nan
    contexts, matches = {}, []
    for coarse_id in np.unique(coarse):
        center_ids = np.flatnonzero(coarse == coarse_id)
        allowed = coarse == coarse_id
        tensor = _cached(directory / f"tensor_{coarse_id}.pkl",
                         lambda: _StoredTensor(center_ids, CellTensor(contacts, types, channels, cfg.sigma_um,
                                                    cfg.aggregation != "sum-sum", cfg), ids[center_ids]))
        pool, pool_weights = program_graph(center_ids, allowed, windows, types, cfg.sigma_um)
        balance = None if comp.balance == "none" else meta.iloc[center_ids][
            "sample_id" if comp.balance == "sample" else "donor_id"].to_numpy()

        def fit_models():
            rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, int(coarse_id)]))
            return [fit_programs(center_ids, tensor.fiber, tensor.n_types, 2 * len(channels.names),
                                 cfg, rng, balance_groups=balance) for _ in range(cfg.repeats)]

        fits = _cached(directory / f"fits_{coarse_id}.pkl", fit_models)
        if any(fit is None for fit in fits):
            cells.iloc[center_ids, cells.columns.get_loc("reference_status")] = "no_communication_signal"
            cells.iloc[center_ids, cells.columns.get_loc("abnormality")] = "no_communication_signal"
            cells.iloc[center_ids, cells.columns.get_loc("niche")] = f"C{coarse_id}.no_signal"
            contexts[int(coarse_id)] = {"cell_ids": ids[center_ids].tolist(), "fits": fits,
                                        "cell_types": names, "lr_pairs": channels.names,
                                        "signaling_types": channels.signaling_types}
            continue
        activity = fits[0].factors[0]
        scores = np.zeros((len(ids), activity.shape[1]), dtype=activity.dtype)
        scores[center_ids] = activity
        pooled = pool_programs(pool, pool_weights, types, scores, cfg.aggregation == "mean-mean")
        result = _reference_analysis(center_ids, composition[center_ids], pooled,
                                     conditions[center_ids], donors[center_ids], comp, cfg,
                                     cfg.seed + int(coarse_id))
        for key, field in (("reference_status", "reference_status"), ("abnormality", "abnormality"),
                           ("match_distance", "distance"), ("abnormality_score", "score")):
            cells.iloc[center_ids, cells.columns.get_loc(key)] = result[field]
        cells.iloc[center_ids, cells.columns.get_loc("niche")] = [f"C{coarse_id}.{x}" for x in result["labels"]]
        matches.extend(result.pop("matches"))
        contexts[int(coarse_id)] = dict(cell_ids=ids[center_ids].tolist(), features=pooled,
                                        pool_weights=pool_weights, fits=fits, cell_types=names,
                                        lr_pairs=channels.names, signaling_types=channels.signaling_types,
                                        **result)
    match_frame = pd.DataFrame(matches, columns=["center_row", "reference_row", "distance", "weight"])
    match_frame["center_id"] = ids.to_numpy()[match_frame.center_row.to_numpy(dtype=int)]
    match_frame["reference_id"] = ids.to_numpy()[match_frame.reference_row.to_numpy(dtype=int)]
    grouping = ["donor_id", "condition", "coarse", "niche"]
    proportions = cells.groupby(grouping, observed=True).size().rename("cells").reset_index()
    combinations = []
    for coarse_id, part in cells.groupby("coarse", sort=True):
        donor_conditions = part[["donor_id", "condition"]].drop_duplicates()
        niches = pd.DataFrame({"niche": part.niche.unique(), "coarse": coarse_id})
        combinations.append(donor_conditions.merge(niches, how="cross"))
    proportions = pd.concat(combinations, ignore_index=True).merge(proportions, on=grouping, how="left")
    proportions["cells"] = proportions.cells.fillna(0).astype(int)
    proportions["fraction"] = proportions.cells / proportions.groupby(["donor_id", "condition", "coarse"]).cells.transform("sum")
    result = ComparativeResult(cells, pd.DataFrame(composition, index=ids, columns=names),
                               contexts, match_frame, proportions, comp, cfg, directory)
    _cached(directory / "result.pkl", lambda: result)
    return result
