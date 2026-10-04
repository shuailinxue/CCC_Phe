from dataclasses import dataclass
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class PreparedInput:
    matrix_path: Path
    matrix_shape: tuple
    retained_indices: np.ndarray
    scale: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    features: pd.DataFrame
    cells: pd.DataFrame
    coordinates: np.ndarray
    cache_dir: Path


def resolve_device(requested="auto"):
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {requested}")
    return str(requested)


def _v1003_cache(config):
    cache = Path(config.v1003_output) / "cache"
    required = [cache / name for name in
                ("ccc.npy", "features.csv", "cells.csv.gz", "coordinates.npy", "input_manifest.json")]
    if all(path.is_file() for path in required):
        return cache
    source = Path(config.v1003_source)
    if str(source) not in sys.path:
        sys.path.append(str(source))
    import phenoniche
    package = source / "phenoniche"
    if str(package) not in phenoniche.__path__:
        phenoniche.__path__.append(str(package))
    from phenoniche.v1003.config import default_config
    from phenoniche.v1003.data import prepare_prime5k_ccc
    prepare_prime5k_ccc(default_config())
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("V1003 did not produce the required validated Prime 5K CCC cache")
    return cache


def deterministic_split(n, validation_fraction, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(1, int(round(n * validation_fraction)))
    return np.sort(order[n_val:]), np.sort(order[:n_val])


def feature_statistics(matrix, rows, chunk_size=8192):
    count = len(rows)
    sums = np.zeros(matrix.shape[1], dtype=np.float64)
    sums2 = np.zeros_like(sums)
    nonzero = np.zeros(matrix.shape[1], dtype=np.int64)
    for start in range(0, count, chunk_size):
        block = np.asarray(matrix[rows[start:start + chunk_size]], dtype=np.float64)
        sums += block.sum(0)
        sums2 += np.square(block).sum(0)
        nonzero += np.count_nonzero(block, axis=0)
    mean = sums / count
    mean_square = sums2 / count
    variance = np.maximum(mean_square - np.square(mean), 0.)
    return mean, mean_square, variance, nonzero / count


def prepare_input(config, force=False):
    source = _v1003_cache(config)
    root = Path(config.output_dir)
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "input_manifest.json"
    source_matrix = source / "ccc.npy"
    matrix = np.load(source_matrix, mmap_mode="r")
    signature = {
        "revision": config.revision, "source": str(source_matrix.resolve()),
        "source_bytes": source_matrix.stat().st_size, "shape": list(matrix.shape),
        "seed": config.seed, "validation_fraction": config.validation_fraction,
        "nonzero_fraction_min": config.nonzero_fraction_min,
        "normalization": "train-feature RMS only; no row normalization",
    }
    files = {name: cache / name for name in
             ("retained_indices.npy", "feature_scale.npy", "train_indices.npy",
              "validation_indices.npy", "feature_filter_audit.csv", "retained_features.csv")}
    if not force and manifest_path.is_file() and all(p.is_file() for p in files.values()):
        if json.loads(manifest_path.read_text()).get("signature") == signature:
            return load_prepared_input(config)

    train, validation = deterministic_split(matrix.shape[0], config.validation_fraction, config.seed)
    mean, mean_square, variance, nonzero_fraction = feature_statistics(matrix, train)
    # A floating-point resolution test removes only numerically constant features.
    variance_floor = np.maximum(np.finfo(np.float32).eps * mean_square, np.finfo(np.float32).tiny)
    retained = (nonzero_fraction >= config.nonzero_fraction_min) & (variance > variance_floor)
    if retained.sum() < config.final_niches:
        raise RuntimeError(f"Only {retained.sum()} CCC features survived the conservative filter")
    retained_indices = np.flatnonzero(retained).astype(np.int64)
    scale = np.sqrt(mean_square[retained]).astype(np.float32)
    scale = np.maximum(scale, np.float32(config.epsilon))
    original_features = pd.read_csv(source / "features.csv")
    if len(original_features) != matrix.shape[1]:
        raise ValueError("CCC feature metadata does not match the cached matrix")
    audit = original_features.copy()
    audit["train_nonzero_fraction"] = nonzero_fraction
    audit["train_mean"] = mean
    audit["train_variance"] = variance
    audit["variance_floor"] = variance_floor
    audit["retained"] = retained
    audit["filter_reason"] = np.where(nonzero_fraction < config.nonzero_fraction_min,
                                      "nonzero_fraction", np.where(variance <= variance_floor,
                                      "near_constant", "retained"))
    audit["feature_scale"] = np.nan
    audit.loc[retained, "feature_scale"] = scale
    retained_features = original_features.iloc[retained_indices].reset_index(drop=True)
    retained_features.insert(0, "v1006_feature_id", np.arange(len(retained_features)))
    np.save(files["retained_indices.npy"], retained_indices)
    np.save(files["feature_scale.npy"], scale)
    np.save(files["train_indices.npy"], train)
    np.save(files["validation_indices.npy"], validation)
    audit.to_csv(files["feature_filter_audit.csv"], index=False)
    retained_features.to_csv(files["retained_features.csv"], index=False)
    # Small metadata are duplicated at the result root for direct audit access.
    audit.to_csv(root / "feature_filter_audit.csv", index=False)
    retained_features.to_csv(root / "retained_features.csv", index=False)
    np.save(root / "feature_scale.npy", scale)
    np.save(root / "train_indices.npy", train)
    np.save(root / "validation_indices.npy", validation)
    (root / "filtered_matrix_metadata.json").write_text(json.dumps({
        "source_matrix": str(source_matrix), "source_shape": list(matrix.shape),
        "retained_shape": [int(matrix.shape[0]), int(retained.sum())],
        "retained_indices": str(files["retained_indices.npy"]),
        "scaling": "training-set feature RMS; reversible; no centering; no row normalization"
    }, indent=2))
    manifest_path.write_text(json.dumps({"signature": signature, "n_retained": int(retained.sum())}, indent=2))
    return load_prepared_input(config)


def load_prepared_input(config):
    source = _v1003_cache(config)
    cache = Path(config.output_dir) / "cache"
    matrix = np.load(source / "ccc.npy", mmap_mode="r")
    return PreparedInput(
        source / "ccc.npy", tuple(matrix.shape), np.load(cache / "retained_indices.npy"),
        np.load(cache / "feature_scale.npy"), np.load(cache / "train_indices.npy"),
        np.load(cache / "validation_indices.npy"), pd.read_csv(cache / "retained_features.csv"),
        pd.read_csv(source / "cells.csv.gz"), np.load(source / "coordinates.npy", mmap_mode="r"), cache)


class CCCDataset(Dataset):
    def __init__(self, prepared, rows):
        self.matrix = np.load(prepared.matrix_path, mmap_mode="r")
        self.rows = np.asarray(rows, dtype=np.int64)
        self.columns = prepared.retained_indices
        self.scale = prepared.scale

    def __len__(self): return len(self.rows)

    def __getitem__(self, item):
        row = self.rows[item]
        x = np.asarray(self.matrix[row, self.columns], dtype=np.float32) / self.scale
        return torch.from_numpy(x.copy()), int(row)
