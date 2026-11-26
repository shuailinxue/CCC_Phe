"""Content-addressed, resumable network shards. No training labels enter this module."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from . import networks


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def key(meta):
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:24]


def versions(names):
    return {"python": platform.python_version(), **{name: version(name) for name in names}}


@contextmanager
def lock(path):
    with open(path / ".lock", "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def save_npz(path, **arrays):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        try:
            np.savez_compressed(stream, **arrays)
            stream.close()
            os.replace(stream.name, path)
        finally:
            Path(stream.name).unlink(missing_ok=True)


def prepare(source, atlas, root, *, coords, types, specimen, layer="X", window=50.,
            fraction=0., null_seed=None, workers=4, reference=None):
    from phenoniche.spatial import channels
    if not np.isfinite(window) or window <= 0 or not 0 <= fraction < 1 or workers < 1:
        raise ValueError("Invalid network configuration")
    meta = dict(source=digest(source), atlas=digest(atlas), coords=coords, types=types,
                specimen=specimen, layer=layer, window=window, holdout=fraction,
                radii={"Contact":20., "ECM":20., "Secreted":100., "Ambiguous":50., "Unknown":50.},
                null_seed=null_seed, reference=digest(reference) if reference else None,
                source_code={Path(p).name:digest(p) for p in
                             [__file__, networks.__file__, channels.__file__]},
                versions=versions(["anndata", "scipy", "numpy", "pandas", "numba"]))
    path = Path(root) / ("network-" + key(meta))
    path.mkdir(parents=True, exist_ok=True)
    with lock(path):
        if (path / "manifest.json").exists():
            if json.loads((path / "manifest.json").read_text()) != meta:
                raise ValueError("Network cache provenance mismatch")
            return path
        a = ad.read_h5ad(source)
        ids = a.obs_names.to_numpy(dtype=str)
        labels = a.obs[types]
        samples = a.obs[specimen] if specimen else pd.Series(["single"] * len(ids))
        xyz = np.asarray(a.obsm[coords], dtype=float)
        if (not len(ids) or not a.obs_names.is_unique or not a.var_names.is_unique or
            labels.isna().any() or samples.isna().any() or not np.isfinite(xyz).all()
            or xyz.shape not in [(len(ids), 2), (len(ids), 3)]):
            raise ValueError("Expected unique IDs and aligned observed types, specimens and coordinates")
        samples = samples.to_numpy(dtype=str)
        labels = labels.to_numpy(dtype=str)
        names = np.unique(labels)
        scales = None
        genes = a.var_names.to_numpy(dtype=str)
        if reference:
            with np.load(reference) as ref:
                names, scales = ref["names"], ref["scales"]
                if not np.array_equal(ref["genes"], genes) or not np.isin(labels, names).all():
                    raise ValueError("Inference requires matching panel and known cell types")
        codes = np.searchsorted(names, labels)
        train, test = networks.split_region(xyz, samples, window, fraction)
        expr, left, right, lr, signaling, scales = networks.expression(
            a.X if layer == "X" else a.layers[layer], genes, pd.read_csv(atlas, sep="\t"),
            train, scales)
        del a
        if null_seed is not None:
            rng = np.random.default_rng(null_seed)
            # Keep profiles on their side of the physical train/test boundary.
            region = np.zeros(len(ids), int)
            for sample in np.unique(samples):
                ix = np.flatnonzero(samples == sample)
                if test[ix].any():
                    region[ix] = xyz[ix, 0] >= xyz[ix[test[ix]], 0].min() - window
            for idx in pd.DataFrame(dict(s=samples,t=codes,b=region)).groupby(["s","t","b"]).indices.values():
                expr[idx] = expr[rng.permutation(idx)]
        radii, group = np.unique([meta["radii"][s] for s in signaling], return_inverse=True)
        rows = networks.window_index(xyz, samples, window)
        e = sparse.csr_matrix(expr)
        args = xyz, codes, len(names), expr, left, right, radii**2, group, e.indptr, e.indices
        shard_size = 256
        def write(start):
            part = path / f"part-{start:08d}.npz"
            if part.exists():
                return
            obs = [networks.observe(ids_in_window, *args) for ids_in_window in rows[start:start+shard_size]]
            mass, support, edges, stats = zip(*obs)
            save_npz(part, ids=ids[start:start+len(obs)], mass=mass, support=support,
                     ptr=np.r_[0,np.cumsum([len(x) for x in edges])],
                     edge=np.concatenate(edges), stats=np.concatenate(stats))
        networks.observe(rows[0], *args)
        with ThreadPoolExecutor(workers) as pool:
            for i, _ in enumerate(pool.map(write, range(0, len(ids), shard_size))):
                if i % 100 == 0:
                    print(f"{path.name} network {i*shard_size}/{len(ids)}", flush=True)
        save_npz(path / "schema.npz", ids=ids, names=names, samples=samples, xyz=xyz,
                 genes=genes, scales=scales, lr=np.array(lr), signaling=signaling, group=group,
                 radii=radii, train=train, test=test, window=window)
        (path / "manifest.tmp").write_text(json.dumps(meta, indent=2))
        os.replace(path / "manifest.tmp", path / "manifest.json")
    return path


class Network:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "manifest.json").read_text())
        with np.load(self.path / "schema.npz") as z:
            for name in z.files:
                setattr(self, name, z[name])
        mass, support, lengths, edge, stats = [], [], [], [], []
        offset = 0
        for part in sorted(self.path.glob("part-*.npz")):
            with np.load(part) as z:
                n = len(z["mass"])
                if not np.array_equal(z["ids"], self.ids[offset:offset+n]):
                    raise ValueError(f"Cell order mismatch in {part}")
                offset += n
                mass.append(z["mass"])
                support.append(z["support"])
                lengths.append(np.diff(z["ptr"]))
                edge.append(z["edge"])
                stats.append(z["stats"])
        if offset != len(self.ids):
            raise ValueError("Incomplete network")
        self.mass, self.support = np.concatenate(mass), np.concatenate(support)
        self.ptr = np.r_[0, np.cumsum(np.concatenate(lengths))]
        self.edge, self.stats = np.concatenate(edge), np.concatenate(stats)
        self.nt, self.nl = len(self.names), len(self.lr)

    def entries(self, ids):
        lengths = self.ptr[ids+1] - self.ptr[ids]
        row = np.repeat(np.arange(len(ids)), lengths)
        offsets = self.ptr[ids] - (np.cumsum(lengths)-lengths)
        ix = np.arange(len(row)) + offsets[row]
        return row, self.edge[ix], self.stats[ix]
