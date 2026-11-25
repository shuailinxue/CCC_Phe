"""Full-volume V2 experiment, resumable at composition and per-context checkpoints."""

import json
import multiprocessing as mp
from pathlib import Path
import pickle
import time
from dataclasses import replace

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from phenoniche.spatial import Settings
from phenoniche.spatial.channels import Channels
from phenoniche.spatial.geometry import (
    composition_partition,
    contexts,
    physical_graph,
)
from phenoniche.spatial.input import lung_coordinates
from phenoniche.spatial.refine import refine_context
from phenoniche.spatial.subclasses import cluster_context, partition_contexts

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(
    "/home/nas3/biod/xueshuailin/CCC_Phe/data/3D_lung_CosMx/extracted"
)
OUT = Path("/data1/xueshuailin/CCC_Phe/workspace/outputs/3D_lung_CosMx/v2_lung_study")
CFG = Settings()
POOLED = replace(CFG, effect_mad=1.5, agreement=0.6, loading_cosine=0.65)
GRAPH = OUT / "geometry_v1.npz"


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def read_input():
    a = ad.read_h5ad(DATA / "preprocessed_objects/preprocessed_objects/cosmx.h5ad")
    xyz = lung_coordinates(a, DATA / "stimwrap_files/stimwrap_files")
    return a, xyz


def prepare(a, xyz):
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path = OUT / "protocol.json"
    protocol = dict(
        settings=vars(CFG),
        cells=len(a),
        genes=a.n_vars,
        input=str(DATA),
        atlas="data/commuspace_human_lr_atlas.tsv",
    )
    if cfg_path.exists():
        previous = json.loads(cfg_path.read_text())
        legacy_input = str(DATA).replace("/CCC_Phe/data/", "/CCC_Phe/workspace/data/", 1)
        if previous.get("input") == legacy_input:
            previous["input"] = str(DATA)
        fields = (
            "window_um",
            "contact_um",
            "sigma_um",
            "composition_neighbors",
            "resolution",
            "seed",
        )
        if any(
            previous[k] != protocol[k] for k in ("cells", "genes", "input", "atlas")
        ) or any(previous["settings"][k] != protocol["settings"][k] for k in fields):
            raise ValueError("Existing geometry protocol differs")
    else:
        cfg_path.write_text(json.dumps(protocol, indent=2))
    if GRAPH.exists():
        return
    types = np.unique(a.obs.celltypes.astype(str), return_inverse=True)[1]
    log("building 50 µm 3D graph")
    graph_path = OUT / "contacts.npz"
    if graph_path.exists():
        graph = sparse.load_npz(graph_path)
    else:
        graph = physical_graph(xyz, np.zeros(len(xyz)), CFG.window_um)
        sparse.save_npz(OUT / "contacts.partial.npz", graph, compressed=False)
        (OUT / "contacts.partial.npz").replace(graph_path)
    log(f"{graph.nnz:,} directed edges; fitting composition Leiden")
    comp, coarse = composition_partition(graph, types, CFG, workers=24)
    with (OUT / "geometry.partial").open("wb") as f:
        np.savez(
            f,
            coarse=coarse,
            composition=comp,
            xyz=xyz,
            ids=a.obs_names.to_numpy(dtype=str),
        )
    (OUT / "geometry.partial").replace(GRAPH)
    log(f"composition complete, {len(np.unique(coarse))} coarse classes")


def _context(key):
    centers = GROUPS[key]
    cfg = RUN_CFG
    dest = RUN_DIR / f"ctx_{key[0]}_{key[1]}.pkl"
    if dest.exists():
        return key, len(centers), "cached"
    start = time.monotonic()
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, *key]))
    stats = {}
    weights = None
    if RUN_DIR.name == "scaled_channels":
        calibration_rng = np.random.default_rng(
            np.random.SeedSequence([cfg.seed, *key, 91])
        )
        rms = None
        n_observed = 0
        for center in calibration_rng.choice(
            centers, min(len(centers), cfg.fit_windows), replace=False
        ):
            values = TENSOR.fibers(center, key)[2]
            squares = np.square(values, dtype=np.float64).sum(axis=0)
            rms = squares if rms is None else rms + squares
            n_observed += len(values)
        rms = np.sqrt(rms / max(1, n_observed))
        if np.any(rms > 0):
            median = np.median(rms[rms > 0])
            weights = np.sqrt(
                np.clip(median / np.maximum(rms, median / 16), 1 / 16, 16)
            ).astype(np.float32)
            np.save(dest.with_suffix(".scaling.npy"), weights)
    proposals, program = refine_context(
        centers,
        key,
        BLOCK,
        CONTACTS,
        TENSOR,
        cfg,
        rng,
        diagnostics=stats,
        lr_weights=weights,
    )
    dest.with_suffix(".json").write_text(json.dumps(stats))
    # Preserve original region audit objects until conflict resolution, including raw LR contrasts.
    tmp = dest.with_suffix(".partial")
    with tmp.open("wb") as f:
        pickle.dump((proposals, program), f, protocol=5)
    tmp.replace(dest)
    return (
        key,
        len(centers),
        f"{len(proposals)} candidates in {time.monotonic() - start:.1f}s",
    )


def fit_volume(a, xyz, cfg=CFG, name="initial", workers=24, pooled=False):
    global GROUPS, CONTACTS, BLOCK, TENSOR, RUN_CFG, RUN_DIR
    from lung3d_v2_fast import FastWindows, verify_kernel
    from phenoniche.spatial import SpatialNiches

    RUN_CFG = cfg
    RUN_DIR = OUT / name
    RUN_DIR.mkdir(exist_ok=True)
    if any(
        getattr(cfg, k) != getattr(CFG, k)
        for k in (
            "window_um",
            "contact_um",
            "sigma_um",
            "resolution",
            "composition_neighbors",
            "seed",
        )
    ):
        raise ValueError("Changed geometry needs a separate composition checkpoint")
    config = RUN_DIR / "settings.json"
    if config.exists() and json.loads(config.read_text()) != vars(cfg):
        raise ValueError("Changed parameters need a new experiment directory")
    config.write_text(json.dumps(vars(cfg), indent=2))
    if (RUN_DIR / "fit/cells.tsv").exists():
        return RUN_DIR / "fit"
    g = np.load(GRAPH)
    if not np.array_equal(g["ids"], a.obs_names) or not np.allclose(xyz, g["xyz"]):
        raise ValueError("Geometry and input cells differ")
    coarse = g["coarse"]
    names, types = np.unique(a.obs.celltypes.astype(str), return_inverse=True)
    CONTACTS = sparse.load_npz(OUT / "contacts.npz")
    GROUPS = contexts(CONTACTS, coarse)
    BLOCK = np.unique(np.floor(xyz / cfg.block_um), axis=0, return_inverse=True)[1]
    atlas = pd.read_csv(ROOT / "data/commuspace_human_lr_atlas.tsv", sep="\t")
    channels = Channels(a.layers["counts"], a.var_names, atlas)
    if pooled:
        GROUPS = {key: ids for key, ids in GROUPS.items() if key[0] == key[1]}
        tensor_classes = np.zeros_like(coarse)
        tensor_groups = {(0, 0): np.arange(len(a))}
    else:
        tensor_classes, tensor_groups = coarse, GROUPS
    TENSOR = FastWindows(
        CONTACTS, tensor_classes, types, channels, cfg.sigma_um, pooled=pooled
    )
    verify_kernel(TENSOR, CONTACTS, tensor_classes, types, channels, cfg, tensor_groups)
    manifest = pd.DataFrame(
        [(a, b, len(v)) for (a, b), v in GROUPS.items()], columns=["A", "B", "windows"]
    )
    manifest.to_csv(RUN_DIR / "contexts.tsv", sep="\t", index=False)
    keys = sorted(GROUPS, key=lambda k: -len(GROUPS[k]))
    log(
        f"{len(keys)} contexts, {len(channels.names)} complete LR channels; {workers} CPU workers"
    )
    with mp.get_context("fork").Pool(workers) as pool:
        for index, (key, n, status) in enumerate(
            pool.imap_unordered(_context, keys), 1
        ):
            log(f"{index}/{len(keys)} context {key} windows={n}: {status}")
    proposals, programs = [], {}
    for key in sorted(GROUPS):
        with (RUN_DIR / f"ctx_{key[0]}_{key[1]}.pkl").open("rb") as f:
            rows, program = pickle.load(f)
        proposals.extend((key, *row) for row in rows)
        if program is not None:
            programs[key] = program
    result = SpatialNiches.from_contexts(
        pd.Index(a.obs_names, name="cell_id"),
        np.repeat("lung", len(a)),
        names,
        types,
        coarse,
        g["composition"],
        CONTACTS,
        proposals,
        programs,
        channels.names,
        cfg,
    )
    result.save(RUN_DIR / "fit")
    log(
        f"final assignment: {len(result.cells)} cells, {result.evidence.region.nunique()} functional regions"
    )
    return RUN_DIR / "fit"


def _subclass_context(key):
    centers = GROUPS[key]
    path = RUN_DIR / f"subclasses_{key[0]}_{key[1]}.pkl"
    if path.exists():
        return key, "cached"
    started = time.monotonic()
    rng = np.random.default_rng(np.random.SeedSequence([RUN_CFG.seed, *key]))
    result = cluster_context(centers, key, BLOCK, TENSOR, RUN_CFG, rng)
    tmp = path.with_suffix(".partial")
    with tmp.open("wb") as handle:
        pickle.dump(result, handle, protocol=5)
    tmp.replace(path)
    return key, (
        f"{result.status} {len(np.unique(result.labels))} classes "
        f"for {len(centers):,} cells in {time.monotonic() - started:.1f}s"
    )


def merge_composition_classes(coarse, composition, limit=8, minimum=500):
    """Merge provisional Leiden classes by nearest Hellinger centroid."""
    labels = coarse.copy()
    points = np.sqrt(composition)
    while True:
        active, sizes = np.unique(labels, return_counts=True)
        small = np.flatnonzero(sizes < minimum)
        if len(active) == 1 or (len(active) <= limit and not len(small)):
            return np.unique(labels, return_inverse=True)[1]
        means = np.array([points[labels == k].mean(axis=0) for k in active])
        distance = np.linalg.norm(means[:, None] - means[None, :], axis=2)
        np.fill_diagonal(distance, np.inf)
        if len(small):
            i = small[sizes[small].argmin()]
            j = distance[i].argmin()
        else:
            i, j = np.unravel_index(distance.argmin(), distance.shape)
        labels[labels == active[i]] = active[j]


def restrict_to_sections(contact, z):
    contact = contact.copy()
    for start in range(0, len(z), 10000):
        stop = min(start + 10000, len(z))
        edge = slice(contact.indptr[start], contact.indptr[stop])
        source = np.repeat(
            np.arange(start, stop), np.diff(contact.indptr[start : stop + 1])
        )
        target = contact.indices[edge]
        # Keep genuine zero-distance edges within each section.
        contact.data[edge] = np.where(z[source] == z[target], contact.data[edge] + 1, 0)
    contact.eliminate_zeros()
    contact.data -= 1
    return contact


def fit_subclasses(a, xyz, workers=20, revised=False, section_only=False):
    global GROUPS, BLOCK, TENSOR, RUN_CFG, RUN_DIR
    from lung3d_v2_fast import FastWindows, verify_kernel
    from phenoniche.spatial import SpatialNiches

    RUN_CFG = replace(CFG, repeats=3, min_cells=30)
    RUN_DIR = OUT / (
        "revised_2d"
        if section_only
        else "revised_3d"
        if revised
        else "communication_subclasses"
    )
    if revised:
        RUN_CFG = replace(
            RUN_CFG,
            rank=5,
            fit_windows=192,
            iterations=30,
            interface_fraction=0.02,
            subclass_stability=0.7,
        )
    RUN_DIR.mkdir(exist_ok=True)
    config = RUN_DIR / "settings.json"
    if config.exists() and json.loads(config.read_text()) != vars(RUN_CFG):
        raise ValueError("Existing subclass run uses different settings")
    config.write_text(json.dumps(vars(RUN_CFG), indent=2))
    if (RUN_DIR / "fit/cells.tsv").exists():
        return RUN_DIR / "fit"
    geometry = np.load(GRAPH)
    if not np.array_equal(geometry["ids"], a.obs_names) or not np.allclose(
        xyz, geometry["xyz"]
    ):
        raise ValueError("Geometry and input cells differ")
    coarse = geometry["coarse"]
    composition = geometry["composition"]
    names, types = np.unique(a.obs.celltypes.astype(str), return_inverse=True)
    contact = sparse.load_npz(OUT / "contacts.npz")
    grouping_graph = contact
    if revised:
        coarse = merge_composition_classes(coarse, composition)
    if section_only:
        contact = restrict_to_sections(contact, xyz[:, 2])
    GROUPS, interfaces = partition_contexts(
        grouping_graph, coarse, RUN_CFG, flank_fraction=0.35 if revised else 0.0
    )
    if revised:
        interface_cells = sum(len(v) for (a1, b1), v in GROUPS.items() if a1 != b1)
        log(
            f"{len(np.unique(coarse))} coarse classes, {interface_cells:,} interface cells "
            f"({interface_cells / len(coarse):.1%})"
        )
    columns = ["A", "B", "cells_a", "cells_b", "fraction_a", "fraction_b", "eligible"]
    pd.DataFrame(interfaces, columns=columns).to_csv(
        RUN_DIR / "interfaces.tsv", sep="\t", index=False
    )
    BLOCK = np.unique(np.floor(xyz / RUN_CFG.block_um), axis=0, return_inverse=True)[1]
    atlas = pd.read_csv(ROOT / "data/commuspace_human_lr_atlas.tsv", sep="\t")
    channels = Channels(a.layers["counts"], a.var_names, atlas)
    TENSOR = FastWindows(
        contact, coarse, types, channels, RUN_CFG.sigma_um, full_interface=revised
    )
    verify_kernel(TENSOR, contact, coarse, types, channels, RUN_CFG, GROUPS)
    keys = sorted(GROUPS, key=lambda key: -len(GROUPS[key]))
    log(
        f"{len(keys)} contexts, {len(channels.names)} LR channels, {workers} CPU workers"
    )
    with mp.get_context("fork").Pool(workers) as pool:
        for i, (key, message) in enumerate(
            pool.imap_unordered(_subclass_context, keys), 1
        ):
            log(f"{i}/{len(keys)} {key}: {message}")
    partitions = {}
    for a1, b1 in keys:
        with (RUN_DIR / f"subclasses_{a1}_{b1}.pkl").open("rb") as handle:
            partitions[a1, b1] = pickle.load(handle)
    result = SpatialNiches.from_partitions(
        pd.Index(a.obs_names, name="cell_id"),
        np.repeat("lung", len(a)),
        names,
        types,
        coarse,
        geometry["composition"],
        partitions,
        channels.names,
        RUN_CFG,
    )
    result.save(RUN_DIR / "fit")
    log(
        f"{len(result.cells):,} unique cell assignments and {result.cells.niche.nunique()} niche classes"
    )
    return RUN_DIR / "fit"


def evaluate(a, xyz, fit):
    fit = Path(fit)
    cells = (
        pd.read_csv(fit / "cells.tsv", sep="\t").set_index("cell_id").loc[a.obs_names]
    )
    assert len(cells) == a.n_obs and cells.index.is_unique and cells.niche.notna().all()
    obs = a.obs
    words = obs.EMT_niche.astype("string").str.strip().str.lower()
    log(f"Author EMT categories: {words.value_counts(dropna=False).to_dict()}")
    positive = words.isin(
        ["true", "1", "emt niche", "emt_niche", "emt", "inside emt niche"]
    )
    if not positive.any() or positive.all():
        raise ValueError("Inspect author EMT categories before evaluation")
    published = obs.niches_3D.astype("string")
    dc = published.eq("DC islands").fillna(False)
    if not dc.any():
        raise ValueError("Dendritic reference not found")
    meta = cells.assign(
        section=obs.section,
        emt=positive,
        dc=dc,
        pseudotime=pd.to_numeric(obs.tumor_pseudotime_rank, errors="coerce"),
    )
    metrics = []
    for target, valid in [
        ("emt", words.notna()),
        ("dc", published.notna() & published.ne("Excluded")),
    ]:
        m = meta.loc[valid]
        truth = m[target]
        for scale, labels in [
            ("composition", m.coarse.astype(str)),
            ("final", m.niche),
            (
                "functional",
                m.niche.where(
                    m.subclass.gt(0) if "subclass" in m else m.niche.str.startswith("F")
                ),
            ),
            ("region", m.get("functional_region")),
        ]:
            if labels is None:
                continue
            groups = truth.groupby(labels, observed=True).agg(["sum", "size"])
            for label, (tp, n) in groups.iterrows():
                total = int(truth.sum())
                metrics.append(
                    (
                        target,
                        scale,
                        label,
                        n,
                        total,
                        tp,
                        tp / n,
                        tp / total,
                        2 * tp / (n + total),
                        tp / (n + total - tp),
                    )
                )
    metrics = pd.DataFrame(
        metrics,
        columns="target scale label cells reference_cells overlap precision recall F1 IoU".split(),
    )
    metrics = metrics.sort_values(
        ["target", "scale", "F1"], ascending=[True, True, False]
    )
    metrics.to_csv(fit.parent / "overlap.tsv", sep="\t", index=False)
    evidence = pd.read_csv(fit / "channels.tsv", sep="\t")
    return meta, metrics, evidence


if __name__ == "__main__":
    import argparse
    import fcntl

    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true")
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--pooled", action="store_true")
    variants.add_argument("--scaled", action="store_true")
    variants.add_argument("--hotspot", action="store_true")
    variants.add_argument("--revised", action="store_true")
    variants.add_argument("--revised-2d", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock_name = (
        "revised_2d.lock"
        if args.revised_2d
        else "revised_3d.lock"
        if args.revised
        else "fit.lock"
    )
    with (OUT / lock_name).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data, coordinates = read_input()
        prepare(data, coordinates)
        if args.fit:
            if args.scaled:
                fit_volume(
                    data, coordinates, POOLED, name="scaled_channels", pooled=True
                )
            elif args.pooled:
                fit_volume(
                    data, coordinates, POOLED, name="pooled_windows", pooled=True
                )
            elif args.hotspot:
                fit_volume(data, coordinates)
            else:
                result = fit_subclasses(
                    data,
                    coordinates,
                    revised=args.revised or args.revised_2d,
                    section_only=args.revised_2d,
                )
                evaluate(data, coordinates, result)
