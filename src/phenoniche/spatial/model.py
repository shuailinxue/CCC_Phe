import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .channels import Channels
from .geometry import composition_partition, physical_graph
from .contexts import delaunay_edges, interface_cells, neighborhood_enrichment
from .cell_programs import contact_support, fit_context
from .settings import Settings
from .programs import cp_device


def _init_context_worker(inputs):
    global _CONTEXT_INPUTS
    _CONTEXT_INPUTS = inputs


def _run_context(task, inputs=None):
    key, centers = task
    graph, contacts, types, channels, cfg = _CONTEXT_INPUTS if inputs is None else inputs
    print(f"开始 {key}，{len(centers):,} 个细胞", flush=True)
    seed = int(np.random.SeedSequence([cfg.seed, *key]).generate_state(1)[0])
    part = fit_context(centers, graph, contacts, types, channels, cfg, seed)
    return key, part


def _context_fingerprint(xyz, ids, types, samples, channels, cfg, coarse):
    import hashlib

    digest = hashlib.sha256()
    arrays = [
        xyz,
        types,
        coarse,
        channels.expression.data,
        channels.expression.indices,
        channels.expression.indptr,
        channels.left,
        channels.right,
    ]
    for array in arrays:
        a = np.ascontiguousarray(array)
        digest.update(str((a.shape, a.dtype)).encode())
        digest.update(a.view(np.uint8))
    for values in (ids, samples, channels.names, channels.signaling_types):
        for value in values:
            digest.update(repr(value).encode())
            digest.update(b"\0")
    digest.update(json.dumps(asdict(cfg), sort_keys=True).encode())
    # Changes to the fitting implementation invalidate old context results.
    for filename in (
        "model.py",
        "cell_programs.py",
        "programs.py",
        "contexts.py",
        "geometry.py",
        "channels.py",
    ):
        digest.update((Path(__file__).parent / filename).read_bytes())
    return digest.hexdigest()


def _fit_contexts(
    groups,
    graph,
    contacts,
    types,
    channels,
    cfg,
    workers,
    checkpoint_dir,
    fingerprint,
):
    import os
    import pickle
    import tempfile
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context

    if workers < 1:
        raise ValueError("workers must be positive")
    inputs = (graph, contacts, types, channels, cfg)
    directory = Path(checkpoint_dir) / fingerprint if checkpoint_dir else None
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
    results, pending = {}, []
    for key, centers in groups.items():
        path = directory / f"context_{key[0]}_{key[1]}.pkl" if directory else None
        if path and path.exists():
            with path.open("rb") as stream:
                part = pickle.load(stream)
            if not np.array_equal(part["ids"], centers):
                raise ValueError(f"Checkpoint cell mismatch in {path}")
            results[key] = part
            print(f"已读取 {key} 进度", flush=True)
        else:
            pending.append((key, centers))

    def commit(key, part):
        if directory:
            fd, temporary = tempfile.mkstemp(
                dir=directory, prefix="context_", suffix=".partial"
            )
            try:
                with os.fdopen(fd, "wb") as stream:
                    pickle.dump(part, stream, protocol=5)
                os.replace(temporary, directory / f"context_{key[0]}_{key[1]}.pkl")
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        results[key] = part
        print(
            f"完成 {len(results)}/{len(groups)} {key}，"
            f"{len(np.unique(part['labels']))} 个 niche，稳定性 {part['stability']:.3f}",
            flush=True,
        )

    pending.sort(key=lambda item: -len(item[1]))
    if workers == 1:
        for task in pending:
            commit(*_run_context(task, inputs))
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn"),
            initializer=_init_context_worker, initargs=(inputs,),
        ) as executor:
            futures = [executor.submit(_run_context, task) for task in pending]
            for future in as_completed(futures):
                commit(*future.result())
    return {key: results[key] for key in groups}


@contextmanager
def _new_output(directory):
    import os
    import tempfile
    from shutil import rmtree

    path = Path(directory)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=path.parent, prefix=path.name + "."))
    try:
        yield tmp
        os.rename(tmp, path)
    finally:
        if tmp.exists():
            rmtree(tmp)


@dataclass
class CoarseAnalysis:
    cells: pd.DataFrame
    composition: pd.DataFrame
    enrichment: pd.DataFrame
    settings: Settings

    def save(self, directory):
        with _new_output(directory) as tmp:
            self._write(tmp)
            (tmp / "settings.json").write_text(json.dumps(asdict(self.settings), indent=2))

    def _write(self, directory):
        from matplotlib import colormaps
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.cells.to_csv(directory / "coarse_cells.tsv", sep="\t")
        self.composition.to_csv(directory / "composition.tsv.gz", sep="\t")
        self.enrichment.to_csv(directory / "enrichment.tsv", sep="\t", index=False, na_rep="NA")
        table = self.enrichment
        classes = np.unique(table[["coarse_a", "coarse_b"]])
        a = np.searchsorted(classes, table.coarse_a)
        b = np.searchsorted(classes, table.coarse_b)
        matrix = np.full((len(classes), len(classes)), np.nan)
        matrix[a, b] = matrix[b, a] = table.z_score.to_numpy()
        finite = matrix[np.isfinite(matrix)]
        limit = max(1, np.max(np.abs(finite))) if finite.size else 1
        figure = Figure(figsize=(7, 6), layout="constrained")
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        cmap = colormaps["RdBu_r"].copy()
        cmap.set_bad("#dddddd")
        image = ax.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit)
        ax.set_xticks(np.arange(len(classes)), classes, rotation=90)
        ax.set_yticks(np.arange(len(classes)), classes)
        ax.set(xlabel="Coarse class", ylabel="Coarse class", title="Spatial adjacency enrichment")
        figure.colorbar(image, ax=ax, label="Permutation Z-score")
        figure.savefig(directory / "enrichment.png", dpi=150)


def _prepare_coarse(cell_ids, coordinates_um, cell_types, specimens, cfg, cache_path):
    xyz = np.asarray(coordinates_um, dtype=float)
    ids = pd.Index(cell_ids, name="cell_id")
    if ids.hasnans or not ids.is_unique or not len(ids):
        raise ValueError("Nonempty unique cell identifiers required")
    if (
        xyz.shape not in [(len(ids), 2), (len(ids), 3)]
        or not np.isfinite(xyz).all()
    ):
        raise ValueError(
            "Expected aligned finite 2D or 3D coordinates in micrometers"
        )
    for values in (cell_types, specimens):
        if np.shape(values) != (len(ids),) or pd.isna(values).any():
            raise ValueError(
                "One observed cell type and specimen per cell is required"
            )
    samples = np.asarray(specimens, dtype=str)
    names, types = np.unique(np.asarray(cell_types, dtype=str), return_inverse=True)
    graph = physical_graph(xyz, samples, cfg.window_um)
    composition, coarse = composition_partition(graph, types, cfg, cache_path=cache_path)
    edges = delaunay_edges(xyz, samples, cfg.delaunay_max_um or cfg.contact_um)
    enrichment = neighborhood_enrichment(edges, coarse, samples, cfg.enrichment_permutations, cfg.seed)
    cells = pd.DataFrame(dict(specimen=samples, cell_type=names[types], coarse=coarse), index=ids)
    preview = CoarseAnalysis(cells, pd.DataFrame(composition, index=ids, columns=names), enrichment, cfg)
    return preview, xyz, ids, samples, names, types, graph, coarse, edges


@dataclass
class SpatialNiches:
    cells: pd.DataFrame
    coarse_analysis: CoarseAnalysis
    evidence: pd.DataFrame
    programs: dict
    lr: list
    settings: Settings
    neighbor_counts: np.ndarray
    neighbor_weights: np.ndarray
    neighbor_counts_by_signaling: np.ndarray
    neighbor_weights_by_signaling: np.ndarray
    lr_signaling_types: np.ndarray

    @property
    def composition(self):
        return self.coarse_analysis.composition

    @staticmethod
    def coarse(cell_ids, coordinates_um, cell_types, specimens, settings=None,
               composition_graph_cache=None):
        """Preview coarse classes and spatial enrichment without fitting CP."""
        return _prepare_coarse(
            cell_ids, coordinates_um, cell_types, specimens, settings or Settings(),
            composition_graph_cache,
        )[0]

    @classmethod
    def fit(
        cls,
        counts,
        genes,
        cell_ids,
        coordinates_um,
        cell_types,
        specimens,
        lr,
        settings=None,
        composition_graph_cache=None,
        workers=1,
        checkpoint_dir=None,
        *,
        interface=None,
    ):
        cfg = settings or Settings()
        device = cp_device(cfg.cp_device)
        if device.type == "cuda" and workers != 1:
            raise ValueError("CUDA CP requires one context worker per device")
        cfg = replace(cfg, cp_device=str(device))
        preview, xyz, ids, samples, names, types, graph, coarse, edges = _prepare_coarse(
            cell_ids, coordinates_um, cell_types, specimens, cfg, composition_graph_cache
        )
        if counts.shape != (len(ids), len(genes)):
            raise ValueError(
                "Count matrix dimensions do not match cell/gene identifiers"
            )
        ch = Channels(counts, pd.Index(genes), lr, cfg.expression_preprocessing, cfg.expression_quantile)
        max_radius = max(cfg.contact_um, cfg.contact_radius_um, cfg.secreted_radius_um, cfg.ecm_radius_um)
        contacts = graph if cfg.window_um == max_radius else physical_graph(xyz, samples, max_radius)
        if interface is None:
            groups = {(int(c), int(c)): np.flatnonzero(coarse == c) for c in np.unique(coarse)}
        else:
            pair = tuple(sorted(interface))
            rows = interface_cells(xyz, samples, coarse, edges, pair, cfg.interface_um)
            groups = {pair: rows}
        fingerprint = _context_fingerprint(
            xyz, ids, types, samples, ch, cfg, coarse
        )
        partitions = _fit_contexts(
            groups,
            graph,
            contacts,
            types,
            ch,
            cfg,
            workers,
            checkpoint_dir,
            fingerprint,
        )
        niche = np.empty(len(ids), dtype=object)
        context = np.empty(len(ids), dtype=object)
        stability = np.zeros(len(ids))
        selected = np.sort(np.concatenate(list(groups.values())))
        evidence = []
        for key, part in partitions.items():
            a, b = key
            parent = f"C{a}" if a == b else f"I{a}_{b}"
            rows, labels = part["ids"], part["labels"]
            context[rows] = parent
            niche[rows] = [f"{parent}.{x + 1}" for x in labels]
            stability[rows] = part["stability"]
            if part["models"]:
                _, neighbor, lr = part["models"][0]
                score = part["cell_scores"][0]
                for cluster in np.unique(labels):
                    members = labels == cluster
                    for program in range(score.shape[1]):
                        for focal_type in np.unique(types[rows[members]]):
                            focal_members = members & (types[rows] == focal_type)
                            mean_activity = float(score[focal_members, program].mean())
                            for neighbor_type in np.argsort(-neighbor[:, program])[:5]:
                                for channel in np.argsort(-lr[:, program])[:10]:
                                    direction = (
                                        "sender"
                                        if channel < len(ch.names)
                                        else "receiver"
                                    )
                                    ligand, receptor = ch.names[channel % len(ch.names)]
                                    evidence.append(
                                        (
                                            f"{parent}.{cluster + 1}",
                                            program,
                                            names[focal_type],
                                            mean_activity,
                                            names[neighbor_type],
                                            direction,
                                            ligand,
                                            receptor,
                                            neighbor[neighbor_type, program],
                                            lr[channel, program],
                                        )
                                    )
        cells = pd.DataFrame(
            dict(
                specimen=samples,
                cell_type=names[types],
                coarse=coarse,
                context=context,
                niche=niche,
                stability=stability,
            ),
            index=ids,
        ).iloc[selected]
        columns = [
            "niche",
            "program",
            "focal_type",
            "mean_focal_activity",
            "neighbor_type",
            "direction",
            "ligand",
            "receptor",
            "neighbor_loading",
            "lr_loading",
        ]
        support = []
        for radius in (cfg.contact_radius_um, cfg.secreted_radius_um,
                       cfg.ecm_radius_um, cfg.contact_um):
            edges = contacts.tocoo()
            keep = edges.data <= radius
            eligible = sparse.csr_matrix((edges.data[keep], (edges.row[keep], edges.col[keep])), shape=contacts.shape)
            support.append(contact_support(eligible, selected, types))
        return cls(
            cells,
            preview,
            pd.DataFrame(evidence, columns=columns),
            partitions,
            ch.names,
            cfg,
            support[-1][0], support[-1][1],
            np.stack([x[0] for x in support], axis=1),
            np.stack([x[1] for x in support], axis=1),
            ch.signaling_types,
        )

    def save(self, directory):
        with _new_output(directory) as tmp:
            self.coarse_analysis._write(tmp)
            self.cells.to_csv(tmp / "cells.tsv", sep="\t")
            self.evidence.to_csv(tmp / "channels.tsv", sep="\t", index=False)
            lr = pd.DataFrame(self.lr, columns=["ligand", "receptor"])
            lr["signaling_type"] = self.lr_signaling_types
            lr.to_csv(tmp / "lr.tsv", sep="\t", index=False)
            (tmp / "settings.json").write_text(
                json.dumps(asdict(self.settings), indent=2)
            )
            np.savez_compressed(
                tmp / "neighbor_support.npz",
                cell_ids=self.cells.index.to_numpy(dtype=str),
                cell_types=self.composition.columns.to_numpy(dtype=str),
                signaling_types=np.array(["Contact", "Secreted", "ECM", "Unknown/Ambiguous"]),
                counts=self.neighbor_counts, weights=self.neighbor_weights,
                counts_by_signaling=self.neighbor_counts_by_signaling,
                weights_by_signaling=self.neighbor_weights_by_signaling,
            )
            ids = self.coarse_analysis.cells.index.to_numpy(dtype=str)
            for (a, b), part in self.programs.items():
                tensor = part["tensor"]
                sparse.save_npz(tmp / f"tensor_{a}_{b}.npz", tensor.values)
                np.savez_compressed(tmp / f"tensor_mask_{a}_{b}.npz",
                    cell_ids=ids[part["ids"]], mask=tensor.mask,
                    radii_um=np.array([group[1] for group in tensor.groups]),
                    **{f"channels_{i}": group[0] for i, group in enumerate(tensor.groups)})
                (tmp / f"rank_evaluation_{a}_{b}.json").write_text(json.dumps(part["rank_evaluation"], indent=2))
                np.savez_compressed(
                    tmp / f"program_support_{a}_{b}.npz",
                    cell_ids=ids[part["ids"]], weights=part["pool_weights"],
                    present=part["pool_weights"] > 0,
                    cell_types=self.composition.columns.to_numpy(dtype=str),
                )
                if part["training"]:
                    _save_training(tmp, a, b, part["training"])
                for rep, (model, activity, feature) in enumerate(zip(
                    part["models"], part["cell_scores"], part["feature_scores"], strict=True
                )):
                    _, neighbor, lr = model
                    np.savez_compressed(
                        tmp / f"programs_{a}_{b}_repeat{rep}.npz",
                        cell_rows=part["ids"],
                        cell_ids=ids[part["ids"]],
                        neighbor=neighbor,
                        lr=lr,
                        focal_activity=activity,
                        neighborhood_activity=feature,
                        **part["representations"][rep],
                    )


def _save_training(directory, a, b, training):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    losses = pd.DataFrame([
        dict(repeat=repeat, **row)
        for repeat, fit in enumerate(training) for row in fit.history
    ])
    losses.to_csv(directory / f"cp_losses_{a}_{b}.tsv", sep="\t", index=False)
    pd.DataFrame([
        dict(repeat=repeat, **fit.metrics) for repeat, fit in enumerate(training)
    ]).to_csv(directory / f"cp_metrics_{a}_{b}.tsv", sep="\t", index=False)
    figure = Figure(figsize=(9, 3), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    for ax, (column, label) in zip(axes, [
        ("reconstruction_loss", "Masked reconstruction"),
        ("orthogonality_loss", "Weighted orthogonality"),
        ("total_loss", "Total objective"),
    ], strict=True):
        for _, group in losses.groupby("repeat", sort=True):
            ax.plot(group.epoch, group[column])
        ax.set(title=label, xlabel="Epoch")
    figure.savefig(directory / f"cp_losses_{a}_{b}.png", dpi=150)
