import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .channels import Channels
from .geometry import composition_partition, physical_graph
from .contexts import partition_contexts
from .cell_programs import fit_context
from .settings import Settings


def _run_context(task):
    key, centers = task
    block, graph, contacts, types, channels, cfg = _CONTEXT_INPUTS
    print(f"开始 {key}，{len(centers):,} 个细胞", flush=True)
    seed = int(np.random.SeedSequence([cfg.seed, *key]).generate_state(1)[0])
    part = fit_context(centers, block, graph, contacts, types, channels, cfg, seed)
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
    for values in (ids, samples, channels.names):
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
    block,
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

    global _CONTEXT_INPUTS
    if workers < 1:
        raise ValueError("workers must be positive")
    _CONTEXT_INPUTS = (block, graph, contacts, types, channels, cfg)
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
            commit(*_run_context(task))
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("fork")
        ) as executor:
            futures = [executor.submit(_run_context, task) for task in pending]
            for future in as_completed(futures):
                commit(*future.result())
    return {key: results[key] for key in groups}


@dataclass
class SpatialNiches:
    cells: pd.DataFrame
    composition: pd.DataFrame
    evidence: pd.DataFrame
    programs: dict
    lr: list
    settings: Settings

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
    ):
        cfg = settings or Settings()
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
        if counts.shape != (len(ids), len(genes)):
            raise ValueError(
                "Count matrix dimensions do not match cell/gene identifiers"
            )
        samples = np.asarray(specimens, dtype=str)
        names, types = np.unique(np.asarray(cell_types, dtype=str), return_inverse=True)
        ch = Channels(counts, pd.Index(genes), lr)
        graph = physical_graph(xyz, samples, cfg.window_um)
        contacts = (
            graph
            if cfg.window_um == cfg.contact_um
            else physical_graph(xyz, samples, cfg.contact_um)
        )
        composition, coarse = composition_partition(
            graph, types, cfg, cache_path=composition_graph_cache
        )
        groups, interfaces = partition_contexts(contacts, coarse, cfg)
        tiles = np.column_stack(
            [np.unique(samples, return_inverse=True)[1], np.floor(xyz / cfg.block_um)]
        )
        _, block = np.unique(tiles, axis=0, return_inverse=True)
        fingerprint = _context_fingerprint(
            xyz, ids, types, samples, ch, cfg, coarse
        )
        partitions = _fit_contexts(
            groups,
            block,
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
        assigned = np.zeros(len(ids), dtype=int)
        programs = {}
        evidence = []
        for key, part in partitions.items():
            a, b = key
            parent = f"C{a}" if a == b else f"I{a}_{b}"
            rows, labels = part["ids"], part["labels"]
            context[rows] = parent
            niche[rows] = [f"{parent}.{x + 1}" for x in labels]
            stability[rows] = part["stability"]
            assigned[rows] += 1
            programs[key] = part
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
        if not np.all(assigned == 1):
            raise ValueError("Each cell must have exactly one context")
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
        )
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
        return cls(
            cells,
            pd.DataFrame(composition, index=ids, columns=names),
            pd.DataFrame(evidence, columns=columns),
            programs,
            ch.names,
            cfg,
        )

    def save(self, directory):
        import os
        import tempfile
        from shutil import rmtree

        path = Path(directory)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=path.parent, prefix=path.name + "."))
        try:
            self.cells.to_csv(tmp / "cells.tsv", sep="\t")
            self.composition.to_csv(tmp / "composition.tsv.gz", sep="\t")
            self.evidence.to_csv(tmp / "channels.tsv", sep="\t", index=False)
            pd.DataFrame(self.lr, columns=["ligand", "receptor"]).to_csv(
                tmp / "lr.tsv", sep="\t", index=False
            )
            (tmp / "settings.json").write_text(
                json.dumps(asdict(self.settings), indent=2)
            )
            for (a, b), part in self.programs.items():
                for rep, (model, activity, feature) in enumerate(
                    zip(part["models"], part["cell_scores"], part["feature_scores"])
                ):
                    _, neighbor, lr = model
                    np.savez_compressed(
                        tmp / f"programs_{a}_{b}_repeat{rep}.npz",
                        cell_rows=part["ids"],
                        neighbor=neighbor,
                        lr=lr,
                        focal_activity=activity,
                        neighborhood_activity=feature,
                    )
            os.rename(tmp, path)
        finally:
            if tmp.exists():
                rmtree(tmp)
