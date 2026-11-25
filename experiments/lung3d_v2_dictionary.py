"""Blind pilot of nonnegative programs over complete, directed LR channels."""

import argparse
from multiprocessing import get_context
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score, silhouette_score

from lung3d_v2 import DATA, GRAPH, OUT, ROOT, log
from lung3d_v2_fast import FastWindows
from phenoniche.spatial.channels import Channels


def fibers_batch(job):
    rows, key = job
    vals, cols, pointer = [], [], [0]
    opps, opcols, oppointer = [], [], [0]
    for cell in rows:
        sender, receiver, score, opportunity = TENSOR.fibers(cell, key)
        pair = sender * TENSOR.n_types + receiver
        i, j = np.nonzero(score)
        vals.append(np.log1p(score[i, j]))
        cols.append(pair[i] * N_LR + j)
        pointer.append(pointer[-1] + len(i))
        opps.append(opportunity.astype(np.float32))
        opcols.append(pair)
        oppointer.append(oppointer[-1] + len(pair))
    index = np.concatenate(cols)
    support = np.concatenate(opcols)
    matrix = sparse.csr_matrix(
        (np.concatenate(vals), index, pointer),
        shape=(len(rows), TENSOR.n_senders * TENSOR.n_types * N_LR),
    )
    op = sparse.csr_matrix(
        (np.concatenate(opps), support, oppointer),
        shape=(len(rows), TENSOR.n_senders * TENSOR.n_types),
    )
    return matrix, op


def gather(pool, ids, key):
    jobs = [
        (chunk, key)
        for chunk in np.array_split(ids, max(1, int(np.ceil(len(ids) / 180))))
    ]
    blocks = list(pool.imap(fibers_batch, jobs))
    return tuple(sparse.vstack(xs, format="csr") for xs in zip(*blocks))


def sampled_tiles(ids, xyz, rng, block_um=90, per_tile=8):
    _, blocks = np.unique(np.floor(xyz[ids] / block_um), axis=0, return_inverse=True)
    order = np.argsort(blocks, kind="stable")
    rowparts = np.split(order, np.cumsum(np.bincount(blocks))[:-1])
    rows = []
    tile = []
    half = []
    for block, part in enumerate(rowparts):
        n = min(len(part) // 2, per_tile // 2)
        if n:
            choice = rng.choice(part, 2 * n, replace=False)
            rows.extend(ids[choice])
            tile.extend([block] * 2 * n)
            half.extend([0] * n + [1] * n)
    return np.asarray(rows), np.asarray(tile), np.asarray(half)


def screen(matrix, blocks, half, budget=1800, tail=False):
    # Replicate neighborhood intensities using two disjoint sets of centers per tile.
    unique, tid = np.unique(blocks, return_inverse=True)
    if len(unique) < 5:
        return np.array([], dtype=int)
    means = []
    for side in (0, 1):
        cols = np.flatnonzero(half == side)
        tiles = tid[cols]
        weights = 1 / np.bincount(tiles)[tiles]
        design = sparse.csr_matrix(
            (weights, (tiles, cols)), shape=(len(unique), len(tid))
        )
        means.append((design @ matrix).tocsr())
    a, b = means
    mean_a = np.asarray(a.mean(axis=0)).ravel()
    mean_b = np.asarray(b.mean(axis=0)).ravel()
    covariance = np.asarray(a.multiply(b).mean(axis=0)).ravel() - mean_a * mean_b
    # Equalize very high-amplitude columns without making sporadic zeros influential.
    priority = covariance / np.sqrt(0.15 + (mean_a + mean_b) / 2)
    hit = np.asarray((a > 0).sum(axis=0)).ravel()
    hit_b = np.asarray((b > 0).sum(axis=0)).ravel()
    if tail:
        shared = a.minimum(b).tocoo()
        excess = np.maximum(
            shared.data - (mean_a[shared.col] + mean_b[shared.col]) / 2, 0
        )
        sizes = np.bincount(shared.col, minlength=matrix.shape[1])
        signal = np.bincount(shared.col, weights=excess**3, minlength=matrix.shape[1])
        # Normalize by replicated occupied tiles to allow localized channels to compete.
        priority = np.cbrt(signal / np.maximum(sizes, 1)) / np.sqrt(
            0.15 + (mean_a + mean_b) / 2
        )
    valid = np.flatnonzero((priority > 0) & (np.minimum(hit, hit_b) >= 5))
    ranked = valid[np.argsort(-priority[valid], kind="stable")]
    per_pair = {}
    take = []
    for column in ranked:
        pair = int(column // N_LR)
        used = per_pair.get(pair, 0)
        if used < 60:
            take.append(column)
            per_pair[pair] = used + 1
        if len(take) >= budget:
            break
    return np.array(sorted(take), dtype=int)


def transform(matrix, opportunity, chosen, mode, baseline=None):
    strength = matrix[:, chosen].toarray().astype(np.float32)
    exposure = opportunity[:, chosen // N_LR].toarray()
    measured = exposure > 0
    if mode == "load":
        # A bounded weighting retains contact abundance without turning it into raw edge count.
        strength *= np.sqrt(exposure / (2 + exposure))
        return strength, None
    if baseline is None:
        baseline = (strength.sum(axis=0) / np.maximum(measured.sum(axis=0), 1)).astype(
            np.float32
        )
    contrast = (strength - baseline) * measured * np.sqrt(exposure / (1 + exposure))
    return np.c_[np.maximum(contrast, 0), np.maximum(-contrast, 0)], baseline


def learn(x, block, rng, exact=False, tail=False):
    """Select rank and partition with spatially disjoint validation tiles and repeated refits."""
    groups = np.unique(block)
    validation_groups = rng.choice(groups, max(1, len(groups) // 5), replace=False)
    validation = np.flatnonzero(np.isin(block, validation_groups))
    dev = np.flatnonzero(~np.isin(block, validation_groups))
    if min(len(validation), len(dev)) < 30:
        return None, []
    eval_ids = rng.choice(validation, min(900, len(validation)), replace=False)
    result = None
    diagnostics = []
    for rank in (8, 16, 24) if tail else (8, 16):
        prepared = []
        for rep in range(2):
            train_groups = rng.choice(
                np.unique(block[dev]),
                max(1, int(0.8 * len(np.unique(block[dev])))),
                replace=False,
            )
            train = dev[np.isin(block[dev], train_groups)]
            if len(train) > 2800:
                train = rng.choice(train, 2800, replace=False)
            model = NMF(
                n_components=rank,
                init="nndsvda",
                max_iter=220 if exact else 100,
                tol=0.001,
                alpha_H=0.0002,
                l1_ratio=0.8,
                random_state=300 + rep,
            )
            score = model.fit_transform(x[train])
            norm = np.maximum(np.linalg.norm(model.components_, axis=1), 1e-8)
            score = score * norm
            val = model.transform(x[eval_ids]) * norm
            relative = np.sqrt(
                np.square(x[eval_ids] - (val / norm) @ model.components_).sum()
                / max(np.square(x[eval_ids]).sum(), 1e-8)
            )
            if exact:
                h = model.components_ / norm[:, None]
                score, val = x[train] @ h.T, x[eval_ids] @ h.T
            scale = np.maximum(np.quantile(score, 0.9, axis=0), 1e-6)
            if exact:
                prepared.append(
                    (model, norm, scale, score / scale, val / scale, relative)
                )
            else:
                prepared.append(
                    (
                        model,
                        norm,
                        scale,
                        np.log1p(score / scale),
                        np.log1p(val / scale),
                        relative,
                    )
                )
        for k in range(2, 8) if tail else (2, 3, 4, 5):
            predictions = []
            estimators = []
            for model, norm, scale, dev_score, validation_score, error in prepared:
                estimator = KMeans(n_clusters=k, n_init=8, random_state=37).fit(
                    dev_score
                )
                estimators.append(estimator)
                predictions.append(estimator.predict(validation_score))
            stability = adjusted_rand_score(*predictions)
            labels = predictions[0]
            silhouette = (
                silhouette_score(prepared[0][4], labels)
                if 2 <= len(np.unique(labels)) < len(labels)
                else -1
            )
            tiny = np.bincount(labels, minlength=k).min() / len(labels)
            error = float(np.mean([p[-1] for p in prepared]))
            diagnostics.append((rank, k, error, stability, silhouette, tiny))
            if stability < 0.65 or silhouette < 0.12 or tiny < 0.025:
                continue
            quality = stability * silhouette
            if result is None or quality > result[0]:
                result = quality, rank, k, prepared[0], estimators[0]
    return result, diagnostics


def feature_labels(chosen, names, lr, polarity=False):
    rows = []
    for feature in chosen:
        pair, lri = divmod(int(feature), len(lr))
        send, receive = divmod(pair, len(names))
        rows.append(
            (names[send % len(names)], names[receive], send // len(names), *lr[lri])
        )
    frame = pd.DataFrame(
        rows, columns=["sender", "receiver", "flank_direction", "ligand", "receptor"]
    )
    if polarity:
        frame = pd.concat(
            [
                frame.assign(signal="above_baseline"),
                frame.assign(signal="below_baseline"),
            ],
            ignore_index=True,
        )
    else:
        frame = frame.assign(signal="opportunity_weighted")
    return frame


def run(args):
    global TENSOR, N_LR
    out = OUT / "directed_dictionary"
    out.mkdir(exist_ok=True)
    a = ad.read_h5ad(DATA / "preprocessed_objects/preprocessed_objects/cosmx.h5ad")
    c = pd.read_csv(
        OUT / "revised_3d/fit/cells.tsv", sep="\t", index_col="cell_id"
    ).loc[a.obs_names]
    xyz = np.load(GRAPH)["xyz"]
    names, types = np.unique(c.cell_type, return_inverse=True)
    graph = sparse.load_npz(OUT / "contacts.npz")
    ch = Channels(
        a.layers["counts"],
        a.var_names,
        pd.read_csv(ROOT / "data/commuspace_human_lr_atlas.tsv", sep="\t"),
    )
    TENSOR = FastWindows(graph, c.coarse.to_numpy(), types, ch, 20, full_interface=True)
    N_LR = len(ch.names)
    with get_context("fork").Pool(16) as pool:
        for context in args.contexts:
            parent, separator, child = context.partition(".")
            rows = np.flatnonzero(c.context.eq(parent))
            if separator:
                assignments = pd.read_csv(
                    out / "exact" / parent / "load_full_cells.tsv", sep="\t"
                ).set_index("cell_id")
                rows = c.index.get_indexer(
                    assignments.index[assignments.subclass.eq(int(child))]
                )
                if np.any(rows < 0):
                    raise ValueError(
                        "Parent assignment IDs do not align with source cells"
                    )
            key = tuple(map(int, parent[1:].split("_")))
            key = (key[0], key[-1])
            pilot = out / ("tree_cache" if separator else "") / context
            pilot.mkdir(parents=True, exist_ok=True)
            dest = pilot
            if args.exact:
                stage = (
                    "tree"
                    if separator
                    else "patch"
                    if args.patch
                    else "tail"
                    if args.tail
                    else "exact"
                )
                dest = out / stage / context
                dest.mkdir(parents=True, exist_ok=True)
            sample, tile, half = sampled_tiles(rows, xyz, np.random.default_rng(252))
            if len(sample) < 100:
                continue
            cached = pilot / "pilot_scores.npz"
            if cached.exists():
                raw = sparse.load_npz(cached)
                opp = sparse.load_npz(pilot / "pilot_opportunities.npz")
            else:
                log(
                    f"{context}: extracting full LR atlas for {len(sample):,} discovery windows"
                )
                raw, opp = gather(pool, sample, key)
                sparse.save_npz(cached, raw)
                sparse.save_npz(pilot / "pilot_opportunities.npz", opp)
            # Hold out identical tiles for unsupervised evaluation; screening sees training tiles only.
            rng = np.random.default_rng(808)
            groups = np.unique(tile)
            hold = rng.choice(groups, max(1, len(groups) // 5), replace=False)
            use = ~np.isin(tile, hold)
            chosen = screen(raw[use], tile[use], half[use], tail=args.tail)
            log(f"{context}: {len(chosen):,} reproducibly varying directed LR channels")
            for mode in (args.mode,) if args.mode else ("load", "contrast"):
                _, baseline = transform(raw[use], opp[use], chosen, mode)
                x, _ = transform(raw, opp, chosen, mode, baseline)
                fit_x, fit_tile = x, tile
                if args.patch:
                    fit_tile, inverse = np.unique(tile, return_inverse=True)
                    means = sparse.csr_matrix(
                        (
                            1 / np.bincount(inverse)[inverse],
                            (inverse, np.arange(len(inverse))),
                        ),
                        shape=(len(fit_tile), len(tile)),
                    )
                    fit_x = (means @ x).astype(np.float32)
                    # A tile's mean feature vector is the object being clustered.
                    x = fit_x[inverse]
                result, diagnostics = learn(
                    fit_x, fit_tile, np.random.default_rng(808), args.exact, args.tail
                )
                pd.DataFrame(
                    diagnostics,
                    columns="rank k holdout_error stability silhouette smallest_class".split(),
                ).to_csv(dest / f"{mode}_selection.tsv", sep="\t", index=False)
                if result is None:
                    log(f"{context} {mode}: no stable split")
                    continue
                quality, rank, k, (model, norm, scale, _, _, _), estimator = result
                h = model.components_ / norm[:, None]
                activity = x @ h.T if args.exact else model.transform(x) * norm
                scores = activity / scale if args.exact else np.log1p(activity / scale)
                labels = estimator.predict(scores)
                # Per cluster exact additive contrast under the nonnegative model:
                # reconstructed difference in each input feature is delta(activity) @ loadings.
                meta = feature_labels(chosen, names, ch.names, mode == "contrast")
                report = []
                if args.exact:
                    d = estimator.transform(scores)
                    competitors = np.argsort(d, axis=1)[:, 1]
                    chosen_centers = estimator.cluster_centers_[labels]
                    runner_centers = estimator.cluster_centers_[competitors]
                    margin_weight = 2 * (chosen_centers - runner_centers) / scale
                    channel_margin = x * (margin_weight @ h)
                    intercept = (runner_centers**2 - chosen_centers**2).sum(axis=1)
                    margin = (
                        d[np.arange(len(d)), competitors] ** 2
                        - d[np.arange(len(d)), labels] ** 2
                    )
                    np.testing.assert_allclose(
                        channel_margin.sum(axis=1) + intercept,
                        margin,
                        atol=0.003,
                        rtol=0.001,
                    )
                    margin_rows = []
                    for label in np.unique(labels):
                        inside = labels == label
                        table = meta.assign(
                            subclass=label + 1,
                            margin_contribution=channel_margin[inside].mean(axis=0),
                            observed_feature_difference=x[inside].mean(axis=0)
                            - x[~inside].mean(axis=0),
                        )
                        margin_rows.append(
                            table.query("margin_contribution > 0").nlargest(
                                30, "margin_contribution"
                            )
                        )
                    pd.concat(margin_rows).to_csv(
                        dest / f"{mode}_decision_channels.tsv", sep="\t", index=False
                    )
                    np.savez_compressed(
                        dest / f"{mode}_centroids.npz",
                        centroids=estimator.cluster_centers_,
                        scale=scale,
                        margin=margin,
                        intercept=intercept,
                    )
                for label in np.unique(labels):
                    inside = labels == label
                    delta = activity[inside].mean(axis=0) - activity[~inside].mean(
                        axis=0
                    )
                    contribution = delta[:, None] * h
                    fitted = contribution.sum(axis=0)
                    observed = x[inside].mean(axis=0) - x[~inside].mean(axis=0)
                    table = meta.assign(
                        subclass=label + 1,
                        fitted_difference=fitted,
                        observed_difference=observed,
                        program=np.abs(contribution).argmax(axis=0) + 1,
                    )
                    report.append(
                        table.loc[(fitted > 0) & (observed > 0)].nlargest(
                            30, "fitted_difference"
                        )
                    )
                pd.concat(report).to_csv(
                    dest / f"{mode}_subclass_channels.tsv", sep="\t", index=False
                )
                meta.to_csv(dest / f"{mode}_feature_index.tsv", sep="\t", index=False)
                np.savez_compressed(
                    dest / f"{mode}_programs.npz",
                    loadings=h,
                    activity=activity,
                    centroids=estimator.cluster_centers_,
                    scale=scale,
                    chosen=chosen,
                )
                pd.DataFrame(dict(cell_id=c.index[sample], subclass=labels + 1)).to_csv(
                    dest / f"{mode}_pilot_cells.tsv", sep="\t", index=False
                )
                log(
                    f"{context} {mode}: rank {rank}, {k} subgroups, pilot validation quality {quality:.3f}"
                )
                if args.full:
                    frames = []
                    sums = np.zeros((k, x.shape[1]), dtype=np.float64)
                    observed_sums = np.zeros_like(sums)
                    class_sizes = np.zeros(k, dtype=int)
                    for batch in np.array_split(
                        rows, max(1, int(np.ceil(len(rows) / 2600)))
                    ):
                        raw_full, opp_full = gather(pool, batch, key)
                        xx, _ = transform(raw_full, opp_full, chosen, mode, baseline)
                        activity = (
                            xx @ h.T if args.exact else model.transform(xx) * norm
                        )
                        projected = (
                            activity / scale
                            if args.exact
                            else np.log1p(activity / scale)
                        )
                        assigned = estimator.predict(projected)
                        if args.exact:
                            distance = estimator.transform(projected)
                            competitor = np.argsort(distance, axis=1)[:, 1]
                            weight = (
                                2
                                * (
                                    estimator.cluster_centers_[assigned]
                                    - estimator.cluster_centers_[competitor]
                                )
                                / scale
                            )
                            contribution = xx * (weight @ h)
                            for label in np.unique(assigned):
                                keep = assigned == label
                                sums[label] += contribution[keep].sum(axis=0)
                                observed_sums[label] += xx[keep].sum(axis=0)
                                class_sizes[label] += keep.sum()
                        frames.append(
                            pd.DataFrame(
                                dict(cell_id=c.index[batch], subclass=assigned + 1)
                            )
                        )
                    pd.concat(frames).to_csv(
                        dest / f"{mode}_full_cells.tsv", sep="\t", index=False
                    )
                    if args.exact:
                        table = []
                        for label in np.flatnonzero(class_sizes):
                            n = class_sizes[label]
                            other = class_sizes.sum() - n
                            if other:
                                observed = (
                                    observed_sums[label] / n
                                    - (observed_sums.sum(axis=0) - observed_sums[label])
                                    / other
                                )
                                tab = meta.assign(
                                    subclass=label + 1,
                                    cells=n,
                                    margin_contribution=sums[label] / n,
                                    observed_feature_difference=observed,
                                )
                                table.append(
                                    tab.query("margin_contribution>0").nlargest(
                                        40, "margin_contribution"
                                    )
                                )
                        if table:
                            pd.concat(table).to_csv(
                                dest / f"{mode}_full_decision_channels.tsv",
                                sep="\t",
                                index=False,
                            )
                    log(f"{context} {mode}: assigned all {len(rows):,} cells")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contexts", nargs="*", default=["C4", "C5", "I0_5"])
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--tail", action="store_true")
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--mode", choices=("load", "contrast"))
    args = parser.parse_args()
    if args.patch and (not args.exact or args.full):
        parser.error("--patch requires --exact and runs on pilot tiles")
    run(args)
