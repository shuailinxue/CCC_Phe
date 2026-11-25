"""Target-blind CP rank scan and an EMT audit performed afterward."""

from dataclasses import replace
from multiprocessing import get_context

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lung3d_v2 import DATA, GRAPH, OUT, ROOT, CFG
from lung3d_v2_fast import FastWindows
from phenoniche.spatial.channels import Channels
from phenoniche.spatial.programs import fit_programs, match_programs, project

RANKS = (3, 5, 8, 12, 16)


def probe(x, labels, layers):
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=600))
    prediction = cross_val_predict(
        pipeline,
        np.log1p(np.nan_to_num(x)),
        labels,
        groups=layers,
        cv=LeaveOneGroupOut(),
        method="predict_proba",
    )[:, 1]
    return roc_auc_score(labels, prediction)


def run_rank(rank):
    cfg = replace(CFG, rank=rank, iterations=40)
    models = [
        fit_programs(
            features,
            TENSOR.n_types,
            cfg,
            np.random.default_rng(512 + rep),
            TENSOR.n_senders,
        )
        for rep, features in enumerate(TRAIN)
    ]
    matches = [match_programs(models[0], m)[2] for m in models[1:]]
    ratios, emt_scores = [], []
    for model in models:
        s, r, h = model
        support, withheld = LR_PART
        masked = (s, r, h[support])
        error, energy = 0.0, 0.0
        for send, recv, value, opp in VALID:
            if not len(send):
                continue
            activity = project((send, recv, value[:, support], opp), masked)
            predicted = ((s[send] * r[recv]) * np.nan_to_num(activity)) @ h[withheld].T
            target = value[:, withheld]
            error += np.square(target - predicted).sum()
            energy += np.square(target).sum()
        ratios.append(np.sqrt(error / energy))
        emt_scores.append(
            probe(np.array([project(f, model) for f in AUDIT]), LABELS, LAYERS)
        )
    return (
        rank,
        *ratios,
        float(np.mean(ratios)),
        float(np.median(matches)),
        float((np.array(matches) >= 0.7).mean()),
        *emt_scores,
        float(np.mean(emt_scores)),
    )


def main():
    global TENSOR, TRAIN, VALID, AUDIT, LABELS, LAYERS, LR_PART
    data = ad.read_h5ad(DATA / "preprocessed_objects/preprocessed_objects/cosmx.h5ad")
    cells = pd.read_csv(
        OUT / "revised_3d/fit/cells.tsv", sep="\t", index_col="cell_id"
    ).loc[data.obs_names]
    xyz = np.load(GRAPH)["xyz"]
    coarse = cells.coarse.to_numpy()
    names, types = np.unique(cells.cell_type, return_inverse=True)
    ch = Channels(
        data.layers["counts"],
        data.var_names,
        pd.read_csv(ROOT / "data/commuspace_human_lr_atlas.tsv", sep="\t"),
    )
    TENSOR = FastWindows(
        load_npz(OUT / "contacts.npz"), coarse, types, ch, 20, full_interface=True
    )
    key = (4, 4)
    candidates = np.flatnonzero(cells.context.eq("C4"))
    blocks = np.floor(xyz[:, :2] / 150).astype(int)
    _, which = np.unique(blocks[candidates], axis=0, return_inverse=True)
    rng = np.random.default_rng(183)
    holdout = rng.choice(
        np.unique(which), max(1, int(0.2 * len(np.unique(which)))), replace=False
    )
    held = candidates[np.isin(which, holdout)]
    dev = candidates[~np.isin(which, holdout)]
    # Training and rank selection use cell positions and expression only.
    raw_train = [
        [TENSOR.fibers(i, key) for i in rng.choice(dev, 192, replace=False)]
        for _ in range(3)
    ]
    rms = np.sqrt(
        np.mean(np.square(np.concatenate([f[2] for f in raw_train[0]])), axis=0)
    )
    typical = np.median(rms[rms > 0])
    weights = np.sqrt(np.clip(typical / np.maximum(rms, typical / 9), 1 / 9, 9))

    def scale(f):
        s, r, y, o = f
        return s, r, y * weights, o

    TRAIN = [[scale(f) for f in group] for group in raw_train]
    del raw_train
    VALID = [scale(TENSOR.fibers(i, key)) for i in rng.choice(held, 256, replace=False)]
    perm = rng.permutation(len(ch.names))
    k = int(0.8 * len(perm))
    LR_PART = np.sort(perm[:k]), np.sort(perm[k:])
    # Author fields enter only the diagnostic AUROC below.
    p = pd.to_numeric(data.obs.tumor_pseudotime_rank, errors="coerce").to_numpy()
    roi = data.obs.EMT_niche.eq("Inside EMT niche").to_numpy()
    surf = (
        data.obs.niches_3D.eq("Tumor surface") & data.obs.celltypes.eq("Tumor cells")
    ).to_numpy()
    eligible = cells.context.eq("C4").to_numpy() & surf
    ids = []
    labels = []
    check_rng = np.random.default_rng(108)
    for z in np.unique(xyz[:, 2]):
        pos = np.flatnonzero(eligible & roi & (p >= 0.75) & (xyz[:, 2] == z))
        neg = np.flatnonzero(eligible & ~roi & (p < 0.75) & (xyz[:, 2] == z))
        n = min(110, len(pos), len(neg))
        if n < 15:
            continue
        ids.extend(check_rng.choice(pos, n, replace=False))
        ids.extend(check_rng.choice(neg, n, replace=False))
        labels.extend([True] * n + [False] * n)
    ids = np.array(ids)
    LABELS = np.array(labels)
    LAYERS = xyz[ids, 2]
    original = [TENSOR.fibers(i, key) for i in ids]
    AUDIT = [scale(f) for f in original]
    # Diagnostic using both broad sender types and every available LR channel.
    receiver = np.flatnonzero(names == "Tumor cells")[0]
    sender = np.flatnonzero(np.isin(names, ["Fibroblasts", "Macrophages"]))
    vectors = np.zeros((len(ids), len(sender), len(ch.names)), dtype=np.float32)
    for row, (s, r, value, _) in enumerate(original):
        for j, t in enumerate(sender):
            take = (s == t) & (r == receiver)
            if take.any():
                vectors[row, j] = value[take][0]
    print(
        "Full LR panel, these two sender types to tumor, held-out-section diagnostic AUROC",
        round(probe(vectors.reshape(len(ids), -1), LABELS, LAYERS), 4),
        flush=True,
    )
    rows = []
    with get_context("fork").Pool(len(RANKS)) as pool:
        for result in pool.imap_unordered(run_rank, RANKS):
            rows.append(result)
            print(result, flush=True)
    report = pd.DataFrame(
        rows,
        columns="rank nrmse1 nrmse2 nrmse3 mean_nrmse matched_cosine_median fraction_matched_ge_0_7 emt_auc1 emt_auc2 emt_auc3 emt_auc_mean".split(),
    ).sort_values("rank")
    report.to_csv(OUT / "emt_c4_rank_scan.tsv", sep="\t", index=False)
    print(report.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
