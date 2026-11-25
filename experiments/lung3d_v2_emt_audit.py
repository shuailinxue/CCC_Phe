"""Audit composition, directed LR and CP on the same EMT comparison cells."""

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lung3d_v2 import DATA, GRAPH, OUT, ROOT
from lung3d_v2_fast import FastWindows
from phenoniche.spatial.channels import Channels


def verify_pair_calculation(tensor, center, xyz, coarse, types, expr, channels, lr):
    """Independent direct enumeration of real directed cell pairs."""
    distances = np.linalg.norm(xyz - xyz[center], axis=1)
    members = np.flatnonzero((distances <= 50) & (coarse == 4))
    pair_distance = np.linalg.norm(xyz[members, None] - xyz[members][None, :], axis=2)
    left, right = np.nonzero((pair_distance <= 50) & ~np.eye(len(members), dtype=bool))
    radial = np.exp(-0.5 * (distances[members] / 20) ** 2)
    w = (
        radial[left]
        * radial[right]
        * np.exp(-0.5 * (pair_distance[left, right] / 20) ** 2)
    )
    i, j = members[left], members[right]
    sender, receiver, values, opportunity = tensor.fibers(center, (4, 4))
    for row, (s, r) in enumerate(zip(sender, receiver)):
        mask = (types[i] == s) & (types[j] == r)
        np.testing.assert_allclose(opportunity[row], w[mask].sum(), rtol=2e-5)
        observed = (
            expr[i[mask, None], channels.left[lr]]
            * expr[j[mask, None], channels.right[lr]]
        )
        expected = (w[mask, None] * observed).sum(axis=0) / w[mask].sum()
        np.testing.assert_allclose(values[row, lr], expected, rtol=3e-5, atol=1e-5)


def audit():
    a = ad.read_h5ad(DATA / "preprocessed_objects/preprocessed_objects/cosmx.h5ad")
    folder = OUT / "revised_3d/fit"
    cells = pd.read_csv(folder / "cells.tsv", sep="\t", index_col="cell_id").loc[
        a.obs_names
    ]
    coarse = cells.coarse.to_numpy()
    geo = np.load(GRAPH)
    xyz, composition = geo["xyz"], geo["composition"]
    names, types = np.unique(cells.cell_type, return_inverse=True)
    roi = a.obs.EMT_niche.eq("Inside EMT niche").to_numpy()
    p = pd.to_numeric(a.obs.tumor_pseudotime_rank, errors="coerce").to_numpy()
    surface = (
        a.obs.niches_3D.eq("Tumor surface") & a.obs.celltypes.eq("Tumor cells")
    ).to_numpy()
    inner = cells.context.eq("C4").to_numpy()
    core = surface & inner & np.isfinite(p)
    actual = pd.crosstab(
        cells.loc[coarse == 4, "cell_type"], roi[coarse == 4], normalize="columns"
    )
    actual.to_csv(OUT / "emt_c4_actual_composition.tsv", sep="\t")
    candidates = []
    rng = np.random.default_rng(108)
    # All comparisons use C4 interior surface tumor centers. Pair the sample counts per section.
    for z in np.unique(xyz[:, 2]):
        pos = np.flatnonzero(core & roi & (p >= 0.75) & (xyz[:, 2] == z))
        neg = np.flatnonzero(core & ~roi & (p < 0.75) & (xyz[:, 2] == z))
        n = min(110, len(pos), len(neg))
        if n >= 15:
            candidates.extend([(i, True) for i in rng.choice(pos, n, replace=False)])
            candidates.extend([(i, False) for i in rng.choice(neg, n, replace=False)])
    ids = np.array([i for i, _ in candidates])
    y = np.array([t for _, t in candidates])
    layer = xyz[ids, 2]
    pd.DataFrame(composition[ids], columns=names).groupby(y).mean().T.to_csv(
        OUT / "emt_c4_window_composition.tsv", sep="\t"
    )
    contact = load_npz(OUT / "contacts.npz")
    atlas = pd.read_csv(ROOT / "data/commuspace_human_lr_atlas.tsv", sep="\t")
    ch = Channels(a.layers["counts"], a.var_names, atlas)
    strict = FastWindows(contact, coarse, types, ch, 20, full_interface=True)
    full = FastWindows(
        contact, np.zeros(len(coarse), dtype=int), types, ch, 20, full_interface=True
    )
    integrins = {"ITGB1", "ITGB4", "ITGB6"}
    lr = np.array(
        [
            i
            for i, (l, r) in enumerate(ch.names)
            if l in {"SPP1", "FN1", "THBS2", "COL1A2", "COL6A3"}
            and integrins.intersection(r.split("+"))
        ]
    )
    typ = {name: i for i, name in enumerate(names)}
    receiver = typ["Tumor cells"]
    sender = [typ["Fibroblasts"], typ["Macrophages"]]
    all_features = {}
    diagnostics = []
    expr = ch.expression.toarray()
    for center in np.r_[ids[y][:4], ids[~y][:4]]:
        verify_pair_calculation(strict, center, xyz, coarse, types, expr, ch, lr)
    print(
        "Direct directed-edge enumeration verified in eight real windows.", flush=True
    )
    for mode, tensor, key in [
        ("model_window", strict, (4, 4)),
        ("full_window", full, (0, 0)),
    ]:
        values, burden, opportunity = [], [], []
        for center in ids:
            s, r, v, o = tensor.fibers(center, key)
            keep = np.isin(s, sender) & (r == receiver)
            total = np.sum(o[keep])
            mass = (o[keep, None] * v[keep][:, lr]).sum(axis=0)
            values.append(mass / total if total else np.zeros(len(lr)))
            burden.append(mass)
            opportunity.append(total)
        values, burden = np.asarray(values), np.asarray(burden)
        all_features[f"{mode}_mean"] = values
        all_features[f"{mode}_total"] = burden
        diagnostics.append(
            (
                mode,
                float(np.median(np.array(opportunity)[y])),
                float(np.median(np.array(opportunity)[~y])),
            )
        )
    receiver_scores = []
    for center in ids:
        start, stop = contact.indptr[center : center + 2]
        nbr = contact.indices[start:stop]
        keep = np.isin(types[nbr], sender)
        nbr = nbr[keep]
        w = np.exp(-0.5 * (contact.data[start:stop][keep] / 20) ** 2)
        receiver_scores.append(
            (w @ expr[nbr][:, ch.left[lr]]) * expr[center, ch.right[lr]]
        )
    all_features["centered_receiver_total"] = np.array(receiver_scores)
    with np.load(folder / "programs_4_4.npz") as k:
        lookup = pd.Index(k["cell_rows"]).get_indexer(ids)
        assert np.all(lookup >= 0)
        all_features["saved_CP"] = np.nan_to_num(k["activity"][lookup])
    all_features["composition"] = composition[ids]
    all_features["composition_plus_LR"] = np.c_[
        composition[ids], all_features["model_window_mean"]
    ]
    all_features["composition_plus_CP"] = np.c_[
        composition[ids], all_features["saved_CP"]
    ]
    rows = []
    for label, x in all_features.items():
        # Leave each section out. This is retrospective discrimination, not a model training target.
        pred = cross_val_predict(
            make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500)),
            np.log1p(x),
            y,
            groups=layer,
            cv=LeaveOneGroupOut(),
            method="predict_proba",
        )[:, 1]
        offset = (np.mean(x[y], axis=0) + 1e-8) / (np.mean(x[~y], axis=0) + 1e-8)
        rows.append(
            (
                label,
                len(ids),
                roc_auc_score(y, pred),
                roc_auc_score(y, x.mean(axis=1)),
                float(np.median(offset)),
            )
        )
    report = pd.DataFrame(
        rows,
        columns=[
            "features",
            "cells",
            "heldout_section_AUROC",
            "mean_score_AUROC",
            "median_channel_mean_ratio",
        ],
    )
    channels_table = pd.DataFrame(
        [
            (
                ch.names[k][0],
                ch.names[k][1],
                before,
                after,
                float(roc_auc_score(y, vals)),
            )
            for k, before, after, vals in zip(
                lr,
                all_features["model_window_mean"][~y].mean(axis=0),
                all_features["model_window_mean"][y].mean(axis=0),
                all_features["model_window_mean"].T,
            )
        ],
        columns=["ligand", "receptor", "control_mean", "EMT_mean", "AUROC"],
    )
    channels_table.to_csv(OUT / "emt_c4_per_channel.tsv", sep="\t", index=False)
    report.to_csv(OUT / "emt_c4_audit.tsv", sep="\t", index=False)
    counts = pd.DataFrame(
        diagnostics,
        columns=["mode", "median_opportunity_high", "median_opportunity_control"],
    )
    counts.to_csv(OUT / "emt_c4_opportunity.tsv", sep="\t", index=False)
    print(actual.T.to_string())
    print(report.round(3).to_string(index=False))
    print(counts.round(3).to_string(index=False))


if __name__ == "__main__":
    audit()
