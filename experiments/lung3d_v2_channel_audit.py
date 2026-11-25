import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score
from dataclasses import replace
from lung3d_v2 import (
    DATA,
    OUT,
    GRAPH,
    ROOT,
    CFG,
    merge_composition_classes,
    restrict_to_sections,
)
from lung3d_v2_fast import FastWindows
from phenoniche.spatial.channels import Channels
from phenoniche.spatial.subclasses import partition_contexts

a = ad.read_h5ad(DATA / "preprocessed_objects/preprocessed_objects/cosmx.h5ad")
f = np.load(GRAPH)
coarse = merge_composition_classes(f["coarse"], f["composition"])
xyz = f["xyz"]
graph = sparse.load_npz(OUT / "contacts.npz")
cfg = replace(CFG, min_cells=30, interface_fraction=0.02)
groups, _ = partition_contexts(graph, coarse, cfg, flank_fraction=0.35)
names, types = np.unique(a.obs.celltypes, return_inverse=True)
channels = Channels(
    a.layers["counts"],
    a.var_names,
    pd.read_csv(ROOT / "data/commuspace_human_lr_atlas.tsv", sep="\t"),
)
p = pd.to_numeric(a.obs.tumor_pseudotime_rank, errors="coerce").to_numpy()
tum = (
    a.obs.celltypes.eq("Tumor cells").to_numpy()
    & a.obs.niches_3D.eq("Tumor surface").to_numpy()
    & np.isfinite(p)
)
roi = a.obs.EMT_niche.eq("Inside EMT niche").to_numpy()
dc = a.obs.niches_3D.eq("DC islands").to_numpy()
targets = {
    "EMT": (tum & roi & (p >= 0.75), tum & ~roi & (p < 0.75)),
    "DC": (dc, ~dc & a.obs.niches_3D.ne("Excluded").to_numpy()),
}
panels = {
    "EMT": (
        ["Fibroblasts", "Macrophages"],
        ["Tumor cells"],
        lambda l, r: (
            l in ["FN1", "SPP1", "THBS2", "COL1A2", "COL6A3"]
            and any(g in r.split("+") for g in ["ITGB1", "ITGB4", "ITGB6"])
        ),
    ),
    "DC": (
        ["Dendritic cells"],
        ["Regulatory T cells", "Cytotoxic T cells"],
        lambda l, r: (
            (l, r) in [("CD274", "PDCD1"), ("CD80", "CTLA4"), ("LGALS9", "HAVCR2")]
        ),
    ),
}
rows = []
matched = {}
for mode, g in [("3D", graph), ("2D", restrict_to_sections(graph, xyz[:, 2]))]:
    k = FastWindows(g, coarse, types, channels, 20, full_interface=True)
    for target, (positive, negative) in targets.items():
        sn, rn, lr_rule = panels[target]
        lr = [i for i, (l, r) in enumerate(channels.names) if lr_rule(l, r)]
        for key, ids in sorted(groups.items(), key=lambda kv: -positive[kv[1]].sum())[
            :3
        ]:
            pos = ids[positive[ids]]
            neg = ids[negative[ids]]
            n = min(len(pos), len(neg), 240)
            if n < 30:
                continue
            rr = np.random.default_rng(np.random.SeedSequence([2027, *key]))
            ids = np.r_[
                rr.choice(pos, n, replace=False), rr.choice(neg, n, replace=False)
            ]
            v = np.full(2 * n, np.nan)
            for ii, idx in enumerate(ids):
                s, r, y, o = k.fibers(idx, key)
                selects = np.isin(names[s % len(names)], sn) & np.isin(names[r], rn)
                if selects.any():
                    v[ii] = np.average(
                        y[selects][:, lr].mean(axis=1), weights=o[selects]
                    )
            ok = np.isfinite(v)
            y = np.arange(2 * n) < n
            score = (
                roc_auc_score(y[ok], v[ok]) if len(np.unique(y[ok])) == 2 else np.nan
            )
            key = tuple(map(int, key))
            matched.setdefault((target, key), {})[mode] = (y, v)
            rows.append(
                (mode, target, str(key), n, ok[:n].mean(), ok[n:].mean(), score)
            )
            print(rows[-1], flush=True)
pd.DataFrame(
    rows,
    columns="geometry target context sample_per_group observed_positive observed_negative LR_AUROC".split(),
).to_csv(OUT / "revised_raw_channel_audit.tsv", sep="\t", index=False)

comparison = []
for (target, key), runs in matched.items():
    y, v3 = runs["3D"]
    y2, v2 = runs["2D"]
    np.testing.assert_array_equal(y, y2)
    both = np.isfinite(v2) & np.isfinite(v3)
    if len(np.unique(y[both])) == 2:
        comparison.append(
            (
                target,
                str(key),
                int(both.sum()),
                roc_auc_score(y[both], v2[both]),
                roc_auc_score(y[both], v3[both]),
            )
        )
pd.DataFrame(
    comparison, columns=["target", "context", "paired_windows", "AUROC_2D", "AUROC_3D"]
).to_csv(OUT / "revised_raw_channel_paired.tsv", sep="\t", index=False)
