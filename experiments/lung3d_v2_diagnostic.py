"""Post-fit checks with published mechanisms. These labels never enter fitting."""

import anndata as ad, numpy as np, pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score
from lung3d_v2 import OUT, DATA, CFG
from lung3d_v2_fast import FastWindows
from phenoniche.spatial.channels import Channels

a = ad.read_h5ad(DATA / "preprocessed_objects/preprocessed_objects/cosmx.h5ad")
x = np.load(OUT / "geometry_v1.npz")
c = x["coarse"]
g = sparse.load_npz(OUT / "contacts.npz")
names, ty = np.unique(a.obs.celltypes, return_inverse=True)
ch = Channels(
    a.layers["counts"],
    a.var_names,
    pd.read_csv("data/commuspace_human_lr_atlas.tsv", sep="\t"),
)
kernels = [
    FastWindows(g, c, ty, ch, CFG.sigma_um),
    FastWindows(g, np.zeros_like(c), ty, ch, CFG.sigma_um, pooled=True),
]
emt = a.obs.EMT_niche.astype(str).str.lower().eq("inside emt niche").to_numpy()
dc = a.obs.niches_3D.astype(str).eq("DC islands").to_numpy()
print("Author reference coarse class fractions")
for target in [emt, dc]:
    print(pd.Series(c[target]).value_counts(normalize=True).head(8).to_dict())
rng = np.random.default_rng(144)
panels = {
    "EMT": (
        emt,
        ["Macrophages", "Fibroblasts"],
        ["Tumor cells"],
        {"SPP1", "FN1", "THBS2"},
        {"ITGB1", "ITGB4", "ITGB6"},
    ),
    "DC": (
        dc,
        ["Dendritic cells"],
        ["Cytotoxic T cells", "Regulatory T cells"],
        {"CD274", "LGALS9", "CD80"},
        {"PDCD1", "HAVCR2", "CTLA4"},
    ),
}
diagnosis = []
for target, (truth, send, recv, lig, rec) in panels.items():
    lr = np.array(
        [
            i
            for i, (l, r) in enumerate(ch.names)
            if l in lig and bool(rec.intersection(r.split("+")))
        ]
    )
    print(target, "covered", [ch.names[i] for i in lr])
    # Compare with same coarse assignments, sampled after model fit for diagnosis.
    positive = rng.choice(np.flatnonzero(truth), min(1000, truth.sum()), replace=False)
    pools = {k: np.flatnonzero(~truth & (c == k)) for k in np.unique(c[positive])}
    negative = np.array([rng.choice(pools[c[i]]) for i in positive])
    for mode, kernel in zip(["strict", "pooled"], kernels):
        score = []
        measured = []
        for i in np.r_[positive, negative]:
            s, r, v, o = kernel.fibers(i, (c[i], c[i]))
            ok = np.isin(names[s], send) & np.isin(names[r], recv)
            measured.append(bool(ok.any()))
            score.append(float(v[ok][:, lr].mean()) if ok.any() else 0.0)
        label = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
        valid = np.array(measured)
        diagnosis.append(
            dict(
                target=target,
                mode=mode,
                n_pairs=len(positive),
                measured_positive=valid[: len(positive)].mean(),
                measured_negative=valid[len(positive) :].mean(),
                observed_AUROC=roc_auc_score(label[valid], np.array(score)[valid])
                if len(np.unique(label[valid])) == 2
                else np.nan,
            )
        )
p = pd.DataFrame(diagnosis)
print(p.to_string(index=False))
p.to_csv(OUT / "postfit_diagnostic.tsv", sep="\t", index=False)
