"""Composition-only clustering with comparator subgroup counts."""

import anndata as ad, pandas as pd, numpy as np
from sklearn.cluster import KMeans
from lung3d_v2 import DATA, OUT, GRAPH

a = ad.read_h5ad(
    DATA / "preprocessed_objects/preprocessed_objects/cosmx.h5ad", backed="r"
)
obs = a.obs
data = pd.read_csv(OUT / "revised_3d/fit/cells.tsv", sep="\t", index_col="cell_id").loc[
    a.obs_names
]
geo = np.load(GRAPH)
arrays = np.sqrt(geo["composition"])
dc = obs.niches_3D.eq("DC islands").to_numpy()
p = obs.tumor_pseudotime_rank.astype(float).to_numpy()
surf = (
    obs.niches_3D.eq("Tumor surface") & obs.celltypes.eq("Tumor cells")
).to_numpy() & np.isfinite(p)
high = obs.EMT_niche.eq("Inside EMT niche").to_numpy() & surf & (p >= 0.75)
rows = []
rng = np.random.default_rng(808)
for context in ["C4", "C5", "I0_5"]:
    ids = np.flatnonzero(data.context.eq(context).to_numpy())
    sampled = rng.choice(ids, min(len(ids), 3000), replace=False)
    for k in [2, 3, 4, 7]:
        y = (
            KMeans(n_clusters=k, random_state=37, n_init=10)
            .fit(arrays[sampled])
            .predict(arrays[ids])
        )
        for label in range(k):
            x = ids[y == label]
            rows.append(
                (
                    context,
                    k,
                    label + 1,
                    len(x),
                    int(dc[x].sum()),
                    dc[x].mean(),
                    int(high[x].sum()),
                    int(surf[x].sum()),
                    np.median(p[x[surf[x]]]) if surf[x].any() else np.nan,
                )
            )
r = pd.DataFrame(
    rows,
    columns="context k subclass cells dc dc_fraction high_emt surface_tumor median_emt".split(),
)
r.to_csv(OUT / "directed_dictionary/composition_only.tsv", sep="\t", index=False)
print(r.round(3).to_string(index=False))
