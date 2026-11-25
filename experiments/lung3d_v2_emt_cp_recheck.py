"""Reconstruct the C4 CP refits and compare program reproducibility."""

import json, numpy as np, pandas as pd, anndata as ad
from scipy.sparse import load_npz
from sklearn.metrics import roc_auc_score
from lung3d_v2 import DATA, OUT, GRAPH, ROOT
from lung3d_v2_fast import FastWindows
from phenoniche.spatial import Settings
from phenoniche.spatial.channels import Channels
from phenoniche.spatial.programs import (
    fit_programs,
    match_programs,
    project,
    sample_blocks,
)

a = ad.read_h5ad(DATA / "preprocessed_objects/preprocessed_objects/cosmx.h5ad")
c = pd.read_csv(OUT / "revised_3d/fit/cells.tsv", sep="\t", index_col="cell_id").loc[
    a.obs_names
]
cfg = Settings(**json.loads((OUT / "revised_3d/settings.json").read_text()))
coarse = c.coarse.to_numpy()
key = (4, 4)
ids = np.flatnonzero(c.context.eq("C4").to_numpy())
xyz = np.load(GRAPH)["xyz"]
block = np.unique(np.floor(xyz / cfg.block_um), axis=0, return_inverse=True)[1]
g = load_npz(OUT / "contacts.npz")
names, ty = np.unique(c.cell_type, return_inverse=True)
atlas = pd.read_csv(ROOT / "data/commuspace_human_lr_atlas.tsv", sep="\t")
ch = Channels(a.layers["counts"], a.var_names, atlas)
t = FastWindows(g, coarse, ty, ch, cfg.sigma_um, full_interface=True)
rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, *key]))
weights = None
models = []
for n in range(cfg.repeats):
    sample = sample_blocks(ids, block, cfg, rng)
    data = [t.fibers(i, key) for i in sample]
    if weights is None:
        rms = np.sqrt(np.mean(np.square(np.concatenate([f[2] for f in data])), axis=0))
        typical = np.median(rms[rms > 0])
        weights = np.sqrt(np.clip(typical / np.maximum(rms, typical / 9), 1 / 9, 9))
    models.append(
        fit_programs(
            [(s, r, y * weights, o) for s, r, y, o in data],
            t.n_types,
            cfg,
            rng,
            t.n_senders,
        )
    )
for i, model in enumerate(models):
    if i:
        lhs, rhs, sim = match_programs(models[0], model)
        print(
            "loading similarity",
            i + 1,
            list(zip(lhs.tolist(), rhs.tolist(), sim.round(3).tolist())),
            flush=True,
        )
p = a.obs.tumor_pseudotime_rank.astype(float).to_numpy()
roi = a.obs.EMT_niche.eq("Inside EMT niche").to_numpy()
surf = (
    a.obs.niches_3D.eq("Tumor surface") & a.obs.celltypes.eq("Tumor cells")
).to_numpy()
pos = ids[surf[ids] & roi[ids] & (p[ids] >= 0.75)]
neg = ids[surf[ids] & ~roi[ids] & (p[ids] < 0.75)]
check = np.r_[rng.choice(pos, 240, replace=False), rng.choice(neg, 240, replace=False)]
y = np.arange(len(check)) < 240
scores = []
for i in check:
    s, r, v, o = t.fibers(i, key)
    scores.append(np.array([project((s, r, v * weights, o), m) for m in models]))
scores = np.array(scores)
with np.load(OUT / "revised_3d/fit/programs_4_4.npz") as data:
    saved = data["activity"][pd.Index(data["cell_rows"]).get_indexer(check)]
    print(
        "saved first fit max diff",
        float(np.nanmax(abs(scores[:, 0] - saved))),
        flush=True,
    )
for i in range(len(models)):
    print(
        "model AUC",
        i + 1,
        [
            round(max(q, 1 - q), 3)
            for q in [
                roc_auc_score(y[np.isfinite(x)], x[np.isfinite(x)])
                for x in scores[:, i, :].T
            ]
        ],
        flush=True,
    )
