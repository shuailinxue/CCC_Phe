"""Measure how much reference signal is present in the fitted scores before region assignment."""

import pickle

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from lung3d_v2 import DATA, OUT

a = ad.read_h5ad(
    DATA / "preprocessed_objects/preprocessed_objects/cosmx.h5ad", backed="r"
)
targets = {
    "EMT": a.obs.EMT_niche.eq("Inside EMT niche").to_numpy(),
    "DC": a.obs.niches_3D.eq("DC islands").to_numpy(),
}


def check(fit):
    rows = []
    for f in fit.glob("programs_*.npz"):
        pack = np.load(f)
        ids = pack["cell_rows"]
        for name, mask in targets.items():
            for comp, v in enumerate(pack["activity"].T):
                ok = np.isfinite(v)
                y = mask[ids][ok]
                val = v[ok]
                if y.sum() < 30 or (~y).sum() < 30 or y.mean() > 0.95:
                    continue
                for direction in [1, -1]:
                    ap = average_precision_score(y, direction * val)
                    rows.append(
                        (
                            name,
                            f.stem,
                            comp,
                            direction,
                            len(y),
                            y.sum(),
                            y.sum() / mask.sum(),
                            y.mean(),
                            ap,
                            roc_auc_score(y, direction * val),
                            ap / y.mean(),
                        )
                    )
    return pd.DataFrame(
        rows,
        columns="target context program sign measured positives positive_coverage prevalence AP AUROC enrichment".split(),
    )


def audit_assignment(fit):
    counts = np.zeros(len(a), dtype=np.int32)
    for path in fit.parent.glob("ctx*.pkl"):
        with path.open("rb") as stream:
            proposals, _ = pickle.load(
                stream
            )  # Checkpoints written by this experiment.
        for ids, *_ in proposals:
            counts[ids] += 1
    labels = (
        pd.read_csv(fit / "cells.tsv", sep="\t").set_index("cell_id").loc[a.obs_names]
    )
    assigned = labels.niche.str.startswith("F").to_numpy()
    rows = [
        (
            name,
            int(mask.sum()),
            int((counts[mask] > 0).sum()),
            int((counts[mask] > 1).sum()),
            int(assigned[mask].sum()),
        )
        for name, mask in {"All": np.ones(len(a), dtype=bool), **targets}.items()
    ]
    return pd.DataFrame(
        rows,
        columns=["target", "reference", "candidate", "overlap_conflict", "assigned"],
    )


for name in ["initial", "pooled_windows", "scaled_channels"]:
    fit = OUT / name / "fit"
    if not fit.exists():
        continue
    audit_assignment(fit).to_csv(
        fit.parent / "assignment_gates.tsv", sep="\t", index=False
    )
    r = check(fit)
    r.to_csv(fit.parent / "postfit_program_checks.tsv", sep="\t", index=False)
    print(
        name,
        r[r.positive_coverage >= 0.15]
        .sort_values("AUROC", ascending=False)
        .groupby("target")
        .head(3)
        .round(3)
        .to_string(index=False),
    )
