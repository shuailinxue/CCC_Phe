"""Post-fit comparison of full 3D and section-only communication on fixed contexts."""

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from lung3d_v2 import DATA, OUT


def direction_agnostic_auc(y, x):
    if min(y.sum(), (~y).sum()) < 10:
        return np.nan
    auc = roc_auc_score(y, x)
    return max(auc, 1 - auc)


def compare():
    data = ad.read_h5ad(
        DATA / "preprocessed_objects/preprocessed_objects/cosmx.h5ad", backed="r"
    )
    obs = data.obs
    pseudo = pd.to_numeric(obs.tumor_pseudotime_rank, errors="coerce")
    surface = (
        obs.niches_3D.eq("Tumor surface")
        & obs.celltypes.eq("Tumor cells")
        & pseudo.notna()
    )
    focus = obs.EMT_niche.eq("Inside EMT niche")
    dc = obs.niches_3D.eq("DC islands")
    high = (surface & focus & pseudo.ge(0.75)).to_numpy()
    low = (surface & ~focus & pseudo.lt(0.75)).to_numpy()
    pseudo_values, surface_values, dc_values = (
        pseudo.to_numpy(),
        surface.to_numpy(),
        dc.to_numpy(),
    )
    summaries = []
    for variant in ("communication_subclasses", "revised_3d", "revised_2d"):
        directory = OUT / variant
        fit = directory / "fit"
        if not fit.exists():
            continue
        cells = (
            pd.read_csv(fit / "cells.tsv", sep="\t")
            .set_index("cell_id")
            .loc[data.obs_names]
        )
        evidence = pd.read_csv(fit / "channels.tsv", sep="\t")
        for label, group in cells.groupby("niche"):
            idx = group.index
            target = surface.loc[idx]
            t = pseudo.loc[idx][target]
            peer = (
                surface
                & cells.context.eq(group.context.iloc[0])
                & cells.niche.ne(label)
            )
            ev = evidence[
                evidence.niche.eq(label) & evidence.subclass_minus_peers.gt(0)
            ]
            integrin = (
                ev.receiver.eq("Tumor cells")
                & ev.sender.isin(["Macrophages", "Fibroblasts"])
                & ev.ligand.isin(["SPP1", "FN1", "THBS2", "COL1A2", "COL6A3"])
                & ev.receptor.str.contains(r"ITGB[146]", na=False)
            )
            checkpoint = ev.sender.eq("Dendritic cells") & (
                ev.ligand + "-" + ev.receptor
            ).isin(["CD274-PDCD1", "CD80-CTLA4", "LGALS9-HAVCR2"])
            summaries.append(
                dict(
                    variant=variant,
                    niche=label,
                    context=group.context.iloc[0],
                    subclass=int(group.subclass.iloc[0]),
                    cells=len(group),
                    surface_tumor=len(t),
                    surface_emt_high=int((t >= 0.75).sum()),
                    surface_emt_high_in_roi=int((t[focus.loc[t.index]] >= 0.75).sum()),
                    surface_pseudotime_median=t.median() if len(t) else np.nan,
                    surface_peer_median=pseudo[peer].median() if peer.any() else np.nan,
                    dc=int(dc.loc[idx].sum()),
                    integrin_channels=int(integrin.sum()),
                    dc_checkpoint_channels=int(checkpoint.sum()),
                )
            )
        rows = []
        for path in fit.glob("programs_*.npz"):
            with np.load(path) as pack:
                ix, activity = pack["cell_rows"], pack["activity"]
            for program, v in enumerate(activity.T):
                valid = np.isfinite(v)
                mt = valid & surface_values[ix]
                dc_here = dc_values[ix][valid]
                correlation = (
                    spearmanr(v[mt], pseudo_values[ix][mt]).statistic
                    if mt.sum() >= 30 and np.ptp(v[mt]) > 0
                    else np.nan
                )
                ap = (
                    max(
                        average_precision_score(dc_here, sign * v[valid])
                        for sign in (1, -1)
                    )
                    if 20 <= dc_here.sum() < len(dc_here)
                    else np.nan
                )
                emt_mask = valid & (high[ix] | low[ix])
                emt_labels = high[ix][emt_mask]
                emt_auc = direction_agnostic_auc(emt_labels, v[emt_mask])
                dc_auc = direction_agnostic_auc(dc_here, v[valid])
                rows.append(
                    (
                        path.stem,
                        program,
                        int(mt.sum()),
                        correlation,
                        int(emt_labels.sum()),
                        emt_auc,
                        int(dc_here.sum()),
                        dc_here.mean(),
                        ap,
                        dc_auc,
                    )
                )
        pd.DataFrame(
            rows,
            columns="context program surface_tumor surface_spearman emt_high_cells emt_best_direction_AUROC dc_cells dc_prevalence dc_best_direction_AP dc_best_direction_AUROC".split(),
        ).to_csv(directory / "program_evaluation.tsv", sep="\t", index=False)
    output = pd.DataFrame(summaries)
    output.to_csv(OUT / "revised_comparison.tsv", sep="\t", index=False)
    return output


if __name__ == "__main__":
    print(
        compare()
        .query("subclass > 0")
        .sort_values(["variant", "surface_emt_high_in_roi"], ascending=[True, False])
        .groupby("variant")
        .head(8)
        .to_string(index=False)
    )
