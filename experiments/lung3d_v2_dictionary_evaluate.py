"""Evaluate learned communication groupings against published lung observations."""

import anndata as ad
import numpy as np
import pandas as pd

from lung3d_v2 import DATA, OUT


def run():
    a = ad.read_h5ad(
        DATA / "preprocessed_objects/preprocessed_objects/cosmx.h5ad", backed="r"
    )
    obs = a.obs
    box = obs.EMT_niche.eq("Inside EMT niche")
    pseudo = pd.to_numeric(obs.tumor_pseudotime_rank, errors="coerce")
    tumor = (
        obs.celltypes.eq("Tumor cells")
        & obs.niches_3D.eq("Tumor surface")
        & pseudo.notna()
    )
    high = tumor & box & pseudo.ge(0.75)
    dc = obs.niches_3D.eq("DC islands")
    root = OUT / "directed_dictionary"
    results = []
    for f in sorted(root.glob("**/*_cells.tsv")):
        mode, coverage, _ = f.stem.split("_")
        flavor = (
            f.parent.parent.name
            if f.parent.parent.name in {"exact", "tail", "patch", "tree"}
            else "projection_nnls"
        )
        index = pd.read_csv(f, sep="\t").set_index("cell_id")
        assert index.index.is_unique
        n_target_high = high[index.index].sum()
        n_target_dc = dc[index.index].sum()
        if flavor in {"exact", "tail", "patch", "tree"}:
            chan = (
                f.parent
                / f"{mode}_{'full_' if coverage == 'full' else ''}decision_channels.tsv"
            )
        else:
            chan = f.parent / f"{mode}_subclass_channels.tsv"
        channels = pd.read_csv(chan, sep="\t") if chan.exists() else None
        for label, g in index.groupby("subclass"):
            ids = g.index
            emt_count = high[ids].sum()
            dc_count = dc[ids].sum()
            at = tumor[ids]
            rest = index.index.difference(ids)
            base = tumor[rest]
            per_slice = pd.crosstab(
                obs.section[index.index], high[index.index]
            ).reindex(columns=[False, True], fill_value=0)[True]
            mapped = pd.crosstab(obs.section[ids], high[ids]).reindex(
                columns=[False, True], fill_value=0
            )[True]
            evidence_emt = evidence_dc = 0
            if channels is not None:
                ch = channels[channels.subclass.eq(label)]
                ch = ch[~ch.signal.eq("below_baseline")]
                evidence_emt = (
                    ch.sender.isin(["Macrophages", "Fibroblasts"])
                    & ch.receiver.eq("Tumor cells")
                    & ch.ligand.isin(["SPP1", "FN1", "THBS2", "COL1A2", "COL6A3"])
                    & ch.receptor.str.contains(r"ITGB[146]", na=False)
                ).sum()
                evidence_dc = (
                    ch.sender.eq("Dendritic cells")
                    & ch.receiver.isin(["Cytotoxic T cells", "Regulatory T cells"])
                    & (ch.ligand + "-" + ch.receptor).isin(
                        ["CD274-PDCD1", "CD80-CTLA4", "LGALS9-HAVCR2"]
                    )
                ).sum()
            results.append(
                (
                    flavor,
                    f.parent.name,
                    mode,
                    coverage,
                    label,
                    len(ids),
                    int(at.sum()),
                    float(pseudo[ids][at].median()),
                    float(pseudo[rest][base].median()),
                    int(emt_count),
                    emt_count / max(int(n_target_high), 1),
                    emt_count / max(int(at.sum()), 1),
                    int((mapped[per_slice > 0] > 0).sum()),
                    int(dc_count),
                    dc_count / max(int(n_target_dc), 1),
                    dc_count / len(ids),
                    int(evidence_emt),
                    int(evidence_dc),
                )
            )
    table = pd.DataFrame(
        results,
        columns=(
            "flavor context mode coverage subclass cells surface_tumor median_emt peer_median_emt "
            "high_emt high_emt_recall_in_context high_emt_fraction_of_surface slices_high_emt dc dc_recall_in_context "
            "dc_fraction integrin_decision_features checkpoint_decision_features"
        ).split(),
    )
    table.to_csv(root / "evaluation.tsv", sep="\t", index=False)
    print(table.round(3).to_string(index=False))
    return table


def export_tree_partition():
    """Export the predeclared load variant over the three fitted contexts."""
    root = OUT / "directed_dictionary"
    base = pd.read_csv(OUT / "revised_3d/fit/cells.tsv", sep="\t").set_index("cell_id")
    base["dictionary_niche"] = base.niche
    paths = [
        (f"C4.{k}", root / "tree" / f"C4.{k}" / "load_full_cells.tsv") for k in (1, 2)
    ]
    paths += [
        (context, root / "exact" / context / "load_full_cells.tsv")
        for context in ("C5", "I0_5")
    ]
    if not all(path.exists() for _, path in paths):
        return None
    assigned = pd.Index([])
    for prefix, path in paths:
        values = pd.read_csv(path, sep="\t").set_index("cell_id").subclass
        if not values.index.is_unique or len(assigned.intersection(values.index)):
            raise ValueError("Overlapping context assignments")
        if not base.loc[values.index, "context"].eq(prefix.split(".")[0]).all():
            raise ValueError("Context mismatch in dictionary assignment")
        assigned = assigned.append(values.index)
        base.loc[values.index, "dictionary_niche"] = prefix + "." + values.astype(str)
    expected = base.index[base.context.isin(["C4", "C5", "I0_5"])]
    if not assigned.symmetric_difference(expected).empty:
        raise ValueError("Incomplete experimental assignments")
    base[["coarse", "context", "niche", "dictionary_niche"]].to_csv(
        root / "tree_load_full_tissue.tsv", sep="\t"
    )
    print(
        f"Exploratory full-tissue map: {len(base):,} cells, {base.dictionary_niche.nunique()} labels"
    )
    return base


if __name__ == "__main__":
    run()
    export_tree_partition()
