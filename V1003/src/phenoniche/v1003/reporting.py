from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import default_config


def _display(value):
    from IPython.display import display
    display(value)


def _markdown(text):
    from IPython.display import Markdown, display
    display(Markdown(text))


def load_artifacts(config=None):
    config = default_config() if config is None else config
    root = Path(config.output_dir)
    required = ["summary.json", "assignments.csv.gz", "niche_counts.csv",
                "top_ccc.csv", "H.npy", "H_cosine_similarity.csv"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"V1003 output is incomplete; missing {missing}. Run ensure_results() first.")
    return {
        "config": config,
        "root": root,
        "summary": json.loads((root / "summary.json").read_text()),
        "assignments": pd.read_csv(root / "assignments.csv.gz"),
        "counts": pd.read_csv(root / "niche_counts.csv"),
        "top_ccc": pd.read_csv(root / "top_ccc.csv"),
        "H": np.load(root / "H.npy"),
        "H_cosine": pd.read_csv(root / "H_cosine_similarity.csv", index_col=0),
    }


def dataset_summary(artifacts):
    summary = artifacts["summary"]
    _display(pd.DataFrame([{
        "Dataset": summary["dataset"],
        "Cells / microenvironments": summary["n_cells"],
        "Retained directed CCC": summary["retained_ccc_features"],
        "K": summary["K"],
    }]))


def model_and_training(artifacts):
    config = artifacts["config"]
    training = artifacts["summary"]["training"]
    _markdown("**Model:** F → 256 → 64 → Softmax(8); nonnegative row-normalized linear decoder H.")
    _display(pd.DataFrame([{
        "Best epoch": training["best_epoch"],
        "Epochs completed": training["epochs_completed"],
        "Final train reconstruction": training["final_train_reconstruction"],
        "Final validation reconstruction": training["final_validation_reconstruction"],
        "Early stopped": training["early_stopped"],
        "Seed": config.seed,
    }]))


def overview_map(artifacts):
    """Side-by-side tissue overview matching the V1002 Prime 5K display style."""
    table = artifacts["assignments"]
    root = artifacts["root"]
    k = artifacts["config"].niches
    cell_types = sorted(table.cell_type.astype(str).unique())
    type_codes = pd.Categorical(table.cell_type.astype(str), categories=cell_types).codes
    type_colors = plt.get_cmap("tab20", len(cell_types))(np.arange(len(cell_types)))
    niche_colors = plt.get_cmap("tab10", k)(np.arange(k))
    niche = table.niche_label.to_numpy(dtype=np.int64)
    xlim = (table.x.min(), table.x.max())
    ylim = (table.y.max(), table.y.min())
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.4), layout="constrained")
    axes[0].scatter(table.x, table.y, c=type_colors[type_codes], s=.22,
                    linewidths=0, rasterized=True)
    axes[1].scatter(table.x, table.y, c=niche_colors[niche - 1], s=.22,
                    linewidths=0, rasterized=True)
    for axis, title in zip(axes, ("Provisional cell types", "V1003 CCC-autoencoder niches")):
        axis.set(xlim=xlim, ylim=ylim, xlabel="x (µm)", ylabel="y (µm)", title=title)
        axis.set_aspect("equal")
    type_handles = [plt.Line2D([0], [0], marker="o", color="none",
                               markerfacecolor=type_colors[index], markeredgewidth=0,
                               markersize=5, label=name)
                    for index, name in enumerate(cell_types)]
    niche_handles = [plt.Line2D([0], [0], marker="o", color="none",
                                markerfacecolor=niche_colors[index], markeredgewidth=0,
                                markersize=5, label=f"Niche {index + 1}")
                     for index in range(k)]
    axes[0].legend(handles=type_handles, loc="upper center", bbox_to_anchor=(.5, -.10),
                   ncol=min(4, len(cell_types)), frameon=False, fontsize=7)
    axes[1].legend(handles=niche_handles, loc="upper center", bbox_to_anchor=(.5, -.10),
                   ncol=4, frameon=False, fontsize=7)
    figure_dir = root / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / "celltype_vs_v1003_niche_overview.png", dpi=600,
                bbox_inches="tight")
    _display(fig)
    plt.close(fig)


def spatial_niches(artifacts):
    table = artifacts["assignments"]
    root = artifacts["root"]
    k = artifacts["config"].niches
    colors = plt.get_cmap("tab10", k)(np.arange(k))
    xlim = (table.x.min(), table.x.max())
    ylim = (table.y.max(), table.y.min())
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    for niche, axis in enumerate(axes.flat, 1):
        selected = table.niche_label.to_numpy() == niche
        axis.scatter(table.x, table.y, c="#dddddd", s=.12, linewidths=0, rasterized=True)
        axis.scatter(table.loc[selected, "x"], table.loc[selected, "y"],
                     color=colors[niche - 1], s=.30, linewidths=0, rasterized=True)
        axis.set(xlim=xlim, ylim=ylim, title=f"Niche {niche} (n={selected.sum():,})")
        axis.set_aspect("equal")
        axis.set_axis_off()
    fig.suptitle("V1003 CCC autoencoder niches", fontsize=14)
    figure_dir = root / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / "niche_spatial_distribution.png", dpi=300, bbox_inches="tight")
    _display(fig)
    plt.close(fig)


def niche_counts(artifacts):
    table = artifacts["counts"].copy()
    fig, axis = plt.subplots(figsize=(6.5, 3.2))
    axis.bar(table.niche.astype(str), table.fraction, color="#4c78a8")
    axis.set(xlabel="Niche", ylabel="Cell fraction", title="Niche assignment size")
    fig.tight_layout()
    _display(fig)
    plt.close(fig)
    _display(table)


def _local_composition(artifacts):
    cache = artifacts["root"] / "cache/local_composition.npy"
    table = artifacts["assignments"]
    cell_types = sorted(table.cell_type.astype(str).unique())
    if cache.is_file():
        values = np.load(cache, mmap_mode="r")
        if values.shape == (len(table), len(cell_types)):
            return values, cell_types
    from scipy.spatial import cKDTree
    coordinates = table[["x", "y"]].to_numpy(np.float32)
    codes = pd.Categorical(table.cell_type.astype(str), categories=cell_types).codes
    tree = cKDTree(coordinates)
    temporary = cache.with_name("local_composition.incomplete.npy")
    values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                       shape=(len(table), len(cell_types)))
    for start in range(0, len(table), 50000):
        stop = min(start + 50000, len(table))
        distance, neighbors = tree.query(coordinates[start:stop], k=30)
        weights = np.exp(-(distance ** 2) / (2 * 20.0 ** 2))
        denominator = weights.sum(1)
        for cell_type in range(len(cell_types)):
            values[start:stop, cell_type] = (
                weights * (codes[neighbors] == cell_type)
            ).sum(1) / denominator
    values.flush(); del values
    temporary.replace(cache)
    return np.load(cache, mmap_mode="r"), cell_types


def niche_celltype_enrichment(artifacts):
    """V1002-style niche by local-cell-type enrichment with entropy dots."""
    from scipy.stats import norm, rankdata, tiecorrect
    composition, cell_types = _local_composition(artifacts)
    labels = artifacts["assignments"].niche_label.to_numpy(np.int64) - 1
    k = artifacts["config"].niches
    sizes = np.bincount(labels, minlength=k)
    baseline = np.asarray(composition).mean(0)
    means = np.empty((k, len(cell_types)), dtype=float)
    pvalues = np.ones_like(means)
    for column in range(len(cell_types)):
        values = np.asarray(composition[:, column], dtype=np.float64)
        ranks = rankdata(values, method="average")
        rank_sum = np.bincount(labels, weights=ranks, minlength=k)
        abundance = np.bincount(labels, weights=values, minlength=k)
        means[:, column] = abundance / sizes
        outside = len(labels) - sizes
        u = rank_sum - sizes * (sizes + 1) / 2
        sd = np.sqrt(sizes * outside * (len(labels) + 1) * tiecorrect(ranks) / 12)
        valid = sd > 0
        pvalues[valid, column] = norm.sf((u[valid] - sizes[valid] * outside[valid] / 2 - .5) / sd[valid])
    enrichment = np.clip((means - baseline[None, :]) /
                         np.maximum(1 - baseline[None, :], 1e-12), 0, 1)
    mass = sizes[:, None] * means
    share = mass / np.maximum(mass.sum(0, keepdims=True), 1e-12)
    entropy = -np.sum(np.where(share > 0, share * np.log(np.maximum(share, 1e-300)), 0), axis=0) / np.log(k)
    order = np.argsort(-entropy, kind="stable")
    enrichment, pvalues, entropy = enrichment[:, order], pvalues[:, order], entropy[order]
    ordered_types = [cell_types[index] for index in order]
    fig = plt.figure(figsize=(11.5, 5.2), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[20, 1.2], height_ratios=[.8, 4.2], hspace=.02, wspace=.08)
    axis = fig.add_subplot(grid[1, 0]); dots = fig.add_subplot(grid[0, 0], sharex=axis)
    colorbar_axis = fig.add_subplot(grid[1, 1]); legend_axis = fig.add_subplot(grid[0, 1])
    image = axis.imshow(enrichment, cmap="YlOrBr", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    for row, column in np.ndindex(enrichment.shape):
        p = pvalues[row, column]
        stars = "***" if p <= 1e-4 else "**" if p <= 1e-3 else "*" if p <= .05 else ""
        if stars and enrichment[row, column] > 0:
            axis.text(column, row, stars, ha="center", va="center", fontsize=8,
                      color="white" if enrichment[row, column] > .65 else "#252525")
    axis.set_xticks(np.arange(len(ordered_types)), ordered_types, rotation=90, fontsize=8)
    axis.set_yticks(np.arange(k), [f"Niche {index}" for index in range(1, k + 1)])
    axis.set(xlabel="Provisional cell type", ylabel="V1003 niche")
    dots.scatter(np.arange(len(entropy)), np.zeros(len(entropy)), s=12 + 70 * entropy,
                 color="#70bde0", linewidths=0); dots.set_axis_off()
    dots.set_title("V1003 niche × local cell-type enrichment")
    fig.colorbar(image, cax=colorbar_axis, label="Relative enrichment (0–1)")
    legend_axis.scatter([.2, .5, .8], [.65] * 3, s=[12, 47, 82], color="#70bde0", linewidths=0)
    legend_axis.text(.5, .12, "Entropy", ha="center", fontsize=8); legend_axis.set_axis_off()
    figure_dir = artifacts["root"] / "figures"; figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / "niche_celltype_enrichment.png", dpi=600, bbox_inches="tight")
    records = []
    for niche in range(k):
        for rank, original in enumerate(order, 1):
            records.append({"niche": niche + 1, "cell_type": cell_types[original],
                            "mean_local_fraction": means[niche, original],
                            "relative_enrichment_0to1": enrichment[niche, rank - 1],
                            "p_greater_raw": pvalues[niche, rank - 1],
                            "across_niche_entropy": entropy[rank - 1]})
    pd.DataFrame(records).to_csv(artifacts["root"] / "niche_celltype_enrichment.csv", index=False)
    _display(fig); plt.close(fig)


def top_ccc_table(artifacts, top=10):
    table = artifacts["top_ccc"]
    _display(table[table["rank"] <= top][["niche", "rank", "ccc", "weight"]].reset_index(drop=True))


def compact_results(artifacts):
    summary = artifacts["summary"]; training = summary["training"]
    _display(pd.DataFrame([{"best_epoch": training["best_epoch"],
                           "train_reconstruction": training["final_train_reconstruction"],
                           "validation_reconstruction": training["final_validation_reconstruction"],
                           "mean_confidence": summary["assignment_confidence"]["mean"],
                           "niche_collapse": summary["niche_collapse"],
                           "duplicated_H": summary["duplicated_H"]}]))
    _display(artifacts["counts"])


def representative_ccc(artifacts, top=15):
    table = artifacts["top_ccc"]
    counts = artifacts["counts"].set_index("niche")
    for niche in range(1, artifacts["config"].niches + 1):
        _markdown(f"**Niche {niche}**")
        row = counts.loc[niche]
        if row.n_cells == 0 or row.mean_usage < artifacts["config"].collapsed_usage_threshold:
            _markdown(
                f"Inactive/orphan factor: assigned cells = **{int(row.n_cells)}**, "
                f"mean usage = **{row.mean_usage:.3g}**. Its normalized H row has numerical "
                "weights but is not reported as a biological niche or marker program."
            )
            continue
        _display(table[(table.niche == niche) & (table["rank"] <= top)][
            ["rank", "ccc", "weight"]].reset_index(drop=True))


def dictionary_heatmap(artifacts, per_niche=3):
    table = artifacts["top_ccc"]
    selected = table[table["rank"] <= per_niche].feature_id.drop_duplicates().astype(int).to_numpy()
    labels = (table.drop_duplicates("feature_id").set_index("feature_id").loc[selected, "ccc"].tolist())
    values = artifacts["H"][:, selected]
    fig, axis = plt.subplots(figsize=(max(8, .34 * len(selected)), 4.2), layout="constrained")
    image = axis.imshow(values, cmap="YlOrBr", aspect="auto", interpolation="nearest")
    axis.set_xticks(np.arange(len(selected)), labels, rotation=90, fontsize=7)
    inactive = set(artifacts["summary"].get("inactive_niches", []))
    axis.set_yticks(np.arange(artifacts["config"].niches),
                    [f"Niche {index}" + (" (inactive)" if index in inactive else "")
                     for index in range(1, artifacts["config"].niches + 1)])
    axis.set_title("Normalized H on representative CCC features (inactive rows are diagnostic only)")
    fig.colorbar(image, ax=axis, label="H weight", fraction=.025)
    figure_dir = artifacts["root"] / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / "representative_H_heatmap.png", dpi=300, bbox_inches="tight")
    _display(fig)
    plt.close(fig)


def confidence_and_usage(artifacts):
    assignment = artifacts["assignments"]
    counts = artifacts["counts"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")
    axes[0].hist(assignment.assignment_confidence, bins=40, color="#59a14f")
    axes[0].set(xlabel="max(z)", ylabel="Cells", title="Assignment confidence")
    axes[1].bar(counts.niche.astype(str), counts.mean_usage, color="#f28e2b")
    axes[1].axhline(1 / artifacts["config"].niches, color="#777777", linestyle=":")
    axes[1].set(xlabel="Niche", ylabel="mean(z)", title="Mean niche usage")
    _display(fig)
    plt.close(fig)
    _markdown("**H pairwise cosine similarity**")
    _display(artifacts["H_cosine"])


def final_summary(artifacts):
    summary = artifacts["summary"]
    active = summary.get("active_niches", list(range(1, artifacts["config"].niches + 1)))
    top = artifacts["top_ccc"].query("rank == 1 and niche in @active").sort_values("niche")
    features = "; ".join(f"Niche {row.niche}: {row.ccc}" for row in top.itertuples())
    _markdown(
        f"- All eight niches used: **{summary['all_niches_assigned']}**\n"
        f"- Niche collapse detected: **{summary['niche_collapse']}**\n"
        f"- Duplicated H detected (cosine ≥ 0.95): **{summary['duplicated_H']}**\n"
        f"- Inactive/orphan factors: **{summary.get('inactive_niches', [])}**; their H rows are not biological marker programs.\n"
        f"- Maximum off-diagonal H cosine: **{summary['maximum_off_diagonal_H_cosine']:.3f}**\n"
        f"- Main representative CCC: {features}"
    )
