from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _niche_colors(k):
    return plt.get_cmap("tab20", k)(np.arange(k))


def training_curves(output):
    output = Path(output); s1 = pd.read_csv(output / "stage1_training_history.csv"); s2 = pd.read_csv(output / "stage2B_training_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(s1.epoch, s1.train_reconstruction, label="train"); axes[0].plot(s1.epoch, s1.validation_reconstruction, label="validation")
    axes[0].set(title="Stage 1 reconstruction", xlabel="Epoch", ylabel="Huber loss"); axes[0].legend(frameon=False)
    axes[1].plot(s2.epoch, s2.validation_ae, label="nonlinear AE")
    axes[1].plot(s2.epoch, s2.validation_linear, label="Z32 H32")
    axes[1].set(title="Stage 2B validation", xlabel="Epoch", ylabel="Huber loss"); axes[1].legend(frameon=False)
    fig.tight_layout(); return fig


def spatial_overview(artifacts):
    assignments = artifacts["assignments"]
    cell_type = pd.Categorical(assignments.cell_type)
    niche = assignments.niche.to_numpy() - 1
    type_colors = plt.get_cmap("tab20", len(cell_type.categories))(np.arange(len(cell_type.categories)))
    k = artifacts["selected_K"]
    niche_colors = _niche_colors(k)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.4), layout="constrained")
    axes[0].scatter(assignments.x, assignments.y, c=type_colors[cell_type.codes], s=.22, linewidths=0, rasterized=True)
    axes[0].set_title("Cell types")
    axes[1].scatter(assignments.x, assignments.y, c=niche_colors[niche], s=.22, linewidths=0, rasterized=True)
    axes[1].set_title(f"V1006 selected niches (K={k})")
    for ax in axes: ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    type_handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=type_colors[i],
                               markeredgewidth=0, markersize=5, label=name)
                    for i, name in enumerate(cell_type.categories)]
    niche_handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=niche_colors[i],
                                markeredgewidth=0, markersize=5, label=f"Niche {i + 1}") for i in range(k)]
    axes[0].legend(handles=type_handles, loc="upper center", bbox_to_anchor=(.5, -.10), ncol=4, frameon=False, fontsize=7)
    axes[1].legend(handles=niche_handles, loc="upper center", bbox_to_anchor=(.5, -.10), ncol=4, frameon=False, fontsize=7)
    figures = Path(artifacts["output"]) / "figures"; figures.mkdir(exist_ok=True)
    fig.savefig(figures / "celltype_vs_v1006_niche_overview.png", dpi=600, bbox_inches="tight")
    return fig


def _local_composition(artifacts):
    assignments = artifacts["assignments"]
    source = Path(artifacts["config"].v1003_output) / "cache/local_composition.npy"
    manifest = Path(artifacts["config"].v1003_output) / "cache/input_manifest.json"
    cell_types = sorted(assignments.cell_type.astype(str).unique())
    if manifest.is_file():
        import json
        cell_types = json.loads(manifest.read_text()).get("cell_types", cell_types)
    if source.is_file():
        values = np.load(source, mmap_mode="r")
        if values.shape == (len(assignments), len(cell_types)): return values, cell_types
    # Fallback reproduces the fixed V1002 30-neighbor, sigma=20 µm composition for reporting only.
    from scipy.spatial import cKDTree
    cache = Path(artifacts["output"]) / "cache/local_composition.npy"; cache.parent.mkdir(exist_ok=True)
    if cache.is_file(): return np.load(cache, mmap_mode="r"), cell_types
    xy = assignments[["x", "y"]].to_numpy(np.float32)
    codes = pd.Categorical(assignments.cell_type.astype(str), categories=cell_types).codes
    tree = cKDTree(xy)
    out = np.lib.format.open_memmap(cache.with_name("local_composition.incomplete.npy"), mode="w+", dtype=np.float32,
                                   shape=(len(xy), len(cell_types)))
    for start in range(0, len(xy), 50000):
        stop = min(start + 50000, len(xy)); distance, neighbors = tree.query(xy[start:stop], k=30)
        weights = np.exp(-(distance ** 2) / (2 * 20.0 ** 2)); denominator = weights.sum(1)
        for t in range(len(cell_types)):
            out[start:stop, t] = (weights * (codes[neighbors] == t)).sum(1) / denominator
    out.flush(); del out; cache.with_name("local_composition.incomplete.npy").replace(cache)
    return np.load(cache, mmap_mode="r"), cell_types


def niche_celltype_enrichment(artifacts):
    """Match the V1002 one-sided Mann–Whitney U heatmap and entropy-dot definition."""
    from scipy.stats import norm, rankdata, tiecorrect
    composition, cell_types = _local_composition(artifacts)
    labels = artifacts["labels"].astype(np.int64); n, k = len(labels), artifacts["selected_K"]
    sizes = np.bincount(labels, minlength=k); baseline = np.asarray(composition).mean(0)
    means = np.empty((k, len(cell_types))); pvalues = np.ones_like(means)
    for t in range(len(cell_types)):
        values = np.asarray(composition[:, t], dtype=np.float64)
        ranks = rankdata(values, method="average")
        rank_sum = np.bincount(labels, weights=ranks, minlength=k)
        means[:, t] = np.bincount(labels, weights=values, minlength=k) / np.maximum(sizes, 1)
        outside = n - sizes; u = rank_sum - sizes * (sizes + 1) / 2
        sd = np.sqrt(sizes * outside * (n + 1) * tiecorrect(ranks) / 12)
        valid = sd > 0
        pvalues[valid, t] = norm.sf((u[valid] - sizes[valid] * outside[valid] / 2 - .5) / sd[valid])
    enrichment = np.clip((means - baseline[None, :]) / np.maximum(1 - baseline[None, :], 1e-12), 0, 1)
    mass = sizes[:, None] * means; share = mass / np.maximum(mass.sum(0, keepdims=True), 1e-12)
    entropy = -np.sum(np.where(share > 0, share * np.log(np.maximum(share, 1e-300)), 0), axis=0) / np.log(k)
    order = np.argsort(-entropy, kind="stable")
    enrichment, pvalues, entropy = enrichment[:, order], pvalues[:, order], entropy[order]
    ordered_types = [cell_types[i] for i in order]
    fig = plt.figure(figsize=(13, 5.6), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[20, 1.5], height_ratios=[.8, 4.2], hspace=.02, wspace=.08)
    ax = fig.add_subplot(grid[1, 0]); top = fig.add_subplot(grid[0, 0], sharex=ax)
    bar_ax = fig.add_subplot(grid[1, 1]); legend_ax = fig.add_subplot(grid[0, 1])
    image = ax.imshow(enrichment, cmap="YlOrBr", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    for row, col in np.ndindex(enrichment.shape):
        p = pvalues[row, col]; stars = "***" if p <= 1e-4 else "**" if p <= 1e-3 else "*" if p <= .05 else ""
        if stars and enrichment[row, col] > 0:
            ax.text(col, row, stars, ha="center", va="center", fontsize=8,
                    color="white" if enrichment[row, col] > .65 else "#252525")
    ax.set_xticks(np.arange(len(ordered_types)), ordered_types, rotation=90, ha="center", fontsize=8)
    ax.set_yticks(np.arange(k), [f"Niche {i}" for i in range(1, k + 1)], fontsize=9)
    ax.set(xlabel="Provisional cell type", ylabel="Niche")
    top.scatter(np.arange(len(entropy)), np.zeros(len(entropy)), s=12 + 70 * entropy,
                color="#70bde0", linewidths=0); top.set_axis_off(); top.set_title("V1006 niche × local cell type")
    fig.colorbar(image, cax=bar_ax, label="Relative enrichment (0–1)")
    legend_ax.scatter([.18, .48, .78], [.65] * 3, s=[12 + 70 * x for x in (0, .5, 1)], color="#70bde0", linewidths=0)
    legend_ax.annotate("", xy=(.88, .4), xytext=(.1, .4), arrowprops=dict(arrowstyle="->", lw=.8, color="#333"))
    legend_ax.text(.5, .12, "Entropy", ha="center", fontsize=8); legend_ax.set_axis_off()
    figures = Path(artifacts["output"]) / "figures"; figures.mkdir(exist_ok=True)
    fig.savefig(figures / "v1006_niche_celltype_enrichment.png", dpi=600, bbox_inches="tight")
    records = []
    for niche in range(k):
        for rank, original in enumerate(order):
            records.append({"niche": niche + 1, "cell_type": cell_types[original], "n_cells": int(sizes[niche]),
                            "mean_local_fraction": means[niche, original], "global_mean_local_fraction": baseline[original],
                            "relative_enrichment_0to1": enrichment[niche, rank], "p_greater_raw": pvalues[niche, rank],
                            "across_niche_entropy": entropy[rank], "entropy_rank": rank + 1})
    table = pd.DataFrame(records); table.to_csv(Path(artifacts["output"]) / "niche_celltype_enrichment.csv", index=False)
    return fig, table


def spatial_each_niche(artifacts):
    table = artifacts["assignments"]; labels = artifacts["labels"]
    k = artifacts["selected_K"]; columns = 4; rows = int(np.ceil(k / columns))
    colors = _niche_colors(k); xlim = (table.x.min(), table.x.max()); ylim = (table.y.max(), table.y.min())
    fig, axes = plt.subplots(rows, columns, figsize=(17, 4.2 * rows), squeeze=False, layout="constrained")
    for niche, ax in enumerate(axes.flat[:k]):
        selected = labels == niche
        ax.scatter(table.x, table.y, c="#d9d9d9", s=.16, linewidths=0, rasterized=True)
        ax.scatter(table.loc[selected, "x"], table.loc[selected, "y"], c=[colors[niche]], s=.38, linewidths=0, rasterized=True)
        ax.set(xlim=xlim, ylim=ylim, title=f"Niche {niche + 1}  (n={selected.sum():,})"); ax.set_aspect("equal"); ax.set_axis_off()
    for ax in axes.flat[k:]: ax.set_axis_off()
    fig.suptitle("V1006: spatial distribution of each niche", fontsize=15)
    figures = Path(artifacts["output"]) / "figures"; figures.mkdir(exist_ok=True)
    fig.savefig(figures / "v1006_each_niche_spatial.png", dpi=600, bbox_inches="tight")
    return fig


def niche_size_plot(artifacts):
    table = artifacts["counts"]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(table.niche.astype(str), table.fraction, color=_niche_colors(len(table)))
    ax.set(xlabel="Niche", ylabel="Cell fraction", title="V1006 niche size")
    fig.tight_layout(); return fig


def representative_h_heatmap(h, features, per_niche=3):
    selected = np.unique(np.concatenate([np.argsort(-row)[:per_niche] for row in h]))
    labels = features.iloc[selected].get("ccc", pd.Series(selected.astype(str))).astype(str).tolist()
    fig, ax = plt.subplots(figsize=(max(7, .35 * len(selected)), max(4, .35 * len(h))))
    sns.heatmap(h[:, selected], cmap="mako", xticklabels=labels, yticklabels=np.arange(1, len(h) + 1), ax=ax)
    ax.set(xlabel="Representative directed CCC", ylabel="Niche", title=f"Selected H (K={len(h)}) CCC programs")
    ax.tick_params(axis="x", rotation=90); fig.tight_layout(); return fig


def model_selection_plot(artifacts):
    table = artifacts["model_selection"]; selected = artifacts["selected_K"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    panels = [("silhouette", "H32 silhouette"),
              ("adjacent_K_stability", "Adjacent-K stability"),
              ("max_niche_cell_fraction", "Largest niche fraction")]
    for ax, (column, label) in zip(axes, panels):
        ax.plot(table.K, table[column], marker="o", ms=3, color="#4c78a8")
        ax.axvline(selected, color="#d62728", ls="--", lw=1.2, label=f"selected K={selected}")
        ax.axvline(8, color="#777777", ls=":", lw=1.1, label="K=8 comparator")
        ax.set(xlabel="K", ylabel=label); ax.set_xticks(table.K)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Cell-level P32/activity niche selection: K=4–15")
    fig.tight_layout()
    figures = Path(artifacts["output"]) / "figures"; figures.mkdir(exist_ok=True)
    fig.savefig(figures / "cell_latent_K_model_selection.png", dpi=600, bbox_inches="tight")
    return fig
