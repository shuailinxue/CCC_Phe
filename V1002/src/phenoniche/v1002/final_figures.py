"""Transparent publication-format figures for the v2 controlled simulation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

OUT = Path("outputs/v1002_final_sim")
COLORS = ["#bbbbbb", "#c43d4b", "#de9d37", "#3b74b3", "#7a68a9", "#4d9d81"]
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(np.arange(-.5, 6.5), 6)


def finish(fig, stem):
    fig.savefig(OUT / (stem + ".pdf"), bbox_inches="tight")
    fig.savefig(OUT / (stem + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def map_panel(ax, xy, values, title, continuous=False):
    if continuous:
        artist = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap="magma", s=9, vmin=0, vmax=max(.5, float(np.quantile(values, .99))), linewidths=0)
    else:
        artist = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=CMAP, norm=NORM, s=9, linewidths=0)
    ax.set(title=title, xlabel="x", ylabel="y", xlim=(-1, 41), ylim=(-1, 46))
    ax.invert_yaxis(); ax.set_aspect("equal")
    if continuous: plt.colorbar(artist, ax=ax, fraction=.04, pad=.02)


def example(data):
    xy, labels = data["coordinates"], data["labels"]
    pred = data["C+I_pred"]
    activity = data["C+I_activity"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4), constrained_layout=True)
    map_panel(axes[0], xy, labels, "A  True niche map")
    map_panel(axes[1], xy, pred, "B  C+I predicted map")
    map_panel(axes[2], xy, activity[:, 1], "C  Risk N1 activity", True)
    map_panel(axes[3], xy, activity[:, 3], "D  Protective N3 activity", True)
    finish(fig, "Figure_example_maps_v2")


def pair_profiles(axc, axi, data, a, b, label):
    hc = data["hc"]
    axc.plot(hc[a], "o-", label=f"N{a}", color=COLORS[a])
    axc.plot(hc[b], "s--", label=f"N{b}", color=COLORS[b])
    axc.set(title=f"{label}: composition", xlabel="Cell type", ylabel="Proportion")
    axc.legend(frameon=False, fontsize=8)
    h1, h2 = data[f"hi{a}"], data[f"hi{b}"]
    top = np.argsort(-(h1 + h2))[:30]
    axi.plot(np.arange(len(top)), h1[top], "-", lw=1.5, color=COLORS[a], label=f"N{a}")
    axi.plot(np.arange(len(top)), h2[top], "-", lw=1.5, color=COLORS[b], label=f"N{b}")
    similarity = np.dot(h1, h2) / max(np.linalg.norm(h1) * np.linalg.norm(h2), 1e-12)
    axi.set(title=f"Directed CCC (cosine {similarity:.2f})", xlabel="Top observed features", ylabel="Mean score")
    axi.legend(frameon=False, fontsize=8)


def complementarity(data):
    xy, labels = data["coordinates"], data["labels"]
    fig = plt.figure(figsize=(16, 8.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 6, width_ratios=(1.1, 1.1, 1, 1, 1, 1))
    for row, (a, b, label) in enumerate(((1, 2, "N1/N2"), (3, 4, "N3/N4"))):
        axc, axi = fig.add_subplot(grid[row, 0]), fig.add_subplot(grid[row, 1])
        pair_profiles(axc, axi, data, a, b, label)
        for col, key, title in ((2, None, "Truth"), (3, "C_only_pred", "C-only"),
                                (4, "I_only_pred", "I-only"), (5, "C+I_pred", "C+I")):
            ax = fig.add_subplot(grid[row, col])
            current = labels if key is None else data[key]
            pair_map = np.where(current == a, a, np.where(current == b, b, 0))
            map_panel(ax, xy, pair_map, f"{label} {title}")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    finish(fig, "Figure_view_complementarity")


def filtering(audit):
    fig, ax = plt.subplots(figsize=(11, 2.8), constrained_layout=True)
    steps = [("Raw LR", audit["raw_LR"]), ("Assay measurable", audit["assay_measurable_LR"]),
             ("Coverage", audit["coverage_directed_CCC"]), ("Spatial opportunity", audit["retained_CCC"])]
    xpos = [0.11, .37, .64, .89]
    for x, (label, count) in zip(xpos, steps):
        ax.text(x, .55, f"{count:,}\n{label}", ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=.6", facecolor="#e8eef4", edgecolor="#56738f"), transform=ax.transAxes)
    for x1, x2 in zip(xpos[:-1], xpos[1:]):
        ax.annotate("", xy=(x2-.075,.55), xytext=(x1+.08,.55), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(.5, .08, f"Raw directed CCC: {audit['raw_directed_CCC']:,}   |   Signal retention recall: {audit['signal_retention_recall']:.3f}",
            ha="center", transform=ax.transAxes, fontsize=10)
    ax.set_axis_off()
    finish(fig, "Figure_filtering")


def main(data, primary, summary):
    xy, labels = data["coordinates"], data["labels"]
    fig, ax = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    map_panel(ax[0, 0], xy, labels, "A  Simulation truth")
    ax[0, 0].text(.02, -.12, "N1/N2: same composition, distinct CCC\nN3/N4: distinct composition, same CCC truth",
                  fontsize=8, transform=ax[0, 0].transAxes, va="top")
    x = np.arange(6)
    methods = ("C+I", "C-only", "I-only")
    for method, color in zip(methods, ("#303f64", "#3b74b3", "#d0902f")):
        rows = [record["methods"][method] for record in primary]
        sens = np.array([[r["per_factor"][i]["sensitivity"] for i in range(6)] for r in rows])
        auc = np.array([[r["per_factor"][i]["auc"] for i in range(6)] for r in rows])
        ax[0, 1].plot(x, sens.mean(0), "o-", label=method, color=color)
        ax[1, 0].plot(x, auc.mean(0), "o-", label=method, color=color)
    for a in (ax[0, 1], ax[1, 0]):
        a.set_xticks(x, ["BG", "N1", "N2", "N3", "N4", "N5"])
        a.set_ylim(0, 1.05)
        a.axhline(.8, lw=.7, ls="--", color="#888888")
    ax[0, 1].set(title="B  Primary niche identification", ylabel="Sensitivity")
    ax[0, 1].legend(frameon=False, fontsize=8)
    ax[1, 0].set(title="D  Primary niche localization", ylabel="AUC")
    groups = ((1, 2, "N1/N2"), (3, 4, "N3/N4"))
    for group_index, (a, b, label) in enumerate(groups):
        values = []
        for method in methods[:3]:
            rows = [record["methods"][method] for record in primary]
            values.append(np.mean([min(r["per_factor"][a]["sensitivity"], r["per_factor"][b]["sensitivity"]) for r in rows]))
        ax[0, 2].bar(np.arange(3) + group_index * 4, values, color=["#303f64", "#3b74b3", "#d0902f"])
    ax[0, 2].set_xticks([1, 5], ["N1/N2", "N3/N4"])
    ax[0, 2].set(title="C  Hard-pair recovery", ylabel="Minimum twin sensitivity", ylim=(0, 1))
    ax[1, 1].set_title("E  ST → bulk transfer")
    ax[1, 2].set_title("F  Clinical phenotype")
    if summary["primary_ST_gate"] and isinstance(summary["bulk"], list):
        bulk = summary["bulk"]
        keys = ("Risk_WB", "Neutral_twin_WB", "Protective_WB")
        matrix = np.array([[r[k] for k in keys] for r in bulk])
        ax[1, 1].bar(range(3), matrix.mean(0), yerr=matrix.std(0, ddof=1), color=[COLORS[1], COLORS[2], COLORS[3]])
        ax[1, 1].set_xticks(range(3), ["Risk", "Neutral", "Protective"])
        ax[1, 1].set(ylabel="WB correlation", ylim=(-1, 1))
        gamma = np.array([r["ALR"]["gamma"] for r in summary["cox"]])
        ax[1, 2].bar(range(5), gamma.mean(0), yerr=gamma.std(0, ddof=1), color=COLORS[1:])
        ax[1, 2].axhline(0, lw=.8, color="black")
        ax[1, 2].set_xticks(range(5), ["Risk", "N2", "Protective", "N4", "N5"])
        ax[1, 2].set_ylabel("ALR Cox gamma")
    else:
        for target in (ax[1, 1], ax[1, 2]):
            target.text(.5, .5, "Not run: primary ST gate failed", ha="center", va="center", transform=target.transAxes)
            target.set_xticks([]); target.set_yticks([])
    finish(fig, "Figure_simulation_main_v2")


def run():
    with np.load(OUT / "primary_figure_data.npz") as values:
        data = {key: values[key] for key in values.files}
    primary = json.loads((OUT / "primary_st_results.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    audit = json.loads((OUT / "filter_audit.json").read_text())
    example(data); complementarity(data); filtering(audit); main(data, primary, summary)


if __name__ == "__main__":
    run()
