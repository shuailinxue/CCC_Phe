"""Oracle and gated recovery figures; unreached stages remain visibly blank."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

OUT = Path("outputs/v1002_identifiability")
COLORS = ["#b7b7b7", "#c74348", "#df9d37", "#3b73ad", "#7965a9", "#4c9a80"]
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(np.arange(-.5, 6.5), 6)


def finish(fig, name):
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(OUT / (name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def spatial(ax, xy, labels, title):
    ax.scatter(xy[:, 0], xy[:, 1], c=labels, s=8, cmap=CMAP, norm=NORM, linewidths=0)
    ax.set(title=title, xlim=(-1, 41), ylim=(-1, 46), xlabel="x", ylabel="y")
    ax.invert_yaxis(); ax.set_aspect("equal")


def profiles(ax, centroids, a, b, title, is_view=False):
    left, right = centroids[a], centroids[b]
    if is_view:
        indices = np.argsort(-np.maximum(left, right))[:25]
        left, right = left[indices], right[indices]
        xlabel = "Top 25 directed CCC features"
    else:
        xlabel = "Cell type"
    x = np.arange(len(left))
    ax.plot(x, left, "o-", lw=1.2, ms=2.5, color=COLORS[a], label=f"N{a}")
    ax.plot(x, right, "s--", lw=1.2, ms=2.5, color=COLORS[b], label=f"N{b}")
    cosine = float(centroids[a] @ centroids[b] / max(np.linalg.norm(centroids[a]) * np.linalg.norm(centroids[b]), 1e-12))
    ax.set(title=f"{title}\ncentroid cosine={cosine:.2f}", xlabel=xlabel, ylabel="Observed mean")
    ax.legend(frameon=False, fontsize=8)


def run():
    audit = json.loads((OUT / "oracle_identifiability.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    passed = summary["Level1_observable_identifiable"]
    if passed:
        with np.load(OUT / "simulation_observed_data_seed31.npz") as source:
            data = {key: source[key] for key in source.files}
        with np.load(OUT / "oracle_centroids.npz") as source:
            centroids = {key: source[key][0] for key in source.files}
    else:
        with np.load(OUT / "oracle_last_attempt_seed31.npz") as source:
            data = {key: source[key] for key in source.files}
        centroids = {"CS": data["CS_centroids"], "IS": data["IS_centroids"]}
    rows = audit["attempts"][-1]["rows"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.8), constrained_layout=True)
    spatial(ax[0, 0], data["coordinates"], data["labels"], "A  True niche map")
    profiles(ax[0, 1], centroids["CS"], 1, 2, "B  Observable CS: N1/N2")
    profiles(ax[0, 2], centroids["IS"], 1, 2, "C  Observable IS: N1/N2", True)
    profiles(ax[1, 0], centroids["CS"], 3, 4, "D  Observable CS: N3/N4")
    profiles(ax[1, 1], centroids["IS"], 3, 4, "E  Observable IS: N3/N4", True)
    labels = ["N1/2 CS", "N1/2 IS", "N3/4 CS", "N3/4 IS", "BG vs niches"]
    values = np.array([[r["N1_N2"]["CS_AUC"], r["N1_N2"]["IS_AUC"],
                        r["N3_N4"]["CS_AUC"], r["N3_N4"]["IS_AUC"],
                        r["background_vs_niche_CS_IS_AUC"]] for r in rows])
    x = np.arange(5)
    ax[1, 2].bar(x, values.mean(0), yerr=values.std(0, ddof=1), color=["#72869b", "#b45a60", "#b45a60", "#72869b", "#5c9d85"])
    ax[1, 2].axhline(.6, color="#888", ls="--", lw=.8)
    ax[1, 2].axhline(.9, color="#888", ls=":", lw=.8)
    ax[1, 2].set(title="F  Held-out oracle AUC\nindependent tissue as test", xticks=x, xticklabels=labels,
                 ylabel="AUC", ylim=(0, 1.04))
    ax[1, 2].tick_params(axis="x", labelrotation=25)
    finish(fig, "Figure_simulation_truth_vs_observed")

    fig, ax = plt.subplots(1, 4, figsize=(14, 4.3), constrained_layout=True)
    spatial(ax[0], data["coordinates"], data["labels"], "True niche map")
    if summary["Level2_ST"] == "not_run":
        for axis, title in zip(ax[1:], ("Composition-only", "CCC-only", "Full-balanced")):
            axis.set_title(title)
            axis.text(.5, .5, "Not run\nOracle gate failed", ha="center", va="center", transform=axis.transAxes)
            axis.set_xticks([]); axis.set_yticks([])
    else:
        reports = json.loads((OUT / "primary_st_results.json").read_text())[0]["methods"]
        for axis, method in zip(ax[1:], ("Composition-only", "CCC-only", "Full-balanced")):
            spatial(axis, data["coordinates"], np.asarray(reports[method]["predicted"]), method)
    finish(fig, "Figure_final_niche_recovery")


if __name__ == "__main__":
    run()
