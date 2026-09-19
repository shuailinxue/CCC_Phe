import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


COLORS = {"Full": "#2F6B9A", "No-filter": "#8A8F98", "Composition-only": "#D28E45", "CCC-only": "#5C9272"}
NICHE_COLORS = ["#D9D9D9", "#B94B5F", "#6A8EBB", "#4D9B78", "#B18C46", "#8069A6"]


def _style():
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "pdf.fonttype": 42, "font.size": 7, "axes.spines.right": False,
                         "axes.spines.top": False, "axes.linewidth": 0.7, "legend.frameon": False})


def _save(fig, stem):
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _method(record, name):
    return next(value for value in record["methods"] if value["Method"] == name)


def _nested_grid(fig, slot, records, metric, methods, ylabel):
    grid = GridSpecFromSubplotSpec(3, 3, subplot_spec=slot, wspace=0.22, hspace=0.30)
    for row, size in enumerate((50, 100, 150)):
        for column, purity in enumerate((0.3, 0.5, 0.7)):
            ax = fig.add_subplot(grid[row, column])
            for method in methods:
                means, sds = [], []
                for noise in (0.0, 0.05, 0.10):
                    values = [_method(record, method)["spatial"][metric] for record in records
                              if record["niche_size"] == size and record["purity"] == purity and record["noise"] == noise]
                    means.append(np.mean(values)); sds.append(np.std(values, ddof=1))
                ax.errorbar((0, 0.05, 0.10), means, yerr=sds, color=COLORS[method], marker="o", ms=2.2, lw=0.9, capsize=1.5)
            ax.set_ylim(0, 1.02)
            ax.set_xticks((0, 0.1))
            ax.tick_params(labelsize=5, length=2)
            if row == 0:
                ax.set_title(f"purity {purity}", fontsize=6)
            if column == 0:
                ax.set_ylabel(f"size {size}\n{ylabel}", fontsize=5.5)
            if row == 2:
                ax.set_xlabel("noise", fontsize=5.5)


def figure_main(output):
    _style()
    output = Path(output)
    records = json.loads((output / "robustness_grid.json").read_text())
    truth = json.loads((output / "simulation_truth.json").read_text())
    primary = next(record for record in records if record["purity"] == 0.5 and record["niche_size"] == 100 and record["noise"] == 0.05 and record["replicate"] == 0)
    fig = plt.figure(figsize=(14, 9.4), constrained_layout=True)
    outer = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.92))
    ax = fig.add_subplot(outer[0, 0])
    coords, labels = np.asarray(truth["coordinates"]), np.asarray(truth["labels"])
    ax.scatter(coords[:, 0], coords[:, 1], c=[NICHE_COLORS[value] for value in labels], s=4, linewidths=0)
    ax.set_title("Simulation landscape", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    hc = np.asarray(truth["HC_true"])
    inset = ax.inset_axes([0.04, 0.02, 0.50, 0.29])
    inset.plot(hc[0], color=NICHE_COLORS[1], marker="o", ms=2, lw=1, label="Niche 1")
    inset.plot(hc[1], color=NICHE_COLORS[2], ls="--", lw=1, label="Niche 2")
    inset.set_title("identical composition", fontsize=5.5); inset.set_xticks([]); inset.tick_params(labelsize=5)
    inset.legend(fontsize=5, ncol=2, loc="upper right")
    ccc = ax.inset_axes([0.60, 0.02, 0.36, 0.29])
    active = truth["active_edges"]
    for niche, color, shift in ((1, NICHE_COLORS[1], -0.17), (2, NICHE_COLORS[2], 0.17)):
        counts = np.zeros(64)
        for edge in active:
            if edge["niche"] == niche:
                counts[edge["sender"] * 8 + edge["receiver"]] += edge["strength"]
        top = np.argsort(-counts)[:4]
        ccc.bar(np.arange(4) + shift, counts[top], width=0.32, color=color, label=f"Niche {niche}")
    ccc.set_title("distinct directed CCC", fontsize=5.5); ccc.set_xticks([]); ccc.tick_params(labelsize=5)
    ax.text(-0.08, 1.03, "A", transform=ax.transAxes, weight="bold", fontsize=10)
    _nested_grid(fig, outer[0, 1], records, "niche_sensitivity", ("Full", "No-filter", "Composition-only", "CCC-only"), "Sensitivity")
    fig.text(0.342, 0.98, "B", weight="bold", fontsize=10)
    _nested_grid(fig, outer[0, 2], records, "localization_AUC", ("Full", "No-filter", "Composition-only"), "Localization AUC")
    fig.text(0.67, 0.98, "C", weight="bold", fontsize=10)
    _nested_grid(fig, outer[1, 0], records, "CCC_AUPRC", ("Full", "No-filter", "CCC-only"), "CCC AUPRC")
    fig.text(0.012, 0.49, "D", weight="bold", fontsize=10)
    ax = fig.add_subplot(outer[1, 1])
    subset = [row for row in records if row["purity"] == 0.5 and row["niche_size"] == 100]
    for method in ("Full", "No-filter", "Composition-only"):
        for index, (key, marker) in enumerate((("Risk_WB", "o"), ("Neutral_WB", "s"), ("Protective_WB", "^"))):
            means, sds = [], []
            for noise in (0.0, 0.05, 0.10):
                values = [_method(row, method)["bulk"][key] for row in subset if row["noise"] == noise]
                means.append(np.mean(values)); sds.append(np.std(values, ddof=1))
            ax.errorbar((0, 0.05, 0.10), means, yerr=sds, color=COLORS[method], marker=marker,
                        ls=("-", "--", ":")[index], lw=1, ms=3, capsize=2,
                        label=f"{method} · {key.replace('_WB','')}")
    ax.set_xlabel("Expression noise"); ax.set_ylabel("WB recovery"); ax.set_ylim(-0.2, 1.02)
    ax.legend(fontsize=5, ncol=2, loc="lower left")
    ax.text(-0.14, 1.03, "E", transform=ax.transAxes, weight="bold", fontsize=10)
    ax = fig.add_subplot(outer[1, 2])
    x = np.arange(3)
    width = 0.22
    for offset, method in enumerate(("Full", "No-filter", "Composition-only")):
        gamma = []
        error = []
        for key in ("gamma_1", "gamma_2", "gamma_3"):
            values = [_method(row, method)["bulk"][key] for row in primary["methods"]] if False else [
                _method(row, method)["bulk"][key] for row in records if row["purity"] == 0.5 and row["niche_size"] == 100 and row["noise"] == 0.05]
            gamma.append(np.mean(values)); error.append(np.std(values, ddof=1))
        ax.bar(x + (offset - 1) * width, gamma, width, yerr=error, color=COLORS[method], label=method, capsize=2)
    ax.axhline(0, color="#333333", lw=0.7)
    ax.set_xticks(x, ("Risk", "Neutral", "Protective")); ax.set_ylabel("Cox coefficient")
    ax.legend(fontsize=5)
    test_c = []
    for method in ("Full", "No-filter", "Composition-only"):
        values = [_method(row, method)["bulk"]["Test_C"] for row in records
                  if row["purity"] == 0.5 and row["niche_size"] == 100 and row["noise"] == 0.05]
        test_c.append(f"{method}: {np.mean(values):.3f}")
    ax.text(0.02, 0.98, "Test C  " + " | ".join(test_c), transform=ax.transAxes, va="top", fontsize=5.2)
    ax.text(-0.14, 1.03, "F", transform=ax.transAxes, weight="bold", fontsize=10)
    handles = [Line2D([0], [0], color=COLORS[name], lw=1.4, marker="o", ms=3, label=name) for name in COLORS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.005), ncol=4, fontsize=6)
    _save(fig, output / "Figure_simulation_main")


def figure_example_maps(output):
    _style()
    output = Path(output)
    records = json.loads((output / "robustness_grid.json").read_text())
    truth = json.loads((output / "simulation_truth.json").read_text())
    primary = next(record for record in records if record["purity"] == 0.5 and record["niche_size"] == 100 and record["noise"] == 0.05 and record["replicate"] == 0)
    spatial = _method(primary, "Full")["spatial"]
    coordinates = np.asarray(truth["coordinates"])
    panels = ((np.asarray(truth["labels"]), "True niche map", "categorical"),
              (np.asarray(spatial["predicted_labels"]), "Predicted niche map", "categorical"),
              (np.asarray(spatial["aligned_WS"])[:, 0], "Risk niche activity", "continuous"),
              (np.asarray(spatial["aligned_WS"])[:, 2], "Protective niche activity", "continuous"))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0), constrained_layout=True)
    for label, (ax, (values, title, kind)) in enumerate(zip(axes.ravel(), panels)):
        if kind == "categorical":
            ax.scatter(coordinates[:, 0], coordinates[:, 1], c=[NICHE_COLORS[int(value)] for value in values], s=7, linewidths=0)
        else:
            image = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=values, cmap="viridis", s=7, linewidths=0)
            fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.text(-0.05, 1.02, chr(65 + label), transform=ax.transAxes, weight="bold", fontsize=9)
    _save(fig, output / "Figure_example_maps")


def figure_risk_network(output):
    _style()
    output = Path(output)
    records = json.loads((output / "robustness_grid.json").read_text())
    truth = json.loads((output / "simulation_truth.json").read_text())
    primary = next(record for record in records if record["purity"] == 0.5 and record["niche_size"] == 100 and record["noise"] == 0.05 and record["replicate"] == 0)
    edges = _method(primary, "Full")["top_CCC_features"][:12]
    names = truth["cell_types"]
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    positions = np.column_stack((np.cos(angles), np.sin(angles)))
    fig, (ax, table_ax) = plt.subplots(1, 2, figsize=(9.0, 4.3), gridspec_kw={"width_ratios": (1.0, 1.35)}, constrained_layout=True)
    for index, (x, y) in enumerate(positions):
        ax.scatter(x, y, s=320, color="#E8EEF3", edgecolor="#40566B", zorder=3)
        ax.text(x, y, names[index], ha="center", va="center", fontsize=6)
    maximum = max(edge["weight"] for edge in edges)
    for edge in edges:
        start, end = positions[edge["sender"]], positions[edge["receiver"]]
        arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
                                lw=0.5 + 2 * edge["weight"] / maximum, color="#B94B5F", alpha=0.65,
                                connectionstyle="arc3,rad=0.13")
        ax.add_patch(arrow)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.axis("off"); ax.set_title("Risk niche directed CCC network")
    table_ax.axis("off")
    rows = [[names[edge["sender"]], names[edge["receiver"]], edge["lr_id"], f"{edge['weight']:.3g}"] for edge in edges]
    table = table_ax.table(cellText=rows, colLabels=("Sender", "Receiver", "LR", "Weight"), loc="center", cellLoc="left")
    table.auto_set_font_size(False); table.set_fontsize(6); table.scale(1, 1.25)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#D0D0D0"); cell.set_linewidth(0.4)
    _save(fig, output / "Figure_risk_niche_CCC")


def figure_ccc_precision_recall(output):
    _style()
    output = Path(output)
    records = json.loads((output / "robustness_grid.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for ax, metric, title in zip(axes, ("CCC_precision", "CCC_recall"), ("Active-edge precision", "Active-edge recall")):
        for method in ("Full", "No-filter", "CCC-only"):
            means, sds = [], []
            for noise in (0.0, 0.05, 0.10):
                values = [_method(row, method)["spatial"][metric] for row in records if row["noise"] == noise]
                means.append(np.mean(values)); sds.append(np.std(values, ddof=1))
            ax.errorbar((0, 0.05, 0.10), means, yerr=sds, color=COLORS[method], marker="o", lw=1, capsize=2, label=method)
        ax.set_title(title); ax.set_xlabel("Expression noise"); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Recovery")
    axes[1].legend(fontsize=6)
    _save(fig, output / "Figure_CCC_precision_recall")


def make_all_figures(output):
    figure_main(output)
    figure_example_maps(output)
    figure_risk_network(output)
    figure_ccc_precision_recall(output)
