"""Model-faithfulness check: reset individual input channels and recompute NNLS margins."""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import nnls
from scipy.spatial.distance import cdist

import lung3d_v2_dictionary as dictionary
from lung3d_v2 import OUT


def run():
    d = OUT / "directed_dictionary/C4"
    with np.load(d / "load_programs.npz") as model:
        H, centers, scale, chosen, original = (
            model[k] for k in ("loadings", "centroids", "scale", "chosen", "activity")
        )
    raw = sparse.load_npz(d / "pilot_scores.npz")
    opportunity = sparse.load_npz(d / "pilot_opportunities.npz")
    dictionary.N_LR = pd.read_csv(OUT / "revised_3d/fit/lr.tsv", sep="\t").shape[0]
    x, _ = dictionary.transform(raw, opportunity, chosen, "load")
    channels = pd.read_csv(d / "load_feature_index.tsv", sep="\t")
    data = pd.read_csv(d / "load_pilot_cells.tsv", sep="\t")
    report = pd.read_csv(d / "load_subclass_channels.tsv", sep="\t")
    coordinate = np.linalg.cholesky(H @ H.T + np.eye(len(H)) * 1e-8)

    def scores(rhs):
        sol = np.linalg.solve(coordinate, rhs.T).T
        return np.array([nnls(coordinate.T, b)[0] for b in sol])

    rng = np.random.default_rng(74)
    rows = []
    for label in sorted(data.subclass.unique()):
        inside = np.flatnonzero(data.subclass.to_numpy() == label)
        selected = rng.choice(inside, min(len(inside), 320), replace=False)
        original_act = scores(x[selected] @ H.T)
        ds = cdist(np.log1p(original_act / scale), centers, "sqeuclidean")
        match = ds.argmin(axis=1) == label - 1
        ids = selected[match]
        if not len(ids):
            continue
        options = ds[match].copy()
        options[:, label - 1] = np.inf
        competitor = options.argmin(axis=1)
        margin = options.min(axis=1) - ds[match, label - 1]
        rhs = x[ids] @ H.T
        lookup = report[report.subclass.eq(label)][channels.columns].drop_duplicates()
        candidates = channels.reset_index().merge(lookup, on=channels.columns.tolist())[
            "index"
        ]
        xmeans = x[data.subclass.ne(label)].mean(axis=0)
        for index in candidates:
            diff = xmeans[index] - x[ids, index]
            new = scores(rhs + diff[:, None] * H[:, index])
            changed = cdist(np.log1p(new / scale), centers, "sqeuclidean")
            post = changed[np.arange(len(ids)), competitor] - changed[:, label - 1]
            rows.append(
                (
                    label,
                    index,
                    len(ids),
                    float(match.mean()),
                    float((margin - post).mean()),
                    float((changed.argmin(axis=1) != label - 1).mean()),
                )
            )
    out = pd.DataFrame(
        rows,
        columns="subclass index checked_cells replay_agreement mean_margin_drop reassignment_fraction".split(),
    ).merge(channels.reset_index(), on="index")
    out.sort_values(["subclass", "mean_margin_drop"], ascending=[True, False]).to_csv(
        d / "load_channel_perturbation.tsv", sep="\t", index=False
    )
    print(
        out.sort_values(["subclass", "mean_margin_drop"], ascending=[True, False])
        .groupby("subclass")
        .head(7)
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    run()
