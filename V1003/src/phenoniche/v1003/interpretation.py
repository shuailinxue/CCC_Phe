from pathlib import Path
import json
import numpy as np
import pandas as pd


def cosine_matrix(dictionary, epsilon=1e-12):
    dictionary = np.asarray(dictionary, dtype=np.float64)
    norm = np.linalg.norm(dictionary, axis=1, keepdims=True)
    normalized = dictionary / np.maximum(norm, epsilon)
    return np.clip(normalized @ normalized.T, -1, 1)


def interpret_dictionary(dictionary, feature_table, top_n):
    dictionary = np.asarray(dictionary)
    if dictionary.ndim != 2 or dictionary.shape[1] != len(feature_table):
        raise ValueError("Dictionary and feature table are inconsistent")
    ccc_records = []
    pair_records = []
    lr_records = []
    for niche in range(dictionary.shape[0]):
        weights = dictionary[niche]
        for rank, index in enumerate(np.argsort(-weights, kind="stable")[:top_n], 1):
            row = feature_table.iloc[int(index)]
            ccc_records.append({
                "niche": niche + 1,
                "rank": rank,
                "feature_id": int(index),
                "ccc": row["ccc"],
                "sender": row["sender"],
                "receiver": row["receiver"],
                "lr_id": row["lr_id"],
                "ligand": row["ligand"],
                "receptor": row["receptor"],
                "weight": float(weights[index]),
            })
        table = feature_table[["sender", "receiver", "lr_id", "ligand", "receptor"]].copy()
        table["weight"] = weights
        pair = table.groupby(["sender", "receiver"], as_index=False, observed=True)["weight"].sum()
        pair = pair.sort_values("weight", ascending=False, kind="stable").head(top_n)
        for rank, row in enumerate(pair.itertuples(index=False), 1):
            pair_records.append({"niche": niche + 1, "rank": rank,
                                 "sender": row.sender, "receiver": row.receiver,
                                 "sender_receiver": f"{row.sender} → {row.receiver}",
                                 "weight": float(row.weight)})
        lr = table.groupby(["lr_id", "ligand", "receptor"], as_index=False, observed=True)["weight"].sum()
        lr = lr.sort_values("weight", ascending=False, kind="stable").head(top_n)
        for rank, row in enumerate(lr.itertuples(index=False), 1):
            lr_records.append({"niche": niche + 1, "rank": rank,
                               "lr_id": row.lr_id, "ligand": row.ligand,
                               "receptor": row.receptor,
                               "lr": f"{row.ligand}–{row.receptor}",
                               "weight": float(row.weight)})
    return pd.DataFrame(ccc_records), pd.DataFrame(pair_records), pd.DataFrame(lr_records)


def export_results(labels, confidence, dictionary, cached, training_summary, config, output_dir):
    output_dir = Path(output_dir)
    counts = np.bincount(labels, minlength=config.niches)
    usage = np.load(output_dir / "Z.npy", mmap_mode="r").mean(axis=0)
    assignment = cached["cells"].copy()
    assignment["x"] = cached["coordinates"][:, 0]
    assignment["y"] = cached["coordinates"][:, 1]
    assignment["niche_label"] = labels.astype(np.int64) + 1
    assignment["assignment_confidence"] = confidence
    assignment["ccc_magnitude"] = cached["magnitude"]
    assignment.to_csv(output_dir / "assignments.csv.gz", index=False, compression="gzip")
    count_table = pd.DataFrame({
        "niche": np.arange(1, config.niches + 1),
        "n_cells": counts,
        "fraction": counts / counts.sum(),
        "mean_usage": usage,
    })
    count_table.to_csv(output_dir / "niche_counts.csv", index=False)
    ccc, pair, lr = interpret_dictionary(dictionary, cached["features"], config.top_ccc)
    active = (counts > 0) & (usage >= config.collapsed_usage_threshold)
    status = pd.DataFrame({
        "niche": np.arange(1, config.niches + 1),
        "n_cells": counts,
        "mean_usage": usage,
        "active_factor": active,
    })
    ccc = ccc.merge(status, on="niche", how="left", validate="many_to_one")
    pair = pair.merge(status, on="niche", how="left", validate="many_to_one")
    lr = lr.merge(status, on="niche", how="left", validate="many_to_one")
    ccc.to_csv(output_dir / "top_ccc.csv", index=False)
    pair.to_csv(output_dir / "top_sender_receiver.csv", index=False)
    lr.to_csv(output_dir / "top_lr.csv", index=False)
    similarity = cosine_matrix(dictionary, config.epsilon)
    pd.DataFrame(similarity,
                 index=[f"Niche {index}" for index in range(1, config.niches + 1)],
                 columns=[f"Niche {index}" for index in range(1, config.niches + 1)]).to_csv(
                     output_dir / "H_cosine_similarity.csv")
    off_diagonal = similarity[np.triu_indices(config.niches, 1)]
    low_usage = np.flatnonzero(usage < config.collapsed_usage_threshold) + 1
    empty_assignment = np.flatnonzero(counts == 0) + 1
    duplicate_pairs = []
    for first, second in zip(*np.triu_indices(config.niches, 1)):
        if similarity[first, second] >= config.duplicate_h_cosine_threshold:
            duplicate_pairs.append({"niche_a": int(first + 1), "niche_b": int(second + 1),
                                    "cosine": float(similarity[first, second])})
    summary = {
        "dataset": "Xenium Prime 5K Breast",
        "seed": config.seed,
        "K": config.niches,
        "n_cells": int(len(labels)),
        "retained_ccc_features": int(dictionary.shape[1]),
        "training": training_summary,
        "mean_niche_usage": [float(value) for value in usage],
        "assignment_confidence": {
            "mean": float(np.mean(confidence)),
            "median": float(np.median(confidence)),
            "q10": float(np.quantile(confidence, 0.10)),
            "q90": float(np.quantile(confidence, 0.90)),
        },
        "all_niches_assigned": bool(np.all(counts > 0)),
        "active_niches": (np.flatnonzero(active) + 1).tolist(),
        "inactive_niches": (np.flatnonzero(~active) + 1).tolist(),
        "niche_collapse": bool(len(low_usage) or len(empty_assignment)),
        "low_usage_niches_below_1pct": low_usage.tolist(),
        "empty_assignment_niches": empty_assignment.tolist(),
        "maximum_off_diagonal_H_cosine": float(off_diagonal.max()),
        "duplicated_H": bool(duplicate_pairs),
        "duplicated_H_pairs_at_or_above_0.95": duplicate_pairs,
        "diagnostic_thresholds": {
            "collapsed_usage": config.collapsed_usage_threshold,
            "duplicated_H_cosine": config.duplicate_h_cosine_threshold,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
