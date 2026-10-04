import json
import numpy as np
import pandas as pd

from .consolidation import cosine_matrix


def _feature_label(row):
    if "ccc" in row and pd.notna(row["ccc"]): return str(row["ccc"])
    return f'{row["sender"]} → {row["receiver"]} | {row["ligand"]}–{row["receptor"]}'


def empirical_profiles(prepared, weights, epsilon=1e-8, chunk_size=8192):
    matrix = np.load(prepared.matrix_path, mmap_mode="r"); k = weights.shape[1]; f = len(prepared.retained_indices)
    weighted = np.zeros((k, f), dtype=np.float64); total = np.zeros(k); global_sum = np.zeros(f)
    for start in range(0, len(weights), chunk_size):
        stop = min(start + chunk_size, len(weights)); q = np.asarray(weights[start:stop], dtype=np.float64)
        x = np.asarray(matrix[start:stop], dtype=np.float64)[:, prepared.retained_indices]
        weighted += q.T @ x; total += q.sum(0); global_sum += x.sum(0)
    mean = weighted / np.maximum(total[:, None], epsilon); global_mean = global_sum / len(weights)
    enrichment = np.log2((mean + epsilon) / (global_mean[None, :] + epsilon))
    return mean.astype(np.float32), enrichment.astype(np.float32)


def export_top_markers(h8, features, output, top_n=15):
    records = []
    for niche in range(h8.shape[0]):
        for rank, feature in enumerate(np.argsort(-h8[niche], kind="stable")[:top_n], 1):
            row = features.iloc[feature]
            records.append({"niche": niche + 1, "rank": rank, "feature_id": int(feature),
                            "ccc": _feature_label(row), "sender": row["sender"], "receiver": row["receiver"],
                            "ligand": row["ligand"], "receptor": row["receptor"], "weight": float(h8[niche, feature])})
    top = pd.DataFrame(records); top.to_csv(output / "top_ccc.csv", index=False)
    for columns, filename, label in [(["sender", "receiver"], "top_sender_receiver.csv", "sender_receiver"),
                                      (["ligand", "receptor"], "top_lr.csv", "lr")]:
        rows = []
        for niche in range(h8.shape[0]):
            table = features[columns].copy(); table["weight"] = h8[niche]
            table = table.groupby(columns, as_index=False).weight.sum().sort_values("weight", ascending=False).head(top_n)
            table.insert(0, "rank", np.arange(1, len(table) + 1)); table.insert(0, "niche", niche + 1)
            table[label] = table[columns].astype(str).agg(" → ".join, axis=1) if label == "sender_receiver" else table[columns].astype(str).agg("–".join, axis=1)
            rows.append(table)
        pd.concat(rows, ignore_index=True).to_csv(output / filename, index=False)
    return top


def export_results(prepared, config, output, arrays, h32, q8, h8, labels, confidence, mapping, checkpoints):
    np.save(output / "Q8.npy", q8); np.save(output / "H8.npy", h8)
    np.save(output / "niche_labels.npy", labels.astype(np.int16)); np.save(output / "assignment_confidence.npy", confidence)
    mapping.to_csv(output / "latent32_to_niche8.csv", index=False)
    similarity = cosine_matrix(h32, config.epsilon)
    pd.DataFrame(similarity, index=np.arange(1, 33), columns=np.arange(1, 33)).to_csv(output / "H32_cosine_similarity.csv")
    empirical, enrichment = empirical_profiles(prepared, q8, config.epsilon)
    np.save(output / "H8_empirical_mean.npy", empirical); np.save(output / "H8_empirical_log2_enrichment.npy", enrichment)
    counts = np.bincount(labels, minlength=config.final_niches); fractions = counts / len(labels)
    pd.DataFrame({"niche": np.arange(1, config.final_niches + 1), "n_cells": counts,
                  "fraction": fractions, "mean_soft_usage": np.asarray(q8).mean(0)}).to_csv(output / "niche_counts.csv", index=False)
    assignments = prepared.cells.copy(); assignments["x"] = np.asarray(prepared.coordinates)[:, 0]; assignments["y"] = np.asarray(prepared.coordinates)[:, 1]
    assignments["niche"] = labels + 1; assignments["assignment_confidence_relative"] = confidence
    assignments["CCC_activity"] = np.asarray(arrays["activity"])
    assignments.to_csv(output / "niche_assignments.csv.gz", index=False, compression="gzip")
    export_top_markers(h8, prepared.features, output, config.top_ccc)
    off = similarity[np.triu_indices_from(similarity, 1)]; stage2 = checkpoints["stage2b"]
    group_ids = mapping.final_niche_id.to_numpy()
    cross_mask = group_ids[:, None] != group_ids[None, :]
    within_mask = (group_ids[:, None] == group_ids[None, :]) & ~np.eye(len(group_ids), dtype=bool)
    summary = {"revision": config.revision,
        "model": "dual-decoder nonlinear AE with magnitude-separated simplex CCC program bottleneck",
        "n_cells": len(labels), "n_input_features": prepared.matrix_shape[1],
        "n_retained_features": len(prepared.retained_indices), "original_CCC_features": prepared.matrix_shape[1],
        "retained_CCC_features": len(prepared.retained_indices), "latent_dim": config.latent_dim,
        "final_niches": config.final_niches, "stage1_best_epoch": int(checkpoints["stage1"]["epoch"]),
        "stage1_train_reconstruction": float(checkpoints["stage1"]["train_reconstruction"]),
        "stage1_validation_reconstruction": float(checkpoints["stage1"]["validation_reconstruction"]),
        "stage2a_best_epoch": int(checkpoints["stage2a"]["epoch"]), "stage2b_best_epoch": int(stage2["epoch"]),
        "stage2b_rollback": bool(stage2["rollback_to_stage2a"]),
        "stage2_train_ae": float(stage2["final_train"]["ae"]),
        "stage2_validation_ae": float(stage2["final_validation"]["ae"]),
        "stage2_validation_linear": float(stage2["final_validation"]["linear"]),
        "stage2_validation_latent_drift": float(stage2["final_validation"]["preserve"]),
        "effective_programs": float(stage2["final_validation"]["effective_programs"]),
        "max_H32_offdiagonal_cosine": float(off.max(initial=-1)),
        "max_H32_cross_niche_cosine": float(similarity[cross_mask].max(initial=-1)),
        "max_H32_within_niche_cosine": float(similarity[within_mask].max(initial=-1)),
        "program32_mean_mixture": np.asarray(arrays["P32"]).mean(0).tolist(),
        "program32_mean_exposure": np.asarray(arrays["Z32"]).mean(0).tolist(),
        "niche_cell_counts": counts.tolist(), "niche_cell_fractions": fractions.tolist(),
        "niche_soft_usage": np.asarray(q8).mean(0).tolist(),
        "mean_assignment_confidence": float(confidence.mean()), "median_assignment_confidence": float(np.median(confidence)),
        "niche_collapse": bool(np.count_nonzero(counts) < config.final_niches),
        "severe_imbalance": bool(fractions.max() > .75 or np.count_nonzero(fractions < .005) > 0)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2)); return summary
