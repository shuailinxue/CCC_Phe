import json
import numpy as np
import pandas as pd


def _feature_label(row):
    if "ccc" in row and pd.notna(row["ccc"]): return str(row["ccc"])
    return f'{row["sender"]} → {row["receiver"]} | {row["ligand"]}–{row["receptor"]}'


def empirical_profiles(prepared, q8, epsilon=1e-8, chunk_size=8192):
    matrix = np.load(prepared.matrix_path, mmap_mode="r"); k = q8.shape[1]; f = len(prepared.retained_indices)
    weighted = np.zeros((k, f), dtype=np.float64); global_sum = np.zeros(f, dtype=np.float64); usage = np.zeros(k)
    for start in range(0, len(q8), chunk_size):
        stop = min(start + chunk_size, len(q8)); q = np.asarray(q8[start:stop], dtype=np.float64)
        x = np.asarray(matrix[start:stop], dtype=np.float64)[:, prepared.retained_indices]
        weighted += q.T @ x; global_sum += x.sum(0); usage += q.sum(0)
    mean = weighted / np.maximum(usage[:, None], epsilon)
    global_mean = global_sum / len(q8)
    log2_enrichment = np.log2((mean + epsilon) / (global_mean[None, :] + epsilon))
    presentation = mean / np.maximum(mean.sum(1, keepdims=True), epsilon)
    return mean.astype(np.float32), presentation.astype(np.float32), log2_enrichment.astype(np.float32), usage


def export_top_markers(presentation, enrichment, empirical_mean, features, output, top_n=15):
    records = []
    for niche in range(presentation.shape[0]):
        order = np.argsort(-enrichment[niche], kind="stable")[:top_n]
        for rank, feature in enumerate(order, 1):
            row = features.iloc[feature]
            records.append({"niche": niche + 1, "rank": rank, "feature_id": int(feature),
                            "ccc": _feature_label(row), "sender": row["sender"], "receiver": row["receiver"],
                            "ligand": row["ligand"], "receptor": row["receptor"],
                            "weight": float(presentation[niche, feature]),
                            "mean_CCC_activity": float(empirical_mean[niche, feature]),
                            "log2_enrichment_vs_global": float(enrichment[niche, feature])})
    top = pd.DataFrame(records); top.to_csv(output / "top_ccc.csv", index=False)
    for columns, filename, label in [(["sender", "receiver"], "top_sender_receiver.csv", "sender_receiver"),
                                      (["ligand", "receptor"], "top_lr.csv", "lr")]:
        rows = []
        for niche in range(presentation.shape[0]):
            table = features[columns].copy(); table["weight"] = presentation[niche]
            table = table.groupby(columns, as_index=False).weight.sum().sort_values("weight", ascending=False).head(top_n)
            table.insert(0, "rank", np.arange(1, len(table) + 1)); table.insert(0, "niche", niche + 1)
            table[label] = table[columns].astype(str).agg(" → ".join, axis=1) if label == "sender_receiver" else table[columns].astype(str).agg("–".join, axis=1)
            rows.append(table)
        pd.concat(rows, ignore_index=True).to_csv(output / filename, index=False)
    return top


def export_results(prepared, config, output, z32, q8, prototypes, checkpoints):
    labels = np.asarray(q8).argmax(1); confidence = np.asarray(q8).max(1)
    empirical, h8, enrichment, soft_usage = empirical_profiles(prepared, q8, config.epsilon)
    np.save(output / "H8_empirical_mean.npy", empirical); np.save(output / "H8.npy", h8)
    np.save(output / "H8_log2_enrichment.npy", enrichment)
    np.save(output / "niche_labels.npy", labels.astype(np.int16)); np.save(output / "assignment_confidence.npy", confidence.astype(np.float32))
    counts = np.bincount(labels, minlength=config.final_niches); fractions = counts / len(labels)
    pd.DataFrame({"niche": np.arange(1, config.final_niches + 1), "n_cells": counts,
                  "fraction": fractions, "mean_soft_usage": np.asarray(q8).mean(0)}).to_csv(output / "niche_counts.csv", index=False)
    assignments = prepared.cells.copy(); assignments["x"] = np.asarray(prepared.coordinates)[:, 0]; assignments["y"] = np.asarray(prepared.coordinates)[:, 1]
    assignments["niche"] = labels + 1; assignments["assignment_confidence_relative"] = confidence
    assignments.to_csv(output / "niche_assignments.csv.gz", index=False, compression="gzip")
    export_top_markers(h8, enrichment, empirical, prepared.features, output, config.top_ccc)
    similarity = prototypes @ prototypes.T; off = similarity[np.triu_indices_from(similarity, 1)]
    pd.DataFrame(similarity, index=np.arange(1, 9), columns=np.arange(1, 9)).to_csv(output / "prototype_cosine_similarity.csv")
    stage2 = checkpoints["stage2"]
    summary = {
        "revision": config.revision, "model": "nonlinear autoencoder plus learnable cosine prototypes",
        "n_cells": len(labels), "n_input_features": prepared.matrix_shape[1],
        "n_retained_features": len(prepared.retained_indices), "original_CCC_features": prepared.matrix_shape[1],
        "retained_CCC_features": len(prepared.retained_indices), "latent_dim": config.latent_dim,
        "prototype_cluster_dim": int(prototypes.shape[1]),
        "final_niches": config.final_niches, "stage1_best_epoch": int(checkpoints["stage1"]["epoch"]),
        "stage1_train_reconstruction": float(checkpoints["stage1"]["train_reconstruction"]),
        "stage1_validation_reconstruction": float(checkpoints["stage1"]["validation_reconstruction"]),
        "stage2_best_epoch": int(stage2["epoch"]),
        "stage2_train_reconstruction": float(stage2["final_train"]["reconstruction"]),
        "stage2_validation_reconstruction": float(stage2["final_validation"]["reconstruction"]),
        "stage2_validation_cluster_loss": float(stage2["final_validation"]["cluster"]),
        "stage2_validation_min_soft_usage": float(stage2["final_validation"]["min_soft_usage"]),
        "stage2_validation_max_hard_fraction": float(stage2["final_validation"]["max_hard_fraction"]),
        "niche_cell_counts": counts.tolist(), "niche_cell_fractions": fractions.tolist(),
        "niche_soft_usage": (soft_usage / soft_usage.sum()).tolist(),
        "mean_assignment_confidence": float(confidence.mean()), "median_assignment_confidence": float(np.median(confidence)),
        "min_assignment_confidence": float(confidence.min()), "prototype_max_offdiagonal_cosine": float(off.max(initial=-1)),
        "niche_collapse": bool(np.count_nonzero(counts) < config.final_niches),
        "severe_imbalance": bool(fractions.max() > .75 or np.count_nonzero(fractions < .005) > 0),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2)); return summary
