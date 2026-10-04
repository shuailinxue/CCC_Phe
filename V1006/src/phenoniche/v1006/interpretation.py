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


def export_model_selection(prepared, config, output, table, candidates, selected_k, reason, comparator=None):
    """Save post-training H32 consolidation results; this never touches AE checkpoints."""
    from pathlib import Path

    output = Path(output)
    selection_dir = output / "model_selection"
    selected_dir = output / "selected_model"
    comparator_dir = output / "K8_comparator"
    for directory in (selection_dir, selected_dir, comparator_dir):
        directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(selection_dir / "K4_15_metrics.csv", index=False)
    mappings = []
    for k, result in candidates.items():
        frame = result["mapping"].copy()
        frame.insert(0, "K", k)
        mappings.append(frame)
    pd.concat(mappings, ignore_index=True).to_csv(selection_dir / "program_to_niche_mappings.csv", index=False)

    def save_candidate(result, directory):
        q, h = result["Q"], result["H"]
        labels, confidence, mapping = result["labels"], result["confidence"], result["mapping"]
        np.save(directory / "Q.npy", q)
        np.save(directory / "H.npy", h)
        np.save(directory / "niche_labels.npy", labels)
        np.save(directory / "assignment_confidence.npy", confidence)
        mapping.to_csv(directory / "program32_to_niche.csv", index=False)
        counts = np.bincount(labels, minlength=h.shape[0])
        count_table = pd.DataFrame({"niche": np.arange(1, h.shape[0] + 1), "n_cells": counts,
                                    "fraction": counts / len(labels), "mean_soft_usage": q.mean(0)})
        count_table.to_csv(directory / "niche_counts.csv", index=False)
        assignments = prepared.cells.copy()
        assignments["x"] = np.asarray(prepared.coordinates)[:, 0]
        assignments["y"] = np.asarray(prepared.coordinates)[:, 1]
        assignments["niche"] = labels + 1
        assignments["assignment_confidence_relative"] = confidence
        activity_path = output / "activity.npy"
        if activity_path.is_file():
            assignments["CCC_activity"] = np.load(activity_path, mmap_mode="r")
        assignments.to_csv(directory / "niche_assignments.csv.gz", index=False, compression="gzip")
        top = export_top_markers(h, prepared.features, directory, config.top_ccc)
        return count_table, top

    selected_counts, _ = save_candidate(candidates[selected_k], selected_dir)
    comparator = candidates[config.final_niches] if comparator is None else comparator
    k8_counts, _ = save_candidate(comparator, comparator_dir)
    selected_row = table.loc[table.K == selected_k].iloc[0]
    k8_row = table.loc[table.K == config.final_niches].iloc[0]
    summary = {
        "consolidation_revision": config.consolidation_revision,
        "method": "MiniBatchKMeans on cell-level sqrt(P32) plus standardized log CCC activity",
        "K8_comparator_method": "original trained Q8_direct architectural grouping",
        "selected_K": int(selected_k), "comparison_K": int(config.final_niches),
        "selection_reason": reason,
        "selected_silhouette": float(selected_row.silhouette),
        "selected_adjacent_K_stability": float(selected_row.adjacent_K_stability),
        "selected_max_niche_cell_fraction": float(selected_row.max_niche_cell_fraction),
        "selected_tiny_niche_fraction": float(selected_row.tiny_niche_fraction),
        "selected_mean_assignment_confidence": float(selected_row.mean_assignment_confidence),
        "selected_mean_H_program_separation": float(selected_row.mean_H_program_separation),
        "selected_niche_cell_counts": selected_counts.n_cells.astype(int).tolist(),
        "selected_niche_cell_fractions": selected_counts.fraction.tolist(),
        "K8_silhouette": float(k8_row.silhouette),
        "K8_adjacent_K_stability": float(k8_row.adjacent_K_stability),
        "K8_max_niche_cell_fraction": float(k8_counts.fraction.max()),
        "K8_tiny_niche_fraction": float(k8_counts.loc[k8_counts.fraction < config.tiny_niche_fraction_threshold, "fraction"].sum()),
        "K8_mean_assignment_confidence": float(np.asarray(comparator["confidence"]).mean()),
        "K8_niche_cell_counts": k8_counts.n_cells.astype(int).tolist(),
        "network_retrained": False,
    }
    (output / "model_selection_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
