from dataclasses import asdict
import json
import math
from pathlib import Path
import resource
import time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from phenoniche.evaluation.metrics import concordance_index, cosine_similarity, pearson_correlation
from phenoniche.evaluation.niche_stability import match_factors
from phenoniche.evaluation.patient_protocol import fit_frozen_cox, split_patients
from phenoniche.v1001.simulation import generate_neighborhood_structured
from phenoniche.v1002.bulk_bridge import atlas_gene_vocabulary, directional_communication_potential
from phenoniche.v1002.large_lr import SparseExpandedMatrix, build_large_spatial_matrix, fit_large_st, summarize_pairs
from phenoniche.v1002.lr_atlas import load_lr_atlas, make_lr_subsets


SEEDS = (31, 32, 33, 34, 35)


def _save(path, value):
    def default(item):
        if isinstance(item, (np.ndarray, torch.Tensor)):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError
    Path(path).write_text(json.dumps(value, indent=2, default=default), encoding="utf-8")


def _key(interaction):
    return interaction.ligand_components, interaction.receptor_components


def _active_specifications(small, seed=20261002):
    rng = np.random.default_rng(seed)
    features = 64 * len(small)
    specific, shared = 120, 30
    order = rng.permutation(features)
    cursor = 0
    specific_sets = []
    for _ in range(6):
        specific_sets.append(order[cursor:cursor + specific])
        cursor += specific
    shared_sets = {}
    for pair in ((0, 2), (1, 3), (4, 5)):
        shared_sets[pair] = order[cursor:cursor + shared]
        cursor += shared
    partner = {0: (0, 2), 2: (0, 2), 1: (1, 3), 3: (1, 3), 4: (4, 5), 5: (4, 5)}
    specs = []
    for niche in range(6):
        selected = np.concatenate((specific_sets[niche], shared_sets[partner[niche]]))
        strength = np.concatenate((rng.uniform(1.0, 1.5, specific), rng.uniform(0.30, 0.45, shared)))
        strength /= strength.sum()
        for index, value in zip(selected, strength):
            pair, lr = divmod(int(index), len(small))
            sender, receiver = divmod(pair, 8)
            specs.append((niche, sender, receiver, _key(small.interactions[lr]), float(value)))
    return specs


def _hi_for_atlas(atlas, specs):
    lookup = {_key(row): index for index, row in enumerate(atlas.interactions)}
    hi = np.full((6, 64, len(atlas)), 1e-8, dtype=np.float32)
    active = []
    for niche, sender, receiver, key, strength in specs:
        lr = lookup[key]
        hi[niche, sender * 8 + receiver, lr] = strength
        active.append((niche, (sender * 8 + receiver) * len(atlas) + lr))
    left, right = hi[0].reshape(-1), hi[1].reshape(-1)
    similarity = float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))
    if similarity >= 0.05:
        raise AssertionError("A/B large-LR truth is not sufficiently distinct")
    return hi, active, similarity


def _molecular_states(w, hi, atlas, seed):
    genes = atlas_gene_vocabulary(atlas)
    gene_index = {gene: index for index, gene in enumerate(genes)}
    state = np.full((6, 8, len(genes)), 0.005, dtype=np.float32)
    for niche in range(6):
        active = np.argwhere(hi[niche] > 1e-7)
        for pair, lr in active:
            sender, receiver = divmod(int(pair), 8)
            value = math.sqrt(float(hi[niche, pair, lr]))
            row = atlas.interactions[int(lr)]
            for gene in row.ligand_components:
                state[niche, sender, gene_index[gene]] += value
            for gene in row.receptor_components:
                state[niche, receiver, gene_index[gene]] += value
    expression = np.einsum("sk,kcg->scg", np.asarray(w), state, optimize=True)
    rng = np.random.default_rng(seed)
    expression = np.maximum(expression + rng.normal(0, 0.005 * np.sqrt(np.mean(expression ** 2)), expression.shape), 0)
    return torch.tensor(expression, dtype=torch.float32), genes


def _potential_stats(expression, genes, atlas, chunk=128):
    square_sum = 0.0
    nonzero = 0
    started = time.perf_counter()
    for start in range(0, len(atlas), chunk):
        value = directional_communication_potential(expression, genes, atlas, lr_start=start, lr_stop=start + chunk)
        square_sum += float(value.square().sum())
        nonzero += int((value > 0).sum())
    elapsed = time.perf_counter() - started
    elements = expression.shape[0] * 64 * len(atlas)
    return math.sqrt(square_sum / elements), nonzero, elements, elapsed


def _scale_matrix(matrix, scale):
    return SparseExpandedMatrix(matrix.base / scale, matrix.lr_count, matrix.active_indices, matrix.active_delta / scale)


def _consensus(results, cs, communication, os):
    factors = [{name: getattr(result, name).numpy() for name in ("WS", "HC", "HI", "HO")} for result in results]
    reference = factors[int(np.argmin([row.history[-1]["weighted_mean_squared_error"] for row in results]))]
    aligned = []
    for current in factors:
        order, _ = match_factors(current, reference)
        aligned.append({name: current[name][order] if name.startswith("H") else current[name][:, order]
                        for name in ("WS", "HC", "HI", "HO")})
    dictionaries = {name: torch.tensor(np.median(np.stack([row[name] for row in aligned]), axis=0), dtype=torch.float32)
                    for name in ("HC", "HI", "HO")}
    scale = dictionaries["HC"].sum(1)
    for name in dictionaries:
        dictionaries[name] /= scale[:, None]
    gram = dictionaries["HC"] @ dictionaries["HC"].T / cs.shape[1]
    gram += dictionaries["HI"] @ dictionaries["HI"].T / communication.shape[1]
    gram += dictionaries["HO"] @ dictionaries["HO"].T / os.shape[1]
    target = cs @ dictionaries["HC"].T / cs.shape[1]
    target += communication.x_h_t(dictionaries["HI"]) / communication.shape[1]
    target += os @ dictionaries["HO"].T / os.shape[1]
    step = 1 / gram.abs().sum(1).max()
    activity = torch.zeros_like(target)
    extrapolated = activity
    momentum = 1.0
    for _ in range(200):
        updated = torch.relu(extrapolated - step * (extrapolated @ gram - target))
        next_momentum = (1 + math.sqrt(1 + 4 * momentum ** 2)) / 2
        extrapolated = updated + (momentum - 1) / next_momentum * (updated - activity)
        activity, momentum = updated, next_momentum
    dictionaries["WS"] = activity
    stability = {}
    for name in ("HI", "WS"):
        values = []
        for left in range(len(aligned)):
            for right in range(left + 1, len(aligned)):
                count = 6
                values.append(np.mean([cosine_similarity(aligned[left][name][:, k], aligned[right][name][:, k])
                                       if name == "WS" else cosine_similarity(aligned[left][name][k], aligned[right][name][k])
                                       for k in range(count)]))
        stability[name] = float(np.mean(values))
    return dictionaries, stability


def _recover(anchor, truth, active):
    learned = {name: anchor[name].numpy() for name in ("WS", "HC", "HI", "HO")}
    target = {name: np.asarray(truth[name]) for name in ("WS", "HC", "HI", "HO")}
    order, _ = match_factors(learned, target)
    rows = []
    active_by_niche = {k: set() for k in range(6)}
    for niche, index in active:
        active_by_niche[niche].add(index)
    for niche, learned_index in enumerate(order):
        scores = learned["HI"][learned_index]
        ranked = np.argsort(-scores)
        labels = np.zeros(len(scores), dtype=np.int8)
        labels[list(active_by_niche[niche])] = 1
        cumulative = np.cumsum(labels[ranked])
        precision = cumulative / np.arange(1, len(scores) + 1)
        auprc = float((precision * labels[ranked]).sum() / labels.sum())
        rows.append({"niche": chr(65 + niche), "HC": cosine_similarity(learned["HC"][learned_index], target["HC"][niche]),
                     "HI": cosine_similarity(scores, target["HI"][niche]),
                     "HO": cosine_similarity(learned["HO"][learned_index], target["HO"][niche]),
                     "WS": pearson_correlation(learned["WS"][:, learned_index], target["WS"][:, niche]),
                     "top50_precision": len(set(ranked[:50]) & active_by_niche[niche]) / 50,
                     "top100_precision": len(set(ranked[:100]) & active_by_niche[niche]) / 100,
                     "top100_recall": len(set(ranked[:100]) & active_by_niche[niche]) / len(active_by_niche[niche]),
                     "AUPRC": auprc, "learned_index": int(learned_index)})
    similarity = np.stack([learned["HI"] @ target["HI"][k] / (np.linalg.norm(learned["HI"], axis=1) * np.linalg.norm(target["HI"][k])) for k in range(2)], 1)
    collision = bool(similarity[:, 0].argmax() == similarity[:, 1].argmax())
    return {"matching": order.tolist(), "per_niche": rows, "overall_HC": float(np.mean([r["HC"] for r in rows])),
            "overall_HI": float(np.mean([r["HI"] for r in rows])), "overall_HO": float(np.mean([r["HO"] for r in rows])),
            "overall_WS": float(np.mean([r["WS"] for r in rows])), "A_B_collision": collision,
            "top100_precision": float(np.mean([r["top100_precision"] for r in rows])),
            "top100_recall": float(np.mean([r["top100_recall"] for r in rows])), "AUPRC": float(np.mean([r["AUPRC"] for r in rows]))}


def _project_expression(expression, genes, atlas, composition, hc, hi, scale, steps=50, chunk=128):
    started = time.perf_counter()
    gram = hc @ hc.T / composition.shape[1] + hi @ hi.T / hi.shape[1]
    target = composition @ hc.T / composition.shape[1]
    shaped = hi.reshape(hi.shape[0], 64, len(atlas))
    for start in range(0, len(atlas), chunk):
        value = directional_communication_potential(expression, genes, atlas, lr_start=start, lr_stop=start + chunk) / scale
        target += value.reshape(value.shape[0], -1) @ shaped[:, :, start:start + value.shape[2]].reshape(hi.shape[0], -1).T / hi.shape[1]
    step = 1 / gram.abs().sum(1).max()
    activity = torch.zeros_like(target)
    extrapolated = activity
    momentum = 1.0
    for _ in range(steps):
        updated = torch.relu(extrapolated - step * (extrapolated @ gram - target))
        next_momentum = (1 + math.sqrt(1 + 4 * momentum ** 2)) / 2
        extrapolated = updated + (momentum - 1) / next_momentum * (updated - activity)
        activity, momentum = updated, next_momentum
    return activity, time.perf_counter() - started


def _run_scale(name, atlas, specs, core, opportunity, local_niche, seeds, output):
    started_total = time.perf_counter()
    hi, active, ab_similarity = _hi_for_atlas(atlas, specs)
    wb, ws = core.structured.truth["WB"].numpy(), core.structured.truth["WS"].numpy()
    bulk_expression, genes = _molecular_states(wb, hi, atlas, 20261002)
    bulk_rms, ib_nnz, ib_elements, bulk_construction = _potential_stats(bulk_expression, genes, atlas)
    cb = core.structured.data.CB
    communication_scale = bulk_rms / float(cb.square().mean().sqrt())
    spatial_started = time.perf_counter()
    spatial = build_large_spatial_matrix(opportunity, local_niche, hi, seed=20261002)
    spatial_construction = time.perf_counter() - spatial_started
    scaled = _scale_matrix(spatial, communication_scale)
    cs = core.structured.data.CS
    topology_scale = float(core.structured.data.OS.square().mean().sqrt() / cs.square().mean().sqrt())
    os = core.structured.data.OS / topology_scale
    results = [fit_large_st(cs, scaled, os, seed=seed) for seed in seeds]
    anchor, stability = _consensus(results, cs, scaled, os)
    truth = {"HC": core.structured.truth["HC"].numpy(), "HI": hi.reshape(6, -1) / communication_scale,
             "HO": core.structured.truth["HO"].numpy() / topology_scale, "WS": ws}
    recovery = _recover(anchor, truth, active)
    bulk_w, projection_seconds = _project_expression(bulk_expression, genes, atlas, cb, anchor["HC"], anchor["HI"], communication_scale)
    matching = recovery["matching"]
    wb_rows = [pearson_correlation(bulk_w[:, matching[k]], wb[:, k]) for k in range(6)]
    split = split_patients(len(wb), seed=90210)
    gamma = fit_frozen_cox(bulk_w[split.train], core.structured.data.time[torch.tensor(split.train)],
                           core.structured.data.event[torch.tensor(split.train)], ridge=0.01)
    cindex = {}
    for partition in ("train", "validation", "test"):
        indices = torch.tensor(getattr(split, partition))
        cindex[partition] = concordance_index(core.structured.data.time[indices].numpy(), core.structured.data.event[indices].numpy(),
                                              (bulk_w[indices] @ gamma).numpy())
    pseudo_expression, pseudo_genes = _molecular_states(ws, hi, atlas, 20261003)
    pseudo_w, pseudo_seconds = _project_expression(pseudo_expression, pseudo_genes, atlas, cs, anchor["HC"], anchor["HI"], communication_scale)
    zero_hi = torch.zeros_like(anchor["HI"])
    composition_w, _ = _project_expression(pseudo_expression, pseudo_genes, atlas, cs, anchor["HC"], zero_hi, communication_scale)
    pseudo_rows = [pearson_correlation(pseudo_w[:, matching[k]], ws[:, k]) for k in range(6)]
    composition_rows = [pearson_correlation(composition_w[:, matching[k]], ws[:, k]) for k in range(6)]
    runtime = {"nLR": len(atlas), "F": 64 * len(atlas), "IS_nnz": int((spatial.base > 0).sum()) * len(atlas),
               "IS_density": float((spatial.base > 0).float().mean()), "HI_parameters": int(6 * 64 * len(atlas)),
               "HI_memory_bytes": int(anchor["HI"].numel() * 4), "peak_RAM_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
               "CCC_construction_seconds": spatial_construction, "ST_epoch_seconds": float(np.mean([r.epoch_seconds for r in results])),
               "total_ST_training_seconds": float(sum(r.training_seconds for r in results)), "bulk_construction_seconds": bulk_construction,
               "bulk_projection_seconds": projection_seconds, "total_scale_seconds": time.perf_counter() - started_total}
    result = {"scale": name, "nLR": len(atlas), "F": 64 * len(atlas), "A_B_truth_HI_cosine": ab_similarity,
              "recovery": recovery, "stability": stability, "bulk": {"overall_WB": float(np.mean(wb_rows)),
              "per_niche_WB": wb_rows, "train_C": cindex["train"], "validation_C": cindex["validation"], "test_C": cindex["test"],
              "gamma_true_order": [float(gamma[matching[k]]) for k in range(6)]},
              "pseudobulk": {"overall": float(np.mean(pseudo_rows)), "per_niche": pseudo_rows,
              "composition_only_overall": float(np.mean(composition_rows)), "composition_only_per_niche": composition_rows,
              "projection_seconds": pseudo_seconds}, "runtime": runtime, "IB_nnz": ib_nnz, "IB_elements": ib_elements}
    _save(output / f"{name.lower()}_lr.json", result)
    return result


def run_stress_test(output, atlas_path):
    output = Path(output)
    atlas = load_lr_atlas(atlas_path)
    subsets = make_lr_subsets(atlas)
    specs = _active_specifications(subsets["Small"])
    core = generate_neighborhood_structured()
    opportunity, local_niche = summarize_pairs(core.neighborhoods, core.cells.cell_type, core.cells.niche_activity)
    results = []
    for name, seeds in (("Small", SEEDS[:3]), ("Medium", SEEDS[:3]), ("Full", SEEDS)):
        results.append(_run_scale(name, subsets[name], specs, core, opportunity, local_niche, seeds, output))
    _save(output / "lr_atlas_audit.json", atlas.audit | {"subsets": {name: len(value) for name, value in subsets.items()}})
    _save(output / "st_recovery.json", {row["scale"]: row["recovery"] for row in results})
    _save(output / "bulk_bridge.json", {row["scale"]: row["bulk"] for row in results})
    _save(output / "pseudobulk_bridge.json", {row["scale"]: row["pseudobulk"] for row in results})
    _save(output / "cox_results.json", {row["scale"]: {key: row["bulk"][key] for key in ("train_C", "validation_C", "test_C", "gamma_true_order")} for row in results})
    _save(output / "large_lr_edge_recovery.json", {row["scale"]: {"top100_precision": row["recovery"]["top100_precision"],
                                                                    "top100_recall": row["recovery"]["top100_recall"],
                                                                    "AUPRC": row["recovery"]["AUPRC"]} for row in results})
    _save(output / "scaling_runtime_memory.json", {row["scale"]: row["runtime"] for row in results})
    table1 = [{"scale": row["scale"], "nLR": row["nLR"], "F": row["F"], "HC": row["recovery"]["overall_HC"],
               "HI": row["recovery"]["overall_HI"], "WS": row["recovery"]["overall_WS"],
               "A_HI": row["recovery"]["per_niche"][0]["HI"], "A_WS": row["recovery"]["per_niche"][0]["WS"],
               "B_HI": row["recovery"]["per_niche"][1]["HI"], "B_WS": row["recovery"]["per_niche"][1]["WS"],
               "collision": row["recovery"]["A_B_collision"], "Top100_precision": row["recovery"]["top100_precision"],
               "Top100_recall": row["recovery"]["top100_recall"], "AUPRC": row["recovery"]["AUPRC"]} for row in results]
    full, medium = results[2], results[1]
    statistical_drop = full["recovery"]["overall_HI"] < medium["recovery"]["overall_HI"] - 0.1 or full["recovery"]["overall_WS"] < medium["recovery"]["overall_WS"] - 0.1
    bulk_drop = full["bulk"]["overall_WB"] < medium["bulk"]["overall_WB"] - 0.1
    computational = full["runtime"]["total_scale_seconds"] > 10 * medium["runtime"]["total_scale_seconds"]
    if statistical_drop and computational:
        conclusion = "E"
    elif computational:
        conclusion = "B"
    elif statistical_drop:
        conclusion = "C"
    elif bulk_drop:
        conclusion = "D"
    else:
        conclusion = "A"
    reference_path = output.parent / "v1001" / "v1001_primary.json"
    v1001_reference = None
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text())["consensus_recovery"]
        v1001_reference = {"label": "V1001-small-program reference", "nLR": 12, "F": 768,
                           "HC": reference["mean_HC_recovery"], "HI": reference["mean_HI_recovery"],
                           "WS": reference["mean_WS_recovery"], "A_HI": reference["per_niche"][0]["HI_recovery"],
                           "A_WS": reference["per_niche"][0]["WS_recovery"], "B_HI": reference["per_niche"][1]["HI_recovery"],
                           "B_WS": reference["per_niche"][1]["WS_recovery"], "collision": reference["HI_only_A_B_collision"]}
    summary = {"status": "complete", "V1001_reference": v1001_reference, "ST_table": table1, "bulk_table": [row["bulk"] | {"scale": row["scale"]} for row in results],
               "pseudobulk_table": [row["pseudobulk"] | {"scale": row["scale"]} for row in results],
               "runtime_table": [row["runtime"] | {"scale": row["scale"]} for row in results], "conclusion": conclusion}
    _save(output / "summary.json", summary)
    return summary
