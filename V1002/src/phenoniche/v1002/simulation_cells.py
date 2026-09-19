from dataclasses import dataclass
import math
import numpy as np
import torch
from phenoniche.v1001.neighborhoods import build_neighborhoods
from phenoniche.v1002.lr_atlas import LRAtlas


CELL_TYPES = ("Tumor", "Macrophage", "T_cell", "Fibroblast", "Endothelial", "B_cell", "Myeloid", "NK_cell")
NICHE_NAMES = ("Risk", "Neutral twin", "Protective", "Neutral 4", "Neutral 5")
CHARACTERISTIC_TYPES = ((0, 1, 2), (0, 1, 2), (2, 3, 4), (4, 5, 6), (5, 6, 7))


@dataclass
class SimulationSpec:
    atlas: LRAtlas
    full_lr_indices: np.ndarray
    genes: tuple
    baseline_mean: np.ndarray
    active_edges: tuple
    hi_truth: np.ndarray
    expression_delta: np.ndarray


@dataclass
class SpatialSimulation:
    coordinates: np.ndarray
    labels: np.ndarray
    cell_types: np.ndarray
    expression: np.ndarray
    hc_truth: np.ndarray
    ws_truth: np.ndarray
    neighborhoods: object
    cs: np.ndarray
    communication: np.ndarray
    opportunity: np.ndarray
    pair_support: np.ndarray
    coverage_mask: np.ndarray
    final_mask: np.ndarray


def build_simulation_spec(full_atlas, seed=1401, panel_seed_lr=150):
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(len(full_atlas))[:panel_seed_lr]
    assay_genes = {gene for index in chosen for gene in full_atlas.interactions[int(index)].ligand_components + full_atlas.interactions[int(index)].receptor_components}
    measurable = np.array([index for index, row in enumerate(full_atlas.interactions)
                           if all(gene in assay_genes for gene in row.ligand_components + row.receptor_components)], dtype=np.int64)
    interactions = tuple(full_atlas.interactions[int(index)] for index in measurable)
    atlas = LRAtlas(interactions, {"n_raw_lr": len(full_atlas), "n_unique_lr": len(interactions)}, full_atlas.source_path)
    genes = tuple(sorted({gene for row in interactions for gene in row.ligand_components + row.receptor_components}))
    gene_index = {gene: index for index, gene in enumerate(genes)}
    baseline = np.full((8, len(genes)), 0.025, dtype=np.float32)
    for gene in range(len(genes)):
        preferred = rng.choice(8, size=2, replace=False)
        baseline[preferred, gene] = rng.uniform(0.8, 1.4, 2)
    lr_count = len(atlas)
    shared_lr = rng.choice(lr_count, size=5, replace=False)
    shared_pairs = rng.choice(64, size=5, replace=False)
    active = []
    used = set()
    for niche in range(5):
        pairs = np.array([a * 8 + b for a in CHARACTERISTIC_TYPES[niche] for b in CHARACTERISTIC_TYPES[niche] if a != b])
        candidates = np.array([int(pair) * lr_count + int(lr) for pair in pairs for lr in range(lr_count)], dtype=np.int64)
        available = np.array([value for value in candidates if value not in used])
        selected = rng.choice(available, size=115, replace=False)
        used.update(selected.tolist())
        for feature in selected:
            active.append((niche, int(feature), float(rng.uniform(0.8, 1.2)), "specific"))
        for pair, lr in zip(shared_pairs, shared_lr):
            active.append((niche, int(pair * lr_count + lr), float(rng.uniform(0.20, 0.35)), "shared"))
    hi = np.full((5, 64 * lr_count), 1e-8, dtype=np.float32)
    delta = np.zeros((5, 8, len(genes)), dtype=np.float32)
    for niche, feature, strength, category in active:
        pair, lr = divmod(feature, lr_count)
        sender, receiver = divmod(pair, 8)
        hi[niche, feature] = strength
        row = atlas.interactions[lr]
        amplitude = 2.8 if category == "specific" else 0.8
        for gene in row.ligand_components:
            baseline[sender, gene_index[gene]] = max(baseline[sender, gene_index[gene]], 0.15)
            delta[niche, sender, gene_index[gene]] += amplitude / len(row.ligand_components)
        for gene in row.receptor_components:
            baseline[receiver, gene_index[gene]] = max(baseline[receiver, gene_index[gene]], 0.15)
            delta[niche, receiver, gene_index[gene]] += amplitude / len(row.receptor_components)
    hi /= hi.sum(1, keepdims=True)
    return SimulationSpec(atlas, measurable, genes, baseline, tuple(active), hi, delta)


def composition_truth(purity):
    values = np.empty((5, 8), dtype=np.float32)
    for niche, characteristic in enumerate(CHARACTERISTIC_TYPES):
        row = np.full(8, (1 - purity) / 5)
        row[list(characteristic)] = purity / 3
        values[niche] = row / row.sum()
    values[1] = values[0]
    return values


def _layout(niche_size, seed):
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(np.arange(45), np.arange(40), indexing="ij")
    coordinates = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
    coordinates += rng.normal(0, 0.16, coordinates.shape)
    centers = np.array([[8, 10], [29, 9], [11, 28], [29, 30], [20, 20]], dtype=float)
    labels = np.zeros(len(coordinates), dtype=np.int64)
    available = np.ones(len(coordinates), dtype=bool)
    for niche, center in enumerate(centers, start=1):
        dx = coordinates[:, 0] - center[0]
        dy = coordinates[:, 1] - center[1]
        angle = 0.45 * niche
        u = np.cos(angle) * dx + np.sin(angle) * dy
        v = -np.sin(angle) * dx + np.cos(angle) * dy
        distance = (u / (1.0 + 0.14 * niche)) ** 2 + (v / (1.5 - 0.08 * niche)) ** 2
        irregularity = 0.32 * np.sin(0.7 * dx + niche) + 0.24 * np.cos(0.55 * dy - niche)
        score = -distance + irregularity + rng.normal(0, 0.08, len(coordinates))
        order = np.argsort(-np.where(available, score, -np.inf))[:niche_size]
        labels[order] = niche
        available[order] = False
    return coordinates, labels


def _cell_types(labels, hc, seed):
    rng = np.random.default_rng(seed)
    result = np.empty(len(labels), dtype=np.int64)
    background = np.full(8, 1 / 8)
    for index, label in enumerate(labels):
        center = background if label == 0 else hc[label - 1]
        local = rng.dirichlet(center * 80)
        result[index] = rng.choice(8, p=local)
    return result


def _expression(labels, cell_types, spec, noise, seed):
    rng = np.random.default_rng(seed)
    mean = spec.baseline_mean[cell_types].astype(np.float64)
    domain = labels > 0
    mean[domain] += spec.expression_delta[labels[domain] - 1, cell_types[domain]]
    if noise > 0:
        mean *= rng.lognormal(0, noise, mean.shape)
    dispersion = 2.0
    probability = dispersion / (dispersion + mean)
    counts = rng.negative_binomial(dispersion, probability).astype(np.float32)
    if noise > 0:
        counts[rng.random(counts.shape) < noise] = 0
    return counts


def _side_expression(expression, genes, atlas):
    index = {gene: position for position, gene in enumerate(genes)}
    ligand = np.empty((len(expression), len(atlas)), dtype=np.float32)
    receptor = np.empty_like(ligand)
    for lr, row in enumerate(atlas.interactions):
        ligand[:, lr] = np.prod(expression[:, [index[g] for g in row.ligand_components]], axis=1) ** (1 / len(row.ligand_components))
        receptor[:, lr] = np.prod(expression[:, [index[g] for g in row.receptor_components]], axis=1) ** (1 / len(row.receptor_components))
    return ligand, receptor


def aggregate_pairwise_ccc(coordinates, cell_types, ligand, receptor, n_types,
                           sigma, members=None, anchor_weights=None, feature_mask=None,
                           batch_size=64, lr_batch_size=32, tau=1e-4, device="cpu",
                           compute_signal=True, communication_out=None):
    """Aggregate ordered physical cell pairs in bounded anchor and LR batches.

    Inputs are either global cell arrays plus ``members`` or local arrays of
    shape (anchor, cell, ...). The same weighted pairs define opportunity and
    CCC; only the identical physical cell is excluded. ``feature_mask`` uses
    flattened (sender type, receiver type, LR) order.
    """
    coordinates = np.asarray(coordinates)
    cell_types = np.asarray(cell_types)
    ligand = np.asarray(ligand)
    receptor = np.asarray(receptor)
    if members is None:
        if coordinates.ndim != 3 or cell_types.ndim != 2 or ligand.ndim != 3:
            raise ValueError("Local inputs must be anchor-by-cell arrays")
        n_anchors, k = cell_types.shape
    else:
        members = np.asarray(members)
        if members.ndim != 2 or coordinates.ndim != 2 or cell_types.ndim != 1 or ligand.ndim != 2:
            raise ValueError("Global inputs require two-dimensional members")
        n_anchors, k = members.shape
        if np.any(members < 0) or np.any(members >= len(cell_types)):
            raise ValueError("members contain invalid cell indices")
    if k < 2 or ligand.shape != receptor.shape or sigma <= 0 or tau < 0:
        raise ValueError("Invalid pairwise CCC dimensions or normalization")
    if anchor_weights is None:
        anchor_weights = np.ones((n_anchors, k), dtype=np.float32)
    else:
        anchor_weights = np.asarray(anchor_weights, dtype=np.float32)
    if anchor_weights.shape != (n_anchors, k):
        raise ValueError("anchor_weights must match neighborhood shape")
    n_lr = ligand.shape[-1]
    if feature_mask is None:
        feature_mask = np.ones((n_types * n_types, n_lr), dtype=bool)
    else:
        feature_mask = np.asarray(feature_mask, dtype=bool).reshape(n_types * n_types, n_lr)
    selected = np.flatnonzero(feature_mask.ravel())
    composition = np.empty((n_anchors, n_types), dtype=np.float32)
    opportunity = np.empty((n_anchors, n_types * n_types), dtype=np.float32)
    communication = None
    if compute_signal:
        communication = (np.empty((n_anchors, len(selected)), dtype=np.float32)
                         if communication_out is None else communication_out)
        if communication.shape != (n_anchors, len(selected)):
            raise ValueError("communication_out has the wrong shape")
    sender_index, receiver_index = np.where(~np.eye(k, dtype=bool))
    sender_pos = torch.as_tensor(sender_index, device=device)
    receiver_pos = torch.as_tensor(receiver_index, device=device)
    pair_count = n_types * n_types
    for start in range(0, n_anchors, batch_size):
        stop = min(start + batch_size, n_anchors)
        index = slice(start, stop)
        local_ids = None if members is None else members[index]
        local_xyz = coordinates[index] if local_ids is None else coordinates[local_ids]
        local_types = cell_types[index] if local_ids is None else cell_types[local_ids]
        local_ligand = ligand[index] if local_ids is None else ligand[local_ids]
        local_receptor = receptor[index] if local_ids is None else receptor[local_ids]
        local_types = torch.as_tensor(local_types.astype(np.int64), device=device)
        xyz = torch.as_tensor(local_xyz, dtype=torch.float32, device=device)
        weights = torch.as_tensor(anchor_weights[index], device=device)
        l = torch.as_tensor(local_ligand, dtype=torch.float32, device=device)
        r = torch.as_tensor(local_receptor, dtype=torch.float32, device=device)
        if torch.any(local_types < 0) or torch.any(local_types >= n_types):
            raise ValueError("cell type outside 0..n_types-1")
        composition[index] = (torch.nn.functional.one_hot(local_types, n_types).float()
                              * weights[:, :, None]).sum(1).div(weights.sum(1)[:, None]).cpu().numpy()
        distance2 = ((xyz[:, sender_pos] - xyz[:, receiver_pos]) ** 2).sum(-1)
        pair_weights = weights[:, sender_pos] * weights[:, receiver_pos]
        pair_weights *= torch.exp(-distance2 / (2 * sigma ** 2))
        if local_ids is not None:
            distinct = local_ids[:, sender_index] != local_ids[:, receiver_index]
            pair_weights *= torch.as_tensor(distinct, device=device)
        pair_type = local_types[:, sender_pos] * n_types + local_types[:, receiver_pos]
        flat_pair = pair_type + torch.arange(stop - start, device=device)[:, None] * pair_count
        flat_pair = flat_pair.reshape(-1)
        opp = torch.zeros((stop - start) * pair_count, device=device)
        opp.scatter_add_(0, flat_pair, pair_weights.reshape(-1))
        opportunity[index] = opp.reshape(stop - start, pair_count).cpu().numpy()
        if not compute_signal:
            continue
        for q0 in range(0, n_lr, lr_batch_size):
            q1 = min(q0 + lr_batch_size, n_lr)
            mask_chunk = feature_mask[:, q0:q1].reshape(-1)
            if not mask_chunk.any():
                continue
            signal = torch.sqrt(torch.clamp(l[:, sender_pos, q0:q1] * r[:, receiver_pos, q0:q1], min=0))
            values = (signal * pair_weights[:, :, None]).reshape(-1, q1 - q0)
            raw = torch.zeros(((stop - start) * pair_count, q1 - q0), device=device)
            raw.scatter_add_(0, flat_pair[:, None].expand_as(values), values)
            normalized = raw.reshape(stop - start, pair_count, q1 - q0) / (opp.reshape(stop - start, pair_count, 1) + tau)
            local_selected = np.flatnonzero(mask_chunk)
            pair_ids, lr_ids = np.divmod(local_selected, q1 - q0)
            global_ids = pair_ids * n_lr + q0 + lr_ids
            output_cols = np.searchsorted(selected, global_ids)
            communication[index, output_cols] = normalized.reshape(stop - start, -1)[:, local_selected].cpu().numpy()
    return composition, communication, opportunity


def _aggregate_views(coordinates, labels, cell_types, ligand, receptor, neighborhoods, device="cuda"):
    cells = len(coordinates)
    k = np.diff(neighborhoods.offsets)[0]
    members = neighborhoods.indices.reshape(cells, k)
    anchor_weights = neighborhoods.weights.reshape(cells, k).astype(np.float32)
    cs, communication, opportunity = aggregate_pairwise_ccc(
        coordinates, cell_types, ligand, receptor, 8, sigma=0.8,
        members=members, anchor_weights=anchor_weights, device=device,
    )
    true_ws = np.zeros((cells, 5), dtype=np.float32)
    local_labels = labels[members]
    for niche in range(5):
        true_ws[:, niche] = ((local_labels == niche + 1) * anchor_weights).sum(1)
    true_ws += 0.002
    true_ws /= true_ws.sum(1, keepdims=True)
    pair_support = (opportunity > 0).sum(0).astype(np.int64)
    return cs, true_ws, communication, opportunity, pair_support


def _coverage_mask(expression, cell_types, spec):
    ligand, receptor = _side_expression(expression, spec.genes, spec.atlas)
    coverage_l = np.stack([(ligand[cell_types == value] > 0).mean(0) for value in range(8)])
    coverage_r = np.stack([(receptor[cell_types == value] > 0).mean(0) for value in range(8)])
    mask = np.zeros((64, len(spec.atlas)), dtype=bool)
    for sender in range(8):
        for receiver in range(8):
            mask[sender * 8 + receiver] = (coverage_l[sender] >= 0.10) & (coverage_r[receiver] >= 0.10)
    return ligand, receptor, mask


def simulate_spatial(spec, purity, niche_size, noise, seed, device="cuda"):
    coordinates, labels = _layout(niche_size, seed)
    hc = composition_truth(purity)
    cell_types = _cell_types(labels, hc, seed + 101)
    expression = _expression(labels, cell_types, spec, noise, seed + 202)
    ligand, receptor, coverage = _coverage_mask(expression, cell_types, spec)
    neighborhoods = build_neighborhoods(coordinates, k=15, sigma=0.8)
    cs, ws, communication, opportunity, pair_support = _aggregate_views(
        coordinates, labels, cell_types, ligand, receptor, neighborhoods, device=device
    )
    supported = pair_support >= max(20, math.ceil(0.01 * len(coordinates)))
    final = coverage & supported[:, None]
    return SpatialSimulation(coordinates, labels, cell_types, expression, hc, ws, neighborhoods, cs,
                             communication, opportunity, pair_support, coverage, final)


def simulate_bulk(spec, hc, noise, seed, patients=320):
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        raw = rng.lognormal(-0.1, 0.50, (patients, 5))
        raw[:, 4] = rng.lognormal(0.1, 1.20, patients)
        pi = raw / raw.sum(1, keepdims=True)
        if abs(np.corrcoef(pi[:, 0], pi[:, 1])[0, 1]) < 0.10:
            break
    else:
        raise RuntimeError("Unable to generate independent twin-niche exposures")
    cb = np.maximum(pi @ hc + rng.normal(0, 0.01 + noise * 0.02, (patients, 8)), 0)
    cb /= cb.sum(1, keepdims=True)
    expression = spec.baseline_mean[None, :, :] + np.einsum("pk,kcg->pcg", pi, spec.expression_delta, optimize=True)
    expression *= rng.lognormal(0, 0.03 + noise, expression.shape)
    expression = np.maximum(expression, 0).astype(np.float32)
    beta = np.array([4.0, 0.0, -4.0, 0.0, 0.0], dtype=np.float32)
    eta = pi @ beta
    failure = rng.exponential(size=patients) / (0.05 * np.exp(eta))
    censor = rng.exponential(scale=np.median(failure) * 1.8, size=patients)
    time = np.maximum(np.minimum(failure, censor), np.finfo(np.float32).tiny).astype(np.float32)
    event = (failure <= censor).astype(np.float32)
    return pi.astype(np.float32), cb.astype(np.float32), expression, time, event, beta, eta.astype(np.float32)


def bulk_potential(expression, spec):
    ligand, receptor = _side_expression(expression.reshape(-1, expression.shape[2]), spec.genes, spec.atlas)
    ligand = ligand.reshape(expression.shape[0], 8, -1)
    receptor = receptor.reshape(expression.shape[0], 8, -1)
    return np.sqrt(ligand[:, :, None, :] * receptor[:, None, :, :]).reshape(expression.shape[0], -1).astype(np.float32)
