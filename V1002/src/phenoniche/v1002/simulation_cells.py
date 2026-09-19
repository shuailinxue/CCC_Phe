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


def _aggregate_views(coordinates, labels, cell_types, ligand, receptor, neighborhoods, device="cuda"):
    cells = len(coordinates)
    lr_count = ligand.shape[1]
    k = np.diff(neighborhoods.offsets)[0]
    members = neighborhoods.indices.reshape(cells, k)
    anchor_weights = neighborhoods.weights.reshape(cells, k).astype(np.float32)
    cs = np.zeros((cells, 8), dtype=np.float32)
    true_ws = np.zeros((cells, 5), dtype=np.float32)
    pair_support = np.zeros(64, dtype=np.int64)
    opportunity = np.zeros((cells, 64), dtype=np.float32)
    local_types = cell_types[members]
    for cell_type in range(8):
        cs[:, cell_type] = ((local_types == cell_type) * anchor_weights).sum(1) / anchor_weights.sum(1)
    local_labels = labels[members]
    for niche in range(5):
        true_ws[:, niche] = ((local_labels == niche + 1) * anchor_weights).sum(1)
    true_ws += 0.002
    true_ws /= true_ws.sum(1, keepdims=True)
    counts = np.stack([(local_types == value).sum(1) for value in range(8)], axis=1)
    for sender in range(8):
        for receiver in range(8):
            present = (counts[:, sender] > 0) & (counts[:, receiver] > 0)
            if sender == receiver:
                present = counts[:, sender] > 1
            pair_support[sender * 8 + receiver] = int(present.sum())
    sender_pos = np.repeat(np.arange(k), k)
    receiver_pos = np.tile(np.arange(k), k)
    valid = sender_pos != receiver_pos
    sender_pos, receiver_pos = sender_pos[valid], receiver_pos[valid]
    ligand_t = torch.tensor(np.sqrt(ligand), device=device)
    receptor_t = torch.tensor(np.sqrt(receptor), device=device)
    coordinates_t = torch.tensor(coordinates, device=device)
    result = np.empty((cells, 64, lr_count), dtype=np.float32)
    for start in range(0, cells, 100):
        stop = min(start + 100, cells)
        selected = torch.tensor(members[start:stop], device=device)
        sender = selected[:, sender_pos]
        receiver = selected[:, receiver_pos]
        types_s = torch.tensor(cell_types, device=device)[sender]
        types_r = torch.tensor(cell_types, device=device)[receiver]
        pair = types_s * 8 + types_r
        anchor_w = torch.tensor(anchor_weights[start:stop], device=device)
        distance = torch.linalg.vector_norm(coordinates_t[sender] - coordinates_t[receiver], dim=2)
        weights = anchor_w[:, sender_pos] * anchor_w[:, receiver_pos] * torch.exp(-(distance ** 2) / (2 * 0.8 ** 2))
        one_hot = torch.nn.functional.one_hot(pair, 64).float().transpose(1, 2)
        raw = torch.bmm(one_hot, ligand_t[sender] * receptor_t[receiver] * weights[:, :, None])
        opp = torch.bmm(one_hot, weights[:, :, None]).squeeze(2)
        normalized = raw / (opp[:, :, None] + 1e-4)
        result[start:stop] = normalized.cpu().numpy()
        opportunity[start:stop] = opp.cpu().numpy()
    return cs, true_ws, result.reshape(cells, -1), opportunity, pair_support


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
