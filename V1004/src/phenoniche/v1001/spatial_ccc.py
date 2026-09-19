from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpatialCCCAggregation:
    composition: np.ndarray
    raw: np.ndarray
    opportunity: np.ndarray
    normalized: np.ndarray
    peripheral_raw: np.ndarray
    audit: dict


def _validate_inputs(neighborhoods, cell_types, cell_parent, cell_niches, hi_tensor, tau, pair_sigma):
    types = np.asarray(cell_types, dtype=np.int64)
    parent = np.asarray(cell_parent, dtype=np.int64)
    niches = np.asarray(cell_niches, dtype=np.float64)
    hi = np.asarray(hi_tensor, dtype=np.float64)
    cells = neighborhoods.cell_coordinates.shape[0]
    if types.shape != (cells,) or parent.shape != (cells,) or niches.shape[0] != cells:
        raise ValueError("Cell attributes must align with cell coordinates")
    if hi.ndim != 4 or hi.shape[0] != niches.shape[1] or hi.shape[1] != hi.shape[2]:
        raise ValueError("HI must have shape K by C by C by Q")
    if (types < 0).any() or (types >= hi.shape[1]).any() or (parent < 0).any():
        raise ValueError("Cell type or parent index is invalid")
    if not np.isfinite(niches).all() or (niches < 0).any() or tau <= 0 or pair_sigma <= 0:
        raise ValueError("Niche activities and aggregation constants must be valid")
    return types, parent, niches, hi


def aggregate_spatial_ccc(neighborhoods, cell_types, cell_parent, cell_niches, hi_tensor,
                          tau=0.0001, baseline=0.001, pair_sigma=0.8):
    types, parent, niches, hi = _validate_inputs(neighborhoods, cell_types, cell_parent, cell_niches,
                                                  hi_tensor, tau, pair_sigma)
    anchors = len(neighborhoods.offsets) - 1
    cell_type_count, programs = hi.shape[1], hi.shape[3]
    composition = np.zeros((anchors, cell_type_count), dtype=np.float64)
    raw = np.zeros((anchors, cell_type_count * cell_type_count, programs), dtype=np.float64)
    opportunity = np.zeros((anchors, cell_type_count * cell_type_count), dtype=np.float64)
    peripheral_raw = np.zeros_like(raw)
    pair_counts = np.zeros(anchors, dtype=np.int64)
    peripheral_counts = np.zeros(anchors, dtype=np.int64)
    for anchor in range(anchors):
        cells, anchor_weights, _ = neighborhoods.members(anchor)
        local_types = types[cells]
        composition[anchor] = np.bincount(local_types, weights=anchor_weights,
                                           minlength=cell_type_count) / anchor_weights.sum()
        size = len(cells)
        sender_position = np.repeat(np.arange(size), size)
        receiver_position = np.tile(np.arange(size), size)
        valid = sender_position != receiver_position
        sender_position = sender_position[valid]
        receiver_position = receiver_position[valid]
        sender = cells[sender_position]
        receiver = cells[receiver_position]
        distance = np.linalg.norm(neighborhoods.cell_coordinates[sender] - neighborhoods.cell_coordinates[receiver], axis=1)
        pair_weight = (anchor_weights[sender_position] * anchor_weights[receiver_position]
                       * np.exp(-(distance ** 2) / (2 * pair_sigma ** 2)))
        pair_feature = types[sender] * cell_type_count + types[receiver]
        opportunity[anchor] = np.bincount(pair_feature, weights=pair_weight,
                                           minlength=cell_type_count * cell_type_count)
        local_niche = 0.5 * (niches[sender] + niches[receiver])
        local_niche /= local_niche.sum(1, keepdims=True)
        pair_dictionary = hi[:, types[sender], types[receiver], :].transpose(1, 0, 2)
        signal = baseline + np.einsum("pk,pkq->pq", local_niche, pair_dictionary, optimize=True)
        peripheral = (parent[sender] != anchor) & (parent[receiver] != anchor)
        for program in range(programs):
            weighted = pair_weight * signal[:, program]
            raw[anchor, :, program] = np.bincount(pair_feature, weights=weighted,
                                                   minlength=cell_type_count * cell_type_count)
            peripheral_raw[anchor, :, program] = np.bincount(pair_feature[peripheral], weights=weighted[peripheral],
                                                              minlength=cell_type_count * cell_type_count)
        pair_counts[anchor] = len(sender)
        peripheral_counts[anchor] = int(peripheral.sum())
    normalized = raw / (opportunity[:, :, None] + tau)
    peripheral_signal = float(peripheral_raw.sum())
    total_signal = float(raw.sum())
    audit = {"anchors": anchors, "cells": len(types), "local_directed_pair_instances": int(pair_counts.sum()),
             "mean_cells_per_neighborhood": float(np.diff(neighborhoods.offsets).mean()),
             "mean_directed_pairs_per_neighborhood": float(pair_counts.mean()),
             "peripheral_pair_instances": int(peripheral_counts.sum()),
             "peripheral_pair_fraction": float(peripheral_counts.sum() / pair_counts.sum()),
             "peripheral_signal_fraction": peripheral_signal / total_signal,
             "peripheral_normalized_l1": float((peripheral_raw / (opportunity[:, :, None] + tau)).sum()),
             "ordered_pairs": True, "self_pairs_excluded": True}
    return SpatialCCCAggregation(composition, raw.reshape(anchors, -1),
                                 np.repeat(opportunity[:, :, None], programs, axis=2).reshape(anchors, -1),
                                 normalized.reshape(anchors, -1), peripheral_raw.reshape(anchors, -1), audit)


def brute_force_spatial_ccc(neighborhoods, cell_types, cell_parent, cell_niches, hi_tensor,
                            tau=0.0001, baseline=0.001, pair_sigma=0.8):
    types, parent, niches, hi = _validate_inputs(neighborhoods, cell_types, cell_parent, cell_niches,
                                                  hi_tensor, tau, pair_sigma)
    anchors = len(neighborhoods.offsets) - 1
    cell_type_count, programs = hi.shape[1], hi.shape[3]
    composition = np.zeros((anchors, cell_type_count), dtype=np.float64)
    raw = np.zeros((anchors, cell_type_count, cell_type_count, programs), dtype=np.float64)
    opportunity = np.zeros((anchors, cell_type_count, cell_type_count), dtype=np.float64)
    peripheral_raw = np.zeros_like(raw)
    for anchor in range(anchors):
        cells, anchor_weights, _ = neighborhoods.members(anchor)
        for position, cell in enumerate(cells):
            composition[anchor, types[cell]] += anchor_weights[position]
        composition[anchor] /= anchor_weights.sum()
        for sender_position, sender in enumerate(cells):
            for receiver_position, receiver in enumerate(cells):
                if sender == receiver:
                    continue
                distance = np.linalg.norm(neighborhoods.cell_coordinates[sender] - neighborhoods.cell_coordinates[receiver])
                weight = (anchor_weights[sender_position] * anchor_weights[receiver_position]
                          * np.exp(-(distance ** 2) / (2 * pair_sigma ** 2)))
                sender_type, receiver_type = types[sender], types[receiver]
                opportunity[anchor, sender_type, receiver_type] += weight
                local_niche = 0.5 * (niches[sender] + niches[receiver])
                local_niche /= local_niche.sum()
                for program in range(programs):
                    value = weight * (baseline + local_niche @ hi[:, sender_type, receiver_type, program])
                    raw[anchor, sender_type, receiver_type, program] += value
                    if parent[sender] != anchor and parent[receiver] != anchor:
                        peripheral_raw[anchor, sender_type, receiver_type, program] += value
    repeated = np.repeat(opportunity[:, :, :, None], programs, axis=3)
    normalized = raw / (repeated + tau)
    return SpatialCCCAggregation(composition, raw.reshape(anchors, -1), repeated.reshape(anchors, -1),
                                 normalized.reshape(anchors, -1), peripheral_raw.reshape(anchors, -1), {})
