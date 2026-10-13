import numpy as np
import torch


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

