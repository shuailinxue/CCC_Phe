"""Fixed-radius HBC1 contexts and chunked physical-pair CCC aggregation.

The context contains every cell within ``radius_um``.  A 5-sigma cutoff is
used only when evaluating the Gaussian *pair* kernel; it never removes a cell
from the context or the composition calculation.  Numerator and opportunity
use exactly the same retained physical pairs and weights.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
import torch


@dataclass(frozen=True)
class RadiusContext:
    offsets: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    radius_um: float
    anchor_sigma_um: float

    @property
    def counts(self):
        return np.diff(self.offsets)

    def members(self, anchor):
        start, stop = self.offsets[anchor:anchor + 2]
        return self.indices[start:stop], self.weights[start:stop]


def build_radius_context(coordinates, radius_um=200.0, anchor_sigma_um=20.0,
                         query_batch_size=512):
    """Pack all radius-neighbor indices in CSR form without a global list of lists."""
    xy = np.asarray(coordinates, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("coordinates must be a finite N-by-2 array")
    if radius_um <= 0 or anchor_sigma_um <= 0 or query_batch_size < 1:
        raise ValueError("radius, sigma and query batch size must be positive")
    tree = cKDTree(xy)
    counts = tree.query_ball_point(xy, radius_um, return_length=True, workers=-1)
    offsets = np.empty(len(xy) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    indices = np.empty(int(offsets[-1]), dtype=np.int32)
    weights = np.empty(int(offsets[-1]), dtype=np.float32)
    for start in range(0, len(xy), query_batch_size):
        stop = min(start + query_batch_size, len(xy))
        selections = tree.query_ball_point(xy[start:stop], radius_um,
                                           return_sorted=True, workers=-1)
        lengths = np.fromiter((len(row) for row in selections), dtype=np.int64,
                              count=stop - start)
        if not np.array_equal(lengths, counts[start:stop]):
            raise RuntimeError("Radius-query lengths changed between passes")
        flat = np.fromiter((j for row in selections for j in row), dtype=np.int32,
                           count=int(lengths.sum()))
        owner = np.repeat(np.arange(start, stop), lengths)
        distance2 = np.sum((xy[flat] - xy[owner]) ** 2, axis=1)
        if np.any(distance2 > (radius_um + 1e-6) ** 2):
            raise RuntimeError("Radius query returned an out-of-radius cell")
        section = slice(offsets[start], offsets[stop])
        indices[section] = flat
        weights[section] = np.exp(-distance2 / (2 * anchor_sigma_um ** 2))
    return RadiusContext(offsets, indices, weights, float(radius_um),
                         float(anchor_sigma_um))


def radius_composition(context, cell_types, n_types):
    """Original anchor-weighted composition over every radius member."""
    types = np.asarray(cell_types)
    if types.ndim != 1 or len(types) != len(context.counts):
        raise ValueError("cell types must match anchor order")
    result = np.empty((len(types), n_types), dtype=np.float32)
    for anchor in range(len(types)):
        ids, w = context.members(anchor)
        result[anchor] = np.bincount(types[ids], weights=w,
                                     minlength=n_types).astype(np.float32)
        result[anchor] /= w.sum()
    return result


def _pair_batch(coordinates, cell_types, ligand, receptor, context, start, stop,
                sigma_um, pair_cutoff_um, device):
    """Make only local padded arrays; never materialize an all-cell distance matrix."""
    lengths = context.counts[start:stop]
    width = int(lengths.max())
    size = stop - start
    ids = np.repeat(np.arange(start, stop, dtype=np.int32)[:, None], width, axis=1)
    weights = np.zeros((size, width), dtype=np.float32)
    for row, anchor in enumerate(range(start, stop)):
        local, w = context.members(anchor)
        ids[row, :len(local)] = local
        weights[row, :len(local)] = w
    xyz = torch.as_tensor(coordinates[ids], dtype=torch.float32, device=device)
    w = torch.as_tensor(weights, device=device)
    types = torch.as_tensor(cell_types[ids].astype(np.int64), device=device)
    valid = torch.arange(width, device=device)[None, :] < torch.as_tensor(lengths, device=device)[:, None]
    distance2 = (xyz[:, :, None, :] - xyz[:, None, :, :]).square().sum(-1)
    valid_pairs = valid[:, :, None] & valid[:, None, :]
    valid_pairs &= ~torch.eye(width, dtype=torch.bool, device=device)[None, :, :]
    if pair_cutoff_um is not None:
        valid_pairs &= distance2 <= pair_cutoff_um ** 2
    batch_index, sender, receiver = torch.where(valid_pairs)
    pair_weight = w[batch_index, sender] * w[batch_index, receiver]
    pair_weight *= torch.exp(-distance2[batch_index, sender, receiver] / (2 * sigma_um ** 2))
    pair_type = types[batch_index, sender], types[batch_index, receiver]
    return ids, batch_index, sender, receiver, pair_weight, pair_type


def aggregate_radius_pairwise_ccc(coordinates, cell_types, ligand, receptor,
                                  n_types, context, sigma_um=20.0,
                                  feature_mask=None, batch_size=16,
                                  lr_batch_size=8, pair_cutoff_um=100.0,
                                  tau=1e-4, device="cpu", compute_signal=True,
                                  communication_out=None, opportunity_input=None):
    """Opportunity and directed CCC from identical, distinct physical pairs.

    ``pair_cutoff_um=5*sigma`` is a numerical Gaussian-tail approximation.
    Set it to ``None`` for exact pair evaluation on small datasets.  Radius
    membership and weighted composition are unaffected by this cutoff.
    """
    xy = np.asarray(coordinates, dtype=np.float32)
    types = np.asarray(cell_types)
    ligand = np.asarray(ligand, dtype=np.float32)
    receptor = np.asarray(receptor, dtype=np.float32)
    n = len(xy)
    if len(types) != n or len(context.counts) != n or ligand.shape != receptor.shape or ligand.shape[0] != n:
        raise ValueError("Cell arrays and radius contexts must share their row order")
    if sigma_um <= 0 or tau < 0 or batch_size < 1 or lr_batch_size < 1:
        raise ValueError("Invalid CCC parameters")
    if pair_cutoff_um is not None and pair_cutoff_um <= 0:
        raise ValueError("Pair cutoff must be positive or None")
    if np.any(types < 0) or np.any(types >= n_types):
        raise ValueError("Cell type outside 0..n_types-1")
    n_lr = ligand.shape[1]
    mask = (np.ones((n_types * n_types, n_lr), dtype=bool) if feature_mask is None
            else np.asarray(feature_mask, dtype=bool).reshape(n_types * n_types, n_lr))
    selected = np.flatnonzero(mask.ravel())
    if opportunity_input is not None:
        if not compute_signal:
            raise ValueError("Precomputed opportunity is only useful with signal")
        opportunity = np.asarray(opportunity_input, dtype=np.float32)
        if opportunity.shape != (n, n_types * n_types):
            raise ValueError("opportunity_input shape is wrong")
    else:
        opportunity = np.empty((n, n_types * n_types), dtype=np.float32)
    communication = None
    if compute_signal:
        communication = (np.empty((n, len(selected)), dtype=np.float32)
                         if communication_out is None else communication_out)
        if communication.shape != (n, len(selected)):
            raise ValueError("communication_out shape is wrong")
    chunk_columns = []
    for q0 in range(0, n_lr, lr_batch_size):
        q1 = min(q0 + lr_batch_size, n_lr)
        pairs, qlocal = np.where(mask[:, q0:q1])
        global_ids = pairs * n_lr + q0 + qlocal
        chunk_columns.append((q0, q1, pairs, qlocal,
                              np.searchsorted(selected, global_ids)))
    pair_count = n_types * n_types
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        ids, b, u, v, pw, (sender_type, receiver_type) = _pair_batch(
            xy, types, ligand, receptor, context, start, stop,
            sigma_um, pair_cutoff_um, device)
        group = b * pair_count + sender_type * n_types + receiver_type
        if opportunity_input is None:
            opp = torch.zeros((stop - start) * pair_count, dtype=torch.float32, device=device)
            opp.scatter_add_(0, group, pw)
            opp = opp.reshape(stop - start, pair_count)
            opportunity[start:stop] = opp.cpu().numpy()
        else:
            opp = torch.as_tensor(opportunity[start:stop], device=device)
        if not compute_signal:
            continue
        local_l = torch.as_tensor(ligand[ids], device=device)
        local_r = torch.as_tensor(receptor[ids], device=device)
        for q0, q1, pairs, qlocal, output_cols in chunk_columns:
            if not len(output_cols):
                continue
            value = torch.sqrt(torch.clamp(local_l[b, u, q0:q1] *
                                           local_r[b, v, q0:q1], min=0))
            value *= pw[:, None]
            raw = torch.zeros(((stop - start) * pair_count, q1 - q0),
                              dtype=torch.float32, device=device)
            raw.scatter_add_(0, group[:, None].expand_as(value), value)
            normalized = raw.reshape(stop - start, pair_count, q1 - q0)
            normalized /= opp[:, :, None] + tau
            communication[start:stop, output_cols] = normalized[:, pairs, qlocal].cpu().numpy()
    return communication, opportunity
