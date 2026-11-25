"""Equivalent sparse LR accumulation for the full-volume experiment."""

import numpy as np
from numba import njit

from phenoniche.spatial.channels import WindowTensor


@njit(cache=True)
def _fiber(
    center,
    a,
    b,
    windows_ptr,
    windows_idx,
    window_weight,
    contact_ptr,
    contact_idx,
    contact_weight,
    coarse,
    types,
    n_types,
    ligptr,
    ligidx,
    ligval,
    expression,
    right,
    full_interface,
    scratch,
):
    begin, end = windows_ptr[center : center + 2]
    members = np.empty(end - begin + 1, np.int32)
    members[0] = center
    scratch[center] = 1.0
    n = 1
    for e in range(begin, end):
        j = windows_idx[e]
        if coarse[j] == a or coarse[j] == b:
            scratch[j] = window_weight[e]
            members[n] = j
            n += 1
    # Sort by code as in the reference implementation, including zero-expression fibers.
    codes = np.full((4 if full_interface else 2) * n_types * n_types, -1, np.int32)
    for i in members[:n]:
        prefix = (
            (coarse[i] != a) * (2 if full_interface else 1) * n_types + types[i]
        ) * n_types
        for e in range(contact_ptr[i], contact_ptr[i + 1]):
            j = contact_idx[e]
            if scratch[j] > 0 and (full_interface or a == b or coarse[i] != coarse[j]):
                codes[
                    prefix
                    + types[j]
                    + (coarse[j] != a) * n_types * n_types * full_interface
                ] = 0
    observed = np.flatnonzero(codes == 0)
    codes[observed] = np.arange(len(observed), dtype=np.int32)
    value = np.zeros((len(observed), len(right)), np.float64)
    opportunity = np.zeros(len(observed))
    for i in members[:n]:
        prefix = (
            (coarse[i] != a) * (2 if full_interface else 1) * n_types + types[i]
        ) * n_types
        for e in range(contact_ptr[i], contact_ptr[i + 1]):
            j = contact_idx[e]
            if scratch[j] == 0 or (
                not full_interface and a != b and coarse[i] == coarse[j]
            ):
                continue
            row = codes[
                prefix
                + types[j]
                + (coarse[j] != a) * n_types * n_types * full_interface
            ]
            w = scratch[i] * scratch[j] * contact_weight[e]
            opportunity[row] += w
            for k in range(ligptr[i], ligptr[i + 1]):
                lr = ligidx[k]
                value[row, lr] += w * ligval[k] * expression[j, right[lr]]
    for i in members[:n]:
        scratch[i] = 0.0
    for row in range(len(observed)):
        value[row] /= opportunity[row]
    return (
        observed // n_types,
        observed % n_types,
        value.astype(np.float32),
        opportunity,
    )


class FastWindows:
    def __init__(
        self, graph, coarse, types, channels, sigma, pooled=False, full_interface=False
    ):
        weight = np.exp(-0.5 * (graph.data / sigma) ** 2)
        self.n_types = int(types.max() + 1)
        lig = channels.expression[:, channels.left].tocsr()
        lig.eliminate_zeros()
        self.args = (
            graph.indptr,
            graph.indices,
            weight,
            graph.indptr,
            graph.indices,
            weight,
            coarse,
            types,
            self.n_types,
            lig.indptr,
            lig.indices,
            lig.data,
            channels.expression.toarray(),
            channels.right,
        )
        self.pooled = pooled
        self.full_interface = full_interface
        self.balance_channels = full_interface
        self.n_senders = (4 if full_interface else 2) * self.n_types
        self.scratch = np.zeros(len(types))

    def fibers(self, center, key):
        return _fiber(
            center,
            *((0, 0) if self.pooled else key),
            *self.args,
            self.full_interface,
            self.scratch,
        )


def verify_kernel(fast, graph, coarse, types, channels, cfg, groups):
    reference = WindowTensor(
        graph,
        graph,
        coarse,
        types,
        channels,
        cfg.sigma_um,
        full_interface=fast.full_interface,
    )
    keys = list(groups)
    for key in keys[:6] + [k for k in keys if k[0] != k[1]][:6]:
        for i in groups[key][:: max(1, len(groups[key]) // 2)][:2]:
            for x, y in zip(fast.fibers(i, key), reference.fibers(i, key)):
                np.testing.assert_allclose(x, y, rtol=3e-5, atol=1e-5)
