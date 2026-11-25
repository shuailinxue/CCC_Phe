"""Window fibers indexed by ordered sender and receiver types; LR is the last axis."""

import re

import numpy as np
from scipy import sparse


class Channels:
    def __init__(self, counts, genes, lr):
        data = sparse.csr_matrix(counts, dtype=np.float32)
        if np.any(~np.isfinite(data.data)) or np.any(data.data < 0):
            raise ValueError("Counts must be finite and nonnegative")
        if not genes.is_unique:
            raise ValueError("Gene names must be unique")
        lookup = dict(zip(genes, range(len(genes))))

        def arm(name):
            parts = (
                (name,)
                if name in lookup
                else tuple(sorted(set(re.split(r"[,_+]", name))))
            )
            return parts if all(g in lookup for g in parts) else ()

        pairs = sorted(
            {
                (arm(l), arm(r))
                for l, r in lr[["ligand", "receptor"]].itertuples(
                    index=False, name=None
                )
            }
        )
        pairs = [(l, r) for l, r in pairs if l and r]
        if not pairs:
            raise ValueError("No complete ligand–receptor pairs on panel")
        total = np.asarray(data.sum(axis=1)).ravel()
        data = data.multiply(10000 / np.maximum(total[:, None], 1)).tocsc()
        arms = sorted({a for pair in pairs for a in pair})
        columns = []
        for parts in arms:
            product = data[:, lookup[parts[0]]].power(1 / len(parts))
            for gene in parts[1:]:
                product = product.multiply(data[:, lookup[gene]].power(1 / len(parts)))
            columns.append(product.sqrt())
        self.expression = sparse.hstack(columns, format="csr")
        index = dict(zip(arms, range(len(arms))))
        self.left, self.right = np.array([(index[l], index[r]) for l, r in pairs]).T
        self.names = [("+".join(l), "+".join(r)) for l, r in pairs]


class WindowTensor:
    def __init__(
        self, windows, contacts, coarse, types, channels, sigma, full_interface=False
    ):
        self.windows, self.contacts = windows, contacts
        self.coarse, self.types = coarse, types
        self.channels, self.sigma = channels, sigma
        self.full_interface = full_interface
        self.n_types = types.max() + 1

    def fibers(self, center, context):
        """Return observed type pairs, opportunity-normalized LR and opportunity weights."""
        a, b = context
        start, stop = self.windows.indptr[center : center + 2]
        members = np.append(self.windows.indices[start:stop], center)
        radial = np.exp(
            -0.5 * (np.append(self.windows.data[start:stop], 0) / self.sigma) ** 2
        )
        keep = np.isin(self.coarse[members], context)
        members, radial = members[keep], radial[keep]
        edges = self.contacts[members][:, members].tocoo()
        i, j = edges.row, edges.col
        if a != b and not self.full_interface:
            keep = self.coarse[members[i]] != self.coarse[members[j]]
        else:
            keep = np.ones(len(i), dtype=bool)
        i, j = i[keep], j[keep]
        weight = (
            radial[i] * radial[j] * np.exp(-0.5 * (edges.data[keep] / self.sigma) ** 2)
        )
        # Keep flank direction distinct even when both sides have the same cell type.
        direction = (self.coarse[members[i]] != a).astype(int)
        if self.full_interface:
            direction = 2 * direction + (self.coarse[members[j]] != a)
        key = (
            direction * self.n_types + self.types[members[i]]
        ) * self.n_types + self.types[members[j]]
        codes, inv = np.unique(key, return_inverse=True)
        opportunity = np.bincount(inv, weights=weight, minlength=len(codes))
        ch = self.channels
        values = np.zeros((len(codes), len(ch.names)), dtype=np.float32)
        if len(i):
            expr = ch.expression[members].toarray()
            for index in range(len(codes)):
                same = inv == index
                senders, si = np.unique(i[same], return_inverse=True)
                receivers, ri = np.unique(j[same], return_inverse=True)
                kernel = sparse.csr_matrix(
                    (weight[same], (si, ri)), shape=(len(senders), len(receivers))
                )
                lhs = expr[senders]
                rhs = kernel @ expr[receivers]
                for first in range(0, len(ch.names), 128):
                    sl = slice(first, first + 128)
                    values[index, sl] = (
                        lhs[:, ch.left[sl]] * rhs[:, ch.right[sl]]
                    ).sum(axis=0) / opportunity[index]
        return codes // self.n_types, codes % self.n_types, values, opportunity
