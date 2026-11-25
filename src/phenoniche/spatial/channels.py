"""Prepare panel-normalized expression for complete ligand–receptor pairs."""

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

        pair_types = {}
        labels = lr.get("signaling_type", ["Unknown"] * len(lr))
        for (l, r), label in zip(lr[["ligand", "receptor"]].itertuples(index=False, name=None), labels, strict=True):
            if label not in {"Contact", "Secreted", "ECM", "Unknown", "Ambiguous"}:
                raise ValueError(f"Invalid signaling type: {label}")
            pair_types.setdefault((arm(l), arm(r)), set()).add(label)
        pairs = sorted(pair_types)
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
        self.signaling_types = np.array([
            next(iter(pair_types[p])) if len(pair_types[p]) == 1 else "Ambiguous"
            for p in pairs
        ])
