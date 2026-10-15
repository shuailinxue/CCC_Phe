"""Label-blind priors and physical-fold isolation shared by new adapters."""
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]


def physical_owners(members, folds):
    owner = np.full(len(folds), -1, np.int8)
    for fold in range(3):
        cells = np.unique(members[folds == fold])
        if (owner[cells] >= 0).any():
            raise ValueError('Physical cells overlap across training/evaluation folds')
        owner[cells] = fold
    return owner


def isolate_graph(graph, owner, keep_diagonal=False):
    g = graph.tocoo()
    keep = owner[g.row] == owner[g.col]
    if not keep_diagonal:
        keep &= g.row != g.col
    return sp.csr_matrix((g.data[keep], (g.row[keep], g.col[keep])), shape=g.shape)


def row_normalize(graph):
    return sp.diags(1 / np.maximum(np.asarray(graph.sum(1)).ravel(), 1)) @ graph


def panel_lr_prior(genes, output=None):
    """Project the entire public LR atlas, never the planted MI-LR table."""
    path = ROOT / 'data/commuspace_human_lr_atlas.tsv'
    full = pd.read_csv(path, sep='\t')
    panel = set(genes)
    matched = full[full.ligand.isin(panel) & full.receptor.isin(panel)].drop_duplicates(['ligand', 'receptor']).reset_index(drop=True)
    if matched.empty:
        raise ValueError('No genuine public LR prior matches the expression panel')
    if output is not None:
        matched.to_csv(Path(output) / 'public_panel_LR_prior.csv', index=False)
    return matched
