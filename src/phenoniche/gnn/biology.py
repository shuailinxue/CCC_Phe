"""Audited priors and observed, direction-specific regulatory activity.

No TF-mRNA activity surrogate, and no trainable gene-expression decoder.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from ..ccc.lr_atlas import canonicalize_complex
from .model import physical_weights

DATABASES = {"CollecTRI", "DoRothEA", "TRRUST", "NicheNet", "OmniPath", "SIGNOR"}


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read_prior(path, manifest, branch, synthetic=False):
    path = Path(path)
    if not path.is_file(): return None, {"status": "missing", "path": str(path)}
    meta = manifest.get(branch, {})
    required = ("database", "version", "url", "sha256", "species")
    if any(not meta.get(key) for key in required): raise ValueError(f"{branch}: incomplete prior provenance")
    if meta['database'] not in DATABASES and not (synthetic and meta['database'] == 'synthetic_test'):
        raise ValueError(f"{branch}: unrecognized prior database")
    if not synthetic and meta['species'].lower() not in ('human', 'homo sapiens'):
        raise ValueError("Only human priors match the current panel")
    if checksum(path) != meta['sha256']: raise ValueError(f"{branch}: prior checksum mismatch")
    table = pd.read_csv(path, sep='\t', keep_default_na=False)
    key = 'tf' if branch == 'upstream' else 'ligand'
    if not {key, 'target', 'weight', 'source', 'evidence'}.issubset(table.columns):
        raise ValueError(f"{branch}: missing gene/weight/evidence columns")
    if table[['source', 'evidence']].eq('').any().any(): raise ValueError("Every prior row needs traceable evidence")
    table['weight'] = pd.to_numeric(table.weight, errors='raise')
    if not np.isfinite(table.weight).all() or (table.weight == 0).any(): raise ValueError("Invalid prior weights")
    for column in (key, 'target'):
        table[column] = table[column].astype(str).str.upper().str.strip()
        if table[column].eq('').any(): raise ValueError("Empty prior gene")
    # Contradictory signed evidence is not silently averaged into a label.
    grouped = table.groupby([key, 'target']).weight
    if (grouped.min() * grouped.max() < 0).any(): raise ValueError("Conflicting signed prior rows")
    table = table.drop_duplicates([key, 'target'])
    return table, {"status": "verified", "rows": len(table), **meta}


@dataclass
class RegulatoryPrior:
    mapping: np.ndarray  # F x M, signed TF-ligand / ligand-response links
    genes: tuple
    weights: np.ndarray  # G x M, observed regulon/response score weights
    modules: pd.DataFrame
    branch: str
    audit: dict


def compile_prior(table, features, panel, branch, min_targets=5, min_coverage=.2):
    panel = tuple(str(g).upper() for g in panel); lookup = {g: i for i, g in enumerate(panel)}
    lr_genes = {g for col in ('ligand', 'receptor') for value in features[col] for g in canonicalize_complex(value)}
    key = 'tf' if branch == 'upstream' else 'ligand'
    modules, columns, gene_weights, skipped = [], [], [], []
    if table is None:
        return RegulatoryPrior(np.zeros((len(features), 0), np.float32), panel,
                               np.zeros((len(panel), 0), np.float32), pd.DataFrame(), branch,
                               {'status': 'missing', 'modules': 0, 'TF_target_coverage': None, 'feature_match_rate': None})
    for entity, links in table.groupby(key, sort=True):
        independent = links.loc[~links.target.isin(lr_genes | {entity})]
        covered = independent.loc[independent.target.isin(lookup)]
        coverage = len(covered) / max(len(independent), 1)
        if len(covered) < min_targets or coverage < min_coverage:
            skipped.append({'entity': entity, 'covered': len(covered), 'available_prior_targets': len(independent), 'reason': 'insufficient_independent_panel_targets'})
            continue
        scores = np.zeros(len(panel), np.float32)
        for row in covered.itertuples(): scores[lookup[row.target]] = row.weight
        scores /= max(float(np.linalg.norm(scores)), 1e-8)
        roles = sorted(features.sender.unique() if branch == 'upstream' else features.receiver.unique())
        for cell_type in roles:
            a = np.zeros(len(features), np.float32)
            for f, row in enumerate(features.itertuples()):
                ligand = canonicalize_complex(row.ligand)
                if branch == 'upstream' and row.sender == cell_type:
                    relations = links.set_index('target').weight
                    a[f] = sum(float(relations.get(g, 0)) for g in ligand) / len(ligand)
                elif branch == 'downstream' and row.receiver == cell_type and entity in ligand:
                    # A ligand-target prior is ligand-level; do not invent receptor specificity.
                    a[f] = 1. / len(ligand)
            if not np.any(a): continue
            a /= np.abs(a).sum()
            columns.append(a); gene_weights.append(scores)
            modules.append({'module': f'{cell_type}|{entity}', 'cell_type': cell_type, 'entity': entity,
                            'entity_on_panel': entity in lookup, 'targets_on_panel': len(covered),
                            'target_coverage': coverage, 'targets': ';'.join(covered.target),
                            'sources': ';'.join(sorted(set(covered.source))),
                            'evidence': ';'.join(sorted(set(covered.evidence))),
                            'communication_link_evidence': ';'.join(sorted(set(links.loc[links.target.isin({g for value in features.iloc[np.flatnonzero(a)].ligand for g in canonicalize_complex(value)})].evidence))) if branch=='upstream' else ';'.join(sorted(set(covered.evidence))),
                            'interpretation': 'TF regulon proxy' if branch == 'upstream' else 'ligand-linked response; receptor specificity unproven'})
    a = np.stack(columns, axis=1) if columns else np.zeros((len(features), 0), np.float32)
    w = np.stack(gene_weights, axis=1) if columns else np.zeros((len(panel), 0), np.float32)
    return RegulatoryPrior(a, panel, w, pd.DataFrame(modules), branch,
        {'status': 'mapped' if columns else 'no_usable_modules', 'modules': len(columns),
         'mapped_features': int(np.any(a != 0, axis=1).sum()), 'feature_match_rate': float(np.any(a != 0, axis=1).mean()), 'uncovered_feature_indices': np.flatnonzero(~np.any(a != 0, axis=1)).tolist(),
         'prior_entities': int(table[key].nunique()), 'entities_on_panel': int(table[key].isin(lookup).groupby(table[key]).max().sum()),
         'prior_target_genes': int(table.target.nunique()), 'target_genes_on_panel': int(table.loc[table.target.isin(lookup)].target.nunique()),
         'target_gene_match_rate': float(table.loc[table.target.isin(lookup)].target.nunique()/max(table.target.nunique(),1)),
         'independent_score_genes': [panel[i] for i in np.flatnonzero(np.any(w != 0, axis=1))],
         'excluded_LR_score_genes': sorted(lr_genes), 'skipped_entities': skipped})


def activity_scores(expression, prior, training_cells, out_path=None):
    """Signed, target-set activity proxy; normalization fitted only on training cells."""
    used = np.flatnonzero(np.any(prior.weights != 0, axis=1))
    if not len(used): return np.zeros((expression.shape[0], 0), np.float32)
    sample = expression[training_cells][:, used].toarray() if hasattr(expression, 'toarray') else expression[np.ix_(training_cells, used)]
    sample = np.log1p(sample); mean = sample.mean(0); scale = np.maximum(sample.std(0), .1)
    shape = (expression.shape[0], prior.mapping.shape[1])
    out = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=shape) if out_path else np.empty(shape, np.float32)
    for start in range(0, len(out), 8192):
        block = expression[start:start+8192, used]
        if hasattr(block, 'toarray'): block = block.toarray()
        out[start:start+len(block)] = ((np.log1p(block) - mean) / scale) @ prior.weights[used]
    if hasattr(out, "flush"): out.flush()
    return out


def aggregate_activity(scores, local_types, coordinates, anchor_weights, cell_ids, features, prior, categories, sigma=20.):
    """Opportunity-normalized sender/receiver activity using the identical physical pairs.

    Does not multiply by ligand/receptor expression: observed targets remain independent.
    """
    weight = physical_weights(coordinates, anchor_weights, sigma, cell_ids)
    b = len(weight); m = prior.mapping.shape[1]
    output = weight.new_zeros((b, m)); mass = weight.new_zeros((b, m))
    categories = list(categories)
    for sender, receiver in features[['sender', 'receiver']].drop_duplicates().itertuples(index=False, name=None):
        pair = weight * (local_types == categories.index(sender))[:, :, None] * (local_types == categories.index(receiver))[:, None, :]
        denominator = pair.sum((1, 2))
        side = pair.sum(2) if prior.branch == 'upstream' else pair.sum(1)
        observed = (scores * side[..., None]).sum(1) / (denominator[:, None] + 1e-4)
        feature_rows = (features.sender == sender) & (features.receiver == receiver)
        contribution = torch.as_tensor(np.abs(prior.mapping[feature_rows]).sum(0), device=weight.device)
        supported = (denominator > 1e-8)[:, None]
        output += observed * contribution * supported
        mass += contribution * supported
    return output / mass.clamp_min(1e-8), mass > 0


def ridge(x, y, penalty=1e-3):
    # Fixed training-only nuisance model; unpenalized intercept.
    regularizer = np.eye(x.shape[1]) * penalty; regularizer[0, 0] = 0
    return np.linalg.solve(x.T @ x + regularizer, x.T @ y)


@dataclass
class BioAlignment:
    prior: RegulatoryPrior
    ccc_baseline: np.ndarray
    observed_baseline: np.ndarray
    slope: np.ndarray
    scale: np.ndarray
    valid: np.ndarray

    @classmethod
    def fit(cls, prior, ccc, nuisance, observed, mask, train):
        bc = ridge(nuisance[train], ccc[train])
        projected = (ccc - nuisance @ bc) @ prior.mapping
        by = np.zeros((nuisance.shape[1], observed.shape[1])); slope = np.zeros(observed.shape[1]); scale = np.ones_like(slope); valid = np.zeros_like(slope, bool)
        for j in range(observed.shape[1]):
            rows = train[mask[train, j]]
            if len(rows) < 5: continue
            by[:, j] = ridge(nuisance[rows], observed[rows, j])
            residual = observed[rows, j] - nuisance[rows] @ by[:, j]
            scale[j] = max(float(np.std(residual)), .1)
            signal = projected[rows, j]
            variance = float(signal @ signal)
            if variance <= 1e-8 or np.std(residual) <= 1e-6: continue
            slope[j] = float(signal @ residual) / (variance + 1e-3)
            valid[j] = True
        return cls(prior, bc.astype(np.float32), by.astype(np.float32), slope.astype(np.float32), scale.astype(np.float32), valid)

    def tensors(self, device):
        return [torch.as_tensor(value, device=device) for value in
                (self.prior.mapping, self.ccc_baseline, self.observed_baseline, self.slope, self.scale, self.valid)]

    def prediction_target(self, reconstruction, nuisance, observed):
        a, bc, by, slope, scale, valid = self.tensors(reconstruction.device)
        prediction = ((reconstruction - nuisance @ bc) @ a) * slope / scale
        target = (observed - nuisance @ by) / scale
        return prediction, target, valid

    def loss(self, reconstruction, nuisance, observed, mask):
        prediction, target, valid = self.prediction_target(reconstruction, nuisance, observed)
        mask = mask & valid[None]
        if not mask.any(): return reconstruction.sum() * 0
        return F.smooth_l1_loss(prediction[mask], target[mask])
