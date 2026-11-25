"""Observed target arrays only: no RNA, GraphData, or GNN configuration."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
from ..data import feature_statistics
from .io import array_identity, digest


@dataclass
class TargetData:
    targets: object
    composition_target: object
    categories: tuple
    feature_ids: tuple
    rows: list
    audit: dict

    def __post_init__(self):
        self.categories = tuple(self.categories)
        self.feature_ids = tuple(self.feature_ids)
        self.rows = [np.asarray(r, dtype=np.int64) for r in self.rows]
        n, f = self.targets.shape
        if self.composition_target.shape != (n, len(self.categories)) or f != len(self.feature_ids):
            raise ValueError('Target dimensions/order metadata do not match')
        if len(set(self.categories)) != len(self.categories) or len(set(self.feature_ids)) != f:
            raise ValueError('Feature/category identities must be unique')
        if len(self.rows) != 3 or not len(self.rows[0]):
            raise ValueError('Supply nonempty training rows and explicit validation/test rows')
        for i, r in enumerate(self.rows):
            if r.ndim != 1 or len(np.unique(r)) != len(r) or (r < 0).any() or (r >= n).any():
                raise ValueError('Invalid center row indices')
            if any(np.intersect1d(r, previous).size for previous in self.rows[:i]):
                raise ValueError('Training/validation/test center rows overlap')
        self._resident = None

    @property
    def schema(self):
        return dict(categories=list(self.categories), feature_ids=list(self.feature_ids))

    def identity(self):
        return dict(schema=self.schema, CCC=array_identity(self.targets),
                    composition=array_identity(self.composition_target),
                    folds=[array_identity(r) for r in self.rows], audit=self.audit)

    def fit_statistics(self):
        """Only TRAINING rows determine the mask, scales and output priors."""
        mean, ms, variance, rho = feature_statistics(self.targets, self.rows[0], chunk_size=2048)
        mask = (rho >= .005) & (variance > np.maximum(np.finfo(np.float32).eps * ms, np.finfo(np.float32).tiny))
        if not mask.any():
            raise ValueError('No variable CCC features in training rows')
        rms = np.sqrt(ms)
        self.feature_scale = np.maximum(rms, max(float(np.median(rms[mask])) * .1, 1e-6)).astype(np.float32)
        self.feature_mask = mask
        self.feature_mean = mean.astype(np.float32)
        self.feature_nonzero_fraction = rho.astype(np.float32)

    def batch(self, rows, device='cpu', include_target=True):
        if self._resident is not None and self._resident[0].device == torch.device(device):
            ids = torch.as_tensor(rows, device=device)
            return dict(target=self._resident[0][ids], composition_target=self._resident[1][ids])
        y = np.asarray(self.targets[rows], dtype=np.float32).copy()
        p = np.asarray(self.composition_target[rows], dtype=np.float32).copy()
        if not np.isfinite(y).all() or (y < 0).any() or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError('CCC and composition must be finite and nonnegative')
        if not np.allclose(p.sum(1), 1., atol=1e-5):
            raise ValueError('Composition rows must be probability distributions')
        return dict(target=torch.as_tensor(y, device=device), composition_target=torch.as_tensor(p, device=device))

    def forward(self, model, batch, reconstruct=True):
        if reconstruct:
            return model(batch['target'], batch['composition_target'])
        return dict(shared=model.encode(batch['target'], batch['composition_target']))

    @classmethod
    def from_matrices(cls, directory, train_rows=None, validation_rows=None, test_rows=None):
        """Read existing audited targets; never rebuild CCC or physical graphs."""
        import pandas as pd
        directory = Path(directory)
        c = np.load(directory / 'ccc.npy', mmap_mode='r')
        p = np.load(directory / 'composition.npy', mmap_mode='r')
        meta = json.loads((directory / 'complete.json').read_text())
        features = pd.read_csv(directory / 'features.csv')
        ids = feature_identities(features)
        rows = [np.arange(len(c)) if train_rows is None else train_rows,
                np.empty(0, np.int64) if validation_rows is None else validation_rows,
                np.empty(0, np.int64) if test_rows is None else test_rows]
        # Physical membership is an optional split audit, never an AE encoder input.
        if any(len(r) for r in rows[1:]) and (directory / 'members.npy').exists():
            members = np.load(directory / 'members.npy', mmap_mode='r')
            sets = [np.unique(members[r]) for r in rows]
            if any(np.intersect1d(sets[i], sets[j]).size for i in range(3) for j in range(i)):
                raise ValueError('Physical microenvironment members cross evaluation folds')
        return cls(c, p, tuple(meta['categories']), ids, rows,
                   dict(source_matrices=str(directory.resolve()), input='observed CCC + composition',
                        RNA_or_graph_encoder=False, metadata_digest=digest(meta)))


def feature_identities(features):
    columns = ['sender', 'receiver', 'ligand', 'receptor']
    if all(c in features for c in columns):
        return tuple(json.dumps(list(row), ensure_ascii=True) for row in features[columns].astype(str).itertuples(index=False, name=None))
    if 'feature_id' in features:
        return tuple(features.feature_id.astype(str))
    raise ValueError('CCC features need typed LR identities or unique feature_id')
