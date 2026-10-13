"""Cell-aligned sparse inputs, reusable neighborhoods, and disjoint spatial splits."""
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree
from ..ccc.config import default_config as ccc_config
from ..ccc.data import load_cached_input, _cache_is_valid, _manifest_signature, cache_paths
from ..data import feature_statistics
from .model import physical_weights
from .inputs import resolve_input


def align_cells(adata, cached):
    ids = adata.obs.cell_id.astype(str)
    expected = cached['cells'].cell_id.astype(str)
    if not ids.is_unique or not expected.is_unique: raise ValueError('Duplicate physical cell IDs')
    order = pd.Index(ids).get_indexer(expected)
    if (order < 0).any() or len(order) != len(ids): raise ValueError('Missing/extra cells in expression vs CCC')
    xy = np.asarray(adata.obsm['spatial'])[order].astype(np.float32)
    if not np.allclose(xy, cached['coordinates'], atol=1e-4, rtol=0): raise ValueError('Cell ID/coordinate alignment failed')
    labels = adata.obs.cell_type_coarse.astype(str).to_numpy()[order]
    if not np.array_equal(labels, cached['cells'].cell_type.astype(str)): raise ValueError('Cell labels differ from CCC cache')
    return order


def neighborhoods(xy, k=30, sigma=20., path=None):
    if len(xy) < k: raise ValueError('Not enough cells for the fixed neighborhood')
    tree = cKDTree(np.asarray(xy, np.float64))
    shape = (len(xy), k)
    if path:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        members = np.lib.format.open_memmap(path/'members.npy', mode='w+', dtype=np.int64, shape=shape)
        weights = np.lib.format.open_memmap(path/'anchor_weights.npy', mode='w+', dtype=np.float32, shape=shape)
    else:
        members = np.empty(shape, np.int64); weights = np.empty(shape, np.float32)
    # Same tree query, sorted membership and float64 anchor distances as ccc.neighborhoods.
    for start in range(0, len(xy), 50000):
        centers = np.asarray(xy[start:start+50000], np.float64)
        _, indices = tree.query(centers, k=k)
        indices.sort(axis=1)
        distance2 = ((np.asarray(xy[indices], np.float64) - centers[:, None])**2).sum(-1)
        members[start:start+len(indices)] = indices
        weights[start:start+len(indices)] = np.exp(-distance2 / (2*sigma**2))
    return members, weights


def spatial_split(xy, members, seed=40700, block_um=400., validation=.2, test=.2, samples=None):
    blocks = np.floor((xy - xy.min(0)) / block_um).astype(np.int64)
    keys = np.asarray([f'{a}:{b}' for a,b in blocks]) if samples is None else np.asarray(samples).astype(str)
    groups, inverse = np.unique(keys, return_inverse=True)
    if len(groups) < 3: raise ValueError('At least three spatial blocks/samples are required')
    order = np.random.default_rng(seed).permutation(len(groups))
    nv = max(1, round(len(groups)*validation)); nt = max(1, round(len(groups)*test))
    if nv+nt >= len(groups): raise ValueError('Too few blocks for train/validation/test')
    group_fold = np.zeros(len(groups), np.int8)
    group_fold[order[:nv]] = 1; group_fold[order[nv:nv+nt]] = 2
    cell_fold = group_fold[inverse]
    folds = np.full(len(xy), -1, np.int8)
    for start in range(0,len(xy),50000):
        stop = min(start+50000,len(xy)); local = members[start:stop]
        own = cell_fold[start:stop]
        # Purge boundary centers, not just centers lying near another block.
        eligible = (cell_fold[local] == own[:,None]).all(1)
        folds[start:stop] = np.where(eligible,own,-1)
    rows = [np.flatnonzero(folds == i) for i in range(3)]
    if any(len(r) < 2 for r in rows): raise ValueError('Spatial boundary purge leaves an empty split; adjust block size')
    used = [np.unique(members[r]) for r in rows]
    if any(np.intersect1d(used[i], used[j]).size for i in range(3) for j in range(i+1,3)):
        raise AssertionError('Physical cells shared between splits')
    return rows, folds, inverse


def expression_block(x, rows, genes, totals):
    block = x[rows][:,genes]
    if sp.issparse(block): block = block.toarray()
    return np.log1p(block * (10000 / np.maximum(totals[rows,None],1))).astype(np.float32)


class Targets:
    def __init__(self, matrix, columns): self.matrix, self.columns = matrix, columns; self.shape=(len(matrix),len(columns))
    def __getitem__(self, rows): return np.asarray(self.matrix[rows],np.float32)[...,self.columns]


@dataclass
class GraphData:
    expression: object
    totals: np.ndarray
    genes: tuple
    encoder_indices: np.ndarray
    gene_mean: np.ndarray
    gene_scale: np.ndarray
    types: np.ndarray
    categories: tuple
    coordinates: np.ndarray
    members: np.ndarray
    weights: np.ndarray
    targets: Targets
    feature_scale: np.ndarray
    feature_mask: np.ndarray
    features: pd.DataFrame
    cells: pd.DataFrame
    rows: list
    folds: np.ndarray
    blocks: np.ndarray
    composition_target: np.ndarray
    audit: dict

    def prepare_device(self, device, cache_dir):
        """Cache static cell inputs once; learned cell representations remain batch-local."""
        if not str(device).startswith('cuda'): return
        cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
        values=np.lib.format.open_memmap(cache_dir/'encoder_features.npy',mode='w+',dtype=np.float32,shape=(len(self.cells),len(self.encoder_indices)+1))
        for start in range(0,len(values),8192):
            rows=np.arange(start,min(start+8192,len(values)))
            x=expression_block(self.expression,rows,self.encoder_indices,self.totals)
            values[rows,:-1]=(x-self.gene_mean)/self.gene_scale
            values[rows,-1]=np.log1p(self.totals[rows])/10
        values.flush()
        tensor=lambda a:torch.as_tensor(np.asarray(a).copy(),device=device)
        self._resident={'expression':tensor(values),
                        'coordinates':tensor(self.coordinates),'members':tensor(self.members),
                        'weights':tensor(self.weights),'composition_target':tensor(self.composition_target)}
        # Assemble targets in bounded chunks rather than a second full host copy.
        target=torch.empty(self.targets.shape,dtype=torch.float32,device=device)
        for start in range(0,len(self.cells),8192):
            stop=min(start+8192,len(self.cells));target[start:stop]=tensor(self.targets[slice(start,stop)])
        self._resident['target']=target
        print(f'Static graph inputs resident on {device}',flush=True)

    def batch(self, rows, device='cpu', include_target=True):
        if hasattr(self,'_resident') and str(device).startswith('cuda'):
            r=self._resident;row=torch.as_tensor(rows,device=device,dtype=torch.long);local=r['members'][row]
            unique,inverse=torch.unique(local.reshape(-1),sorted=True,return_inverse=True)
            batch={'expression':r['expression'][unique],
                   'local_index':inverse.reshape(local.shape),'coordinates':r['coordinates'][local],
                   'anchor_weights':r['weights'][row],'cell_ids':local}
            if include_target:
                batch['target']=r['target'][row]
                batch['composition_target']=r['composition_target'][row]
            return batch
        local = np.asarray(self.members[rows]); unique, inverse = np.unique(local, return_inverse=True)
        x = expression_block(self.expression, unique, self.encoder_indices, self.totals)
        x = (x - self.gene_mean) / self.gene_scale
        # Raw library size supplies intensity information lost by compositional expression normalization.
        x = np.column_stack((x,np.log1p(self.totals[unique])/10)).astype(np.float32)
        tensor = lambda x: torch.as_tensor(np.asarray(x).copy(),device=device)
        batch = {'expression':tensor(x),
                 'local_index':tensor(inverse.reshape(local.shape)), 'coordinates':tensor(self.coordinates[local]),
                 'anchor_weights':tensor(self.weights[rows]), 'cell_ids':tensor(local),'unique':unique}
        if include_target:
            batch['target']=tensor(self.targets[rows])
            batch['composition_target']=tensor(self.composition_target[rows])
        return batch

    def forward(self,model,batch,**kwargs):
        return model(*(batch[k] for k in ('expression','local_index','coordinates','anchor_weights','cell_ids')),**kwargs)


def prepare_graph_data(config):
    """Called only after the audit gate. Reads legacy caches; never rebuilds or modifies them."""
    import anndata as ad
    inp = resolve_input(config,ccc_config())
    if not _cache_is_valid(cache_paths(inp),_manifest_signature(inp)): raise ValueError('Source CCC cache has invalid provenance')
    cached = load_cached_input(inp); adata = ad.read_h5ad(inp.processed_h5ad)
    order = align_cells(adata,cached)
    x = sp.csr_matrix(adata.X[order],dtype=np.float32)
    if not np.isfinite(x.data).all() or (x.data<0).any(): raise ValueError('Expression must be finite nonnegative counts')
    genes = tuple(str(g).upper() for g in adata.var_names)
    if len(set(genes)) != len(genes): raise ValueError('Ambiguous uppercase gene symbols')
    output = Path(config.output_dir); cache=output/'cache'; cache.mkdir(parents=True,exist_ok=True)
    xy=np.asarray(cached['coordinates']); members,weights=neighborhoods(xy,config.n_neighbors,config.sigma_um,cache)
    samples = adata.obs[config.sample_column].astype(str).to_numpy()[order] if config.sample_column else None
    if samples is not None and any(len(set(samples[m]))>1 for m in members): raise ValueError('Neighborhoods cross sample identities; audit spatial units')
    rows,folds,blocks=spatial_split(xy,members,config.split_seed,config.block_um,config.validation_fraction,config.test_fraction,samples)
    columns=(np.load(Path(config.legacy_output)/'cache/retained_indices.npy') if config.use_legacy_feature_mask
             else np.arange(len(cached['features']),dtype=np.int64))
    if len(set(columns))!=len(columns) or np.any(np.diff(columns)<=0): raise ValueError('Legacy retained feature order is invalid')
    features=cached['features'].iloc[columns].reset_index(drop=True)
    targets=Targets(cached['matrix'],columns)
    mean,ms,var,nonzero=feature_statistics(cached['matrix'],rows[0])
    mask=(nonzero[columns]>=.005)&(var[columns]>np.maximum(np.finfo(np.float32).eps*ms[columns],np.finfo(np.float32).tiny))
    if not mask.any(): raise ValueError('No effective training CCC features')
    scale=np.maximum(np.sqrt(ms[columns]),1e-6).astype(np.float32)
    audit={'split':'spatial/sample blocks with all-member boundary purge',
           'split_sizes':[len(r) for r in rows],'purged_centers':int((folds<0).sum()),
           'fixed_features':len(columns),'effective_features':int(mask.sum()),
           'source_signature':cached['manifest']['signature'],'CCC_matrix_path':str(cached['paths']['ccc']),
           'cell_order_sha256':hashlib.sha256(pd.util.hash_pandas_object(cached['cells'].cell_id,index=False).to_numpy().tobytes()).hexdigest(),
           'encoder_inputs':'expression and spatial graph only; cell types are supervision/analysis only',
           'architecture':config.architecture}
    available=np.arange(len(genes))
    train_cells=np.unique(members[rows[0]])
    sample=np.random.default_rng(config.split_seed).choice(train_cells,min(len(train_cells),20000),replace=False)
    totals=np.asarray(x.sum(1)).ravel().astype(np.float32)
    sample_x=expression_block(x,sample,available,totals)
    chosen=np.argsort(-sample_x.var(0),kind='stable')[:config.encoder_genes]
    encoder_indices=available[chosen];selected=sample_x[:,chosen]
    gene_mean=selected.mean(0);gene_scale=np.maximum(selected.std(0),.1)
    categories=tuple(sorted(cached['cells'].cell_type.unique()))
    types=pd.Categorical(cached['cells'].cell_type,categories=categories).codes.astype(np.int64)
    composition_target=np.lib.format.open_memmap(cache/'composition_target.npy',mode='w+',dtype=np.float32,shape=(len(xy),len(categories)))
    for start in range(0,len(xy),config.batch_size):
        stop=min(start+config.batch_size,len(xy));local=np.asarray(members[start:stop])
        pair=physical_weights(torch.from_numpy(xy[local]),torch.from_numpy(np.asarray(weights[start:stop]).copy()),config.sigma_um,torch.from_numpy(local))
        composition_target[start:stop]=pair_composition(torch.from_numpy(types[local]),pair,len(categories)).numpy()
    composition_target.flush()
    valid_composition=np.asarray(composition_target).sum(1)>0
    audit['composition_undefined_centers']=int((~valid_composition).sum())
    audit['composition_valid_by_split']=[int(valid_composition[r].sum()) for r in rows]
    audit['composition_missing_policy']='Zero physical-pair opportunity: undefined composition, excluded from composition loss/metrics; CCC supervision unchanged'
    if audit['composition_undefined_centers']:
        print(f"Undefined composition for {audit['composition_undefined_centers']} centers with zero physical-pair opportunity; skipping their composition supervision",flush=True)
    audit['encoder_genes']=[genes[i] for i in encoder_indices]
    audit['composition_definition']='sum of outgoing physical-pair weights per cell type / total directed physical-pair weight'
    audit['categories']=list(categories)
    data=GraphData(x,totals,genes,encoder_indices,gene_mean,gene_scale,types,categories,xy,members,weights,targets,scale,mask,features,cached['cells'],rows,folds,blocks,composition_target,audit)
    np.savez(cache/'preprocessing.npz',encoder_indices=encoder_indices,gene_mean=gene_mean,gene_scale=gene_scale,feature_scale=scale,feature_mask=mask,columns=columns,folds=folds,blocks=blocks)
    audit['cell_annotation']=str(adata.uns.get('cell_type_annotation','unspecified'))
    (cache/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    features.to_csv(cache/'features.csv',index=False)
    return data


def pair_composition(local_types, pair_weights, n_types):
    """Exact CCC weights; all-zero rows mark undefined composition, not a distribution."""
    if not torch.isfinite(pair_weights).all() or (pair_weights<0).any():
        raise ValueError('Invalid physical-pair weights')
    degree=pair_weights.sum(2)
    total=degree.sum(1,keepdim=True)
    # Gaussian weights can underflow for isolated cells. Do not invent a
    # uniform composition or alter the original CCC weights to avoid this.
    denominator=torch.where(total>0,total,torch.ones_like(total))
    return torch.stack([(degree*(local_types==t)).sum(1) for t in range(n_types)],dim=1)/denominator
