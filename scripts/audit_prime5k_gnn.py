"""Reproduce source label provenance and verify cached CCC without changing inputs."""
from pathlib import Path
import sys, json, ast, hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from phenoniche.storage import STORAGE_ROOT,OUTPUT_DIR
from phenoniche.ccc.config import default_config
from phenoniche.ccc.data import load_cached_input,_manifest_signature,_cache_is_valid,cache_paths
from phenoniche.ccc.lr_atlas import canonicalize_complex
from phenoniche.gnn.data import align_cells,neighborhoods
from phenoniche.gnn.model import physical_weights
from phenoniche.gnn.losses import pair_ccc
from phenoniche.gnn.biology import checksum
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.inputs import resolve_input

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'config/gnn.json'))
    cfg=GNNConfig.from_json(parser.parse_args().config)
    dest=Path(cfg.audit_file).parent;dest.mkdir(parents=True,exist_ok=True)
    inp=resolve_input(cfg,default_config());assert _cache_is_valid(cache_paths(inp),_manifest_signature(inp))
    cached=load_cached_input(inp);adata=ad.read_h5ad(inp.processed_h5ad);order=align_cells(adata,cached)
    assert np.array_equal(order,np.arange(adata.n_obs))
    assert adata.uns['dataset']=='Xenium_Prime_Breast_Cancer_FFPE'
    assert np.isfinite(adata.X.data).all() and (adata.X.data>=0).all()
    for start in range(0,len(cached['matrix']),8192):
        block=np.asarray(cached['matrix'][start:start+8192]);assert np.isfinite(block).all() and (block>=0).all()
    # Replay the original fixed-marker annotation function, not its I/O/training entry.
    label_source=Path('/home/nas3/biod/xueshuailin/CCC_Phe/backups/pre_cleanup_20261013/V1002/scripts/prepare_xenium_prime_5k_breast.py')
    tree=ast.parse(label_source.read_text());selected=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MARKERS' for t in node.targets): selected.append(node)
        if isinstance(node,ast.FunctionDef) and node.name=='annotate_clusters': selected.append(node)
    namespace={'np':np,'pd':pd,'sparse':sparse}
    exec(compile(ast.Module(body=selected,type_ignores=[]),'source_annotation_audit','exec'),namespace)
    clusters=np.array([int(str(x).split()[-1])-1 if str(x).startswith('Cluster ') else -1 for x in adata.obs.official_graph_cluster])
    labels,table,coverage=namespace['annotate_clusters'](adata.X,adata.var_names,clusters)
    assert np.array_equal(labels,adata.obs.cell_type_coarse.astype(str))
    table.to_csv(dest/'label_replay.csv',index=False)
    columns=(np.load(Path(cfg.legacy_output)/'cache/retained_indices.npy') if cfg.use_legacy_feature_mask else np.arange(len(cached['features'])))
    features=cached['features'].iloc[columns].reset_index(drop=True)
    assert len(columns)>0 and len(set(columns))==len(columns) and np.all(np.diff(columns)>0)
    panel={str(g).upper() for g in adata.var_names}
    assert all(set(canonicalize_complex(v))<=panel for col in ('ligand','receptor') for v in features[col])
    members,weights=neighborhoods(np.asarray(cached['coordinates']))
    rows=np.sort(np.random.default_rng(40700).choice(adata.n_obs,64,replace=False));local=members[rows]
    unique,inverse=np.unique(local,return_inverse=True);raw=adata.X[unique].toarray();lookup={str(g).upper():i for i,g in enumerate(adata.var_names)}
    sides=[]
    for col in ('ligand','receptor'):
        values=np.column_stack([np.prod(raw[:,[lookup[g] for g in canonicalize_complex(v)]],axis=1)**(1/len(canonicalize_complex(v))) for v in features[col]]).astype(np.float32)
        sides.append(torch.from_numpy(values[inverse.reshape(local.shape)]))
    types=pd.Categorical(cached['cells'].cell_type,categories=cached['manifest']['cell_types']).codes.astype(np.int64)
    physical=physical_weights(torch.from_numpy(np.asarray(cached['coordinates'])[local]),torch.from_numpy(weights[rows]),20.,torch.from_numpy(local))
    reproduced=pair_ccc(*sides,torch.from_numpy(types[local]),features,cached['manifest']['cell_types'],physical).numpy()
    expected=np.asarray(cached['matrix'][rows])[:,columns]
    assert np.allclose(reproduced,expected,rtol=1e-4,atol=2e-5)
    evidence={'dataset':str(adata.uns['dataset']),'scope':'independent_downstream_niche','cells':adata.n_obs,'genes':adata.n_vars,
              'cell_ID_coordinate_type_alignment':'exact','CCC_matrix_finite_nonnegative':True,'feature_count':len(columns),'fully_measurable_feature_LR':True,
              'CCC_exact_pairwise_replay_rows':rows.tolist(),'CCC_max_absolute_difference':float(np.abs(reproduced-expected).max()),
              'label_replay_match_fraction':float(np.mean(labels==adata.obs.cell_type_coarse.astype(str))),
              'label_source_sha256':checksum(label_source),'label_status':'provisional broad marker labels; not official 10x annotation',
              'label_method_inputs':['expression counts','official expression graph clusters','fixed broad cell-type markers'],
              'outcome_or_TLS_input_used':False,'mask_source_sha256':checksum(Path(cfg.legacy_output)/'cache/retained_indices.npy' if cfg.use_legacy_feature_mask else cached['paths']['features']),
              'clinical_Cox_analysis_run':False,'BayesPrism_and_patient_labels_used':False,
              'expression_coverage_fraction':inp.expression_coverage_fraction,
              'feature_mask_policy':'historical AE retained universe' if cfg.use_legacy_feature_mask else 'full source CCC universe; effective mask/RMS estimated only on GNN training spatial blocks'}
    ep=dest/'study_evidence.json';ep.write_text(json.dumps(evidence,indent=2)+'\n')
    proof={'status':'verified','evidence_file':str(ep),'sha256':checksum(ep)}
    study={k:proof.copy() for k in ('st_anchor','feature_mask','cell_labels')}
    study['scope']='independent_downstream_niche'
    for role in ('bayesprism','clinical_endpoint','sample_matching'):
        study[role]={'status':'not_applicable','reason':'Single Xenium tissue specimen; independent unsupervised downstream niche analysis uses no bulk cohort, patient outcomes or clinical matching. Spatial cell/CCC ID matching was verified separately.'}
    (dest/'study_audit.json').write_text(json.dumps(study,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))
