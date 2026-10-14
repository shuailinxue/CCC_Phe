"""Restartable fixed-K benchmark on the saved simulation, without truth tuning."""
from pathlib import Path
import hashlib,json,time,importlib.metadata as metadata
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from phenoniche.storage import STORAGE_ROOT
from phenoniche.simulation.response_simulation import checksum
from phenoniche.simulation.evaluation import label_mapping,mapped,partition_scores
from .adapters import clean_input,graph_from_members,graphst,cellcharter

ROOT=Path(__file__).resolve().parent


def cluster(training,full,config,scale=True):
    if scale:
        preprocessing=StandardScaler().fit(training);training=preprocessing.transform(training);full=preprocessing.transform(full)
    model=MiniBatchKMeans(n_clusters=config['fixed_k'],random_state=config['seed'],n_init=10,batch_size=min(1024,len(training))).fit(training)
    return model.predict(full).astype(np.int16)


def comparison_metrics(truth,labels,train,test):
    edge=np.isin(truth,[2,3]);records=[]
    for method,predicted in labels.items():
        mapping=label_mapping(predicted,truth,train);aligned=mapped(predicted,mapping)
        ab=label_mapping(predicted,truth,train[edge[train]])
        for scope,rows in [('all_centers_descriptive',np.arange(len(truth))),('common_spatial_test',test)]:
            records.append({'method':method,'scope':scope,'K':len(np.unique(predicted)),**partition_scores(truth,predicted,aligned,rows,ab,train[edge[train]])})
    return pd.DataFrame(records)


def common_test_rows(members,graph,folds):
    # GraphST reconstruction uses two graph hops. Purge evaluation centers
    # whose common max-depth input overlaps another fold or an unassigned cell.
    owner=np.full(len(folds),-1,np.int8)
    for fold in range(3):
        cells=np.unique(members[folds==fold])
        if (owner[cells]>=0).any():raise ValueError('Original physical-cell splits overlap')
        owner[cells]=fold
    g=graph.maximum(graph.T).tocsr();keep=[]
    for center in np.flatnonzero(folds==2):
        first=np.r_[center,g.indices[g.indptr[center]:g.indptr[center+1]]]
        second=np.unique(np.concatenate([first,*[g.indices[g.indptr[i]:g.indptr[i+1]] for i in first]]))
        if (owner[second]==2).all():keep.append(center)
    rows=np.asarray(keep,np.int64)
    if len(rows)<2:raise ValueError('Common two-hop spatial purge leaves no evaluable test centers')
    return rows


def run_comparison(experiment_output,config_path=None):
    experiment=Path(experiment_output);config=json.loads(Path(config_path or ROOT/'config.json').read_text())
    if config['fixed_k']!=4 or config['cellcharter_layers']!=1:raise ValueError('Current common two-hop protocol requires fixed K=4 and CellCharter one-hop KNN30')
    summary=json.loads((experiment/'summary.json').read_text());source=Path(summary['source_CCC_results'])
    out=experiment/'baseline_comparison'
    if out.is_symlink() or not out.resolve().is_relative_to(STORAGE_ROOT.resolve()):raise ValueError('Baseline artifacts must be stored directly on NAS')
    out.mkdir(exist_ok=True)
    inputs={name:checksum(source/name) for name in ('injected.h5ad','gnn/checkpoint.pt','raw_clusters.csv')}
    inputs['joint/Z']=checksum(experiment/'gnn/Z.npy');inputs['joint/clusters']=checksum(experiment/'raw_clusters.csv')
    implementation={p.name:checksum(p) for p in (ROOT/'adapters.py',ROOT/'benchmark.py')}
    versions={p:metadata.version(p) for p in ('cellcharter','torchgmm','torch','anndata','scanpy')}
    manifest={'config':config,'inputs':inputs,'implementation':implementation,'versions':versions,'GraphST_source':json.loads((ROOT/'vendor/graphst/provenance.json').read_text())}
    cfg=out/'config.json'
    if cfg.exists() and json.loads(cfg.read_text())!=manifest:raise ValueError('Baseline protocol/input/code changed: use a distinct NAS result directory to preserve previous runs')
    cfg.write_text(json.dumps(manifest,indent=2)+'\n')
    if (out/'complete.json').exists() and all((out/n).is_file() for n in ('labels.csv','metrics.csv','protocol.json')):return out
    original=ad.read_h5ad(source/'injected.h5ad')
    cache=experiment/'anchor_inputs/input';members=np.load(cache/'members.npy');folds=np.load(cache/'folds.npy')
    train=np.flatnonzero(folds==0);train_cells=np.unique(members[train]);graph=graph_from_members(members)
    # Strip all truth/type/MI annotations before calling either official model.
    a=clean_input(original.X,list(original.var_names),original.obsm['spatial'],original.obs.sample_id)
    if set(a.obs.columns)!={'sample_id'}:raise AssertionError('Forbidden annotations passed to baseline')
    test=common_test_rows(members,graph,folds)
    np.savez(out/'split_rows.npz',train=train,train_cells=train_cells,test=test)
    source_labels=pd.read_csv(source/'raw_clusters.csv');joint=pd.read_csv(experiment/'raw_clusters.csv')
    cells=original.obs.cell_id.astype(str).to_numpy()
    if not np.array_equal(source_labels.cell_id.astype(str),cells) or not np.array_equal(joint.cell_id.astype(str),cells):raise ValueError('Saved prediction cell order mismatch')
    labels={name:source_labels[name].to_numpy() for name in ('Cell Composition','Expression PCA','CCC-AE')}
    labels['CCC-only GNN']=source_labels['CCC_Phe GNN (fixed K)'].to_numpy()
    labels['CCC + gene GNN']=joint['CCC_Phe GNN (fixed K)'].to_numpy();timings={}
    for name in ('GraphST','CellCharter (PCA)'):
        directory=out/('graphst' if name=='GraphST' else 'cellcharter');directory.mkdir(exist_ok=True)
        started=time.monotonic()
        if (directory/'complete.json').exists():
            labels[name]=np.load(directory/'labels.npy');timings[name]=json.loads((directory/'complete.json').read_text())['runtime_seconds'];continue
        if name=='GraphST':
            embedding,training=graphst(a,graph,train_cells,config,directory)
            positions=np.searchsorted(train_cells,train);labels[name]=cluster(training[positions],embedding,config,scale=False)
        else:
            embedding,training,labels[name]=cellcharter(a,graph,train_cells,train,config,directory)
            labels['CellCharter (PCA; common KMeans)']=cluster(training,embedding,config)
        if embedding.shape[0]!=len(a) or not np.isfinite(embedding).all():raise ValueError('Invalid actual baseline embedding')
        np.save(directory/'embedding.npy',embedding);np.save(directory/'labels.npy',labels[name])
        if name.startswith('CellCharter'):np.save(directory/'common_kmeans_labels.npy',labels['CellCharter (PCA; common KMeans)'])
        timings[name]=time.monotonic()-started;(directory/'complete.json').write_text(json.dumps({'runtime_seconds':timings[name],'manifest':manifest},indent=2)+'\n')
    cc=out/'cellcharter/common_kmeans_labels.npy'
    if cc.exists():labels['CellCharter (PCA; common KMeans)']=np.load(cc)
    # Ground truth is consumed only now, after model fitting and clustering.
    truth=original.obs.true_niche.to_numpy();comparison_metrics(truth,labels,train,test).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame({'cell_id':cells,**labels}).to_csv(out/'labels.csv',index=False)
    protocol={'same_cells':len(a),'original_expression_genes':a.n_vars,'appended_response_genes_used':False,'ground_truth_used_for_training_or_K_selection':False,
        'representation_fit':'Original GNN training physical cells only; held-out expression excluded from fitted parameters and normalization',
        'clustering_fit':'Original GNN training centers only; fixed K=4 predeclared for all methods',
        'common_graph':'Original directed KNN30 including center; GraphST symmetrizes and normalizes using upstream sparse implementation; CellCharter uses directed one-hop mean aggregation',
        'cellcharter_variant':'Published aggregate_neighbors + Cluster(GMM), PCA32 input; scVI is not used or silently emulated',
        'graphst_variant':'Upstream sparse Slide implementation; output reconstructed expression embedding (author emb) + KMeans; inductive graph inference adapter',
        'spatial_test':'Original spatial split + common two-hop dependency purge for GraphST; identical evaluation centers for every method',
        'common_test_centers':len(test),'common_test_true_region_counts':{str(k):int(v) for k,v in zip(*np.unique(truth[test],return_counts=True))},
        'inputs_differ':'GNNs: CCC/type-resolved expression supervision. AE: precomputed CCC input. Composition: type labels. Other expression baselines: counts/coordinates only.',
        'reconstruction_comparison':'Spatial baselines do not provide directed CCC reconstruction; no invented CCC reconstruction scores',
        'seeds':[config['seed']],'scope':'One simulation and one model seed; preliminary comparison, not a multi-dataset benchmark',
        'runtime_seconds':timings}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (out/'complete.json').write_text(json.dumps({'manifest':manifest},indent=2)+'\n')
    return out
