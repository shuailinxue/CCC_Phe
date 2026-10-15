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
from .protocol import physical_owners
from .cellcharter_adapter import cellcharter as cellcharter_scvi, safe_test_rows
import scipy.sparse as sp
from .scniche_adapter import scniche
from .nichecompass_adapter import nichecompass
from .spaniche_adapter import spaniche
from .banksy_adapter import banksy
from .stagate_adapter import stagate
from .cache import reuse_completed

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
    if config['fixed_k']!=4:raise ValueError('The common fixed-K benchmark requires K=4')
    summary=json.loads((experiment/'summary.json').read_text());source=Path(summary['source_CCC_results'])
    subdir=config.get('result_subdir','baseline_comparison')
    if Path(subdir).name!=subdir:raise ValueError('Baseline result_subdir must be a single directory name')
    out=experiment/subdir
    if out.is_symlink() or not out.resolve().is_relative_to(STORAGE_ROOT.resolve()):raise ValueError('Baseline artifacts must be stored directly on NAS')
    out.mkdir(exist_ok=True)
    inputs={name:checksum(source/name) for name in ('injected.h5ad','gnn/checkpoint.pt','raw_clusters.csv')}
    inputs['joint/Z']=checksum(experiment/'gnn/Z.npy');inputs['joint/clusters']=checksum(experiment/'raw_clusters.csv')
    cc_source=Path(summary.get('CCC_only_niche_results_dir',source))
    if cc_source!=source:inputs['CCC_only_refined_clusters']=checksum(cc_source/'raw_clusters.csv')
    implementation={str(p.relative_to(ROOT)):checksum(p) for p in (ROOT/'adapters.py',ROOT/'benchmark.py',ROOT/'protocol.py',ROOT/'scniche_adapter.py',ROOT/'nichecompass_adapter.py',ROOT/'spaniche_adapter.py',ROOT/'spaniche_runner.R',ROOT/'banksy_adapter.py',ROOT/'stagate_adapter.py',ROOT/'cache.py',ROOT/'cellcharter_adapter.py',ROOT/'scvi_runner.py')}
    for folder in ('scniche','spaniche','intnmf','banksy','stagate'):
        for p in (ROOT/'vendor'/folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.R','.json'):
                implementation[str(p.relative_to(ROOT))]=checksum(p)
    inputs['public_LR_prior']=checksum(ROOT.parent/'data/commuspace_human_lr_atlas.tsv')
    versions={p:metadata.version(p) for p in ('cellcharter','torchgmm','torch','anndata','scanpy')}
    for method,package in [('scNiche','dgl'),('NicheCompass','nichecompass')]:
        if method in config.get('additional_methods',[]):versions[package]=metadata.version(package)
    if 'STAGATE' in config.get('additional_methods',[]):versions['torch-geometric']=metadata.version('torch-geometric')
    if config['cellcharter_representation']=='scVI':
        import subprocess
        versions['cellcharter_scvi_runtime']=subprocess.check_output([config['cellcharter_scvi_python'],'-c','import importlib.metadata as m; print(m.version("scvi-tools")+";lightning="+m.version("lightning"))'],text=True).strip()
    manifest={'config':config,'inputs':inputs,'implementation':implementation,'versions':versions,'GraphST_source':json.loads((ROOT/'vendor/graphst/provenance.json').read_text())}
    cfg=out/'config.json'
    if cfg.exists() and json.loads(cfg.read_text())!=manifest:raise ValueError('Baseline protocol/input/code changed: use a distinct NAS result directory to preserve previous runs')
    cfg.write_text(json.dumps(manifest,indent=2)+'\n')
    if (out/'complete.json').exists() and all((out/n).is_file() for n in ('labels.csv','metrics.csv','protocol.json')):return out
    if config.get('reuse_result_subdir'):
        previous=config['reuse_result_subdir']
        if Path(previous).name!=previous:raise ValueError('Reuse result subdirectory must be a single name')
        reused=reuse_completed(experiment/previous,out,manifest)
        if reused:print('Reused audited, unchanged baseline results: '+', '.join(reused),flush=True)
    original=ad.read_h5ad(source/'injected.h5ad')
    cache=experiment/'anchor_inputs/input';members=np.load(cache/'members.npy');folds=np.load(cache/'folds.npy')
    train=np.flatnonzero(folds==0);train_cells=np.unique(members[train]);graph=graph_from_members(members)
    owner=physical_owners(members,folds)
    # Strip all truth/type/MI annotations before calling either official model.
    a=clean_input(original.X,list(original.var_names),original.obsm['spatial'],original.obs.sample_id)
    if set(a.obs.columns)!={'sample_id'}:raise AssertionError('Forbidden annotations passed to baseline')
    test=common_test_rows(members,graph,folds)
    np.savez(out/'split_rows.npz',train=train,train_cells=train_cells,test=test)
    source_labels=pd.read_csv(source/'raw_clusters.csv');joint=pd.read_csv(experiment/'raw_clusters.csv')
    cells=original.obs.cell_id.astype(str).to_numpy()
    if not np.array_equal(source_labels.cell_id.astype(str),cells) or not np.array_equal(joint.cell_id.astype(str),cells):raise ValueError('Saved prediction cell order mismatch')
    labels={name:source_labels[name].to_numpy() for name in ('Cell Composition','Expression PCA','CCC-AE')}
    labels['CCC-only GNN']=pd.read_csv(cc_source/'raw_clusters.csv')['CCC_Phe GNN (fixed K)'].to_numpy()
    labels['CCC + gene GNN']=joint['CCC_Phe GNN (fixed K)'].to_numpy();timings={}
    reuse=Path(summary['unrefined_results'])/subdir if 'unrefined_results' in summary else None
    if reuse and (reuse/'config.json').exists():
        old_manifest=json.loads((reuse/'config.json').read_text())
        compatible=old_manifest['config']==config and old_manifest['versions']==versions and old_manifest['implementation']['adapters.py']==implementation['adapters.py'] and all(old_manifest['inputs'].get(key)==inputs[key] for key in ('injected.h5ad','gnn/checkpoint.pt','raw_clusters.csv'))
        if compatible:
            import shutil
            for name in ('graphst','cellcharter'):
                if (reuse/name/'complete.json').exists() and not (out/name).exists():shutil.copytree(reuse/name,out/name)
        labels['CCC-only GNN (unrefined)']=source_labels['CCC_Phe GNN (fixed K)'].to_numpy()
        labels['CCC + gene GNN (unrefined)']=pd.read_csv(Path(summary['unrefined_results'])/'raw_clusters.csv')['CCC_Phe GNN (fixed K)'].to_numpy()
    cc_name='CellCharter (scVI)' if config['cellcharter_representation']=='scVI' else 'CellCharter (PCA)'
    methods=['GraphST',cc_name,*config.get('additional_methods',[])]
    if len(set(methods))!=len(methods) or set(methods)-{'GraphST','CellCharter (PCA)','CellCharter (scVI)','scNiche','NicheCompass','SpaNiche','BANKSY','STAGATE'}:
        raise ValueError('Unknown or duplicate baseline method')
    directories={'GraphST':'graphst','CellCharter (PCA)':'cellcharter','CellCharter (scVI)':'cellcharter','scNiche':'scniche','NicheCompass':'nichecompass','SpaNiche':'spaniche','BANKSY':'banksy','STAGATE':'stagate'}
    from phenoniche.simulation.generation import TYPES
    types=pd.Categorical(original.obs.cell_type,categories=TYPES).codes
    if (types<0).any():raise ValueError('Unrecognized observed cell types')
    for name in methods:
        directory=out/directories[name];directory.mkdir(exist_ok=True)
        started=time.monotonic()
        if (directory/'complete.json').exists():
            labels[name]=np.load(directory/'labels.npy');timings[name]=json.loads((directory/'complete.json').read_text())['runtime_seconds'];continue
        if name=='GraphST':
            embedding,training=graphst(a,graph,train_cells,config,directory)
            positions=np.searchsorted(train_cells,train);labels[name]=cluster(training[positions],embedding,config,scale=False)
        elif name==cc_name:
            adapter=cellcharter_scvi if config['cellcharter_representation']=='scVI' else cellcharter
            embedding,training,labels[name]=adapter(a,graph,train_cells,train,config,directory)
            labels[cc_name.replace(')', '; common KMeans)')]=cluster(training,embedding,config)
        else:
            print(f'Fitting {name} on {len(train_cells)} training physical cells',flush=True)
            proof=directory/'reuse_training.json'
            if proof.exists():
                previous=json.loads(proof.read_text())['manifest']
                if previous['inputs']!=inputs or previous['versions']!=versions:
                    raise ValueError('Reused baseline training inputs or versions differ')
            if name=='scNiche':embedding,training=scniche(a,graph,train_cells,types,TYPES,owner,config,directory)
            elif name=='NicheCompass':embedding,training=nichecompass(a,graph,train_cells,config,directory)
            elif name=='SpaNiche':embedding,training=spaniche(a,members,train_cells,types,TYPES,owner,config,directory)
            elif name=='BANKSY':embedding,training=banksy(a,train_cells,owner,config,directory)
            else:embedding,training=stagate(a,graph,train_cells,config,directory)
            positions=np.searchsorted(train_cells,train)
            labels[name]=cluster(training[positions],embedding,config)
        if embedding.shape[0]!=len(a) or not np.isfinite(embedding).all():raise ValueError('Invalid actual baseline embedding')
        np.save(directory/'embedding.npy',embedding);np.save(directory/'labels.npy',labels[name])
        if name.startswith('CellCharter'):np.save(directory/'common_kmeans_labels.npy',labels[cc_name.replace(')', '; common KMeans)')])
        timings[name]=time.monotonic()-started;(directory/'complete.json').write_text(json.dumps({'runtime_seconds':timings[name],'manifest':manifest},indent=2)+'\n')
    cc=out/'cellcharter/common_kmeans_labels.npy'
    if cc.exists():labels[cc_name.replace(')', '; common KMeans)')]=np.load(cc)
    if 'BANKSY' in methods:
        safe=np.load(out/'banksy/inference_safe_centers.npy')
        test=test[safe[test]]
        if len(test)<2:raise ValueError('BANKSY dependency purge leaves no common spatial test centers')
        np.savez(out/'split_rows.npz',train=train,train_cells=train_cells,test=test)
    if config['cellcharter_representation']=='scVI':
        safe=safe_test_rows(sp.load_npz(out/'cellcharter/spatial_connectivities.npz'),owner,config['cellcharter_layers'])
        test=test[safe[test]]
        if len(test)<2:raise ValueError('CellCharter native graph purge leaves no common test centers')
        np.savez(out/'split_rows.npz',train=train,train_cells=train_cells,test=test)
    # Ground truth is consumed only now, after model fitting and clustering.
    truth=original.obs.true_niche.to_numpy();comparison_metrics(truth,labels,train,test).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame({'cell_id':cells,**labels}).to_csv(out/'labels.csv',index=False)
    protocol={'same_cells':len(a),'original_expression_genes':a.n_vars,'appended_response_genes_used':False,'ground_truth_used_for_training_or_K_selection':False,
        'representation_fit':'Original GNN training physical cells only; held-out expression excluded from fitted parameters and normalization',
        'clustering_fit':'Original GNN training centers only; fixed K=4 predeclared for all methods',
        'common_graph':'Original directed KNN30 including center; GraphST symmetrizes and normalizes using upstream sparse implementation; CellCharter scVI uses official per-sample Delaunay and three-hop aggregation; PCA control uses one-hop KNN',
        'cellcharter_variant':('Official scVI10 (ZINB/raw counts) + Delaunay percentile99 + three-hop aggregate_neighbors + official GMM' if config['cellcharter_representation']=='scVI' else 'PCA variant'),
        'cellcharter_name':cc_name,
        'graphst_variant':'Upstream sparse Slide implementation; output reconstructed expression embedding (author emb) + KMeans; inductive graph inference adapter',
        'spatial_test':'Original spatial split + common two-hop dependency purge for GraphST + actual BANKSY and native CellCharter donor-footprint checks when enabled; identical evaluation centers for every method',
        'common_test_centers':len(test),'common_test_true_region_counts':{str(k):int(v) for k,v in zip(*np.unique(truth[test],return_counts=True))},
        'inputs_differ':'GNNs: CCC/type-resolved expression supervision. AE: precomputed CCC input. Composition: type labels. GraphST/CellCharter/BANKSY/STAGATE: counts+coordinates. scNiche: counts+coordinates+observed types. NicheCompass: counts+coordinates+public LR programs/de novo programs. SpaNiche: counts+coordinates+observed types+public LR prior.',
        'additional_methods':config.get('additional_methods',[]),
        'new_baseline_adapters':{name:json.loads((out/directories[name]/'adapter_audit.json').read_text()) for name in config.get('additional_methods',[])},
        'spatial_scale':'GNN/GraphST/scNiche/STAGATE use shared KNN30 graph; CellCharter scVI uses native Delaunay percentile99 with three-hop mean aggregation. BANKSY mean/AGF use native K15/K30; training donors isolated, full-graph inference, common testing checks actual neighbor footprint. SpaNiche radius is training median KNN30 reach; second view uses 2x radius; graph sigma=20um. Method-native topology/weights retained, not silently forced identical.',
        'inference_extensions':'scNiche graph fusion uses same-size native batches within each physical fold. SpaNiche has no official heldout inference: freeze training H and infer W using its NNLS + graph objective in each heldout fold. No fitted baseline parameters see heldout expression.',
        'reconstruction_comparison':'Spatial baselines do not provide directed CCC reconstruction; no invented CCC reconstruction scores',
        'seeds':[config['seed']],'scope':'One simulation and one model seed; preliminary comparison, not a multi-dataset benchmark',
        'runtime_seconds':timings,'GNN_niche_postprocessing':summary.get('niche_refinement'),
        'raw_GNN_controls_in_metric_table':'CCC-only GNN (unrefined)' in labels}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (out/'complete.json').write_text(json.dumps({'manifest':manifest},indent=2)+'\n')
    return out
