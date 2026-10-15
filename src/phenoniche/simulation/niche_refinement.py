"""Recluster saved simulation programs without refitting or changing decoders."""
from pathlib import Path
from dataclasses import replace
import json,shutil
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from ..storage import STORAGE_ROOT
from ..gnn.evaluation import discover
from ..gnn.spatial_refinement import spatial_cluster_features
from .config import SimulationConfig
from .adapter import adapt,gnn_config
from .baselines import dictionary_program_projection
from .response_simulation import checksum


def refine_saved_niches(experiment_output,output_dir,mixing=.2):
    previous=Path(experiment_output);out=Path(output_dir)
    if out.is_symlink() or not out.resolve().is_relative_to(STORAGE_ROOT.resolve()):raise ValueError('Direct NAS output required')
    if out.resolve().is_relative_to(previous.resolve()) or previous.resolve().is_relative_to(out.resolve()):raise ValueError('Refinement must preserve previous results')
    old=json.loads((previous/'summary.json').read_text());source=Path(old['source_CCC_results'])
    cfg=json.loads((previous/'config.json').read_text());base=SimulationConfig(**cfg['base_config'])
    anchor={name:checksum(source/name) for name in ('gnn/checkpoint.pt','gnn/Z.npy','gnn/H.npy','raw_clusters.csv','injected.h5ad','input/ccc.npy')}
    saved={str(p.relative_to(previous)):checksum(p) for folder in ('gnn','expression','anchor_inputs/input') for p in (previous/folder).rglob('*') if p.is_file() and 'cache' not in p.parts}
    manifest={'previous_results':str(previous),'previous_config':cfg,'previous_summary':old,'input_hashes':saved,'anchor_hashes':anchor,'mixing':mixing,
        'implementation':{p.name:checksum(p) for p in (Path(__file__),Path(__file__).parents[1]/'gnn/spatial_refinement.py',Path(__file__).parents[1]/'gnn/evaluation.py')}}
    out.mkdir(parents=True,exist_ok=True);identity=out/'refinement_config.json'
    if identity.exists() and json.loads(identity.read_text())!=manifest:raise ValueError('Refinement configuration changed; preserve previous outputs')
    identity.write_text(json.dumps(manifest,indent=2)+'\n')
    if (out/'niche_refinement_complete.json').exists():
        done=json.loads((out/'niche_refinement_complete.json').read_text())
        if done['manifest']==manifest and all(checksum(out/name)==digest for name,digest in done['files'].items()):return out
    # Copy prior artifacts, excluding figures/baselines and irrelevant caches.
    # Existing complete outputs are never overwritten; no symbolic links.
    for name in ('gnn','expression','anchor_inputs','ground_truth'):
        shutil.copytree(previous/name,out/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('cache','encoder_features.npy'))
    for p in previous.iterdir():
        if p.is_file() and p.suffix in ('.npy','.npz','.csv','.h5ad','.json') and p.name not in ('complete.json','notebook_completed.json'):
            shutil.copy2(p,out/p.name)
    a=ad.read_h5ad(source/'injected.h5ad');data=adapt(a,pd.read_csv(source/'lr.csv'),base,out/'anchor_inputs')
    if checksum(out/'anchor_inputs/input/ccc.npy')!=anchor['input/ccc.npy']:raise ValueError('CCC supervision changed')
    frames=[];method_labels={}
    for family,folder in [('CCC-only GNN',source/'gnn'),('CCC + gene GNN',out/'gnn')]:
        z=np.load(folder/'Z.npy');h=np.load(folder/'H.npy')
        x=z@dictionary_program_projection(h,data.feature_scale)
        pooled=spatial_cluster_features(x,np.asarray(data.members),np.asarray(data.weights),data.rows,mixing)
        labels=KMeans(n_clusters=base.fixed_k,n_init=10,random_state=base.seed).fit(pooled[data.rows[0]]).predict(pooled)
        method_labels[family]=labels
        if family=='CCC-only GNN':
            cc=out/'ccc_only';cc.mkdir(exist_ok=True)
            pd.DataFrame({'cell_id':a.obs.cell_id.to_numpy(),'CCC_Phe GNN (fixed K)':labels}).to_csv(cc/'raw_clusters.csv',index=False)
            continue
        arrays={name:np.load(folder/(name+'.npy')) for name in ('Z','P','activity')}
        np.save(folder/'clustering_features.npy',pooled)
        gcfg=replace(gnn_config(base,folder),output_dir=str(folder))
        k,adaptive,_=discover(data,arrays,h,gcfg,program_projection=dictionary_program_projection(h,data.feature_scale),clustering_values=pooled,full_kmeans=True)
        import torch
        cp=torch.load(folder/'niche_model.pt',map_location='cpu',weights_only=False);cp['local_mixing']=mixing;cp['clustering_feature_definition']='spatial_cluster_features(Z @ program_projection)';torch.save(cp,folder/'niche_model.pt')
        pd.DataFrame({'cell_id':a.obs.cell_id.to_numpy(),'CCC_Phe GNN (fixed K)':labels,'CCC_Phe GNN (selected K)':adaptive}).to_csv(out/'raw_clusters.csv',index=False)
    # Only now read region truth: evaluation only, never clustering or K choice.
    from .evaluation import label_mapping,mapped,partition_scores
    truth=a.obs.true_niche.to_numpy();edge=np.isin(truth,[2,3]);train=data.rows[0];etr=train[edge[train]]
    for name,pred in [*method_labels.items(),('CCC + gene GNN (selected K)',adaptive)]:
        alignment=mapped(pred,label_mapping(pred,truth,train));ab=label_mapping(pred,truth,etr)
        for scope,r in [('heldout_test',data.rows[2]),('all_transductive',np.arange(len(a)))]:
            frames.append({'method':name if name=='CCC-only GNN' else name.replace('CCC + gene GNN','CCC + gene GNN (fixed K)') if name=='CCC + gene GNN' else name,'evaluation':scope,'K':len(np.unique(pred)),**partition_scores(truth,pred,alignment,r,ab,etr)})
    pd.DataFrame(frames).to_csv(out/'metrics.csv',index=False)
    top=[];compositions=[]
    for niche in np.unique(adaptive):
        r=np.flatnonzero(adaptive==niche);mean_z=z[r].mean(0);ccc=mean_z@h
        for rank,f in enumerate(np.argsort(-ccc)[:10],1):
            feature=data.features.iloc[f];p=int(np.argmax(mean_z*h[:,f]));top.append({'niche':int(niche)+1,'rank':rank,'sender':feature.sender,'receiver':feature.receiver,'LR':f'{feature.ligand}–{feature.receptor}','LR_id':feature.lr_id,'mean_reconstructed_CCC':float(ccc[f]),'main_program':p+1})
        compositions.append(np.asarray(data.composition_target[r]).mean(0))
    pd.DataFrame(top).to_csv(out/'niche_top10_CCC.csv',index=False)
    pd.DataFrame(compositions,index=np.unique(adaptive)+1,columns=data.categories).rename_axis('niche').to_csv(out/'niche_cell_composition.csv')
    updated={**old,'selected_K':int(k),'CCC_only_niche_results_dir':str(out/'ccc_only'),
        'niche_refinement':{'solver':'full KMeans','local_mixing':mixing,'donors':'Existing KNN30 Gaussian weights, >=5 eligible donors; same physical fold for held-out centers','raw_Z_H_CCC_unchanged':True},'unrefined_results':str(previous)}
    (out/'summary.json').write_text(json.dumps(updated,indent=2)+'\n')
    for name,digest in anchor.items():
        if checksum(source/name)!=digest:raise RuntimeError('CCC anchor changed: '+name)
    for name in ('gnn/Z.npy','gnn/H.npy','gnn/checkpoint.pt','expression/prediction.npy','expression/checkpoint.pt'):
        if checksum(out/name)!=saved[name]:raise RuntimeError('Model artifact changed: '+name)
    required=('summary.json','raw_clusters.csv','metrics.csv','niche_top10_CCC.csv','niche_cell_composition.csv','ccc_only/raw_clusters.csv','gnn/K_selection.csv','gnn/selected_model/niche_labels.npy')
    (out/'niche_refinement_complete.json').write_text(json.dumps({'manifest':manifest,'files':{name:checksum(out/name) for name in required}},indent=2)+'\n')
    return out
