"""Independent two-section cohort runs using the existing production CCC/GNN."""
from pathlib import Path
from functools import partial
import json,hashlib
import numpy as np,pandas as pd,torch
from . import colitis as shared
from .atopic_dermatitis import prepare_tissue
from .dermatomyositis import clustering_values
from ..gnn.config import GNNConfig
from ..gnn.training import fit,infer
from ..gnn.evaluation import discover
from ..gnn.losses import ccc_loss


def input_root(source):
    source=Path(source);c=json.loads((source/'cohort_config.json').read_text())
    return Path(c.get('prepared_data_output',source))/'input'


def make_ccc_objective(data,config):
    return partial(ccc_loss,kind='huber',delta=3.,nonzero_fraction=data.feature_nonzero_fraction,
                   nonzero_weight_max=config.ccc_nonzero_weight_max)


def refresh_joint_tables(out):
    """Reported centers, never physical halo cells, define niche summaries."""
    out=Path(out);frame=pd.read_csv(out/'assignments.csv.gz')
    if 'analysis_center' not in frame:return
    valid=frame.analysis_center.to_numpy(bool);labels=frame.niche.to_numpy();z=np.load(out/'gnn/Z.npy',mmap_mode='r');h=np.load(out/'gnn/H.npy')
    cache=input_root(out);comp=np.load(cache/'composition.npy',mmap_mode='r');features=pd.read_csv(cache/'features.csv');k=int(labels.max())
    pd.crosstab(frame.loc[valid,'condition'],frame.loc[valid,'niche'],normalize='index').to_csv(out/'condition_niche_fractions.csv')
    means=np.stack([z[(labels==i)&valid].mean(0) for i in range(1,k+1)])
    pd.DataFrame(means,index=np.arange(1,k+1),columns=np.arange(1,33)).to_csv(out/'niche_program_activity.csv')
    np.save(out/'niche_composition.npy',np.stack([comp[(labels==i)&valid].mean(0) for i in range(1,k+1)]))
    records=[]
    for i,mean in enumerate(means,1):
        signal=mean@h
        for rank,j in enumerate(np.argsort(-signal)[:10],1):
            f=features.iloc[j];records.append({'niche':i,'rank':rank,'sender':f.sender,'receiver':f.receiver,'LR':f.ligand+'–'+f.receptor,'mean_reconstructed_CCC':float(signal[j])})
    pd.DataFrame(records).to_csv(out/'niche_top10_CCC.csv',index=False)


def run(config_path):
    torch.set_num_threads(8);c=shared.read_config(config_path);out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=True)
    sig=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
    if (out/'complete.json').exists():
        if json.loads((out/'complete.json').read_text())['config_signature']!=sig:raise ValueError('Completed configuration differs')
        refresh_joint_tables(out)
        return out
    previous=out/'cohort_config.json'
    if previous.exists() and json.loads(previous.read_text())!=c:raise ValueError('Refusing to mix different configurations')
    previous.write_text(json.dumps(c,indent=2))
    prepared=Path(c.get('prepared_data_output',out))
    if (prepared/'input/preprocessing.npz').exists():
        prepared_config=json.loads((prepared/'cohort_config.json').read_text())
        input_keys=('dataset','data_dir','annotation_donor_counts','counts_source','sections','lr_atlas','seed',
            'core_link_um','minimum_core_cells','coverage_fraction','opportunity_fraction','feature_nonzero_fraction',
            'feature_scope','excluded_ccc_types','encoder_genes','split_unit','evaluation_centers_per_patient','healthy_roi_centers','case_roi_centers')
        if any(c.get(key)!=prepared_config.get(key) for key in input_keys):raise ValueError('Reused input-defining configuration differs')
        a,data=shared.load_prepared(prepared_config,prepared)
    else:
        if c.get('dataset')=='kidney_dkd':
            from .kidney_dkd import prepare_tissue as prepare_kidney
            tissue=prepare_kidney(c,out)
        else:tissue=prepare_tissue(c,out)
        a,data=shared.prepare(c,out,tissue=tissue)
    budget=c.get('diagnostic_training_centers')
    if budget:
        rng=np.random.default_rng(c['seed']);data.rows[0]=np.sort(rng.choice(data.rows[0],min(budget,len(data.rows[0])),replace=False))
        data.rows[1]=np.sort(rng.choice(data.rows[0],min(1000,len(data.rows[0])),replace=False))
        data.audit['split']='Subsampled joint discovery diagnostic; monitor strictly within optimized centers'
        data.audit['optimized_centers_actual']=len(data.rows[0])
    g=GNNConfig(seed=c['seed'],programs=32,hidden=c['hidden'],encoder_genes=len(data.encoder_indices),
        architecture='expression_gnn_residual_v2',message_layers=c['message_layers'],batch_size=c['batch_size'],
        max_epochs=c['epochs'],patience=c['patience'],device=c['device'],lambda_comp=0.,lambda_reg=.005,
        dictionary_initialization='nmf',output_dir=str(out/'gnn'),k_min=4,k_max=15,
        clustering_sample_size=12000,silhouette_sample_size=1200,near_optimal_score_tolerance=0.,
        ccc_nonzero_weight_max=c.get('ccc_nonzero_weight_max',1.),dictionary_parameterization=c.get('dictionary_parameterization','softplus'),
        dictionary_coordinates=c.get('dictionary_coordinates','native'),dictionary_lr=c.get('dictionary_lr',.00005))
    model=shared.restore_checkpoint(data,g) if (out/'gnn/checkpoint.pt').exists() else fit(data,g,rescale_activity=True,ccc_objective=make_ccc_objective(data,g))
    if getattr(data,'_resident',None) is None:data.prepare_device(next(model.parameters()).device,out/'gnn/cache')
    arrays=infer(model,data,out/'gnn',c['batch_size']);z=np.asarray(arrays['Z']);h=np.load(out/'gnn/H.npy')
    values,projection=clustering_values(data,arrays,h,c)
    k,labels,_=discover(data,arrays,h,g,program_projection=projection,clustering_values=values,full_kmeans=True)
    frame=a.obs.reset_index(drop=True).copy();frame['niche']=labels+1;frame['x']=data.coordinates[:,0];frame['y']=data.coordinates[:,1]
    frame.to_csv(out/'assignments.csv.gz',index=False)
    pd.crosstab(frame.condition,frame.niche,normalize='index').to_csv(out/'condition_niche_fractions.csv')
    means=np.stack([z[labels==i].mean(0) for i in range(k)]);pd.DataFrame(means,index=np.arange(1,k+1),columns=np.arange(1,33)).to_csv(out/'niche_program_activity.csv')
    np.save(out/'niche_composition.npy',np.stack([data.composition_target[labels==i].mean(0) for i in range(k)]))
    rows=[]
    for i in range(k):
        signal=means[i]@h
        for rank,j in enumerate(np.argsort(-signal)[:10],1):
            feature=data.features.iloc[j];rows.append({'niche':i+1,'rank':rank,'sender':feature.sender,'receiver':feature.receiver,'LR':feature.ligand+'–'+feature.receptor,'mean_reconstructed_CCC':float(signal[j])})
    pd.DataFrame(rows).to_csv(out/'niche_top10_CCC.csv',index=False)
    prediction=np.lib.format.open_memmap(out/'ccc_reconstruction.npy',mode='w+',dtype=np.float32,shape=data.targets.shape)
    for start in range(0,len(z),1024):prediction[start:start+1024]=z[start:start+1024]@h
    prediction.flush();del prediction
    summary={**data.audit,'selected_K':k,'programs':32,'conditions':c.get('conditions',['NL','AD']),
        'control_label':c.get('control_label','Nonlesional AD skin; not a healthy volunteer'),
        'case_label':c.get('case_label','Lesional AD skin'),'group_replicates':1,'loss':'Feature-scaled CCC Huber(delta=3) + 0.005 program regularization',
        'source_paper':c.get('source_paper','10.1038/s41590-025-02267-8'),'model_architecture_changed':False,
        'ccc_nonzero_weight_max':g.ccc_nonzero_weight_max,'dictionary_parameterization':g.dictionary_parameterization}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));refresh_joint_tables(out)
    (out/'complete.json').write_text(json.dumps({'config_signature':sig},indent=2))
    print('PAIRED MODEL COMPLETE',out,flush=True);return out
