"""Expression-only composition GNN, independent of the clinical CCC/Cox mainline."""
from pathlib import Path
import json,subprocess,sys
import numpy as np
import pandas as pd
import torch
from .config import GNNConfig
from .audit import audit_study,protect_output
from .data import prepare_graph_data
from .training import fit,infer,epoch
from .model import model_from_config
from .evaluation import discover,diagnostics

REQUIRED=('checkpoint.pt','config.json','summary.json','training.csv','heldout_metrics.json',
          'Z.npy','P.npy','activity.npy','H.npy','B.npy','CCC_reconstruction.npy','composition_prediction.npy',
          'reconstruction_metrics.json','reconstruction_feature_metrics.csv','K_selection.csv',
          'selected_model/niche_assignments.csv.gz','selected_model/Q.npy','selected_model/H.npy',
          'selected_model/program_contribution.csv','selected_model/top_ccc.csv','programs/top_ccc.csv',
          'cache/preprocessing.npz','cache/features.csv','cache/composition_target.npy','audit.json','diagnostics.json')


def require_audit(config):
    report=audit_study(config)
    if not report['ready_for_training']: raise RuntimeError('Real analysis blocked: '+'; '.join(report['blockers']))
    return report


def results_complete(config):
    output=Path(config.output_dir)
    if not all((output/p).is_file() for p in REQUIRED): return False
    if GNNConfig(**json.loads((output/'config.json').read_text())).to_dict()!=config.to_dict():
        raise ValueError('Saved result configuration differs; choose an independent output directory')
    return json.loads((output/'summary.json').read_text()).get('architecture')==config.architecture


def load_checkpoint(data,config):
    payload=torch.load(Path(config.output_dir)/'checkpoint.pt',map_location='cpu',weights_only=False)
    if payload.get('architecture')!=config.architecture or GNNConfig(**payload['config']).to_dict()!=config.to_dict():
        raise ValueError('Existing checkpoint belongs to a different model/configuration; it will not be overwritten')
    for key in ('source_signature','cell_order_sha256'):
        if payload['data_audit'][key]!=data.audit[key]: raise ValueError('Checkpoint input identity changed')
    for key in ('encoder_indices','gene_mean','gene_scale','feature_scale','feature_mask','folds'):
        if not np.array_equal(payload[key],getattr(data,key)): raise ValueError('Checkpoint preprocessing changed: '+key)
    for key,default in [('expression_normalization','library_log'),('library_mean',0.),('library_scale',10.)]:
        if payload.get(key,default)!=getattr(data,key):raise ValueError('Checkpoint expression transform changed: '+key)
    if config.cell_state_components:
        saved=payload.get('cell_state_preprocessing',{})
        current=data.cell_state_preprocessing or {}
        if saved.keys()!=current.keys() or any(not np.array_equal(saved[key],current[key]) for key in current):raise ValueError('Checkpoint cell-state RNA preprocessing changed')
    if config.ccc_nonzero_weight_max>1:
        for key in ('feature_mean','feature_nonzero_fraction'):
            if key not in payload or not np.array_equal(payload[key],getattr(data,key)):raise ValueError('Checkpoint training loss preprocessing changed: '+key)
    if payload['features']!=data.features.to_dict('list') or tuple(payload['categories'])!=data.categories: raise ValueError('Checkpoint feature/type order changed')
    model=model_from_config(data,config)
    model.load_state_dict(payload['model_state'])
    from ..data import resolve_device
    return model.to(resolve_device(config.device)),payload


def export_reconstruction(data,arrays,h,b,config):
    output=Path(config.output_dir);n=len(data.cells);f=len(data.features)
    c=np.lib.format.open_memmap(output/'CCC_reconstruction.incomplete.npy',mode='w+',dtype=np.float32,shape=(n,f))
    pi=np.lib.format.open_memmap(output/'composition_prediction.incomplete.npy',mode='w+',dtype=np.float32,shape=(n,len(data.categories)))
    records={i:{'n':0,'comp_n':0,'sum':np.zeros(f),'square':np.zeros(f),'sse':np.zeros(f),'huber':0.,'kl':0.,'comp_sse':0.,
                'positive_huber':0.,'positive_n':0,'zero_huber':0.,'zero_n':0,'weighted_huber':0.,'zero_baseline':0.,'mean_baseline':0.} for i in range(3)}
    mask=data.feature_mask;scale=data.feature_scale
    rho=data.feature_nonzero_fraction
    if config.ccc_nonzero_weight_max>1 and rho is None:raise ValueError('Missing training feature support for weighted reconstruction metrics')
    boost=np.ones(int(mask.sum())) if config.ccc_nonzero_weight_max==1 else np.clip(np.sqrt((1-rho[mask])/np.maximum(rho[mask],1e-8)),1,config.ccc_nonzero_weight_max)
    normalizer=np.ones_like(boost) if rho is None else 1-rho[mask]+rho[mask]*boost
    huber=lambda error:np.where(np.abs(error)<=1,.5*error**2,np.abs(error)-.5)
    for start in range(0,n,4096):
        stop=min(start+4096,n);c[start:stop]=np.asarray(arrays['Z'][start:stop])@h;pi[start:stop]=np.asarray(arrays['P'][start:stop])@b
        actual=data.targets[slice(start,stop)].astype(np.float64);estimate=np.asarray(c[start:stop],np.float64)
        for fold,r in records.items():
            keep=data.folds[start:stop]==fold;a=actual[keep];p=estimate[keep];r['n']+=len(a)
            r['sum']+=a.sum(0);r['square']+=(a*a).sum(0);r['sse']+=((a-p)**2).sum(0)
            error=(a[:,mask]-p[:,mask])/scale[mask];loss=huber(error);positive=a[:,mask]>0;weight=np.where(positive,boost,1)/normalizer
            r['huber']+=loss.sum();r['weighted_huber']+=(loss*weight).sum()
            r['positive_huber']+=loss[positive].sum();r['positive_n']+=int(positive.sum())
            r['zero_huber']+=loss[~positive].sum();r['zero_n']+=int((~positive).sum())
            r['zero_baseline']+=(huber(a[:,mask]/scale[mask])*weight).sum()
            if data.feature_mean is not None:r['mean_baseline']+=(huber((a[:,mask]-data.feature_mean[mask])/scale[mask])*weight).sum()
            truth=np.asarray(data.composition_target[start:stop][keep],np.float64);pred=np.asarray(pi[start:stop][keep],np.float64)
            valid=truth.sum(1)>0;truth=truth[valid];pred=pred[valid];r['comp_n']+=len(truth)
            r['kl']+=(truth*(np.log(np.maximum(truth,1e-8))-np.log(np.maximum(pred,1e-8)))).sum();r['comp_sse']+=((truth-pred)**2).sum()
    c.flush();pi.flush();del c,pi
    for name in ('CCC_reconstruction','composition_prediction'): (output/(name+'.incomplete.npy')).replace(output/(name+'.npy'))
    feature=data.features.copy();feature['effective_mask']=mask;metrics={}
    for fold,name in enumerate(('train','validation','test')):
        r=records[fold];sst=r['square']-r['sum']**2/max(r['n'],1)
        r2=1-np.divide(r['sse'],sst,out=np.full_like(sst,np.nan),where=sst>1e-12);feature[name+'_R2']=r2
        metrics[name]={'n':r['n'],'scaled_huber':r['huber']/max(r['n']*mask.sum(),1),
                       'median_feature_R2':float(np.nanmedian(r2[mask]))}
        metrics[name].update(weighted_huber=r['weighted_huber']/max(r['n']*mask.sum(),1),
            positive_scaled_huber=r['positive_huber']/max(r['positive_n'],1),zero_scaled_huber=r['zero_huber']/max(r['zero_n'],1),
            target_nonzero_fraction=r['positive_n']/max(r['n']*mask.sum(),1),
            zero_predictor_weighted_huber=r['zero_baseline']/max(r['n']*mask.sum(),1),
            gain_over_zero_predictor=1-r['weighted_huber']/max(r['zero_baseline'],1e-12),
            native_relative_RMSE=float(np.sqrt(r['sse'][mask].sum()/max(r['square'][mask].sum(),1e-12))))
        if data.feature_mean is not None:metrics[name].update(
            train_mean_predictor_weighted_huber=r['mean_baseline']/max(r['n']*mask.sum(),1),
            gain_over_train_mean=1-r['weighted_huber']/max(r['mean_baseline'],1e-12))
        if config.lambda_comp>0:metrics[name].update({'composition_valid_n':r['comp_n'],
                       'composition_KL':r['kl']/r['comp_n'] if r['comp_n'] else None,
                       'composition_RMSE':float(np.sqrt(r['comp_sse']/(r['comp_n']*len(data.categories)))) if r['comp_n'] else None})
    feature.to_csv(output/'reconstruction_feature_metrics.csv',index=False)
    (output/'reconstruction_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')


def run(config):
    require_audit(config)
    if results_complete(config): return json.loads((Path(config.output_dir)/'summary.json').read_text())
    data=prepare_graph_data(config);output=Path(config.output_dir)
    if (output/'checkpoint.pt').is_file():
        model,payload=load_checkpoint(data,config)
        if not (output/'heldout_metrics.json').is_file():
            test=epoch(model,data,data.rows[2],config)
            (output/'heldout_metrics.json').write_text(json.dumps({'best_epoch':payload['best_epoch'],'test':test},indent=2)+'\n')
    else: model=fit(data,config)
    arrays=infer(model,data,output,config.batch_size);h=model.dictionary.detach().cpu().numpy();b=model.composition_dictionary.detach().cpu().numpy()
    k,labels,_=discover(data,arrays,h,config)
    summary=diagnostics(data,arrays,h,labels,config)
    export_reconstruction(data,arrays,h,b,config)
    summary.update(selected_K=k,architecture=config.architecture,lambda_comp=config.lambda_comp,
                   encoder_gene_selection=data.audit.get('encoder_gene_selection'),ccc_nonzero_weight_max=config.ccc_nonzero_weight_max,
                   phenotype_supervision=False,legacy_results_modified=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def inference(config):
    require_audit(config);data=prepare_graph_data(config);model,_=load_checkpoint(data,config)
    return infer(model,data,config.output_dir,config.batch_size)


def ensure_results(config_path):
    """Called by Notebook Run All, never on notebook generation/import."""
    config_path=Path(config_path).resolve();config=GNNConfig.from_json(config_path);protect_output(config)
    from .inputs import resolve_input
    from ..ccc.config import default_config
    from ..ccc.data import prepare_prime5k_ccc,cache_paths,_cache_is_valid,_manifest_signature
    inp=resolve_input(config,default_config())
    if not _cache_is_valid(cache_paths(inp),_manifest_signature(inp)):
        if cache_paths(inp)['manifest'].is_file(): raise ValueError('Existing CCC cache differs; refusing to overwrite it')
        prepare_prime5k_ccc(inp)
    # The audit script replays outcome-blind labels and exact directed CCC on real rows.
    if not Path(config.audit_file).is_file():
        script=Path(__file__).resolve().parents[3]/'scripts/audit_prime5k_gnn.py'
        subprocess.run([sys.executable,str(script),'--config',str(config_path)],check=True,cwd=str(script.parent.parent))
    return run(config)
