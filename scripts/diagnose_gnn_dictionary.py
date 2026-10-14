"""Controlled, sampled real-data optimization diagnosis; never runs all centers."""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.audit import audit_study
from phenoniche.gnn.data import GraphData, Targets, align_cells
from phenoniche.gnn.inputs import resolve_input
from phenoniche.ccc.config import default_config
from phenoniche.ccc.data import load_cached_input
from phenoniche.gnn.model import model_from_config
from phenoniche.gnn.training import epoch, make_optimizer
from phenoniche.training import set_seed


def load_data(config):
    report=audit_study(config,write=False)
    if not report['ready_for_training']:raise RuntimeError(report['blockers'])
    cached=load_cached_input(resolve_input(config,default_config()))
    x=ad.read_h5ad(resolve_input(config,default_config()).processed_h5ad)
    order=align_cells(x,cached);expression=sp.csr_matrix(x.X[order],dtype=np.float32)
    cache=Path(config.output_dir)/'cache';pre=np.load(cache/'preprocessing.npz')
    audit=json.loads((cache/'audit.json').read_text())
    if audit['source_signature']!=cached['manifest']['signature']:raise ValueError('Source preprocessing identity mismatch')
    genes=tuple(str(g).upper() for g in x.var_names);categories=tuple(audit['categories'])
    types=pd.Categorical(cached['cells'].cell_type,categories=categories).codes.astype(np.int64)
    features=pd.read_csv(cache/'features.csv')
    if not features.equals(cached['features'].iloc[pre['columns']].reset_index(drop=True)):raise ValueError('CCC feature ordering mismatch')
    folds=pre['folds'];rows=[np.flatnonzero(folds==f) for f in range(3)]
    return GraphData(expression,np.asarray(expression.sum(1)).ravel(),genes,pre['encoder_indices'],pre['gene_mean'],pre['gene_scale'],
        types,categories,cached['coordinates'],np.load(cache/'members.npy',mmap_mode='r'),np.load(cache/'anchor_weights.npy',mmap_mode='r'),
        Targets(cached['matrix'],pre['columns']),pre['feature_scale'],pre['feature_mask'],features,cached['cells'],rows,folds,pre['blocks'],
        np.load(cache/'composition_target.npy',mmap_mode='r'),audit,pre['feature_nonzero_fraction'],pre['feature_mean'])


def resident(data,cache,device):
    values=np.load(Path(cache)/'encoder_features.npy',mmap_mode='r')
    if values.shape!=(len(data.cells),len(data.encoder_indices)+1):raise ValueError('Encoder cache shape mismatch')
    tensor=lambda a:torch.as_tensor(np.asarray(a).copy(),device=device)
    data._resident={key:tensor(value) for key,value in {'expression':values,'coordinates':data.coordinates,
        'members':data.members,'weights':data.weights,'composition_target':data.composition_target}.items()}
    data._resident['target']=torch.empty(data.targets.shape,device=device)
    for start in range(0,len(data.cells),8192):
        stop=min(start+8192,len(data.cells));data._resident['target'][start:stop]=tensor(data.targets[slice(start,stop)])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='config/gnn.json')
    parser.add_argument('--output',required=True);parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--train',type=int,default=8192);parser.add_argument('--validation',type=int,default=4096)
    parser.add_argument('--device',default='cuda:2');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=args.resume)
    cfg=GNNConfig.from_json(args.config);source=Path(cfg.output_dir)/'training_best.pt'
    frozen=out/'starting_checkpoint.pt'
    payload=torch.load(frozen if args.resume else source,map_location='cpu',weights_only=False)
    if GNNConfig(**payload['config']).to_dict()!=cfg.to_dict():raise ValueError('Checkpoint config mismatch')
    if not args.resume:torch.save(payload,frozen)
    torch.set_num_threads(8);data=load_data(cfg);resident(data,Path(cfg.output_dir)/'cache',args.device)
    rng=np.random.default_rng(90210);train=rng.choice(data.rows[0],min(args.train,len(data.rows[0])),replace=False)
    valid=rng.choice(data.rows[1],min(args.validation,len(data.rows[1])),replace=False)
    np.savez(out/'sampled_rows.npz',train=train,validation=valid)
    arms=[('softplus',None),('projected_1e-5',1e-5),('projected_5e-5',5e-5),('projected_2e-4',2e-4),
        ('projected_1e-3',1e-3),('softplus_scaled_activity',None),('projected_scaled_activity',2e-4),
        ('feature_scaled_2e-4',2e-4),('feature_scaled_1e-3',1e-3)]
    records=pd.read_csv(out/'comparison.csv').to_dict('records') if args.resume else [];summary=[]
    for name,rate in arms:
        previous=[r for r in records if r['arm']==name]
        if previous and max(r['epoch'] for r in previous)==args.epochs:
            summary.append(min(previous,key=lambda r:r['validation_total']));continue
        records=[r for r in records if r['arm']!=name]
        config=replace(cfg,device=args.device,dictionary_parameterization='softplus' if rate is None else 'projected',dictionary_lr=rate or cfg.dictionary_lr,
            dictionary_coordinates='feature_scaled' if name.startswith('feature_scaled') else 'native')
        model=model_from_config(data,config)
        state=copy.deepcopy(payload['model_state'])
        if hasattr(model,'dictionary_scale'):state['dictionary_scale']=model.dictionary_scale
        model.load_state_dict(state)
        if rate is not None:
            old=model_from_config(data,cfg);old.load_state_dict(payload['model_state'])
            with torch.no_grad():
                h=old.dictionary
                if hasattr(model,'dictionary_scale'):
                    h=h/model.dictionary_scale;h=h/h.sum(1,keepdim=True)
                model.raw_dictionary.copy_(h)
            del old
        model.to(args.device);activity_scale=1.;hook=None
        if 'scaled_activity' in name:
            activity_scale=max(float(model.activity_head.bias.detach().abs()),1.)
            with torch.no_grad():
                model.activity_head.weight.div_(activity_scale);model.activity_head.bias.div_(activity_scale)
            hook=model.activity_head.register_forward_hook(lambda module,args,value:value*activity_scale)
        optimizer=make_optimizer(model,config);best=float('inf');best_metrics=None
        for number in range(args.epochs+1):
            set_seed(cfg.seed+number);start=time.monotonic()
            tr=epoch(model,data,train,config,optimizer,number) if number else {}
            va=epoch(model,data,valid,config,None,number)
            record={'arm':name,'epoch':number,'seconds':time.monotonic()-start,
                **{'train_'+k:v for k,v in tr.items()},**{'validation_'+k:v for k,v in va.items()}}
            records.append(record);pd.DataFrame(records).to_csv(out/'comparison.csv',index=False)
            score=va['ccc']+config.lambda_reg*va['regularization']
            print(name,number,'CCC',round(va['ccc'],6),'pos',round(va['ccc_positive_huber'],6),'zero',round(va['ccc_zero_huber'],6),'gain',round(va['ccc_gain_over_mean'],4),flush=True)
            if score<best:
                best=score;best_metrics=record
                state=copy.deepcopy(model.state_dict())
                for key in ('activity_head.weight','activity_head.bias'):state[key]*=activity_scale
                torch.save({'model_state':state,'config':config.to_dict(),'epoch':number,'activity_parameter_scale':activity_scale,
                    'source_epoch':payload['epoch'],'sampled_train_rows':train,'sampled_validation_rows':valid},out/(name+'_best.pt'))
        summary.append(best_metrics);del model,optimizer;torch.cuda.empty_cache()
    (out/'summary.json').write_text(json.dumps({'source_epoch':payload['epoch'],'train_centers':len(train),'validation_centers':len(valid),
        'test_set_used':False,'source_checkpoint_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'best_by_arm':summary},indent=2)+'\n')


if __name__=='__main__':main()
