"""Warm-start RNA-only PCA input comparison, on the same sampled microenvironments."""
import argparse,copy,json,time
from dataclasses import replace
from pathlib import Path
import numpy as np,pandas as pd,torch
from diagnose_gnn_dictionary import load_data
from diagnose_gnn_feature_scope import state_diagnostics
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.cell_state import prepare_cell_states
from phenoniche.gnn.model import model_from_config
from phenoniche.gnn.training import epoch,make_optimizer,initialize
from phenoniche.training import set_seed


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--train',type=int,default=8192);p.add_argument('--validation',type=int,default=4096);p.add_argument('--patience',type=int,default=8);p.add_argument('--from-scratch',action='store_true');a=p.parse_args()
    source=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    payload=torch.load(source/'starting_checkpoint.pt',map_location='cpu',weights_only=False);cfg=replace(GNNConfig(**payload['config']),device='cuda:2',cell_state_components=64,patience=a.patience)
    if a.from_scratch:cfg=replace(cfg,dictionary_initialization='log1p_active_kmeans')
    torch.set_num_threads(8);data=load_data(cfg);sampled=np.load(source/'sampled_rows.npz');train=sampled['train'];valid=sampled['validation']
    if a.train!=8192 or a.validation!=4096:
        if a.train>=len(data.rows[0]):raise ValueError('This diagnosis must sample fewer centers than the full training set')
        rng=np.random.default_rng(90210);train=rng.choice(data.rows[0],a.train,replace=False);valid=rng.choice(data.rows[1],a.validation,replace=False)
    np.savez(out/'sampled_rows.npz',train=train,validation=valid)
    cells=np.unique(data.members[np.r_[train,valid]])
    prepare_cell_states(data,cfg,out/'cache',cells);data.prepare_subset_device(np.r_[train,valid],cfg.device)
    records=[];summaries=[]
    for lr_pool in [False,True]:
        set_seed(cfg.seed);config=replace(cfg,program_pair_pooling=lr_pool,lr_pair_pooling=lr_pool)
        model=model_from_config(data,config);state=model.state_dict();source_state=copy.deepcopy(payload['model_state'])
        old_weight=source_state['cell_encoder.0.weight'];weight=torch.zeros_like(state['cell_encoder.0.weight'])
        weight[:,:len(data.encoder_indices)]=old_weight[:,:-1];weight[:,-1]=old_weight[:,-1];source_state['cell_encoder.0.weight']=weight
        if a.from_scratch:
            current_rows=data.rows;data.rows=[train,valid,data.rows[2]]
            model=initialize(data,config);data.rows=current_rows
        else:state.update(source_state);model.load_state_dict(state)
        model.to(cfg.device)
        optimizer=make_optimizer(model,config);best=float('inf');stale=0;name='RNA_PCA64_LR' if lr_pool else 'RNA_PCA64'
        for e in range(a.epochs+1):
            set_seed(cfg.seed+e);start=time.monotonic();tr=epoch(model,data,train,config,optimizer,e) if e else {};va=epoch(model,data,valid,config,None,e)
            r={'arm':name,'epoch':e,'seconds':time.monotonic()-start,**{'train_'+k:v for k,v in tr.items()},**{'validation_'+k:v for k,v in va.items()}}
            records.append(r);pd.DataFrame(records).to_csv(out/'comparison.csv',index=False)
            print(name,e,'train',round(tr.get('ccc',float('nan')),6),'val',round(va['ccc'],6),'gain',round(va['ccc_gain_over_mean'],4),flush=True)
            if va['total']<best-1e-7:
                best=va['total'];stale=0
                torch.save({'model_state':copy.deepcopy(model.state_dict()),'config':config.to_dict(),'epoch':e,'cell_state_preprocessing':data.cell_state_preprocessing},out/(name+'_best.pt'))
            else:
                stale+=1
                if stale>=cfg.patience:break
        saved=torch.load(out/(name+'_best.pt'),map_location=cfg.device,weights_only=False);model.load_state_dict(saved['model_state'])
        summaries.append({'arm':name,'best_epoch':saved['epoch'],'validation_diagnostics':state_diagnostics(model,data,valid,config)})
        (out/'summary.json').write_text(json.dumps({'train_centers':len(train),'validation_centers':len(valid),'test_set_used':False,
            'cell_type_or_CCC_in_PCA':False,'RNA_PCA':data.audit['cell_state_RNA_PCA'],'models':summaries},indent=2)+'\n')
        del model,optimizer;torch.cuda.empty_cache()


if __name__=='__main__':main()
