"""Warm-start CCC error-scale ablation, selected on one fixed validation metric."""
import argparse,copy,json,time
from dataclasses import replace
from functools import partial
from pathlib import Path
import numpy as np,pandas as pd,torch
from diagnose_gnn_dictionary import load_data
from diagnose_gnn_feature_scope import state_diagnostics
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.model import model_from_config
from phenoniche.gnn.losses import ccc_loss
from phenoniche.gnn.training import epoch,make_optimizer
from phenoniche.training import set_seed


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=60);a=p.parse_args()
    source=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    payload=torch.load(source/'starting_checkpoint.pt',map_location='cpu',weights_only=False);cfg=replace(GNNConfig(**payload['config']),device='cuda:2')
    torch.set_num_threads(8);data=load_data(cfg);sampled=np.load(source/'sampled_rows.npz');train=sampled['train'];valid=sampled['validation']
    data.prepare_subset_device(np.r_[train,valid],cfg.device);records=[];summaries=[]
    for kind,delta in [('huber',3.),('mse',1.)]:
        set_seed(cfg.seed);model=model_from_config(data,cfg).to(cfg.device);model.load_state_dict(payload['model_state'])
        optimizer=make_optimizer(model,cfg);best=float('inf');stale=0;name='huber3' if kind=='huber' else 'mse'
        objective=partial(ccc_loss,kind=kind,delta=delta,nonzero_fraction=data.feature_nonzero_fraction,nonzero_weight_max=cfg.ccc_nonzero_weight_max)
        for e in range(a.epochs+1):
            set_seed(cfg.seed+e);start=time.monotonic();tr=epoch(model,data,train,cfg,optimizer,e,objective) if e else {};va=epoch(model,data,valid,cfg,None,e)
            r={'arm':name,'epoch':e,'seconds':time.monotonic()-start,**{'train_'+k:v for k,v in tr.items()},**{'validation_fixed_huber1_'+k:v for k,v in va.items()}}
            records.append(r);pd.DataFrame(records).to_csv(out/'comparison.csv',index=False)
            print(name,e,'train objective',round(tr.get('ccc',float('nan')),6),'val fixed Huber1',round(va['ccc'],6),'positive',round(va['ccc_positive_huber'],6),flush=True)
            if va['total']<best-1e-7:
                best=va['total'];stale=0
                torch.save({'model_state':copy.deepcopy(model.state_dict()),'config':cfg.to_dict(),'epoch':e,'experiment_loss':{'kind':kind,'delta':delta}},out/(name+'_best.pt'))
            else:
                stale+=1
                if stale>=cfg.patience:break
        saved=torch.load(out/(name+'_best.pt'),map_location=cfg.device,weights_only=False);model.load_state_dict(saved['model_state'])
        summaries.append({'arm':name,'best_epoch':saved['epoch'],'validation_diagnostics':state_diagnostics(model,data,valid,cfg)})
        (out/'summary.json').write_text(json.dumps({'train_centers':len(train),'validation_centers':len(valid),'test_set_used':False,
            'source_epoch':payload['epoch'],'same_validation_metric':'weighted Huber delta=1 for every arm','models':summaries},indent=2)+'\n')
        del model,optimizer;torch.cuda.empty_cache()


if __name__=='__main__':main()
