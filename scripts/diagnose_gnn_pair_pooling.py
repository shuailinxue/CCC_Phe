"""Warm-start control of expression-only program-specific pair pooling, on sampled centers."""
import argparse,copy,json,time
from dataclasses import replace
from pathlib import Path
import numpy as np,pandas as pd,torch
from diagnose_gnn_dictionary import load_data
from diagnose_gnn_feature_scope import state_diagnostics
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.model import model_from_config
from phenoniche.gnn.training import epoch,make_optimizer
from phenoniche.training import set_seed


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=60);p.add_argument('--lr-pooling',action='store_true');p.add_argument('--no-dropout',action='store_true');a=p.parse_args()
    source=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    payload=torch.load(source/'starting_checkpoint.pt',map_location='cpu',weights_only=False);cfg=replace(GNNConfig(**payload['config']),device='cuda:2')
    if a.no_dropout:cfg=replace(cfg,dropout=0.)
    torch.set_num_threads(8);data=load_data(cfg);sampled=np.load(source/'sampled_rows.npz');train=sampled['train'];valid=sampled['validation']
    data.prepare_subset_device(np.r_[train,valid],cfg.device)
    records=[];summaries=[]
    for pooling in [False,True]:
        set_seed(cfg.seed);config=replace(cfg,program_pair_pooling=pooling,lr_pair_pooling=pooling and a.lr_pooling,max_epochs=a.epochs)
        model=model_from_config(data,config);state=model.state_dict();state.update(payload['model_state']);model.load_state_dict(state);model.to(cfg.device)
        optimizer=make_optimizer(model,config);best=float('inf');stale=0;name='pair_pooling' if pooling else 'control'
        for e in range(a.epochs+1):
            set_seed(cfg.seed+e);start=time.monotonic();tr=epoch(model,data,train,config,optimizer,e) if e else {};va=epoch(model,data,valid,config,None,e)
            r={'arm':name,'epoch':e,'seconds':time.monotonic()-start,**{'train_'+k:v for k,v in tr.items()},**{'validation_'+k:v for k,v in va.items()}}
            records.append(r);pd.DataFrame(records).to_csv(out/'comparison.csv',index=False)
            print(name,e,'train',round(tr.get('ccc',float('nan')),6),'val',round(va['ccc'],6),'gain',round(va['ccc_gain_over_mean'],4),flush=True)
            if va['total']<best-1e-7:
                best=va['total'];stale=0
                torch.save({'model_state':copy.deepcopy(model.state_dict()),'config':config.to_dict(),'epoch':e},out/(name+'_best.pt'))
            else:
                stale+=1
                if stale>=config.patience:break
        saved=torch.load(out/(name+'_best.pt'),map_location=cfg.device,weights_only=False);model.load_state_dict(saved['model_state'])
        summaries.append({'arm':name,'best_epoch':saved['epoch'],'validation_diagnostics':state_diagnostics(model,data,valid,config)})
        (out/'summary.json').write_text(json.dumps({'train_centers':len(train),'validation_centers':len(valid),'test_set_used':False,
            'source_epoch':payload['epoch'],'identical_initial_predictions':True,'models':summaries},indent=2)+'\n')
        del model,optimizer;torch.cuda.empty_cache()


if __name__=='__main__':main()
