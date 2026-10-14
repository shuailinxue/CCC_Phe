"""Exact-prediction-preserving rescaling of the library feature, sampled warm-start comparison."""
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
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=60);a=p.parse_args()
    source=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    payload=torch.load(source/'starting_checkpoint.pt',map_location='cpu',weights_only=False);cfg=replace(GNNConfig(**payload['config']),device='cuda:2')
    torch.set_num_threads(8);data=load_data(cfg);sampled=np.load(source/'sampled_rows.npz');train=sampled['train'];valid=sampled['validation']
    physical=np.unique(data.members[data.rows[0]]);library=np.log1p(data.totals[physical]);mean=float(library.mean());scale=max(float(library.std()),.1)
    records=[];summaries=[]
    for standardized in [False,True]:
        set_seed(cfg.seed);config=replace(cfg,standardize_library_size=standardized,max_epochs=a.epochs);d=copy.copy(data)
        model=model_from_config(d,config);model.load_state_dict(payload['model_state'])
        if standardized:
            with torch.no_grad():
                layer=model.cell_encoder[0];original=layer.weight[:,-1].clone()
                layer.bias.add_(original*(mean-d.library_mean)/d.library_scale)
                layer.weight[:,-1].mul_(scale/d.library_scale)
            d.library_mean=mean;d.library_scale=scale
        d.prepare_subset_device(np.r_[train,valid],cfg.device);model.to(cfg.device)
        optimizer=make_optimizer(model,config);best=float('inf');stale=0;name='standardized' if standardized else 'control'
        for e in range(a.epochs+1):
            set_seed(cfg.seed+e);start=time.monotonic();tr=epoch(model,d,train,config,optimizer,e) if e else {};va=epoch(model,d,valid,config,None,e)
            r={'arm':name,'epoch':e,'seconds':time.monotonic()-start,**{'train_'+k:v for k,v in tr.items()},**{'validation_'+k:v for k,v in va.items()}}
            records.append(r);pd.DataFrame(records).to_csv(out/'comparison.csv',index=False)
            print(name,e,'train',round(tr.get('ccc',float('nan')),6),'val',round(va['ccc'],6),'gain',round(va['ccc_gain_over_mean'],4),flush=True)
            if va['total']<best-1e-7:
                best=va['total'];stale=0
                torch.save({'model_state':copy.deepcopy(model.state_dict()),'config':config.to_dict(),'epoch':e,'library_mean':d.library_mean,'library_scale':d.library_scale},out/(name+'_best.pt'))
            else:
                stale+=1
                if stale>=config.patience:break
        saved=torch.load(out/(name+'_best.pt'),map_location=cfg.device,weights_only=False);model.load_state_dict(saved['model_state'])
        summaries.append({'arm':name,'best_epoch':saved['epoch'],'validation_diagnostics':state_diagnostics(model,d,valid,config)})
        (out/'summary.json').write_text(json.dumps({'train_centers':len(train),'validation_centers':len(valid),'test_set_used':False,
            'source_epoch':payload['epoch'],'library_train_mean':mean,'library_train_scale':scale,'models':summaries},indent=2)+'\n')
        del model,optimizer,d;torch.cuda.empty_cache()


if __name__=='__main__':main()
