"""Sampled input-scale ablation with unchanged genes, CCC target and model/loss."""
import argparse,copy,json,time
from dataclasses import replace
from pathlib import Path
import numpy as np,pandas as pd,torch
from diagnose_gnn_dictionary import load_data
from diagnose_gnn_feature_scope import state_diagnostics
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.data import expression_block
from phenoniche.gnn.training import initialize,epoch,make_optimizer


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=60);a=p.parse_args()
    source=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    payload=torch.load(source/'starting_checkpoint.pt',map_location='cpu',weights_only=False);cfg=replace(GNNConfig(**payload['config']),device='cuda:2')
    torch.set_num_threads(8);data=load_data(cfg);sampled=np.load(source/'sampled_rows.npz');train=sampled['train'];valid=sampled['validation']
    physical=np.unique(data.members[data.rows[0]]);sample=np.random.default_rng(cfg.split_seed).choice(physical,min(len(physical),20000),replace=False)
    library=np.log1p(data.totals[physical]);data.library_mean=float(library.mean());data.library_scale=max(float(library.std()),.1)
    records=[];summaries=[]
    for mode in ['library_log','log_counts']:
        d=copy.copy(data);d.expression_normalization=mode
        config=replace(cfg,expression_normalization=mode,standardize_library_size=True,max_epochs=a.epochs)
        if mode=='log_counts':
            x=expression_block(d.expression,sample,d.encoder_indices,d.totals,mode);d.gene_mean=x.mean(0);d.gene_scale=np.maximum(x.std(0),.1)
        d.rows=[train,valid,d.rows[2]];d.prepare_subset_device(np.r_[train,valid],cfg.device)
        np.savez(out/(mode+'_preprocessing.npz'),encoder_indices=d.encoder_indices,gene_mean=d.gene_mean,gene_scale=d.gene_scale,library_mean=d.library_mean,library_scale=d.library_scale)
        model=initialize(d,config).to(cfg.device);optimizer=make_optimizer(model,config);best=float('inf');stale=0
        for e in range(a.epochs+1):
            start=time.monotonic();tr=epoch(model,d,train,config,optimizer,e) if e else {};va=epoch(model,d,valid,config,None,e)
            r={'mode':mode,'epoch':e,'seconds':time.monotonic()-start,**{'train_'+k:v for k,v in tr.items()},**{'validation_'+k:v for k,v in va.items()}}
            records.append(r);pd.DataFrame(records).to_csv(out/'comparison.csv',index=False)
            print(mode,e,'train',round(tr.get('ccc',float('nan')),6),'val',round(va['ccc'],6),'gain',round(va['ccc_gain_over_mean'],4),flush=True)
            if va['total']<best-1e-7:
                best=va['total'];stale=0
                torch.save({'model_state':copy.deepcopy(model.state_dict()),'config':config.to_dict(),'epoch':e},out/(mode+'_best.pt'))
            else:
                stale+=1
                if stale>=config.patience:break
        saved=torch.load(out/(mode+'_best.pt'),map_location=cfg.device,weights_only=False);model.load_state_dict(saved['model_state'])
        summaries.append({'mode':mode,'best_epoch':saved['epoch'],'validation_diagnostics':state_diagnostics(model,d,valid,config)})
        (out/'summary.json').write_text(json.dumps({'train_centers':len(train),'validation_centers':len(valid),'test_set_used':False,
            'fixed_gene_indices':True,'library_mean':d.library_mean,'library_scale':d.library_scale,'models':summaries},indent=2)+'\n')
        del model,optimizer,d;torch.cuda.empty_cache()


if __name__=='__main__':main()
