"""Sampled GNN program-capacity comparison; no full-data training or test-set selection."""
import argparse,copy,json,time
from dataclasses import replace
from pathlib import Path
import numpy as np,pandas as pd,torch
from diagnose_gnn_dictionary import load_data,resident
from diagnose_gnn_feature_scope import state_diagnostics
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.training import initialize,epoch,make_optimizer


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=60);p.add_argument('--resume',action='store_true')
    p.add_argument('--hidden',type=int,default=128);p.add_argument('--initializations',nargs='+',choices=['rms_kmeans','log1p_active_kmeans'],default=['rms_kmeans','log1p_active_kmeans'])
    p.add_argument('--scope',choices=['all','common'],default='all');p.add_argument('--ranks',type=int,nargs='+',default=[32,128]);a=p.parse_args()
    source=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=a.resume)
    payload=torch.load(source/'starting_checkpoint.pt',map_location='cpu',weights_only=False);cfg=replace(GNNConfig(**payload['config']),device='cuda:2',hidden=a.hidden)
    torch.set_num_threads(8);data=load_data(cfg)
    if a.scope=='common':
        old=pd.read_csv('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/microenvironment_gnn/cache/features.csv')
        keys=set(zip(old.sender,old.receiver,old.lr_id))
        data.feature_mask=data.feature_mask & np.array([k in keys for k in zip(data.features.sender,data.features.receiver,data.features.lr_id)])
    np.save(out/'supervision_mask.npy',data.feature_mask)
    sampled=np.load(source/'sampled_rows.npz');train=sampled['train'];valid=sampled['validation'];data.rows=[train,valid,data.rows[2]]
    data.prepare_subset_device(np.r_[train,valid],cfg.device)
    records=pd.read_csv(out/'comparison.csv').to_dict('records') if a.resume else []
    for r in records:
        if 'initialization' not in r:r['initialization']='rms_kmeans'
    summaries=json.loads((out/'summary.json').read_text())['models'] if a.resume else []
    for s in summaries:
        if 'initialization' not in s:s['initialization']='rms_kmeans'
    for initialization,rank in [(i,r) for i in a.initializations for r in a.ranks]:
        if any(s['programs']==rank and s['initialization']==initialization for s in summaries):continue
        config=replace(cfg,programs=rank,max_epochs=a.epochs,dictionary_initialization=initialization)
        name=f'{initialization}_rank{rank}'
        model=initialize(data,config).to(cfg.device);optimizer=make_optimizer(model,config);best=float('inf');stale=0
        np.save(out/f'initial_H_{name}.npy',model.dictionary.detach().cpu().numpy())
        (out/f'initialization_{name}.json').write_text(json.dumps(model.initialization_audit,indent=2)+'\n')
        for e in range(a.epochs+1):
            start=time.monotonic();tr=epoch(model,data,train,config,optimizer,e) if e else {};va=epoch(model,data,valid,config,None,e)
            r={'initialization':initialization,'programs':rank,'epoch':e,'seconds':time.monotonic()-start,**{'train_'+k:v for k,v in tr.items()},**{'validation_'+k:v for k,v in va.items()}}
            records.append(r);pd.DataFrame(records).to_csv(out/'comparison.csv',index=False)
            print(name,e,'train',round(tr.get('ccc',float('nan')),6),'val',round(va['ccc'],6),'gain',round(va['ccc_gain_over_mean'],4),flush=True)
            if va['total']<best-1e-7:
                best=va['total'];stale=0
                torch.save({'model_state':copy.deepcopy(model.state_dict()),'config':config.to_dict(),'epoch':e},out/f'gnn_{name}_best.pt')
            else:
                stale+=1
                if stale>=config.patience:break
        saved=torch.load(out/f'gnn_{name}_best.pt',map_location=cfg.device,weights_only=False);model.load_state_dict(saved['model_state'])
        summaries.append({'programs':rank,'initialization':initialization,'best_epoch':saved['epoch'],'validation_diagnostics':state_diagnostics(model,data,valid,config)})
        (out/'summary.json').write_text(json.dumps({'train_centers':len(train),'validation_centers':len(valid),'test_set_used':False,'from_scratch':True,
            'hidden':a.hidden,'scope':a.scope,'effective_features':int(data.feature_mask.sum()),'fixed':'same expression, graph, rows, training-statistics, loss and optimizer; program count and initialization as reported','models':summaries},indent=2)+'\n')
        del model,optimizer;torch.cuda.empty_cache()


if __name__=='__main__':main()
