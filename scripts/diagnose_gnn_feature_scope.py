"""Sampled feature-scope ablation with fixed inputs/network/checkpoint and validation rows."""
import argparse,copy,json
from dataclasses import replace
from functools import partial
from pathlib import Path
import numpy as np,pandas as pd,torch
from diagnose_gnn_dictionary import load_data,resident
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.model import model_from_config
from phenoniche.gnn.training import epoch,make_optimizer
from phenoniche.gnn.losses import ccc_loss
from phenoniche.training import set_seed


def state_diagnostics(model,data,rows,config):
    ps=[];zs=[];predictions=[];targets=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(rows),config.batch_size):
            batch=data.batch(rows[start:start+config.batch_size],config.device);r=data.forward(model,batch)
            ps.append(r['P'].cpu().numpy());zs.append(r['Z'].cpu().numpy())
            predictions.append(r['reconstruction'].cpu().numpy());targets.append(batch['target'].cpu().numpy())
    p=np.concatenate(ps);z=np.concatenate(zs);pred=np.concatenate(predictions);target=np.concatenate(targets)
    def loss(estimate):return float(ccc_loss(torch.from_numpy(estimate),torch.from_numpy(target),data.feature_scale,data.feature_mask,
        nonzero_fraction=data.feature_nonzero_fraction,nonzero_weight_max=8))
    rng=np.random.default_rng(614)
    sst=((target-target.mean(0))**2).sum(0);sse=((target-pred)**2).sum(0);valid=data.feature_mask&(sst>1e-10)
    r2=1-sse[valid]/sst[valid]
    return {'mean_max_program_fraction':float(p.max(1).mean()),'effective_mean_programs':float(np.exp(-(p.mean(0)*np.log(np.maximum(p.mean(0),1e-12))).sum())),
        'mean_program_sd':float(p.std(0).mean()),'intensity_cv':float(z.sum(1).std()/z.sum(1).mean()),
        'CCC':loss(pred),'shuffled_prediction_CCC':loss(pred[rng.permutation(len(pred))]),
        'constant_prediction_CCC':loss(np.broadcast_to(pred.mean(0),pred.shape).copy()),'median_feature_R2':float(np.median(r2)),
        'aggregate_R2':float(1-sse[valid].sum()/sst[valid].sum())}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--epochs',type=int,default=20);args=parser.parse_args()
    root=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    payload=torch.load(root/'starting_checkpoint.pt',map_location='cpu',weights_only=False);cfg=replace(GNNConfig(**payload['config']),device='cuda:2')
    torch.set_num_threads(8);data=load_data(cfg);resident(data,Path(cfg.output_dir)/'cache',cfg.device)
    sampled=np.load(root/'sampled_rows.npz');train=sampled['train'];valid=sampled['validation']
    old=pd.read_csv('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/microenvironment_gnn/cache/features.csv')
    old_keys=set(zip(old.sender,old.receiver,old.lr_id));common=np.array([key in old_keys for key in zip(data.features.sender,data.features.receiver,data.features.lr_id)])&data.feature_mask
    common_data=copy.copy(data);common_data.feature_mask=common
    arms=[('expanded_weight8',data,8),('expanded_weight1',data,1),('common_weight8',common_data,8)]
    history=[];summaries=[]
    for name,training_data,cap in arms:
        config=replace(cfg,ccc_nonzero_weight_max=cap)
        model=model_from_config(data,config);model.load_state_dict(payload['model_state']);model.to(cfg.device);optimizer=make_optimizer(model,config)
        best=float('inf');best_record=None
        for e in range(args.epochs+1):
            set_seed(cfg.seed+e)
            tr=epoch(model,training_data,train,config,optimizer,e) if e else {}
            # Report both target scopes with exactly the same weighted metric.
            va=epoch(model,data,valid,cfg,None,e);co=epoch(model,common_data,valid,cfg,None,e)
            r={'arm':name,'epoch':e,**{'train_'+k:v for k,v in tr.items()},**{'full_'+k:v for k,v in va.items()},**{'common_'+k:v for k,v in co.items()}}
            history.append(r);pd.DataFrame(history).to_csv(out/'comparison.csv',index=False)
            print(name,e,'full',round(va['ccc'],6),'common',round(co['ccc'],6),'common pos',round(co['ccc_positive_huber'],6),flush=True)
            score=(co if name.startswith('common') else va)['total']
            if score<best:
                best=score;best_record=r
                torch.save({'model_state':copy.deepcopy(model.state_dict()),'config':config.to_dict(),'epoch':e},out/(name+'_best.pt'))
        best_payload=torch.load(out/(name+'_best.pt'),map_location=cfg.device,weights_only=False);model.load_state_dict(best_payload['model_state'])
        summaries.append({'best':best_record,'full_diagnostics':state_diagnostics(model,data,valid,cfg),
            'common_diagnostics':state_diagnostics(model,common_data,valid,cfg)})
        (out/'summary.json').write_text(json.dumps({'source_epoch':payload['epoch'],'train_centers':len(train),'validation_centers':len(valid),
            'common_features':int(common.sum()),'expanded_features':int(data.feature_mask.sum()),'test_set_used':False,'arms':summaries},indent=2)+'\n')
        del model,optimizer;torch.cuda.empty_cache()


if __name__=='__main__':main()
