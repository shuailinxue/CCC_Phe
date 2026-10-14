"""Physical-opportunity label audit and sampled CCC-only supervision comparison."""
import argparse,copy,json,time,types
from dataclasses import replace
from pathlib import Path
import numpy as np,pandas as pd,torch
from diagnose_gnn_dictionary import load_data
from diagnose_gnn_feature_scope import state_diagnostics
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.model import model_from_config,physical_weights
from phenoniche.gnn.losses import ccc_loss
from phenoniche.gnn.training import epoch,make_optimizer
from phenoniche.training import set_seed


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=60);a=p.parse_args()
    source=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    payload=torch.load(source/'starting_checkpoint.pt',map_location='cpu',weights_only=False);cfg=replace(GNNConfig(**payload['config']),device='cuda:2')
    torch.set_num_threads(8);data=load_data(cfg);sampled=np.load(source/'sampled_rows.npz');train=sampled['train'];valid=sampled['validation'];centers=np.unique(np.r_[train,valid])
    data.prepare_subset_device(centers,cfg.device);lookup=np.full(len(data.cells),-1,np.int64);lookup[centers]=np.arange(len(centers))
    t=len(data.categories);send=np.array([data.categories.index(x) for x in data.features.sender]);receive=np.array([data.categories.index(x) for x in data.features.receiver]);columns=torch.as_tensor(send*t+receive,device=cfg.device)
    availability=torch.empty((len(centers),t*t),dtype=torch.bool,device=cfg.device)
    for start in range(0,len(centers),256):
        chosen=centers[start:start+256];local=np.asarray(data.members[chosen]);xy=torch.as_tensor(np.asarray(data.coordinates)[local].copy(),device=cfg.device,dtype=torch.float64)
        anchor=torch.as_tensor(np.asarray(data.weights)[chosen].copy(),device=cfg.device,dtype=torch.float64)
        weight=physical_weights(xy,anchor,cfg.sigma_um,torch.as_tensor(local,device=cfg.device))
        labels=torch.nn.functional.one_hot(torch.as_tensor(data.types[local],device=cfg.device),t).double()
        opportunity=labels.transpose(1,2)@weight@labels
        availability[start:start+len(chosen)]=opportunity.flatten(1)>0
    tr_index=torch.as_tensor(lookup[train],device=cfg.device);va_index=torch.as_tensor(lookup[valid],device=cfg.device)
    observed=availability[tr_index][:,columns];actual=torch.as_tensor(data.targets[train],device=cfg.device)
    if ((actual>1e-6)&~observed).any():raise AssertionError('Positive CCC without its physical sender/receiver opportunity')
    fraction=observed.float().mean(0);conditional_rho=((actual>0)&observed).sum(0)/observed.sum(0).clamp_min(1)
    conditional_scale=(actual.square().sum(0)/observed.sum(0).clamp_min(1)).sqrt().clamp_min(1e-6)
    ratio=((1-conditional_rho)/conditional_rho.clamp_min(1e-8)).sqrt().clamp(1,cfg.ccc_nonzero_weight_max)
    normalizer=1-conditional_rho+conditional_rho*ratio
    audit={}
    for name,indices in [('train',tr_index),('validation',va_index)]:
        truth=torch.as_tensor(data.targets[train if name=='train' else valid],device=cfg.device)[:,data.feature_mask]
        observed_here=availability[indices][:,columns][:,data.feature_mask]
        zeros=truth==0
        audit[name]={'opportunity_present_fraction':float(observed_here.float().mean()),'CCC_nonzero_fraction':float((truth>0).float().mean()),
            'fraction_zero_entries_with_absent_opportunity':float((zeros&~observed_here).sum()/zeros.sum()),
            'nonzero_given_physical_opportunity':float(((truth>0)&observed_here).sum()/observed_here.sum())}
    (out/'opportunity_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print('opportunity audit',audit,flush=True)
    np.savez(out/'training_loss_statistics.npz',opportunity_fraction=fraction.cpu().numpy(),conditional_nonzero=conditional_rho.cpu().numpy(),conditional_scale=conditional_scale.cpu().numpy())
    original_batch=data.batch;records=[];summaries=[]
    for name,gamma,lr_pool in [('balanced1',1.,False),('balanced1_lr',1.,True),('balanced025_lr',.25,True)]:
        set_seed(cfg.seed);config=replace(cfg,program_pair_pooling=lr_pool,lr_pair_pooling=lr_pool)
        def batch_with_weights(self,rows,device='cpu',include_target=True):
            batch=original_batch(rows,device,include_target)
            if include_target:
                indices=torch.as_tensor(lookup[np.asarray(rows)],device=cfg.device)
                available=availability[indices][:,columns]
                active=1/fraction.clamp_min(1e-6)
                absent=gamma*normalizer/(1-fraction).clamp_min(1e-6)
                total=1+gamma*(fraction<1).float()
                batch['ccc_entry_weights']=torch.where(available,active,absent)/total
            return batch
        data.batch=types.MethodType(batch_with_weights,data)
        def objective(prediction,target,scale,mask,batch):
            return ccc_loss(prediction,target,conditional_scale,mask,nonzero_fraction=conditional_rho,nonzero_weight_max=cfg.ccc_nonzero_weight_max,entry_weights=batch['ccc_entry_weights'])
        objective.uses_batch=True
        model=model_from_config(data,config);state=model.state_dict();state.update(payload['model_state']);model.load_state_dict(state);model.to(cfg.device)
        optimizer=make_optimizer(model,config);best=float('inf');stale=0
        for e in range(a.epochs+1):
            set_seed(cfg.seed+e);start=time.monotonic();tr=epoch(model,data,train,config,optimizer,e,objective) if e else {};va=epoch(model,data,valid,cfg,None,e)
            r={'arm':name,'epoch':e,'seconds':time.monotonic()-start,**{'train_conditional_'+k:v for k,v in tr.items()},**{'validation_original_'+k:v for k,v in va.items()}}
            records.append(r);pd.DataFrame(records).to_csv(out/'comparison.csv',index=False)
            print(name,e,'conditional train',round(tr.get('ccc',float('nan')),6),'val original',round(va['ccc'],6),'native RMSE',round(va['ccc_relative_RMSE'],4),flush=True)
            if va['total']<best-1e-7:
                best=va['total'];stale=0
                torch.save({'model_state':copy.deepcopy(model.state_dict()),'config':config.to_dict(),'epoch':e,'supervision':'availability-balanced conditional RMS','structural_zero_weight':gamma},out/(name+'_best.pt'))
            else:
                stale+=1
                if stale>=cfg.patience:break
        saved=torch.load(out/(name+'_best.pt'),map_location=cfg.device,weights_only=False);model.load_state_dict(saved['model_state'])
        summaries.append({'arm':name,'best_epoch':saved['epoch'],'validation_diagnostics':state_diagnostics(model,data,valid,config)})
        (out/'summary.json').write_text(json.dumps({'train_centers':len(train),'validation_centers':len(valid),'test_set_used':False,'opportunity_audit':audit,
            'unchanged':'native CCC values, original reconstruction validation metric and dictionary, expression-only inputs, CCC+regularization only','models':summaries},indent=2)+'\n')
        del model,optimizer;torch.cuda.empty_cache()
    data.batch=original_batch


if __name__=='__main__':main()
