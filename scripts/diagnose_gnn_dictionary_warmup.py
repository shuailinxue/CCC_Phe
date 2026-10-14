"""CCC-only factor dictionary warmup, then expression/spatial GNN; no latent-label loss."""
import argparse,copy,json,math,time
from dataclasses import replace
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
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--factors',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=60);p.add_argument('--activity',choices=['separated','independent'],default='separated');a=p.parse_args()
    source=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    payload=torch.load(source/'starting_checkpoint.pt',map_location='cpu',weights_only=False);cfg=replace(GNNConfig(**payload['config']),device='cuda:2')
    torch.set_num_threads(8);data=load_data(cfg);sampled=np.load(source/'sampled_rows.npz');train=sampled['train'];valid=sampled['validation']
    data.prepare_subset_device(np.r_[train,valid],cfg.device);old=model_from_config(data,cfg).to(cfg.device);old.load_state_dict(payload['model_state']);old.eval()
    with torch.no_grad():base_z=torch.cat([data.forward(old,data.batch(train[s:s+cfg.batch_size],cfg.device))['Z'] for s in range(0,len(train),cfg.batch_size)])
    target=data.targets[train];target_tensor=torch.as_tensor(target,device=cfg.device);del old
    records=[];summaries=[]
    for rank in [32,128]:
        set_seed(cfg.seed);config=replace(cfg,programs=rank,max_epochs=a.epochs,activity_parameterization=a.activity)
        effective=np.load(Path(a.factors)/f'H_rank{rank}.npy');h=np.zeros((rank,len(data.features)),np.float32);h[:,data.feature_mask]=effective
        if not np.allclose(h.sum(1),1.,atol=1e-5):raise ValueError('Factor dictionary order/scale mismatch')
        ht=torch.as_tensor(h,device=cfg.device)
        z=torch.nn.Parameter(torch.cat((base_z,torch.zeros(len(train),rank-32,device=cfg.device)),1));optim=torch.optim.Adam([z],lr=.05)
        for step in range(301):
            loss=ccc_loss(z@ht,target_tensor,data.feature_scale,data.feature_mask,nonzero_fraction=data.feature_nonzero_fraction,nonzero_weight_max=8)
            if step==300:break
            optim.zero_grad();loss.backward();optim.step()
            with torch.no_grad():z.clamp_(min=0)
        # Only training CCC fits initialization. Validation latent oracle files
        # are never read; validation remains expression-only prediction.
        prior=z.detach().mean(0).clamp_min(1e-6);prior/=prior.sum()
        activity=max(float(z.detach().sum(1).median()),1e-4)
        model=model_from_config(data,config).to(cfg.device);state=model.state_dict()
        for key,value in payload['model_state'].items():
            if key not in ('raw_dictionary','raw_composition','composition_head.weight','composition_head.bias') and key in state and state[key].shape==value.shape:state[key]=value
        model.load_state_dict(state)
        with torch.no_grad():
            model.raw_dictionary.copy_(torch.log(torch.expm1(ht.clamp_min(1e-7))))
            model.composition_head.weight.zero_();model.composition_head.bias.copy_(prior.log())
            model.activity_head.weight.zero_();model.activity_head.bias.fill_(activity if activity>20 else math.log(math.expm1(activity)))
            if config.activity_parameterization=='independent':
                values=(activity*prior).clamp_min(1e-6)
                model.program_head.weight.zero_();model.program_head.bias.copy_(torch.where(values>20,values,torch.log(torch.expm1(values.clamp_max(20)))))
        np.save(out/f'initial_H_rank{rank}.npy',model.dictionary.detach().cpu().numpy());np.save(out/f'training_factor_prior_rank{rank}.npy',prior.cpu().numpy())
        model.raw_dictionary.requires_grad_(False);optimizer=make_optimizer(model,config);best=float('inf');stale=0
        for e in range(a.epochs+1):
            if e==11:model.raw_dictionary.requires_grad_(True)
            set_seed(cfg.seed+e);start=time.monotonic();tr=epoch(model,data,train,config,optimizer,e) if e else {};va=epoch(model,data,valid,config,None,e)
            r={'programs':rank,'epoch':e,'dictionary_frozen':not model.raw_dictionary.requires_grad,'seconds':time.monotonic()-start,**{'train_'+k:v for k,v in tr.items()},**{'validation_'+k:v for k,v in va.items()}}
            records.append(r);pd.DataFrame(records).to_csv(out/'comparison.csv',index=False)
            print('factor-init',rank,e,'train',round(tr.get('ccc',float('nan')),6),'val',round(va['ccc'],6),'gain',round(va['ccc_gain_over_mean'],4),flush=True)
            if va['total']<best-1e-7:
                best=va['total'];stale=0
                torch.save({'model_state':copy.deepcopy(model.state_dict()),'config':config.to_dict(),'epoch':e},out/f'gnn_rank{rank}_best.pt')
            else:
                stale+=1
                if e>18 and stale>=config.patience:break
        saved=torch.load(out/f'gnn_rank{rank}_best.pt',map_location=cfg.device,weights_only=False);model.load_state_dict(saved['model_state'])
        summaries.append({'programs':rank,'best_epoch':saved['epoch'],'validation_diagnostics':state_diagnostics(model,data,valid,config)})
        (out/'summary.json').write_text(json.dumps({'train_centers':len(train),'validation_centers':len(valid),'test_set_used':False,
            'activity_parameterization':a.activity,'factor_init_training_targets_only':True,'GNN_loss':'CCC + regularization; no latent-factor matching loss','models':summaries},indent=2)+'\n')
        del model,optimizer,z,optim;torch.cuda.empty_cache()


if __name__=='__main__':main()
