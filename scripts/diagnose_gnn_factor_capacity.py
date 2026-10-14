"""Target-informed factor diagnostics: these losses are oracle fits, not GNN validation performance."""
import argparse,copy,json
from dataclasses import replace
from pathlib import Path
import numpy as np,pandas as pd,torch
from diagnose_gnn_dictionary import load_data,resident
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.model import model_from_config
from phenoniche.gnn.losses import ccc_loss,regularization
from phenoniche.training import set_seed


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=300);a=p.parse_args()
    source=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    payload=torch.load(source/'starting_checkpoint.pt',map_location='cpu',weights_only=False);cfg=replace(GNNConfig(**payload['config']),device='cuda:2')
    torch.set_num_threads(8);set_seed(cfg.seed);data=load_data(cfg);resident(data,Path(cfg.output_dir)/'cache',cfg.device)
    rows=np.load(source/'sampled_rows.npz');tr=rows['train'];va=rows['validation'];mask=data.feature_mask;scale=torch.as_tensor(data.feature_scale[mask],device=cfg.device)
    model=model_from_config(data,cfg).to(cfg.device);model.load_state_dict(payload['model_state']);model.eval()
    def initial_z(indices):
        results=[]
        with torch.no_grad():
            for start in range(0,len(indices),cfg.batch_size):results.append(data.forward(model,data.batch(indices[start:start+cfg.batch_size],cfg.device))['Z'])
        return torch.cat(results)
    ztr=initial_z(tr);zva=initial_z(va);h=model.dictionary.detach()[:,mask];del model
    ctr=data._resident['target'][torch.as_tensor(tr,device=cfg.device)][:,mask];cva=data._resident['target'][torch.as_tensor(va,device=cfg.device)][:,mask]
    # Free GPU memory used by expression and full-tissue targets; fit sampled matrices only.
    del data._resident;torch.cuda.empty_cache()
    rho=data.feature_nonzero_fraction[mask];feature_mask=np.ones(int(mask.sum()),bool)
    objective=lambda prediction,target:ccc_loss(prediction,target,scale,feature_mask,nonzero_fraction=rho,nonzero_weight_max=8)
    records=[];summary=[]
    def fit_z(z0,target,dictionary,steps):
        z=torch.nn.Parameter(z0.clone());optim=torch.optim.Adam([z],lr=.05)
        best=float('inf');best_z=None
        for k in range(steps+1):
            loss=objective(z@dictionary,target)
            if float(loss.detach())<best:best=float(loss.detach());best_z=z.detach().clone()
            if k==steps:break
            optim.zero_grad();loss.backward();optim.step()
            with torch.no_grad():z.clamp_(min=0)
        return best,best_z
    fixed_loss,fixed_z=fit_z(zva,cva,h,a.steps)
    initial=float(objective(zva@h,cva));print('GNN',initial,'fixed-H oracle',fixed_loss,flush=True)
    for rank in [32,64,128]:
        set_seed(cfg.seed);extra=rank-h.shape[0]
        if extra:
            picks=torch.as_tensor(np.random.default_rng(cfg.seed).choice(len(tr),extra,replace=False),device=cfg.device)
            added=ctr[picks].clone().clamp_min(1e-9);added/=added.sum(1,keepdim=True)
            initial_h=torch.cat((h,added));initial_z=torch.cat((ztr,torch.zeros(len(tr),extra,device=cfg.device)),1)
        else:initial_h=h.clone();initial_z=ztr.clone()
        d=initial_h/scale;d/=d.sum(1,keepdim=True)
        d=torch.nn.Parameter(d);z=torch.nn.Parameter(initial_z)
        optim=torch.optim.Adam([{'params':[d],'lr':.0002},{'params':[z],'lr':.05}]);best=float('inf');best_d=None;best_z=None
        for step in range(a.steps+1):
            dictionary=d*scale;dictionary=dictionary/dictionary.sum(1,keepdim=True).clamp_min(1e-12)
            loss=objective(z@dictionary,ctr);pi=z/z.sum(1,keepdim=True).clamp_min(1e-12)
            total=loss+cfg.lambda_reg*regularization({'H':dictionary,'P':pi})
            if float(total.detach())<best:best=float(total.detach());best_d=d.detach().clone();best_z=z.detach().clone()
            if step%50==0:
                records.append({'rank':rank,'step':step,'train_oracle_CCC':float(loss.detach())});pd.DataFrame(records).to_csv(out/'training_oracle.csv',index=False)
                print('rank',rank,'step',step,'train oracle',round(float(loss.detach()),6),flush=True)
            if step==a.steps:break
            optim.zero_grad();total.backward();optim.step()
            with torch.no_grad():
                z.clamp_(min=0);d.clamp_(min=1e-12);d.div_(d.sum(1,keepdim=True))
        learned_h=best_d*scale;learned_h/=learned_h.sum(1,keepdim=True)
        start_z=torch.cat((zva,torch.zeros(len(va),extra,device=cfg.device)),1) if extra else zva
        val_loss,latent=fit_z(start_z,cva,learned_h,a.steps)
        train_loss=float(objective(best_z@learned_h,ctr))
        summary.append({'rank':rank,'train_target_informed_loss':train_loss,'validation_target_informed_latent_fit_loss':val_loss})
        np.save(out/f'H_rank{rank}.npy',learned_h.cpu().numpy());np.save(out/f'validation_oracle_Z_rank{rank}.npy',latent.cpu().numpy())
        (out/'summary.json').write_text(json.dumps({'gnn_validation_CCC':initial,'fixed_H_validation_latent_oracle_CCC':fixed_loss,
            'train_centers':len(tr),'validation_centers':len(va),'important':'Validation CCC targets optimize free latent factors: oracle diagnostic, NOT independent prediction performance.',
            'test_set_used':False,'ranks':summary},indent=2)+'\n')
        print('rank',rank,'train',train_loss,'validation oracle',val_loss,flush=True)


if __name__=='__main__':main()
