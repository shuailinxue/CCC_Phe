"""Type-conditioned expression dictionary on frozen CCC programs."""
from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from ..gnn.model import physical_weights
from ..data import resolve_device
from ..training import set_seed


def weighted_expression_targets(adata,data,base_gene_count,batch_size=256):
    raw=np.asarray(adata.X,np.float32)
    if len(raw)!=len(data.cells) or not np.array_equal(adata.obs.cell_id.astype(str),data.cells.cell_id.astype(str)):
        raise ValueError('Expression/cell order mismatch')
    if not np.allclose(adata.obsm['spatial'],data.coordinates,atol=1e-4,rtol=0):raise ValueError('Expression coordinate mismatch')
    # Original-panel library size: appended response genes cannot rescale LR
    # expression or leak into the already trained CCC encoder through totals.
    library=np.maximum(raw[:,:base_gene_count].sum(1),1)
    normalized=raw*(10000/library[:,None]);n=len(raw);t=len(data.categories)
    target=np.zeros((n,t,raw.shape[1]),np.float32);present=np.zeros((n,t),bool);normalization=np.zeros((n,t),np.float32)
    for start in range(0,n,batch_size):
        stop=min(start+batch_size,n);local=np.asarray(data.members[start:stop])
        pair=physical_weights(torch.from_numpy(data.coordinates[local]),torch.from_numpy(np.asarray(data.weights[start:stop]).copy()),20.,torch.from_numpy(local))
        degree=pair.sum(2).numpy()
        for k in range(t):
            weights=degree*(data.types[local]==k);mass=weights.sum(1);valid=mass>1e-12
            mean=(normalized[local]*weights[...,None]).sum(1)/np.maximum(mass[:,None],1e-12)
            target[start:stop,k]=np.log1p(mean);present[start:stop,k]=valid
            normalization[start:stop,k]=(weights*(10000/library[local])).sum(1)/np.maximum(mass,1e-12)
    if not np.isfinite(target).all():raise ValueError('Nonfinite spatial expression target')
    return target,present,normalization


def response_split_rows(data):
    """Extra purge: target cells' response-generating neighbors cannot cross folds."""
    owner=np.full(len(data.cells),-1,np.int8)
    for fold,rows in enumerate(data.rows):
        used=np.unique(data.members[rows])
        if (owner[used]>=0).any():raise ValueError('Physical cells shared across source CCC folds')
        owner[used]=fold
    result=[]
    for fold,rows in enumerate(data.rows):
        kept=[]
        for start in range(0,len(rows),256):
            centers=rows[start:start+256]
            # First hop contributes expression; second hop generates responses.
            dependency=data.members[np.asarray(data.members[centers])]
            valid=(owner[dependency]==fold).all(axis=(1,2))
            kept.extend(centers[valid])
        result.append(np.asarray(kept,dtype=np.int64))
    if any(len(rows)<2 for rows in result):raise ValueError('Response two-hop boundary purge leaves too few centers; enlarge spatial blocks/samples')
    return result


class FrozenProgramExpressionDecoder(nn.Module):
    """One decoder, conditioned on type. No GNN/CCC parameter is registered."""
    def __init__(self,program_mean,program_scale,baseline,feature_scale,feature_mask,detach_programs=True):
        super().__init__()
        self.detach_programs=detach_programs
        for name,value in [('program_mean',program_mean),('program_scale',program_scale),('baseline',baseline),
                           ('feature_scale',feature_scale),('feature_mask',feature_mask)]:
            self.register_buffer(name,torch.as_tensor(value))
        b=np.maximum(np.asarray(baseline),1e-6)
        self.register_buffer('baseline_logit',torch.as_tensor(b+np.log(-np.expm1(-b))))
        self.dictionary=nn.Parameter(torch.zeros(baseline.shape[0],len(program_mean),baseline.shape[1]))
    def forward(self,z,normalization):
        # detach is deliberate: expression loss cannot change CCC-defined Z.
        u=(torch.log1p(z.detach() if self.detach_programs else z)-self.program_mean)/self.program_scale
        signal=torch.einsum('np,tpg->ntg',u,self.dictionary)
        return torch.log1p(normalization[:,:,None]*F.softplus(self.baseline_logit[None]+signal))
    def loss(self,z,target,present,normalization):
        mask=present[:,:,None]&self.feature_mask[None]
        if not mask.any():return self.dictionary.sum()*0,0
        error=(self(z,normalization)-target)/self.feature_scale[None]
        return F.huber_loss(error[mask],torch.zeros_like(error[mask]),reduction='mean'),int(mask.sum())


def preprocessing(z,target,present,normalization,train):
    baseline=np.zeros(target.shape[1:],np.float32);scale=np.ones_like(baseline);mask=np.zeros_like(baseline,bool)
    for t in range(target.shape[1]):
        rows=train[present[train,t]]
        if len(rows):
            y=target[rows,t];factor=normalization[rows,t,None]
            baseline[t]=(factor*np.expm1(y)).sum(0)/np.maximum((factor*factor).sum(),1e-12)
            residual=y-np.log1p(factor*baseline[t]);var=residual.var(0);scale[t]=np.maximum(np.sqrt(var),.15)
            mask[t]=(var>1e-8)&((y>0).mean(0)>=.005)
    if not mask.any():raise ValueError('No variable expression features in training blocks')
    u=np.log1p(np.asarray(z[train]));return u.mean(0),np.maximum(u.std(0),.1),baseline,scale,mask


def fit_expression(z,target,present,normalization,rows,config,output):
    output=Path(output);output.mkdir(exist_ok=True,parents=True);device=resolve_device(config.device)
    values=preprocessing(z,target,present,normalization,rows[0]);model=FrozenProgramExpressionDecoder(*values).to(device)
    tensors=[torch.as_tensor(np.asarray(a).copy(),device=device) for a in (z,target,present,normalization)]
    checkpoint=output/'checkpoint.pt'
    if checkpoint.exists():
        payload=torch.load(checkpoint,map_location=device,weights_only=False)
        if payload['config']!=config.to_dict():raise ValueError('Expression checkpoint configuration mismatch')
        for key,value in zip(('program_mean','program_scale','baseline','feature_scale','feature_mask'),values):
            if not np.array_equal(payload[key],value):raise ValueError('Expression preprocessing changed: '+key)
        model.load_state_dict(payload['model_state'])
    else:
        set_seed(config.seed);optimizer=torch.optim.Adam(model.parameters(),lr=config.lr)
        history=[];best=float('inf');state=None;stale=0
        for epoch in range(config.expression_epochs):
            losses={};start_rows=np.random.default_rng(config.seed+epoch).permutation(rows[0])
            for name,indices in [('train',start_rows),('validation',rows[1])]:
                total=0.;entries=0;model.train(name=='train')
                with torch.set_grad_enabled(name=='train'):
                    for start in range(0,len(indices),config.batch_size):
                        batch=indices[start:start+config.batch_size]
                        loss,count=model.loss(*(a[batch] for a in tensors))
                        if name=='train':
                            objective=loss+config.lambda_dictionary*model.dictionary.abs().mean()
                            optimizer.zero_grad(set_to_none=True);objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                        total+=float(loss.detach())*count;entries+=count
                losses[name]=total/max(entries,1)
            history.append({'epoch':epoch+1,**losses});pd.DataFrame(history).to_csv(output/'training.csv',index=False)
            print(f'Expression epoch {epoch+1}: train={losses["train"]:.5f}, validation={losses["validation"]:.5f}',flush=True)
            if losses['validation']<best-1e-7:
                best=losses['validation'];stale=0;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
                torch.save({'model_state':state,'best_epoch':best_epoch,'validation_loss':best},output/'training_best.pt')
            else:
                stale+=1
                if stale>=config.patience:break
        model.load_state_dict(state)
        payload={'model_state':state,'best_epoch':best_epoch,'validation_loss':best,'config':config.to_dict(),
                 **dict(zip(('program_mean','program_scale','baseline','feature_scale','feature_mask'),values))}
        torch.save(payload,checkpoint)
    model.eval();prediction=np.empty_like(target)
    with torch.no_grad():
        for start in range(0,len(z),config.batch_size):
            stop=min(start+config.batch_size,len(z));prediction[start:stop]=model(tensors[0][start:stop],tensors[3][start:stop]).cpu().numpy()
    baseline_prediction=np.log1p(normalization[:,:,None]*values[2][None])
    np.save(output/'prediction.npy',prediction);np.save(output/'baseline_prediction.npy',baseline_prediction);np.save(output/'baseline.npy',values[2]);np.save(output/'dictionary.npy',model.dictionary.detach().cpu().numpy())
    np.savez(output/'preprocessing.npz',**dict(zip(('program_mean','program_scale','baseline','feature_scale','feature_mask'),values)))
    return prediction,baseline_prediction,values[4],payload['best_epoch']
