"""Simplex training with fixed observed mass, staged geometry and validation checkpoints."""
import copy,json,time
import numpy as np
import pandas as pd
import torch
from .tensor_model import DeepTensorCCC,sparse_tensor
from .ccc_tensor import entry_holdout
from .losses import geometry_loss,group_sparsity
from .diagnostics import latent_audit
from .data import json_write

def split_mask(rows,cols,validation=False):
    hold=entry_holdout(rows,cols)
    partition=((np.asarray(rows,dtype=np.int64)*31+np.asarray(cols,dtype=np.int64)*17)%2)==0
    return hold & (partition if validation else ~partition)

def observed_mass(train):
    return (np.asarray(train.sum(1)).ravel()/.9).astype(np.float32)

def normalized_input(a):
    mass=observed_mass(a)
    denom=np.where(mass>0,mass,1.)
    return a.multiply((a.shape[1]/denom)[:,None]).tocsr(),mass

def encode_rows(model,train,ids,device,batch=1024):
    values=[]
    with torch.no_grad():
        for start in range(0,len(ids),batch):
            inp,_=normalized_input(train[ids[start:start+batch]])
            values.append(model(sparse_tensor(inp,device)).cpu().numpy())
    return np.concatenate(values)

def validation_loss(model,x,train,ids,device,chunk):
    z=encode_rows(model,train,ids,device);h=model.H.detach().cpu().numpy();mass=observed_mass(train[ids]);total=0.;count=0;f=x.shape[1]
    for lo in range(0,f,chunk):
        hi=min(lo+chunk,f);target=x[ids,lo:hi].toarray()/np.where(mass>0,mass,1.)[:,None]
        diff=np.abs(((z@h[:,lo:hi])*(mass>0)[:,None]-target)*f);mask=split_mask(ids[:,None],np.arange(lo,hi)[None],True)
        d=diff[mask];total+=np.where(d<1,.5*d*d,d-.5).sum(dtype=np.float64);count+=len(d)
    return float(total/max(count,1))

def train_model(x,train,context,g0,config,seed,prior,folder):
    from .experiment import seed_all
    tag='tensor' if prior else 'no_prior';prefix=folder/f'seed_{seed}_{tag}';seed_all(seed);device=config['device']
    C,_,L=context['shape'];modes=context['features'][['sender_id','receiver_id','lr_index']].to_numpy()
    model=DeepTensorCCC(int(C),int(L),modes,K=config['K']).to(device)
    if PathExists(str(prefix)+'_training.json'):
        model.load_state_dict(torch.load(prefix.with_suffix('.pt'),map_location=device,weights_only=True))
        return model,np.load(str(prefix)+'_Z.npy'),np.load(str(prefix)+'_H.npy')
    opt=torch.optim.Adam(model.parameters(),lr=config['lr']);rng=np.random.default_rng(seed);n,f=train.shape
    mass=observed_mass(train);degree=np.diff(g0.indptr);edge_mean=float(g0.sum()/n)
    ids=np.sort(np.random.default_rng(456).choice(n,min(1024,n),replace=False))
    audit_ids=np.sort(np.random.default_rng(789).choice(n,min(4096,n),replace=False))
    audit_neighbor=sample_neighbors(g0,audit_ids,np.random.default_rng(321))[0]
    audit_random=np.random.default_rng(654).integers(n,size=len(audit_ids))
    history=[];best=float('inf');best_state=None;best_epoch=0;stale=0
    batch=max(8,min(config['batch_size'],int(8_000_000/max(f,1))))
    warm=config['warmup_epochs'];maximum=warm+config['graph_epochs']
    warm_path=folder/f'seed_{seed}_warmup.pt';start_epoch=0
    if prior and warm_path.exists():
        state=torch.load(warm_path,map_location=device,weights_only=True)
        if state['config']!=config:raise ValueError('Warm-up configuration mismatch')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        history=state['history'];best=state['best'];best_epoch=state['best_epoch'];best_state=state['best_state']
        rng.bit_generator.state=state['rng'];start_epoch=warm
    for epoch in range(start_epoch,maximum):
        start_time=time.time();model.train();order=rng.permutation(n);totals=np.zeros(5);steps=0
        graph_weight=config['lambda_graph']*min(1.,max(0.,(epoch-warm+1)/config['graph_ramp_epochs'])) if prior else 0.
        for start in range(0,n,batch):
            rows=order[start:start+batch];inp,_=normalized_input(train[rows]);z=model(sparse_tensor(inp,device));h=model.H
            target_rows=x[rows];rec=z.sum()*0;denominator=0
            for lo in range(0,f,config['feature_chunk']):
                hi=min(lo+config['feature_chunk'],f)
                target=torch.as_tensor(target_rows[:,lo:hi].toarray()/np.where(mass[rows]>0,mass[rows],1.)[:,None]*f,device=device)
                mask=~entry_holdout(rows[:,None],np.arange(lo,hi)[None]);denominator+=int(mask.sum())
                error=torch.nn.functional.smooth_l1_loss((z@h[:,lo:hi])*torch.as_tensor((mass[rows]>0)[:,None],device=device)*f,target,reduction='none')
                rec+=(error*torch.as_tensor(mask,device=device)).sum()
            rec/=max(denominator,1);graph=z.sum()*0
            if graph_weight:
                neighbor,weight=sample_neighbors(g0,rows,rng)
                other,_=normalized_input(train[neighbor]);zn=model(sparse_tensor(other,device))
                # Degree importance correction estimates the mean weighted G0 edge loss.
                weight=weight*degree[rows]/max(edge_mean,1e-12)
                graph=geometry_loss(z,zn,torch.as_tensor(weight,device=device))
            entropy=-(h*torch.log(h.clamp_min(1e-30))).sum(1).mean()/max(np.log(f),1.)
            group=group_sparsity(h,model.s*model.C+model.r,model.C**2)/config['K']
            loss=rec+graph_weight*graph+config['lambda_sparse']*entropy+config['lambda_group']*group
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite simplex loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
            totals+=np.array([v.detach().item() for v in [loss,rec,graph,entropy,group]]);steps+=1
        model.eval();val=validation_loss(model,x,train,ids,device,config['feature_chunk'])
        za=encode_rows(model,train,audit_ids,device);zn=encode_rows(model,train,audit_neighbor,device);zr=encode_rows(model,train,audit_random,device)
        record=dict(epoch=epoch+1,stage='warmup' if epoch<warm else 'geometry',graph_weight=graph_weight,validation_reconstruction=val,
                    neighbor_distance=float(((za-zn)**2).sum(1).mean()),random_distance=float(((za-zr)**2).sum(1).mean()),
                    **dict(zip(['total','reconstruction','graph','sparse','group'],(totals/steps).tolist())),**latent_audit(za))
        history.append(record);pd.DataFrame(history).to_csv(str(prefix)+'_loss.csv',index=False)
        if val<best-config['early_stop_min_delta']:
            best=val;best_epoch=epoch+1;best_state=copy.deepcopy(model.state_dict());stale=0
            torch.save(best_state,str(prefix)+'_best.pt')
        elif epoch>=warm:stale+=1
        if not prior and epoch+1==warm:
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'history':history,'best':best,
                        'best_epoch':best_epoch,'best_state':best_state,'rng':rng.bit_generator.state,'config':config},warm_path)
        print(folder.name,tag,seed,'epoch',epoch+1,'val',round(val,6),'graph_weight',round(graph_weight,5),'seconds',round(time.time()-start_time),flush=True)
        if epoch>=warm+min(config['graph_ramp_epochs'],config['graph_epochs'])-1 and stale>=config['patience']:break
    model.load_state_dict(best_state);model.eval();z=encode_rows(model,train,np.arange(n),device);h=model.H.detach().cpu().numpy()
    np.save(str(prefix)+'_Z.npy',z);np.save(str(prefix)+'_H.npy',h);torch.save(model.state_dict(),prefix.with_suffix('.pt'))
    json_write(str(prefix)+'_scale.json',{'scale':1.,'convention':'fixed observed row mass; H and Z sum to one'})
    json_write(str(prefix)+'_training.json',{'best_epoch':best_epoch,'epochs_run':len(history),'best_validation':best,'lowrank_enabled':False,'prior':prior,'warmup_reused':bool(start_epoch),'best_graph_weight':history[best_epoch-1]['graph_weight']})
    return model,z,h

def sample_neighbors(graph,rows,rng):
    degree=np.diff(graph.indptr)[rows];valid=degree>0
    index=graph.indptr[rows]+(rng.random(len(rows))*np.maximum(degree,1)).astype(int)
    neighbors=rows.copy();weights=np.zeros(len(rows),np.float32)
    neighbors[valid]=graph.indices[index[valid]];weights[valid]=graph.data[index[valid]]
    return neighbors,weights

def PathExists(path):
    from pathlib import Path
    return Path(path).exists()
