"""Fixed full-data experiment. No phenotype or test-driven parameter selection."""
from pathlib import Path
import os,json,time,random
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import NMF
from .data import ROOT,json_write
from .ccc_tensor import prepare,training_only,entry_holdout
from .tensor_model import DeepTensorCCC,sparse_tensor
from .losses import nonconvex_lowrank,geometry_loss,group_sparsity
from .graph import knn_graph,fuse_graphs
from .clustering import leiden
from .evaluation import spatial_coherence,stability_tables,program_edges

DEFAULT={ 'seed':40700,'seeds':[40700,40701,40702,40703,40704], 'K':16,'Kc':8,'epochs':5,'batch_size':256,
 'lr':.001,'lambda_lowrank':.001,'lambda_graph':.01,'lambda_sparse':.001,'lambda_group':.001,
 'epsilon':.001,'knn':15,'resolution':1.,'alpha':.2,'alphas':[0.,.1,.2,.3],
 'neighbors':30,'sigma':20.,'ccc_batch':64,'ccc_store_rows':1024,'feature_chunk':2048,
 'device':'cuda:1' if torch.cuda.is_available() and torch.cuda.device_count()>1 else ('cuda:0' if torch.cuda.is_available() else 'cpu')}

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def randomized_pca(x,k=16,seed=40700):
    """Centered randomized PCA, streaming sparse products. No dense N×F materialization."""
    n,f=x.shape;q=min(k+8,f,n-1);rng=np.random.default_rng(seed)
    mean=np.asarray(x.mean(0)).ravel();omega=rng.normal(size=(f,q)).astype(np.float32)/q**.5
    y=np.asarray(x@omega)-mean@omega
    qmat=np.linalg.qr(y,mode='reduced')[0];del y,omega
    b=np.asarray(x.T@qmat).T-qmat.sum(0)[:,None]*mean[None]
    _,_,vt=np.linalg.svd(b,full_matrices=False);v=vt[:min(k,len(vt))]
    z=np.asarray(x@v.T)-mean@v.T
    return z.astype(np.float32),v.astype(np.float32),mean.astype(np.float32)

def heldout_error(x,train,encode,decode,seed=123,n_rows=1024,chunk=2048):
    rng=np.random.default_rng(seed);ids=np.sort(rng.choice(x.shape[0],min(n_rows,x.shape[0]),replace=False))
    z=encode(train[ids]);se=ae=hub=0.;count=0
    for f in range(0,x.shape[1],chunk):
        stop=min(f+chunk,x.shape[1]);target=x[ids,f:stop].toarray()
        pred=decode(z,f,stop);mask=entry_holdout(ids[:,None],np.arange(f,stop)[None])
        diff=np.abs(pred-target)[mask];se+=float(np.square(diff,dtype=np.float64).sum());ae+=float(diff.sum(dtype=np.float64))
        hub+=float(np.where(diff<1,.5*diff**2,diff-.5).sum(dtype=np.float64));count+=len(diff)
    return {'heldout_MSE':se/count,'heldout_MAE':ae/count,'heldout_Huber':hub/count,'heldout_entries':count}

def train_model(x,train,context,g0,config,seed,prior,folder):
    tag='tensor' if prior else 'no_prior';prefix=folder/f'seed_{seed}_{tag}'
    modes=context['features'][['sender_id','receiver_id','lr_index']].to_numpy()
    C,_,L=context['shape'];seed_all(seed);device=config['device']
    model=DeepTensorCCC(int(C),int(L),modes,K=config['K']).to(device)
    if prefix.with_suffix('.pt').exists():
        model.load_state_dict(torch.load(prefix.with_suffix('.pt'),map_location=device,weights_only=True))
        return model,np.load(str(prefix)+'_Z.npy'),np.load(str(prefix)+'_H.npy')
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    # One common training-only RMS preserves relative edge magnitudes.
    scale=np.sqrt(float(np.square(train.data,dtype=np.float64).sum())/(train.shape[0]*train.shape[1]))
    scale=max(scale,1e-8)
    row_nnz=np.diff(g0.indptr);rng=np.random.default_rng(seed);history=[];n,f=train.shape
    batch=max(8,min(config['batch_size'],int(8_000_000/max(f,1))))
    for epoch in range(config['epochs']):
        order=rng.permutation(n);totals=np.zeros(6);steps=0;start_time=time.time()
        for start in range(0,n,batch):
            ids=order[start:start+batch];b=len(ids)
            # Neighbor endpoints sampled from fixed CCC-only G0, including out-of-batch endpoints.
            pick=g0.indptr[ids]+(rng.random(b)*np.maximum(row_nnz[ids],1)).astype(int)
            valid=row_nnz[ids]>0;neighbor=ids.copy();weight=np.zeros(b,np.float32)
            neighbor[valid]=g0.indices[pick[valid]];weight[valid]=g0.data[pick[valid]]
            inp=train[ids].copy();inp.data/=scale
            z=model(sparse_tensor(inp,device))
            rec=torch.zeros((),device=device)
            for lo in range(0,f,config['feature_chunk']):
                hi=min(lo+config['feature_chunk'],f)
                target=torch.as_tensor(x[ids,lo:hi].toarray()/scale,device=device)
                mask=torch.as_tensor(~entry_holdout(ids[:,None],np.arange(lo,hi)[None]),device=device)
                loss=torch.nn.functional.smooth_l1_loss(model.decode(z,lo,hi),target,reduction='none')
                rec+=(loss*mask).sum()/(b*f*.9)
            h=model.H
            low=nonconvex_lowrank(z,config['epsilon']) if prior else z.sum()*0
            graph=z.sum()*0
            if prior:
                other=train[neighbor].copy();other.data/=scale
                zn=model(sparse_tensor(other,device));graph=geometry_loss(z,zn,torch.as_tensor(weight,device=device))
            sparsity=h.abs().mean();group=group_sparsity(h,model.s*model.C+model.r,model.C**2)/(config['K']*model.C**2)
            total=rec+config['lambda_lowrank']*low+config['lambda_graph']*graph+config['lambda_sparse']*sparsity+config['lambda_group']*group
            if not torch.isfinite(total):raise FloatingPointError(f'{tag} seed {seed} nonfinite loss')
            optimizer.zero_grad();total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
            totals+=np.array([q.detach().item() for q in (total,rec,low,graph,sparsity,group)]);steps+=1
        history.append(dict(epoch=epoch+1,**dict(zip(['total','reconstruction','lowrank','graph','sparse','group'],totals/steps))))
        pd.DataFrame(history).to_csv(str(prefix)+'_loss.csv',index=False)
        print(folder.name,tag,seed,'epoch',epoch+1,'loss',round(totals[0]/steps,5),'seconds',round(time.time()-start_time),flush=True)
    model.eval();zs=[]
    with torch.no_grad():
        for start in range(0,n,1024):
            inp=train[start:start+1024].copy();inp.data/=scale;zs.append(model(sparse_tensor(inp,device)).cpu().numpy())
    z=np.concatenate(zs);h=model.H.detach().cpu().numpy()*scale
    np.save(str(prefix)+'_Z.npy',z);np.save(str(prefix)+'_H.npy',h)
    torch.save(model.state_dict(),prefix.with_suffix('.pt'));json_write(str(prefix)+'_scale.json',{'scale':scale})
    return model,z,h

def run_dataset(dataset,config=None):
    config=dict(DEFAULT if config is None else config);folder=ROOT/'outputs'/dataset;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'config.json'
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Existing config differs; do not overwrite results')
    json_write(path,config);context=prepare(dataset,config);x=context['X'];train=training_only(x)
    features=context['features'];cache=folder/'cache';P=context['P'];coords=context['coords'];spatial=context['spatial_neighbors']
    print(dataset,'loaded sparse',x.shape,'nnz',x.nnz,flush=True)
    if (cache/'pca.npz').exists():
        p=np.load(cache/'pca.npz');flat_z,v,mean=p['Z'],p['V'],p['mean']
    else:
        flat_z,v,mean=randomized_pca(train,k=config['K']);np.savez(cache/'pca.npz',Z=flat_z,V=v,mean=mean)
    if (cache/'G0.npz').exists():g0=sparse.load_npz(cache/'G0.npz')
    else:g0=knn_graph(flat_z,config['knn'],config['seed']);sparse.save_npz(cache/'G0.npz',g0)
    flat_error=heldout_error(x,train,lambda a:np.asarray(a@v.T)-mean@v.T,lambda z,l,h:z@v[:,l:h]+mean[l:h])
    labels={};programs={};metrics=[]
    for seed in config['seeds']:
        seed_all(seed)
        nmf=NMF(n_components=config['Kc'],init='random',random_state=seed,max_iter=300)
        U=nmf.fit_transform(P).astype(np.float32)
        gc=knn_graph(U,config['knn'],seed)
        comp_labels=leiden(gc,seed,config['resolution']);flat_labels=leiden(g0,seed,config['resolution'])
        for method,lab in [('Composition-only',comp_labels),('CCC-flat',flat_labels)]:
            labels[(method,seed)]=lab;np.save(folder/f'seed_{seed}_{method}_labels.npy',lab)
            metrics.append({'method':method,'seed':seed,'n_niches':len(np.unique(lab)),
                            'spatial_agreement':spatial_coherence(lab,spatial),**(flat_error if method=='CCC-flat' else {})})
        for prior in [False,True]:
            tag='tensor' if prior else 'no_prior'
            model,z,h=train_model(x,train,context,g0,config,seed,prior,folder)
            scale=json.loads((folder/f'seed_{seed}_{tag}_scale.json').read_text())['scale']
            def encode(a):
                b=a.copy();b.data/=scale
                with torch.no_grad():return model(sparse_tensor(b,config['device'])).cpu().numpy()
            err=heldout_error(x,train,encode,lambda a,l,r:a@h[:,l:r])
            gz=knn_graph(z,config['knn'],seed)
            for method,alpha in ([('CCC-tensor',0.),('Proposed',.2)] if prior else [('CCC-autoencoder-no-prior',0.)]):
                g=fuse_graphs(gz,gc,alpha);lab=leiden(g,seed,config['resolution'])
                labels[(method,seed)]=lab;programs[(method,seed)]=h
                np.save(folder/f'seed_{seed}_{method}_labels.npy',lab)
                metrics.append({'method':method,'seed':seed,'n_niches':len(np.unique(lab)),
                   'spatial_agreement':spatial_coherence(lab,spatial),'H_fraction_below_1pct_row_max':float((h<.01*h.max(1,keepdims=True)).mean()),**err})
            if seed==config['seed']:
                program_edges(h,features).to_csv(folder/f'{tag}_top_edges.csv',index=False)
                if prior:
                    sensitivity=[]
                    for alpha in config['alphas']:
                        lab=leiden(fuse_graphs(gz,gc,alpha),seed,config['resolution'])
                        sensitivity.append({'alpha':alpha,'n_niches':len(np.unique(lab)), 'spatial_agreement':spatial_coherence(lab,spatial)})
                    pd.DataFrame(sensitivity).to_csv(folder/'alpha_sensitivity.csv',index=False)
                    lab=labels[('Proposed',seed)];niches=np.unique(lab)
                    pd.DataFrame(np.stack([P[lab==k].mean(0) for k in niches])).to_csv(folder/'niche_composition.csv',index_label='niche')
                    pd.DataFrame(np.stack([z[lab==k].mean(0) for k in niches])).to_csv(folder/'niche_program_activity.csv',index_label='niche')
                    niche_h=np.stack([z[lab==k].mean(0)@h for k in niches]);program_edges(niche_h,features).rename(columns={'program':'niche'}).to_csv(folder/'niche_top_edges.csv',index=False)
            del model
            if torch.cuda.is_available():torch.cuda.empty_cache()
        pd.DataFrame(metrics).to_csv(folder/'baseline_metrics.csv',index=False)
    stability_tables(labels,programs).to_csv(folder/'stability.csv',index=False)
    json_write(folder/'complete.json',{'status':'complete','five_seeds':config['seeds']})
    return pd.DataFrame(metrics)
