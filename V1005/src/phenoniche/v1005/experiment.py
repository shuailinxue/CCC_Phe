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
from .simplex_training import train_model,split_mask,normalized_input
from .diagnostics import graph_audit,dictionary_audit
from .graph import knn_graph,fuse_graphs
from .clustering import leiden
from .evaluation import spatial_coherence,stability_tables,program_edges

DEFAULT={ 'seed':40700,'seeds':[40700,40701,40702,40703,40704], 'K':16,'Kc':8,'warmup_epochs':15,'graph_epochs':25,'graph_ramp_epochs':10,'patience':6,'early_stop_min_delta':1e-5,'training_revision':'simplex_staged','batch_size':256,
 'lr':.001,'lambda_graph':.01,'lambda_sparse':.001,'lambda_group':.001,
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
        pred=decode(z,f,stop);mask=split_mask(ids[:,None],np.arange(f,stop)[None])
        diff=np.abs(pred-target)[mask];se+=float(np.square(diff,dtype=np.float64).sum());ae+=float(diff.sum(dtype=np.float64))
        hub+=float(np.where(diff<1,.5*diff**2,diff-.5).sum(dtype=np.float64));count+=len(diff)
    return {'heldout_MSE':se/count,'heldout_MAE':ae/count,'heldout_Huber':hub/count,'heldout_entries':count}

def run_dataset(dataset,config=None,only_seed=None):
    config=dict(DEFAULT if config is None else config);folder=ROOT/'outputs'/dataset;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'config.json'
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Existing config differs; do not overwrite results')
    if not path.exists():json_write(path,config)
    context=prepare(dataset,config);x=context['X'];train=training_only(x)
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
        if only_seed is not None and seed!=only_seed:continue
        seed_all(seed)
        nmf=NMF(n_components=config['Kc'],init='random',random_state=seed,max_iter=300)
        U=nmf.fit_transform(P).astype(np.float32)
        gc=knn_graph(U,config['knn'],seed)
        comp_labels=leiden(gc,seed,config['resolution']);flat_labels=leiden(g0,seed,config['resolution'])
        for method,lab in [('Composition-only',comp_labels),('CCC-flat',flat_labels)]:
            labels[(method,seed)]=lab;np.save(folder/f'seed_{seed}_{method}_labels.npy',lab)
            metrics.append({'method':method,'seed':seed,**graph_audit(gc if method=='Composition-only' else g0,lab),
                            'spatial_agreement':spatial_coherence(lab,spatial),**(flat_error if method=='CCC-flat' else {})})
        for prior in [False,True]:
            tag='tensor' if prior else 'no_prior'
            model,z,h=train_model(x,train,context,g0,config,seed,prior,folder)
            def encode(a):
                b,mass=normalized_input(a)
                with torch.no_grad():return model(sparse_tensor(b,config['device'])).cpu().numpy(),mass
            err=heldout_error(x,train,encode,lambda a,l,r:(a[0]@h[:,l:r])*a[1][:,None])
            gz=knn_graph(z,config['knn'],seed)
            for method,alpha in ([('CCC-tensor',0.),('Proposed',.2)] if prior else [('CCC-autoencoder-no-prior',0.)]):
                g=fuse_graphs(gz,gc,alpha);lab=leiden(g,seed,config['resolution'])
                labels[(method,seed)]=lab;programs[(method,seed)]=h
                np.save(folder/f'seed_{seed}_{method}_labels.npy',lab)
                metrics.append({'method':method,'seed':seed,**graph_audit(g,lab),
                   'spatial_agreement':spatial_coherence(lab,spatial),'H_fraction_below_1pct_row_max':float((h<.01*h.max(1,keepdims=True)).mean()),**err})
            if seed==config['seed']:
                dictionary_audit(h,features,folder,tag)
                program_edges(h,features).to_csv(folder/f'{tag}_top_edges.csv',index=False)
                if prior:
                    sensitivity=[]
                    for alpha in config['alphas']:
                        ag=fuse_graphs(gz,gc,alpha)
                        lab=labels[('CCC-tensor',seed)] if alpha==0 else (labels[('Proposed',seed)] if alpha==.2 else leiden(ag,seed,config['resolution']))
                        sensitivity.append({'alpha':alpha,**graph_audit(ag,lab), 'spatial_agreement':spatial_coherence(lab,spatial)})
                    pd.DataFrame(sensitivity).to_csv(folder/'alpha_sensitivity.csv',index=False)
                    lab=labels[('Proposed',seed)];niches=np.unique(lab)
                    pd.DataFrame(np.stack([P[lab==k].mean(0) for k in niches])).to_csv(folder/'niche_composition.csv',index_label='niche')
                    pd.DataFrame(np.stack([z[lab==k].mean(0) for k in niches])).to_csv(folder/'niche_program_activity.csv',index_label='niche')
                    niche_h=np.stack([z[lab==k].mean(0)@h for k in niches]);program_edges(niche_h,features).rename(columns={'program':'niche'}).to_csv(folder/'niche_top_edges.csv',index=False)
            del model
            if torch.cuda.is_available():torch.cuda.empty_cache()
        pd.DataFrame(metrics).to_csv(folder/(f'baseline_metrics_seed_{seed}.csv' if only_seed is not None else 'baseline_metrics.csv'),index=False)
    if only_seed is not None:return pd.DataFrame(metrics)
    stability_tables(labels,programs).to_csv(folder/'stability.csv',index=False)
    json_write(folder/'complete.json',{'status':'complete','five_seeds':config['seeds']})
    return pd.DataFrame(metrics)
