"""Train-fitted niche discovery, biological controls, and composition diagnostics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.metrics import adjusted_rand_score,silhouette_score,r2_score
from ..consolidation import _latent_block,_soft_membership,_scale01,cosine_matrix
from ..interpretation import export_top_markers


def discover(data,arrays,h,config):
    train,val,test=data.rows
    rng=np.random.default_rng(config.seed)
    fit_rows=np.sort(rng.choice(train,min(len(train),config.clustering_sample_size),replace=False))
    val_rows=np.sort(rng.choice(val,min(len(val),config.silhouette_sample_size),replace=False))
    if min(len(fit_rows),len(val_rows))<=config.k_max: raise ValueError('Too few disjoint centers for K=4–15 evaluation')
    center=float(np.median(np.log1p(arrays['activity'][fit_rows])));scale=max(float(np.std(np.log1p(arrays['activity'][fit_rows]))),.1)
    latent=lambda rows:_latent_block(arrays['P'],arrays['activity'],rows,center,scale,config.activity_weight)
    train_x=latent(fit_rows);val_x=latent(val_rows);models={};records=[]
    for k in range(config.k_min,config.k_max+1):
        model=MiniBatchKMeans(n_clusters=k,random_state=config.seed,n_init=5,batch_size=4096).fit(train_x)
        other=MiniBatchKMeans(n_clusters=k,random_state=config.seed+1,n_init=5,batch_size=4096).fit(train_x)
        labels=model.predict(val_x);distances=model.transform(train_x);membership_scale=max(float(np.median(distances.min(1)**2)),1e-8)
        counts=np.bincount(labels,minlength=k);fractions=counts/len(labels)
        silhouette=float(silhouette_score(val_x,labels)) if 1<len(set(labels))<len(labels) else -1.
        records.append({'K':k,'silhouette':silhouette,'seed_stability':adjusted_rand_score(labels,other.predict(val_x)),
                        'max_niche_cell_fraction':float(fractions.max()),'tiny_niche_fraction':float(fractions[counts<max(2,.005*len(labels))].sum()),
                        'active_niches':int((counts>0).sum())})
        models[k]=(model,membership_scale)
    table=pd.DataFrame(records)
    table['selection_score']=.4*_scale01(table.silhouette)+.3*_scale01(table.seed_stability)+.2*_scale01(1-table.max_niche_cell_fraction)+.1*_scale01(1-table.tiny_niche_fraction)
    table['eligible']=(table.active_niches==table.K)&(table.max_niche_cell_fraction<=.75)&(table.tiny_niche_fraction<=.01)
    eligible=table[table.eligible]
    if eligible.empty: eligible=table[table.max_niche_cell_fraction<=table.max_niche_cell_fraction.min()+.05]
    best=eligible.selection_score.max();k=int(eligible[eligible.selection_score>=best-config.near_optimal_score_tolerance].K.min())
    table['selected']=table.K==k; model,soft_scale=models[k]
    output=Path(config.output_dir);selected=output/'selected_model';selected.mkdir(exist_ok=True)
    q=np.lib.format.open_memmap(selected/'Q.npy',mode='w+',dtype=np.float32,shape=(len(data.cells),k))
    labels=np.empty(len(q),np.int16)
    for start in range(0,len(q),50000):
        stop=min(start+50000,len(q));distance=model.transform(latent(slice(start,stop)))
        q[start:stop]=_soft_membership(distance,soft_scale);labels[start:stop]=distance.argmin(1)
    q.flush();table.to_csv(output/'K_selection.csv',index=False)
    np.save(selected/'niche_labels.npy',labels)
    # Program contributions and dictionary are weighted by observed native Z, not expression.
    contribution=np.asarray(q[train],np.float64).T@np.asarray(arrays['Z'][train],np.float64)
    contribution/=np.maximum(np.asarray(q[train]).sum(0)[:,None],1e-8)
    profile=contribution@h;profile/=np.maximum(profile.sum(1,keepdims=True),1e-8)
    np.save(selected/'H.npy',profile.astype(np.float32))
    pd.DataFrame(contribution,index=np.arange(1,k+1),columns=np.arange(1,h.shape[0]+1)).to_csv(selected/'program_contribution.csv',index_label='niche')
    export_top_markers(profile,data.features,selected)
    program_dir=output/'programs';program_dir.mkdir(exist_ok=True)
    export_top_markers(h,data.features,program_dir)
    for path in program_dir.glob('*.csv'):
        frame=pd.read_csv(path).rename(columns={'niche':'program'});frame.to_csv(path,index=False)
    assignments=data.cells.copy();assignments['x']=data.coordinates[:,0];assignments['y']=data.coordinates[:,1]
    assignments['niche']=labels+1;assignments['assignment_confidence_relative']=q.max(1);assignments['CCC_activity']=arrays['activity'];assignments['split']=data.folds
    assignments.to_csv(selected/'niche_assignments.csv.gz',index=False)
    counts=np.bincount(labels,minlength=k)
    pd.DataFrame({'niche':np.arange(1,k+1),'n_cells':counts,'fraction':counts/len(labels)}).to_csv(selected/'niche_counts.csv',index=False)
    # Store the fitted clustering model for new-data inference without refitting K.
    import torch
    torch.save({'model':model,'membership_scale':soft_scale,'activity_center':center,'activity_scale':scale,'K':k},output/'niche_model.pt')
    return k,labels,table


def composition(data):
    return data.composition_target


def diagnostics(data,arrays,h,labels,config):
    train,val,test=data.rows;out={}; comp=composition(data)
    # Held-out R² and nuisance-only niche prediction quantify composition recovery.
    z=np.log1p(np.asarray(arrays['Z']))
    valid=np.asarray(comp).sum(1)>0;comp_train=train[valid[train]];comp_test=test[valid[test]]
    out['composition_undefined_centers']=int((~valid).sum())
    out['composition_to_Z_test_R2']=None
    if len(comp_train)>=2 and len(comp_test)>=2:
        estimator=Ridge(alpha=1).fit(comp[comp_train],z[comp_train])
        out['composition_to_Z_test_R2']=float(r2_score(z[comp_test],estimator.predict(comp[comp_test]),multioutput='variance_weighted'))
    if len(comp_train) and len(comp_test) and len(set(labels[comp_train]))>1:
        classifier=LogisticRegression(max_iter=500,random_state=config.seed).fit(comp[comp_train],labels[comp_train])
        out['composition_only_niche_test_accuracy']=float(classifier.score(comp[comp_test],labels[comp_test]))
    out['dominant_niche_test_fraction']=float(np.bincount(labels[test]).max()/len(test))
    tree=cKDTree(data.coordinates[test]);distance,neighbor=tree.query(data.coordinates[test],k=2)
    good=distance[:,1]<=3*config.sigma_um
    out['test_spatial_neighbor_agreement']=float((labels[test][good]==labels[test[neighbor[:,1]]][good]).mean()) if good.any() else None
    p=np.asarray(arrays['P'][test]);mean=p.mean(0)
    out['effective_programs']=float(np.exp(-(mean*np.log(np.maximum(mean,1e-12))).sum()))
    cosine=cosine_matrix(h);out['maximum_dictionary_cosine']=float(cosine[np.triu_indices(len(h),1)].max())
    out['dictionary_entropy']=float((-(h*np.log(np.maximum(h,1e-12))).sum(1)).mean())
    out['interpretation']='Single-tissue association diagnostics; composition is auxiliary supervision, not evidence of causality or phenotype.'
    (Path(config.output_dir)/'diagnostics.json').write_text(json.dumps(out,indent=2)+'\n')
    return out


def match_programs(left,right):
    similarity=np.clip((left/np.maximum(np.linalg.norm(left,axis=1,keepdims=True),1e-8))@(right/np.maximum(np.linalg.norm(right,axis=1,keepdims=True),1e-8)).T,-1,1)
    a,b=linear_sum_assignment(-similarity)
    jaccard=[]
    for i,j in zip(a,b):
        x=set(np.argsort(-left[i])[:15]);y=set(np.argsort(-right[j])[:15]);jaccard.append(len(x&y)/max(len(x|y),1))
    return {'matched_program_cosine_mean':float(similarity[a,b].mean()),'top_CCC_Jaccard_mean':float(np.mean(jaccard)),'matched_programs':len(a)}


def plot_spatial(data,labels,k,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)/'figures';output.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    for ax,color,title in zip(axes,(data.types,labels),('Cell types',f'GNN communication niches (K={k})')):
        ax.scatter(*data.coordinates.T,c=color,s=.2,cmap='tab20',rasterized=True,linewidths=0)
        ax.set_title(title);ax.set_aspect('equal');ax.invert_yaxis();ax.set(xlabel='x (µm)',ylabel='y (µm)')
    fig.savefig(output/'spatial_niches.png',dpi=250);plt.close(fig)
