"""Summarize saved fixed-model diagnostics without fitting any global model."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,silhouette_score
import umap
from phenoniche.v1.storage import save_npz,digest,key,versions

p=argparse.ArgumentParser()
for arg in ('diagnosis','joint','communication'):
    p.add_argument('--'+arg,required=True,type=Path)
args=p.parse_args()
meta=dict(source=digest(__file__),local=digest(args.diagnosis/'local.npz'),
          versions=versions(['umap-learn','scikit-learn','numpy','matplotlib']),
          fit=[digest(q/'model.pt') for q in (args.joint,args.communication)])
dest=args.diagnosis/('summary-'+key(meta))
dest.mkdir(exist_ok=True)
(dest/'manifest.json').write_text(json.dumps(meta,indent=2))
z=np.load(args.diagnosis/'local.npz')
cells=np.load(args.joint/'cells.npz')
comm=np.load(args.communication/'cells.npz')
vis=np.load(args.joint/'sampling.npz')['visits']
train=cells['train']
idx=z['index']
old=cells['niche'][idx]
cand=z['candidate'].argmax(1)
refined=z['shared_refined'].argmax(1)
assert np.array_equal(z['ids'],cells['ids'][idx])
hist=[pd.read_csv(q/'loss.tsv',sep='\t') for q in (args.joint,args.communication)]
feat=np.log1p(np.vstack([z['a0'],z['a_free']]))
n=len(idx)
if not (dest/'embedding.npz').exists():
    xy=umap.UMAP(n_neighbors=20,min_dist=.2,random_state=2026,n_jobs=1).fit_transform(feat)
    groups=[KMeans(4,random_state=2026,n_init=20).fit_predict(x) for x in np.split(feat,2)]
    save_npz(dest/'embedding.npz',ids=z['ids'],embedding=xy,clusters=np.stack(groups))
with np.load(dest/'embedding.npz') as f:
    xy,groups=f['embedding'],f['clusters']
def summary(x):
    return np.quantile(x,[.1,.5,.9]).tolist()
def delta(x,y):
    return float(np.mean(x!=y))
metrics=dict(n=n,gradient_quantiles={k:summary(z[k]) for k in ('grad_comm','grad_prior','grad_raw','grad_total','refined_grad','free_kkt')},
             median_comm_prior_ratio=float(np.median(z['grad_comm']/np.maximum(z['grad_prior'],1e-12))),
             mean_comm_loss=[float(z[k].mean()) for k in ('comm0','comm_free')],
             activity_mean=[float(z[k].mean()) for k in ('a0','a_free')],
             activity_variance=[z[k].var(0).tolist() for k in ('a0','a_free')],
             silhouette=[float(silhouette_score(x,y)) for x,y in zip(np.split(feat,2),groups)],
             cluster_sizes=[np.bincount(x,minlength=4).tolist() for x in groups],
             activity_cluster_ARI=adjusted_rand_score(*groups),
             activity_cluster_ARI_to_original=[adjusted_rand_score(x,old) for x in groups],
             candidate_change=delta(old,cand),candidate_ARI=adjusted_rand_score(old,cand),
             refined_change=delta(old,refined),refined_ARI=adjusted_rand_score(old,refined),
             mean_shared_score_gain=float(np.mean(logsumexp(z['shared_refined'],1)-logsumexp(z['shared0'],1))),
             omega=z['comm_omega'].tolist(),eta_pairwise=pdist(z['comm_eta']).tolist(),
             comm_var=comm['activity'].var(0).tolist(),cosine=summary(z['comm_cosine'][np.triu_indices(12,1)]),
             sampling=dict(total=len(train),eligible=int(train.sum()),visited=int((vis>0).sum()),max_visits=int(vis.max()),
                           by_final_niche=[float((vis[train&(cells['niche']==k)]>0).mean()) for k in range(4)]),
             last_epoch=[h.iloc[-1].to_dict() for h in hist])
(dest/'metrics.json').write_text(json.dumps(metrics,indent=2))
pd.crosstab(old,cand).to_csv(dest/'candidate_transition.tsv',sep='\t')
plt.rcParams.update({'font.size':9,'axes.spines.right':False,'axes.spines.top':False,'pdf.fonttype':42})
fig,ax=plt.subplots(2,3,figsize=(12,7),layout='constrained')
for a,h,title in zip(ax[0,:2],hist,['Joint model','Communication-only model']):
    for col in ('composition','occurrence','intensity','latent','regularization','prior'):
        a.plot(h.epoch,h[col],label=col)
    a.set(yscale='symlog',title=title,xlabel='Epoch',ylabel='Mean objective term')
ax[0,0].legend(fontsize=7,ncol=2)
a=ax[0,2]
a.boxplot([z[k].clip(1e-12) for k in ('grad_raw','grad_comm','grad_prior','grad_total','refined_grad')],showfliers=False)
a.set_xticks(range(1,6),['Raw CCC','Scaled CCC','Prior','Total','Refined'],rotation=20)
a.set(yscale='log',title='Local MAP gradients',ylabel='Per-center L2 norm')
for i,a in enumerate(ax[1,:2]):
    sub=np.split(xy,2)[i]
    a.scatter(*sub.T,c=old,cmap='viridis',s=4)
    a.set(xlim=(xy[:,0].min()-1,xy[:,0].max()+1),ylim=(xy[:,1].min()-1,xy[:,1].max()+1),
          title=['Saved activity','Communication-refitted activity'][i],xlabel='Shared UMAP 1',ylabel='Shared UMAP 2')
    a.text(.02,.02,'Color: original niche',transform=a.transAxes,fontsize=8)
a=ax[1,2]
for k,label in [('a0','Saved'),('a_free','Communication refit')]:
    a.plot(range(1,13),np.maximum(z[k].var(0),1e-12),'o-',label=label)
a.set(yscale='log',xlabel='Program',ylabel='Across-center variance',title='Program activity dispersion');a.legend()
fig.savefig(dest/'diagnostics.png',dpi=180);fig.savefig(dest/'diagnostics.pdf')
print(dest)
fig,ax=plt.subplots(2,2,figsize=(9,7),layout='constrained')
transition=pd.crosstab(old,cand).reindex(index=range(4),columns=range(4),fill_value=0).to_numpy()
ax[0,0].imshow(transition,cmap='Blues')
for i,j in np.ndindex(4,4):
    ax[0,0].text(j,i,str(transition[i,j]),ha='center',va='center',color='white' if transition[i,j]>transition.max()*.55 else 'black')
ax[0,0].set(xlabel='Candidate-specific MAP',ylabel='Saved assignment',title='Paired assignments',xticks=range(4),yticks=range(4))
ax[0,1].bar(range(4),z['comm_omega'],color='#487e94')
ax[0,1].set(title='Communication-only mixture',xticks=range(4),xlabel='Niche',ylabel='Fitted mixture weight')
im=ax[1,0].imshow(z['comm_cosine'],vmin=0,vmax=1,cmap='Blues')
ax[1,0].set(title='Communication-only dictionary',xlabel='Program',ylabel='Program')
fig.colorbar(im,ax=ax[1,0],label='Cosine similarity')
sp=cells['xyz']
ax[1,1].scatter(*sp[::20,:2].T,s=.7,c='#bcbcbc',rasterized=True)
ax[1,1].scatter(*sp[vis>0,:2].T,s=.6,c='#46738e',rasterized=True)
ax[1,1].set(title='Visited centers over tissue',aspect='equal',xlabel='Input X',ylabel='Input Y')
fig.savefig(dest/'assignment_sampling.png',dpi=180);fig.savefig(dest/'assignment_sampling.pdf')
