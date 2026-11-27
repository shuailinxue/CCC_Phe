"""Post-fit validation of sampled ovarian niches with original fine/pathology labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from phenoniche.v1.storage import digest,key,save_npz,versions
from ovca_network import OV


def enrichment(y, labels, strata, rng):
    """Conditional permutations preserve type and 500-µm spatial block."""
    cats, z = np.unique(labels,return_inverse=True)
    k = 4
    groups = [np.flatnonzero(strata==s) for s in np.unique(strata)]
    def table(values):
        return np.bincount(y*len(cats)+values,minlength=k*len(cats)).reshape(k,-1)
    observed=table(z)
    expected=np.zeros_like(observed,dtype=float)
    for g in groups:
        expected+=np.outer(np.bincount(y[g],minlength=k),np.bincount(z[g],minlength=len(cats)))/len(g)
    draws=[]
    for _ in range(999):
        shuffled=z.copy()
        for g in groups:
            shuffled[g]=rng.permutation(z[g])
        draws.append(table(shuffled))
    p=(1+(np.abs(np.stack(draws)-expected)>=np.abs(observed-expected)).sum(0))/1000
    order=np.argsort(p.ravel())
    q=np.empty(p.size)
    q[order]=np.minimum.accumulate((p.ravel()[order]*p.size/np.arange(1,p.size+1))[::-1])[::-1].clip(0,1)
    return pd.DataFrame(dict(niche=np.repeat(np.arange(k),len(cats)),label=np.tile(cats,k),
                             observed=observed.ravel(),expected=expected.ravel(),p=p.ravel(),q=q,
                             log2_enrichment=np.log2((observed.ravel()+.5)/(expected.ravel()+.5))))


p=argparse.ArgumentParser()
for name in ('joint','composition','communication','null','diagnosis','paired-null'):
    p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args()
paths={k:getattr(a,k) for k in ('joint','composition','communication','null')}
meta=dict(script=digest(__file__),source=digest(OV),runs={k:digest(v/'cells.npz') for k,v in paths.items()},
          diagnostic=digest(a.diagnosis/'local.npz'),paired=digest(a.paired_null/'metrics.json'),
          versions=versions(['numpy','pandas','anndata','matplotlib','umap-learn','scikit-learn']))
dest=a.joint.parent/('report-'+key(meta))
dest.mkdir(exist_ok=True)
(dest/'manifest.json').write_text(json.dumps(meta,indent=2))
cells={k:dict(np.load(v/'cells.npz')) for k,v in paths.items()}
j=cells['joint']
xyz=j['xyz']; y=j['niche']; ids=j['ids']
assert all(np.array_equal(v['ids'],ids) and np.array_equal(v['test'],j['test']) and np.array_equal(v['mass'],j['mass']) for v in cells.values())
test=j['test']
obs=ad.read_h5ad(OV,backed='r')
ann=obs.obs.loc[ids,['major_celltype','minor_celltype','path_region']].copy()
obs.file.close()
blocks=np.floor(xyz/500).astype(int)
block=np.array([f'{i}:{j}' for i,j in blocks])
ann.assign(block=block,heldout=test,**{k:v['niche'] for k,v in cells.items()}).to_csv(dest/'annotation_assignments.tsv',sep='\t')
metrics={}
tables=[]
for name,run in paths.items():
    m=json.loads((run/'metrics.json').read_text())
    yp=cells[name]['niche']
    m.update(ARI_joint=adjusted_rand_score(y,yp),ARI_joint_holdout=adjusted_rand_score(y[test],yp[test]))
    hist=pd.read_csv(run/'loss.tsv',sep='\t')
    m.update(final_loss=hist.iloc[-1].to_dict(),first_loss=hist.iloc[0].to_dict())
    n=np.load(run/'niches.npz')
    m.update(omega=n['omega'].tolist(),eta_distances=pdist(n['eta']).tolist())
    vis=np.load(run/'sampling.npz')['visits']
    m.update(visited=int((vis>0).sum()),visits_quantiles=np.quantile(vis[j['train']],[0,.5,1]).tolist())
    metrics[name]=m
    if name not in ('joint','composition'):
        continue
    for region,select in [('all',np.ones(len(ids),bool)),('holdout',test)]:
        for ct in ann.major_celltype.unique():
            g=select & ann.major_celltype.eq(ct).to_numpy()
            if g.sum()<20 or ann.minor_celltype[g].nunique()<2:
                continue
            e=enrichment(yp[g],ann.minor_celltype[g].to_numpy(dtype=str),block[g],np.random.default_rng(73))
            tables.append(e.assign(model=name,scope=region,broad_type=ct))
        g=select & ann.path_region.ne('Unassigned').to_numpy()
        strata=np.char.add(block[g],ann.major_celltype[g].to_numpy(dtype=str))
        e=enrichment(yp[g],ann.path_region[g].to_numpy(dtype=str),strata,np.random.default_rng(73))
        tables.append(e.assign(model=name,scope=region,broad_type='Pathology'))
table=pd.concat(tables,ignore_index=True)
table.to_csv(dest/'enrichment.tsv',sep='\t',index=False)
metrics['null_on_real_holdout']=json.loads((a.paired_null/'metrics.json').read_text())
z=dict(np.load(a.diagnosis/'local.npz'))
ix=pd.Index(ids).get_indexer(z['ids'])
assert (ix>=0).all()
v=[z[k].var(0) for k in ('a0','a_free')]
metrics['diagnostic']=dict(gradient={k:np.quantile(z[k],[.1,.5,.9]).tolist() for k in ('grad_raw','grad_comm','grad_prior','grad_total','refined_grad')},
    ratio=float(np.median(z['grad_comm']/np.maximum(z['grad_prior'],1e-12))),variance=[x.tolist() for x in v],
    comm_loss=[float(z[k].mean()) for k in ('comm0','comm_free')],
    free_zero_fraction=float((z['a_free']==0).mean()),kkt_max=float(z['free_kkt'].max()),
    kkt_fraction=float(np.mean(z['free_kkt']<1e-5)))
if not (dest/'umap.npz').exists():
    import umap
    vals=np.log1p(np.vstack([z['a0'],z['a_free']]))
    emb=umap.UMAP(n_neighbors=20,min_dist=.2,random_state=73,n_jobs=1).fit_transform(vals)
    groups=[KMeans(4,n_init=20,random_state=73).fit_predict(x) for x in np.split(vals,2)]
    save_npz(dest/'umap.npz',xy=emb,ids=z['ids'],groups=groups)
embedding=np.load(dest/'umap.npz')
metrics['diagnostic']['activity_cluster_ARI']=[adjusted_rand_score(y[ix],c) for c in embedding['groups']]
(dest/'metrics.json').write_text(json.dumps(metrics,indent=2))
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.right':False,'axes.spines.top':False})
palette=['#4477aa','#cc6677','#228833','#aa8844']
def save(fig,name):
    for ext in ('pdf','png'):
        fig.savefig(dest/(name+'.'+ext),dpi=180)
    plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
for ax,(name,c) in zip(axes.flat,cells.items()):
    for k,color in enumerate(palette):
        selected=c['niche']==k
        ax.scatter(*xyz[selected].T,s=4,color=color,label=f'{k} ({selected.sum()})',rasterized=True)
    ax.set(title=name,aspect='equal',xlabel='X (µm)',ylabel='Y (µm)')
    ax.legend(fontsize=7,frameon=False,ncol=2)
save(fig,'spatial_niches')
fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
for ax,name in zip(axes[0,:2],('joint','communication')):
    h=pd.read_csv(paths[name]/'loss.tsv',sep='\t')
    for k in ('composition','occurrence','intensity','latent','regularization'):
        ax.plot(h.epoch,h[k],label=k)
    ax.set(title=name,yscale='symlog',xlabel='Epoch',ylabel='Mean objective term')
    ax.legend(fontsize=7,frameon=False)
ax=axes[0,2]
ax.boxplot([z[k].clip(1e-12) for k in ('grad_raw','grad_comm','grad_prior')],showfliers=False)
ax.set(xticks=[1,2,3],xticklabels=['Raw LR','Calibrated LR','Mixture prior'],yscale='log',ylabel='Gradient L2 norm')
lim=embedding['xy']
for ax,title,xy in zip(axes[1,:2],('Saved activity','Communication-only local refit'),np.split(lim,2)):
    ax.scatter(*xy.T,c=np.array(palette)[y[ix]],s=5,rasterized=True)
    ax.set(title=title,xlim=(lim[:,0].min()-1,lim[:,0].max()+1),ylim=(lim[:,1].min()-1,lim[:,1].max()+1),xlabel='Shared UMAP 1',ylabel='Shared UMAP 2')
for x,name in zip(v,['Saved','Local refit']):
    axes[1,2].plot(range(1,13),np.maximum(x,1e-12),'o-',label=name)
axes[1,2].set(yscale='log',xlabel='Program',ylabel='Across-center variance')
axes[1,2].legend(frameon=False)
save(fig,'loss_gradient_activity')
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,name in zip(axes,('composition','joint')):
    n=np.load(paths[name]/'niches.npz')
    im=ax.imshow(n['pi'],aspect='auto',vmin=0,vmax=1,cmap='Blues')
    ax.set(xticks=range(len(n['names'])),xticklabels=n['names'],yticks=range(4),ylabel='Niche',title=name)
    ax.tick_params(axis='x',labelrotation=40)
fig.colorbar(im,ax=axes,label='Prototype proportion')
save(fig,'niche_composition')
program=pd.read_csv(paths['joint']/'top_edges.tsv',sep='\t')
program=program[program.program==np.argmax(v[0])].copy()
program['loading']=np.hypot(program.occurrence,program.intensity)
program=program.nlargest(8,'loading')
program.to_csv(dest/'representative_edges.tsv',sep='\t',index=False)
fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
for i,r in enumerate(program.itertuples()):
    color='#4477aa'
    ax.add_patch(FancyArrowPatch((.27,i),(.73,i),arrowstyle='-|>',mutation_scale=12,color=color,lw=1+2*r.loading))
    ax.text(.24,i,r.sender,ha='right',va='center')
    ax.text(.76,i,r.receiver,ha='left',va='center')
    ax.text(.5,i+.17,f'{r.ligand} → {r.receptor}',ha='center',fontsize=7)
ax.set(xlim=(0,1),ylim=(-.5,len(program)-.3),title=f'Program {program.program.iloc[0]+1} · strongest supported directed edges')
ax.axis('off')
save(fig,'directed_program')
print(dest)
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained',sharey=True)
for ax,name in zip(axes,('composition','joint')):
    t=table.query("model==@name and scope=='all' and broad_type=='Malignant'")
    mat=t.pivot(index='label',columns='niche',values='log2_enrichment')
    counts=t.pivot(index='label',columns='niche',values='observed').reindex_like(mat)
    im=ax.imshow(mat,aspect='auto',cmap='RdBu_r',vmin=-2,vmax=2)
    for i,j in np.ndindex(mat.shape):
        ax.text(j,i,str(counts.iloc[i,j]),ha='center',va='center',fontsize=8)
    ax.set(title=name,xticks=range(4),xlabel='Niche',yticks=range(len(mat)),yticklabels=mat.index)
fig.colorbar(im,ax=axes,label='Log₂ enrichment within spatial blocks')
save(fig,'malignant_states')
