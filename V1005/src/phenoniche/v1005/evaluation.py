import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from itertools import combinations

def match_programs(a,b):
    a=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
    b=b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12)
    similarity=a@b.T;i,j=linear_sum_assignment(-similarity)
    return similarity,i,j

def spatial_coherence(labels,neighbors):
    rows=np.repeat(np.arange(len(labels)),neighbors.shape[1]);cols=neighbors.ravel();keep=rows!=cols
    return float(np.mean(labels[rows[keep]]==labels[cols[keep]]))

def stability_tables(labels,programs):
    rows=[]
    for (method,a),(method2,b) in combinations(labels,2):
        if method!=method2:continue
        la,lb=labels[(method,a)],labels[(method,b)]
        sim=np.nan
        if (method,a) in programs and (method,b) in programs:
            s,i,j=match_programs(programs[(method,a)],programs[(method,b)]);sim=float(s[i,j].mean())
        rows.append({'method':method,'seed_a':a,'seed_b':b,'ARI':adjusted_rand_score(la,lb),
                     'NMI':normalized_mutual_info_score(la,lb),'program_matched_cosine':sim})
    return pd.DataFrame(rows)

def program_edges(h,features,top=10):
    rows=[]
    for k in range(len(h)):
        for rank,f in enumerate(np.argsort(-h[k],kind='stable')[:top],1):
            rows.append({'program':k+1,'rank':rank,**features.iloc[f].to_dict(),'weight':float(h[k,f])})
    return pd.DataFrame(rows)

def representation_diagnostics(dataset):
    """Report redundancy explicitly; high cross-seed cosine alone can reward collapse."""
    from .data import ROOT
    folder=ROOT/'outputs'/dataset;rows=[]
    for path in sorted(folder.glob('seed_*_H.npy')):
        stem=str(path)[:-6];h=np.load(path);z=np.load(stem+'_Z.npy').astype(np.float64)
        normalized=h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-12)
        similarities=normalized@normalized.T;offdiag=similarities[~np.eye(len(h),dtype=bool)]
        sv=np.linalg.svd(z-z.mean(0),compute_uv=False);energy=sv**2
        sv_sum=float(sv.sum());energy_sum=float(energy.sum())
        p=sv/sv_sum if sv_sum>0 else np.zeros_like(sv)
        from .diagnostics import latent_audit
        sampled=np.random.default_rng(789).choice(len(z),min(4096,len(z)),replace=False)
        rows.append({'model':path.name[:-6],**latent_audit(z[sampled]), 'H_row_sum_max_error':float(np.max(np.abs(h.sum(1)-1))),
                     'Z_row_sum_max_error':float(np.max(np.abs(z.sum(1)-1))),
                     'H_other_program_cosine_median':float(np.median(offdiag)),
                     'H_other_program_cosine_max':float(np.max(offdiag)),
                     'centered_Z_first_component_variance_fraction':float(energy[0]/energy_sum) if energy_sum>0 else np.nan,
                     'centered_Z_effective_rank':float(np.exp(-np.sum(p*np.log(np.maximum(p,1e-12))))) if sv_sum>0 else 0.,
                     'maximum_latent_column_std':float(z.std(0).max()),
                     'H_fraction_below_1pct_row_max':float((h<.01*h.max(1,keepdims=True)).mean())})
    t=pd.DataFrame(rows);t.to_csv(folder/'representation_diagnostics.csv',index=False);return t

def finalize_table_labels(dataset):
    """Make saved niche IDs agree with zero-based Leiden labels and spatial maps."""
    import json
    from .data import ROOT
    folder=ROOT/'outputs'/dataset;types=json.loads((folder/'tensor_audit.json').read_text())['types']
    c=pd.read_csv(folder/'niche_composition.csv')
    c=c.rename(columns={str(i):t for i,t in enumerate(types)})
    c.to_csv(folder/'niche_composition.csv',index=False)
    activity=pd.read_csv(folder/'niche_program_activity.csv')
    activity=activity.rename(columns={str(i):f'Program {i+1}' for i in range(activity.shape[1]-1)})
    activity.to_csv(folder/'niche_program_activity.csv',index=False)
    edges=pd.read_csv(folder/'niche_top_edges.csv');expected=set(c.niche)
    if set(edges.niche)!=expected and set(edges.niche-1)==expected:edges['niche']-=1
    if set(edges.niche)!=expected:raise ValueError('Niche edge-table IDs disagree with composition table')
    edges.to_csv(folder/'niche_top_edges.csv',index=False)

    metrics_path=folder/'baseline_metrics.csv'
    if metrics_path.exists():
        metrics=pd.read_csv(metrics_path);fractions=[]
        for row in metrics.itertuples():
            labels=np.load(folder/f'seed_{row.seed}_{row.method}_labels.npy')
            fractions.append(float(np.bincount(labels).max()/len(labels)))
        metrics['largest_niche_fraction']=fractions
        metrics.to_csv(metrics_path,index=False)
