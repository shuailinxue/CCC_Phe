"""Post-training niche graph fusion and deterministic resolution selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from .data import ROOT,json_write
from .graph import knn_graph,normalize_edge_mass,fuse_three_graphs
from .clustering import leiden_sweep
from .diagnostics import graph_audit
from .evaluation import spatial_coherence

SEED=40700
ALPHA=.2
PRIMARY_BETA=.1
BETAS=(0.,.05,.1,.2)
BASE_RESOLUTIONS=(.005,.01,.02,.03,.05,.08,.1,.15,.2,.3,.5)
EXTRA_RESOLUTIONS=(.0001,.0002,.0005,.001,.002)
TINY_LIMIT=.01

def graph_statistics(name,graph):
    degree=np.diff(graph.indptr);weighted=np.asarray(graph.sum(1)).ravel()
    return {'graph':name,'n_nodes':graph.shape[0],'undirected_edges':graph.nnz//2,
            'directed_edge_mass':float(graph.sum(dtype=np.float64)),
            'edge_mass_per_node':float(graph.sum(dtype=np.float64)/graph.shape[0]),
            'mean_degree':float(degree.mean()),'median_degree':float(np.median(degree)),
            'mean_weighted_degree':float(weighted.mean())}

def _row(resolution,graph,labels,quality,spatial_neighbors):
    return {'resolution':float(resolution),**quality,**graph_audit(graph,labels),
            'spatial_agreement':spatial_coherence(labels,spatial_neighbors)}

def _add_adjacency_metrics(table,labels):
    table=table.copy();table['ARI_previous']=np.nan;table['NMI_previous']=np.nan
    table['ARI_next']=np.nan;table['NMI_next']=np.nan
    for i in range(1,len(labels)):
        ari=adjusted_rand_score(labels[i-1],labels[i]);nmi=normalized_mutual_info_score(labels[i-1],labels[i])
        table.loc[i,'ARI_previous']=ari;table.loc[i,'NMI_previous']=nmi
        table.loc[i-1,'ARI_next']=ari;table.loc[i-1,'NMI_next']=nmi
    table['adjacent_stability']=table[['ARI_previous','NMI_previous','ARI_next','NMI_next']].mean(axis=1)
    previous=table.n_niches.shift(1);following=table.n_niches.shift(-1)
    table['adjacent_count_relative_change']=pd.concat([
        (table.n_niches-previous).abs()/table.n_niches,
        (following-table.n_niches).abs()/table.n_niches],axis=1).mean(axis=1)
    return table

def select_resolution(table):
    """Apply range/tiny filter, then plateau, then quality criteria without visual choice."""
    candidate=table[(table.n_niches.between(5,25)) & (table.fraction_cells_niches_lt20<=TINY_LIMIT)].copy()
    if candidate.empty:return None,'No resolution produced 5–25 niches with tiny-niche fraction <= 1%.'
    stable=candidate[candidate.adjacent_count_relative_change<=.25].copy()
    if stable.empty:
        maximum=candidate.adjacent_stability.max();stable=candidate[candidate.adjacent_stability==maximum].copy()
        plateau='No multi-resolution count plateau met the <=25% change rule; used the most stable eligible point.'
    else:
        cutoff=stable.adjacent_stability.quantile(.75);stable=stable[stable.adjacent_stability>=cutoff].copy()
        plateau='Selected within the upper quartile of eligible stability-plateau points.'
    for col in ['modularity','spatial_agreement']:
        lo,hi=stable[col].min(),stable[col].max();stable[col+'_scaled']=1. if hi==lo else (stable[col]-lo)/(hi-lo)
    lo,hi=stable.fraction_cells_niches_lt20.min(),stable.fraction_cells_niches_lt20.max()
    stable['tiny_scaled']=1. if hi==lo else 1-(stable.fraction_cells_niches_lt20-lo)/(hi-lo)
    lo,hi=stable.adjacent_stability.min(),stable.adjacent_stability.max()
    stable['stability_scaled']=1. if hi==lo else (stable.adjacent_stability-lo)/(hi-lo)
    stable['selection_score']=.4*stable.stability_scaled+.25*stable.modularity_scaled+.25*stable.spatial_agreement_scaled+.1*stable.tiny_scaled
    chosen=stable.sort_values(['selection_score','modularity','spatial_agreement','resolution'],ascending=[False,False,False,True],kind='stable').iloc[0]
    return int(chosen.name),plateau+' Equal fixed quality terms: stability 0.40, modularity 0.25, spatial agreement 0.25, tiny-niche control 0.10.'

def _composition_graph(P,out,k=15):
    representation=out/'cache/composition_seed_40700.npy';graph_path=out/'cache/G_comp_seed_40700.npz'
    if representation.exists():U=np.load(representation)
    else:
        U=NMF(n_components=8,init='random',random_state=SEED,max_iter=300).fit_transform(P).astype(np.float32)
        np.save(representation,U)
    if graph_path.exists():return sparse.load_npz(graph_path)
    graph=knn_graph(U,k,SEED);sparse.save_npz(graph_path,graph);return graph

def _cached_knn(values,path,k=15):
    if path.exists():return sparse.load_npz(path)
    graph=knn_graph(values,k,SEED);sparse.save_npz(path,graph);return graph

def run_graph_selection(dataset):
    out=ROOT/'outputs'/dataset;target=out/'graph_selection';target.mkdir(parents=True,exist_ok=True)
    context=np.load(out/'cache/context.npz');coords=context['coords'];P=context['P'];spatial_neighbors=context['spatial_neighbors']
    config=json.loads((out/'config.json').read_text());k=int(config['knn'])
    z=np.load(out/'seed_40700_tensor_Z.npy')
    gccc=_cached_knn(z,out/'cache/G_ccc_latent_seed_40700.npz',k)
    gcomp=_composition_graph(P,out,k)
    gspatial=_cached_knn(coords,out/'cache/G_spatial_seed_40700.npz',k)
    normalized=[normalize_edge_mass(g) for g in [gccc,gcomp,gspatial]]
    pd.DataFrame([graph_statistics(n,g) for n,g in zip(['G_ccc','G_comp','G_spatial'],normalized)]).to_csv(target/'graph_statistics.csv',index=False)
    final=fuse_three_graphs(gccc,gcomp,gspatial,ALPHA,PRIMARY_BETA)
    resolutions=list(BASE_RESOLUTIONS)
    labels,qualities=leiden_sweep(final,resolutions,SEED)
    table=_add_adjacency_metrics(pd.DataFrame([_row(r,final,l,q,spatial_neighbors) for r,l,q in zip(resolutions,labels,qualities)]),labels)
    if not table.n_niches.between(5,25).any():
        resolutions=sorted(set(EXTRA_RESOLUTIONS+BASE_RESOLUTIONS))
        labels,qualities=leiden_sweep(final,resolutions,SEED)
        table=_add_adjacency_metrics(pd.DataFrame([_row(r,final,l,q,spatial_neighbors) for r,l,q in zip(resolutions,labels,qualities)]),labels)
    selected_index,reason=select_resolution(table)
    table['eligible']=(table.n_niches.between(5,25))&(table.fraction_cells_niches_lt20<=TINY_LIMIT)
    table['selected']=False
    summary={'dataset':dataset,'seed':SEED,'alpha':ALPHA,'beta':PRIMARY_BETA,'selected':selected_index is not None,'selection_reason':reason}
    if selected_index is not None:
        table.loc[selected_index,'selected']=True;chosen=table.loc[selected_index];selected_labels=labels[selected_index]
        np.save(target/'selected_labels.npy',selected_labels)
        summary.update({k:(int(chosen[k]) if k in ['n_niches','min_niche_size'] else float(chosen[k])) for k in
                        ['resolution','n_niches','spatial_agreement','median_niche_size','min_niche_size','fraction_cells_niches_lt20','modularity','leiden_objective']})
        beta_rows=[]
        for beta in BETAS:
            graph=fuse_three_graphs(gccc,gcomp,gspatial,ALPHA,beta)
            ls,qs=leiden_sweep(graph,[chosen.resolution],SEED);beta_rows.append({'beta':beta,**_row(chosen.resolution,graph,ls[0],qs[0],spatial_neighbors)})
        pd.DataFrame(beta_rows).to_csv(target/'beta_sensitivity.csv',index=False)
    else:
        summary.update({'resolution':None,'n_niches':None,'spatial_agreement':None,'median_niche_size':None,'min_niche_size':None,'fraction_cells_niches_lt20':None})
        pd.DataFrame(columns=['beta']).to_csv(target/'beta_sensitivity.csv',index=False)
    table.to_csv(target/'resolution_sweep.csv',index=False);json_write(target/'summary.json',summary)
    print(dataset,json.dumps(summary),flush=True)
    return summary
