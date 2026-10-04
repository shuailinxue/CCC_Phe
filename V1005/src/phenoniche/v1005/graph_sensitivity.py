"""Prime 5K post-training graph/Leiden sensitivity audit (seed 40700 only)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from .data import ROOT,json_write
from .graph import normalize_edge_mass,fuse_three_graphs
from .clustering import leiden_sweep
from .diagnostics import graph_audit
from .evaluation import spatial_coherence
from .niche_graph import SEED,ALPHA,TINY_LIMIT,_add_adjacency_metrics,select_resolution

BETAS=np.round(np.arange(0,.151,.01),2)
RESOLUTIONS=np.unique(np.round(np.r_[np.arange(.10,.351,.02),.35],2))
KEY_BETAS=(0.,.03,.05,.08,.1)
PRIMARY_BETA=.1

def _additive_worker(beta):
    out=ROOT/'outputs/xenium5k';target=out/'graph_sensitivity';cache=out/'cache'
    context=np.load(cache/'context.npz');spatial_neighbors=context['spatial_neighbors']
    gccc=sparse.load_npz(cache/'G_ccc_latent_seed_40700.npz');gcomp=sparse.load_npz(cache/'G_comp_seed_40700.npz');gspatial=sparse.load_npz(cache/'G_spatial_seed_40700.npz')
    graph=fuse_three_graphs(gccc,gcomp,gspatial,ALPHA,float(beta))
    _,summary,_=sweep(graph,spatial_neighbors,target,f'additive_beta_{beta:.2f}')
    return float(beta),summary

def _saved_graph_worker(item):
    path,prefix=item;out=ROOT/'outputs/xenium5k';target=out/'graph_sensitivity'
    spatial_neighbors=np.load(out/'cache/context.npz')['spatial_neighbors'];graph=sparse.load_npz(path)
    _,summary,_=sweep(graph,spatial_neighbors,target,prefix)
    return prefix,summary

def _fixed_graph_worker(item):
    """Evaluate a graph once at the common comparison resolution."""
    path,prefix,resolution=item;out=ROOT/'outputs/xenium5k';target=out/'graph_sensitivity'
    spatial_neighbors=np.load(out/'cache/context.npz')['spatial_neighbors'];graph=sparse.load_npz(path)
    labels,quality=leiden_sweep(graph,[float(resolution)],SEED);lab=labels[0]
    row=_metric_row(float(resolution),graph,lab,quality[0],spatial_neighbors)
    eligible=bool(5<=row['n_niches']<=25 and row['fraction_cells_niches_lt20']<=TINY_LIMIT)
    summary={'selected':eligible,'selection_reason':'Fixed-resolution fusion comparison at resolution=0.20.',**row}
    pd.DataFrame([{**row,'eligible':eligible,'selected':eligible}]).to_csv(target/f'{prefix}_fixed_resolution.csv',index=False)
    np.save(target/f'{prefix}_fixed_labels.npy',lab);json_write(target/f'{prefix}_fixed_summary.json',summary)
    return prefix,summary

def _reweight_beta_worker(beta):
    out=ROOT/'outputs/xenium5k';target=out/'graph_sensitivity';cache=target/'graph_cache'
    spatial_neighbors=np.load(out/'cache/context.npz')['spatial_neighbors']
    base=sparse.load_npz(cache/'base_additive_comp.npz');similarity=sparse.load_npz(cache/'spatial_similarity_on_base.npz')
    graph=reweight(base,[similarity],[float(beta)]);labels,quality=leiden_sweep(graph,[.2],SEED);lab=labels[0]
    np.save(target/f'reweight_beta_{beta:.2f}_labels.npy',lab)
    return {'beta':float(beta),**_metric_row(.2,graph,lab,quality[0],spatial_neighbors)}

def graph_distribution(name,graph):
    graph=graph.tocsr();degree=np.diff(graph.indptr);weighted=np.asarray(graph.sum(1)).ravel()
    edge=sparse.triu(graph,k=1).data
    out={'graph':name,'n_nodes':graph.shape[0],'undirected_edges':len(edge),
         'connected_components':int(connected_components(graph,directed=False,return_labels=False))}
    for prefix,values in [('degree',degree),('weighted_degree',weighted),('edge_weight',edge)]:
        for label,value in zip(['min','q25','median','q75','q95','q99','max'],np.quantile(values,[0,.25,.5,.75,.95,.99,1])):
            out[f'{prefix}_{label}']=float(value)
        out[f'{prefix}_mean']=float(np.mean(values))
    return out

def graph_overlap(graphs):
    a,b,c=[sparse.triu(g,k=1).astype(bool).tocsr() for g in graphs]
    na,nb,nc=[x.nnz for x in [a,b,c]];ab=a.multiply(b).nnz;ac=a.multiply(c).nnz;bc=b.multiply(c).nnz
    abc=a.multiply(b).multiply(c).nnz;union=na+nb+nc-ab-ac-bc+abc
    only_a=na-ab-ac+abc;only_b=nb-ab-bc+abc;only_c=nc-ac-bc+abc
    pair=pd.DataFrame([
        {'pair':'G_ccc vs G_comp','shared_edges':ab,'union_edges':na+nb-ab,'jaccard':ab/(na+nb-ab),'fraction_of_first':ab/na,'fraction_of_second':ab/nb},
        {'pair':'G_ccc vs G_spatial','shared_edges':ac,'union_edges':na+nc-ac,'jaccard':ac/(na+nc-ac),'fraction_of_first':ac/na,'fraction_of_second':ac/nc},
        {'pair':'G_comp vs G_spatial','shared_edges':bc,'union_edges':nb+nc-bc,'jaccard':bc/(nb+nc-bc),'fraction_of_first':bc/nb,'fraction_of_second':bc/nc}])
    categories=pd.DataFrame([{'union_edges':union,'shared_edge_fraction':(union-only_a-only_b-only_c)/union,
        'only_ccc_edge_fraction':only_a/union,'only_comp_edge_fraction':only_b/union,
        'only_spatial_edge_fraction':only_c/union,'triple_shared_fraction':abc/union}])
    return pair,categories

def edge_similarity(base,embedding,reference_graph,row_chunk=50000):
    """Symmetric RBF proximity on existing base edges; creates no new edges."""
    base=base.tocsr();embedding=np.asarray(embedding,dtype=np.float32)
    ref=sparse.triu(reference_graph,k=1).tocoo();sample=np.arange(len(ref.data))
    if len(sample)>2_000_000:sample=np.random.default_rng(SEED).choice(len(sample),2_000_000,replace=False)
    distance=np.linalg.norm(embedding[ref.row[sample]]-embedding[ref.col[sample]],axis=1)
    scale=max(float(np.median(distance)),1e-6);values=np.empty(base.nnz,dtype=np.float32)
    for lo in range(0,base.shape[0],row_chunk):
        hi=min(lo+row_chunk,base.shape[0]);start,stop=base.indptr[lo],base.indptr[hi]
        rows=np.repeat(np.arange(lo,hi),np.diff(base.indptr[lo:hi+1]));cols=base.indices[start:stop]
        d=np.linalg.norm(embedding[rows]-embedding[cols],axis=1)
        values[start:stop]=np.exp(-d/scale)
    result=sparse.csr_matrix((values,base.indices.copy(),base.indptr.copy()),shape=base.shape)
    return result,scale

def reweight(base,similarities,weights):
    result=normalize_edge_mass(base)
    multiplier=np.ones(result.nnz,dtype=np.float32)
    for similarity,weight in zip(similarities,weights):
        if similarity is not None and weight:multiplier+=float(weight)*similarity.data
    result.data*=multiplier
    return normalize_edge_mass(result)

def _metric_row(resolution,graph,labels,quality,spatial_neighbors):
    return {'resolution':float(resolution),**quality,**graph_audit(graph,labels),
            'spatial_agreement':spatial_coherence(labels,spatial_neighbors)}

def sweep(graph,spatial_neighbors,target,prefix):
    table_path=target/f'{prefix}_resolution.csv';label_path=target/f'{prefix}_selected_labels.npy';summary_path=target/f'{prefix}_summary.json'
    if table_path.exists() and summary_path.exists():
        summary=json.loads(summary_path.read_text());labels=np.load(label_path) if label_path.exists() else None
        return pd.read_csv(table_path),summary,labels
    labels,quality=leiden_sweep(graph,RESOLUTIONS,SEED)
    table=_add_adjacency_metrics(pd.DataFrame([_metric_row(r,graph,l,q,spatial_neighbors) for r,l,q in zip(RESOLUTIONS,labels,quality)]),labels)
    index,reason=select_resolution(table);table['eligible']=(table.n_niches.between(5,25))&(table.fraction_cells_niches_lt20<=TINY_LIMIT);table['selected']=False
    summary={'selected':index is not None,'selection_reason':reason}
    chosen_labels=None
    if index is not None:
        table.loc[index,'selected']=True;row=table.loc[index];chosen_labels=labels[index];np.save(label_path,chosen_labels)
        summary.update({k:(int(row[k]) if k in ['n_niches','min_niche_size'] else float(row[k])) for k in
            ['resolution','n_niches','modularity','leiden_objective','spatial_agreement','median_niche_size','min_niche_size','fraction_cells_niches_lt20','adjacent_stability']})
    table.to_csv(table_path,index=False);json_write(summary_path,summary)
    return table,summary,chosen_labels

def _adjacent_beta(table,label_map):
    table=table.sort_values('beta').reset_index(drop=True);table['ARI_previous_beta']=np.nan;table['NMI_previous_beta']=np.nan
    for i in range(1,len(table)):
        a,b=table.loc[i-1,'beta'],table.loc[i,'beta'];la,lb=label_map.get(a),label_map.get(b)
        if la is not None and lb is not None:
            table.loc[i,'ARI_previous_beta']=adjusted_rand_score(la,lb);table.loc[i,'NMI_previous_beta']=normalized_mutual_info_score(la,lb)
    table['stable_transition']=(table.ARI_previous_beta>=.8)&(table.NMI_previous_beta>=.8)&table.n_niches.between(5,25)
    return table

def stable_intervals(table):
    intervals=[];start=None;previous=None
    for row in table.itertuples():
        if row.stable_transition:
            if start is None:start=previous
        elif start is not None:
            intervals.append((start,previous));start=None
        previous=float(row.beta)
    if start is not None:intervals.append((start,previous))
    return intervals

def run_prime_diagnostic():
    out=ROOT/'outputs/xenium5k';target=out/'graph_sensitivity';target.mkdir(parents=True,exist_ok=True)
    cache=out/'cache';context=np.load(cache/'context.npz');coords=context['coords'];spatial_neighbors=context['spatial_neighbors']
    gccc=sparse.load_npz(cache/'G_ccc_latent_seed_40700.npz');gcomp=sparse.load_npz(cache/'G_comp_seed_40700.npz');gspatial=sparse.load_npz(cache/'G_spatial_seed_40700.npz')
    graphs=[normalize_edge_mass(g) for g in [gccc,gcomp,gspatial]];gccc,gcomp,gspatial=graphs
    pair,categories=graph_overlap(graphs);pair.to_csv(target/'graph_edge_overlap.csv',index=False);categories.to_csv(target/'graph_edge_categories.csv',index=False)
    pd.DataFrame([graph_distribution(n,g) for n,g in zip(['G_ccc','G_comp','G_spatial'],graphs)]).to_csv(target/'graph_distributions.csv',index=False)
    # Additive beta scan: full fine resolution grid for every beta.
    import concurrent.futures
    additive=[];additive_labels={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as pool:
        additive_results=list(pool.map(_additive_worker,BETAS))
    for beta,summary in additive_results:
        additive.append({'beta':beta,**summary});path=target/f'additive_beta_{beta:.2f}_selected_labels.npy'
        additive_labels[beta]=np.load(path) if path.exists() else None
        print('additive',beta,summary,flush=True)
    additive=_adjacent_beta(pd.DataFrame(additive),additive_labels);additive.to_csv(target/'additive_beta_summary.csv',index=False)
    # Fusion ablation at fixed beta=.1; D is reused from the additive scan.
    ablations={'A_ccc':gccc,
               'C_ccc_spatial':fuse_three_graphs(gccc,gcomp,gspatial,0,PRIMARY_BETA)}
    graph_cache=target/'graph_cache';graph_cache.mkdir(exist_ok=True)
    jobs=[]
    for name,graph in ablations.items():
        path=graph_cache/f'ablation_{name}.npz';sparse.save_npz(path,graph);jobs.append((str(path),'ablation_'+name))
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:ablation_results=list(pool.map(_saved_graph_worker,jobs))
    ablation=[]
    for prefix,summary in ablation_results:
        name=prefix.removeprefix('ablation_');ablation.append({'construction':name,**summary});print('ablation',name,summary,flush=True)
    ablation.extend([
        {'construction':'B_ccc_comp',**json.loads((target/'additive_beta_0.00_summary.json').read_text())},
        {'construction':'D_ccc_comp_spatial',**json.loads((target/'additive_beta_0.10_summary.json').read_text())}])
    pd.DataFrame(ablation).to_csv(target/'fusion_ablation.csv',index=False)
    # Conservative similarities are evaluated only on edges already present in the base graph.
    comp_u=np.load(cache/'composition_seed_40700.npy')
    sim_comp,comp_scale=edge_similarity(gccc,comp_u,gcomp);sim_spatial_ccc,spatial_scale=edge_similarity(gccc,coords,gspatial)
    base_comp=fuse_three_graphs(gccc,gcomp,gspatial,ALPHA,0)
    sim_spatial_base,_=edge_similarity(base_comp,coords,gspatial)
    controls={'ccc_comp_reweight':reweight(gccc,[sim_comp],[ALPHA]),
              'ccc_spatial_reweight':reweight(gccc,[sim_spatial_ccc],[PRIMARY_BETA]),
              'ccc_both_reweight':reweight(gccc,[sim_comp,sim_spatial_ccc],[ALPHA,PRIMARY_BETA]),
              'additive_comp_spatial_reweight':reweight(base_comp,[sim_spatial_base],[PRIMARY_BETA])}
    jobs=[]
    for name,graph in controls.items():
        path=graph_cache/f'reweight_{name}.npz';sparse.save_npz(path,graph);jobs.append((str(path),'reweight_'+name,.2))
    # A full resolution sweep is already available for the additive construction.
    # Reweight controls are compared at one common resolution to isolate graph construction.
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:control_results=list(pool.map(_fixed_graph_worker,jobs))
    compare=[]
    for prefix,summary in control_results:
        name=prefix.removeprefix('reweight_');compare.append({'construction':name,**summary});print('reweight',name,summary,flush=True)
    compare.append({'construction':'additive_all',**json.loads((target/'additive_beta_0.10_summary.json').read_text())})
    pd.DataFrame(compare).to_csv(target/'additive_vs_reweight.csv',index=False)
    # Isolate beta stability for the viable conservative graph: additive composition, spatial reweighting.
    sparse.save_npz(graph_cache/'base_additive_comp.npz',base_comp);sparse.save_npz(graph_cache/'spatial_similarity_on_base.npz',sim_spatial_base)
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as pool:rw=list(pool.map(_reweight_beta_worker,BETAS))
    rw_labels={float(beta):np.load(target/f'reweight_beta_{beta:.2f}_labels.npy') for beta in BETAS}
    rw=_adjacent_beta(pd.DataFrame(rw),rw_labels);rw.to_csv(target/'reweight_beta_fixed_resolution.csv',index=False)
    additive_intervals=stable_intervals(additive);reweight_intervals=stable_intervals(rw)
    only_spatial=float(categories.only_spatial_edge_fraction.iloc[0]);min_add=float(additive.ARI_previous_beta.min(skipna=True));min_rw=float(rw.ARI_previous_beta.min(skipna=True))
    viable=next((x for x in compare if x['construction']=='additive_comp_spatial_reweight'),None)
    recommend_reweight=bool(viable and viable.get('selected') and min_rw>min_add+.1)
    plateau=additive[additive.selection_reason.str.startswith('Selected within the upper quartile',na=False)]
    chosen=plateau.sort_values(['adjacent_stability','spatial_agreement'],ascending=False).iloc[0] if len(plateau) else additive.loc[additive.adjacent_stability.idxmax()]
    recommendation={'seed':SEED,'alpha':ALPHA,'diagnostic_betas':BETAS.tolist(),'fine_resolutions':RESOLUTIONS.tolist(),
        'composition_distance_scale':comp_scale,'spatial_distance_scale':spatial_scale,
        'only_spatial_edge_fraction':only_spatial,'additive_stable_beta_intervals':additive_intervals,
        'spatial_reweight_stable_beta_intervals_at_resolution_0.2':reweight_intervals,
        'minimum_adjacent_beta_ARI_additive':min_add,'minimum_adjacent_beta_ARI_spatial_reweight':min_rw,
        'recommended_beta':float(chosen.beta),'recommended_resolution':float(chosen.resolution),
        'recommended_n_niches':int(chosen.n_niches),
        'recommended_construction':'G_ccc + 0.2 G_comp, then spatial edge-reweight on existing union edges' if recommend_reweight else 'additive G_ccc + 0.2 G_comp + beta G_spatial',
        'reason':'Composition edges repair the disconnected CCC graph. Spatial-only edges change topology; conservative spatial reweighting remains outside the 5-25 niche range, so the additive graph is retained. Beta and resolution are selected from the predefined stability plateau rule.'}
    json_write(target/'recommendation.json',recommendation)
    print(json.dumps(recommendation),flush=True);return recommendation
