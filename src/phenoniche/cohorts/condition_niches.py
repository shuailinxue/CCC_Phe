"""Shared control niches plus spatially coherent exceptional communication states.

This is an explicit downstream background/query comparison, not a supervised GNN
or a condition-blind joint clustering. Thresholds and spatial-size rules apply
identically to BOTH sections. A control exceptional region is retained if found.
"""
from pathlib import Path
import json
import numpy as np,pandas as pd,scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,adjusted_rand_score
from .case_novelty import matched_distance,purged_background_split
from ..gnn.communication_metric import communication_embedding


def spatial_region_filter(mask,members,coordinates,section,minimum=60,max_distance=60.):
    """Same connected-region size rule in every section; isolated cells stay out."""
    mask=np.asarray(mask,bool);rows=np.flatnonzero(mask);keep=np.zeros(len(mask),bool)
    if not len(rows):return keep
    position=np.full(len(mask),-1,int);position[rows]=np.arange(len(rows))
    neighbors=members[rows];valid=mask[neighbors]&(section[rows,None]==section[neighbors])
    valid&=np.linalg.norm(coordinates[rows,None]-coordinates[neighbors],axis=-1)<=max_distance
    first=np.broadcast_to(np.arange(len(rows))[:,None],neighbors.shape)[valid];last=position[neighbors[valid]]
    graph=sp.coo_matrix((np.ones(len(first)),(first,last)),shape=(len(rows),len(rows))).tocsr()
    _,labels=connected_components(graph,directed=False);sizes=np.bincount(labels)
    keep[rows]=sizes[labels]>=minimum;return keep

def composition_local_threshold(query,reference,scores,quantile,neighbors=256):
    """Calibrate against similar healthy mixtures, rather than a global tail.

    A rare heterogeneous immune neighborhood must not set the rejection
    threshold for every ordinary epithelial neighborhood. No case scores or
    desired niche counts are used to set these local thresholds.
    """
    tree=cKDTree(np.sqrt(np.maximum(reference,0)));result=np.empty(len(query),float)
    for start in range(0,len(query),1024):
        _,ids=tree.query(np.sqrt(np.maximum(query[start:start+1024],0)),k=min(neighbors,len(reference)))
        ids=np.asarray(ids).reshape(min(1024,len(query)-start),-1)
        result[start:start+len(ids)]=np.quantile(np.asarray(scores)[ids],quantile,axis=1)
    return np.maximum(result,1e-8)


def _calibrated_score(values,composition,fit,calibration,quantile,local_composition=None):
    score=matched_distance(values,values[fit],composition,composition[fit])
    threshold=max(float(np.quantile(score[calibration],quantile)),1e-8)
    ratio=score/threshold
    reverse=matched_distance(values[fit],values[calibration],composition[fit],composition[calibration])
    reverse_threshold=max(float(np.quantile(reverse,quantile)),1e-8);ratio[fit]=reverse/reverse_threshold
    if local_composition is not None:
        threshold_rows=composition_local_threshold(local_composition,local_composition[calibration],score[calibration],quantile)
        ratio=score/threshold_rows
        reverse_rows=composition_local_threshold(local_composition[fit],local_composition[fit],reverse,quantile)
        ratio[fit]=reverse/reverse_rows
    return ratio,threshold,reverse_threshold


def discover_condition_niches(cells,p,activity,h,feature_scale,composition,members,weights,output,
                             control='NL',case='AD',seed=52173,quantile=.99,minimum_region=60,calibration_mode='global'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);n=len(cells);allrows=np.arange(n)
    xy=cells[['x','y']].to_numpy(float);section=cells.image.astype(str).to_numpy()
    valid=cells.analysis_center.to_numpy(bool) if 'analysis_center' in cells else np.ones(n,bool)
    fit,calibration=purged_background_split(cells,members,control,seed,250.)
    # Relative CCC is primary here. Total activity is retained in native Z and
    # reported separately; it cannot alone label an assay-depth change as novel.
    values,_,metric=communication_embedding(p,activity,h,feature_scale,allrows,activity_weight=0.)
    rng=np.random.default_rng(seed);evalrows=np.sort(rng.choice(calibration,min(1200,len(calibration)),replace=False));models={};records=[]
    for k in range(4,min(15,len(fit)//60)+1):
        km=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(values[fit]);other=KMeans(n_clusters=k,n_init=10,random_state=seed+1).fit(values[fit])
        labels=km.predict(values[evalrows]);counts=np.bincount(labels,minlength=k)
        sil=float(silhouette_score(values[evalrows],labels)) if len(set(labels))>1 else -1.
        stable=float(adjusted_rand_score(labels,other.predict(values[evalrows])))
        records.append({'K':k,'silhouette':sil,'stability':stable,'eligible':bool((counts>=max(2,.005*len(evalrows))).all()),'score':.7*sil+.3*stable})
        models[k]=km
    selection=pd.DataFrame(records);eligible=selection[selection.eligible]
    best=(eligible if len(eligible) else selection).sort_values(['score','K'],ascending=[False,True]).iloc[0]
    k=int(best.K);base=models[k].predict(values)+1;selection['selected']=selection.K==k;selection.to_csv(out/'background_K_selection.csv',index=False)
    comp=np.asarray(composition,np.float32);sqrt=np.sqrt(np.maximum(comp,0))
    tree=cKDTree(sqrt[fit]);d,_=tree.query(sqrt,k=min(5,len(fit)));composition_distance=np.mean(np.atleast_2d(d),axis=1)
    comp_threshold=max(float(np.quantile(composition_distance[calibration],quantile)),1e-8)
    comp_ratio=composition_distance/comp_threshold
    reverse=np.mean(cKDTree(sqrt[calibration]).query(sqrt[fit],k=5)[0],axis=1)
    comp_ratio[fit]=reverse/max(float(np.quantile(reverse,quantile)),1e-8)
    composition_distance[fit]=reverse
    # Fixed absolute Hellinger tolerance additionally prevents calling distant
    # cell mixtures 'matched' just because they are each other's nearest match.
    common_support=composition_distance/np.sqrt(2)<=.15
    matching=comp.copy()
    if 'local_log_library' in cells:
        lib=cells.local_log_library.to_numpy(float);center=np.median(lib[fit]);scale=max(float(lib[fit].std()),.1)
        matching=np.column_stack([sqrt,.15*(lib-center)/scale])
        # matched_distance square-roots its matching arguments; encode the
        # desired nonnegative distance coordinates by squaring them.
        matching=np.column_stack([comp,(.15*(lib-lib.min())/scale)**2])
    if calibration_mode not in ('global','composition_local'):raise ValueError('Unknown control calibration')
    if calibration_mode=='composition_local':
        local_threshold=composition_local_threshold(comp,comp[calibration],composition_distance[calibration],quantile)
        comp_ratio=composition_distance/local_threshold
        comp_ratio[fit]=reverse/composition_local_threshold(comp[fit],comp[fit],reverse,quantile)
    comm_ratio,comm_threshold,reverse_threshold=_calibrated_score(values,matching,fit,calibration,quantile,
        local_composition=comp if calibration_mode=='composition_local' else None)
    raw_comp=(comp_ratio>1)&valid;raw_comm=(comm_ratio>1)&common_support&~raw_comp&valid
    raw=raw_comp|raw_comm
    support=(raw[members]*weights).sum(1)/np.maximum(weights.sum(1),1e-8)
    candidate=spatial_region_filter(raw&(support>=.5),members,xy,section,minimum_region)
    ids=np.flatnonzero(candidate);novel=np.zeros(n,np.int16);novel_records=[]
    if len(ids)>=minimum_region:
        fitrows=np.sort(rng.choice(ids,min(4000,len(ids)),replace=False));choices=[]
        distinct=len(np.unique(values[fitrows],axis=0))
        for nk in range(1,min(6,len(ids)//minimum_region,distinct)+1):
            km=KMeans(n_clusters=nk,n_init=10,random_state=seed).fit(values[fitrows]);labels=km.predict(values[ids])
            counts=np.bincount(labels,minlength=nk);eligible_partition=counts.min()>=minimum_region
            sil=float(silhouette_score(values[fitrows],km.labels_,sample_size=min(1500,len(fitrows)),random_state=seed)) if 1<len(set(km.labels_))<len(fitrows) else -.01
            novel_records.append({'K':nk,'silhouette':sil,'eligible':bool(eligible_partition)})
            if eligible_partition:choices.append((sil,nk,labels))
        if choices:
            _,nk,labels=max(choices,key=lambda v:(v[0],-v[1]));novel[ids]=labels+1
            # A cluster with 30 scattered control cells and a 1000-cell case
            # region is not a 30-cell control niche. Apply region size per label.
            for label in range(1,nk+1):
                keep=spatial_region_filter(novel==label,members,xy,section,minimum_region)
                novel[(novel==label)&~keep]=0
    detail=cells.copy();detail['background_niche']=base;detail['novel_niche']=novel
    detail['composition_score_ratio']=comp_ratio;detail['communication_score_ratio']=comm_ratio
    detail['composition_common_support']=common_support;detail['composition_Hellinger_distance']=composition_distance/np.sqrt(2)
    detail['novelty_spatial_support']=support
    detail['label']=[f'R{j}' if j else f'B{i}' for i,j in zip(base,novel)]
    detail.to_csv(out/'assignments.csv.gz',index=False)
    summaries=[]
    for label in sorted(set(novel)-{0}):
        hit=(novel==label)&valid;nh=hit&(cells.condition.to_numpy()==control);nd=hit&(cells.condition.to_numpy()==case)
        hf=nh.sum()/max(int(((cells.condition==control)&valid).sum()),1);df=nd.sum()/max(int(((cells.condition==case)&valid).sum()),1)
        comm_share=float(raw_comm[nd].mean()) if nd.any() else 0.
        summaries.append({'niche':f'R{label}','control_fraction':hf,'case_fraction':df,'control_cells':int(nh.sum()),'case_cells':int(nd.sum()),
            'case_common_composition_fraction':float(common_support[nd].mean()) if nd.any() else 0.,
            'conditional_communication_fraction':comm_share,'interpretation':'communication_change_with_composition_support' if comm_share>=.5 else 'composition_change_or_mixed',
            'observed_absent_in_control':bool(nh.sum()==0),'case_enriched':bool(df>2*hf and df>=.005)})
    summary=pd.DataFrame(summaries,columns=['niche','control_fraction','case_fraction','control_cells','case_cells',
        'case_common_composition_fraction','conditional_communication_fraction','interpretation','observed_absent_in_control','case_enriched'])
    summary.to_csv(out/'candidate_niches.csv',index=False)
    pd.DataFrame(novel_records).to_csv(out/'novel_K_selection.csv',index=False)
    audit={'control':control,'case':case,'encoder_uses_condition':False,'joint_GNN_refit':False,
        'background_K':k,'control_calibration_rows':calibration.tolist(),'control_fit_rows':fit.tolist(),
        'physical_cell_disjoint_calibration':True,'quantile':quantile,'min_connected_cells_per_section_and_label':minimum_region,
        'calibration_mode':calibration_mode,'local_calibration_neighbors':256 if calibration_mode=='composition_local' else None,
        'matching':'Original spatial-pair-weighted composition + weak local library size nuisance matching',
        'max_Hellinger_for_common_composition':.15,'communication_forward_threshold':comm_threshold,'communication_reverse_threshold':reverse_threshold,
        'total_activity_used_for_novelty':False,'metric':metric,'case_labels_used_for_novel_clustering':False,
        'note':'Downstream control-background comparison; same exception thresholds and region rules in both sections; zero case-specific niches is allowed'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2));return detail,summary


def ensure_condition_results(source,control='NL',case='AD',calibration_mode='composition_local'):
    """Cached post-fit analysis; no GNN training or weight modifications."""
    import hashlib
    from .matched_niche_evidence import export_matched_evidence
    from .paired_skin import input_root
    source=Path(source);out=source/'condition_niches';out.mkdir(exist_ok=True)
    if not (source/'complete.json').exists():raise RuntimeError('Finish the GNN run first')
    digest=hashlib.sha256((source/'gnn/checkpoint.pt').read_bytes()).hexdigest()
    code=hashlib.sha256(b''.join(Path(__file__).with_name(name).read_bytes() for name in
        ('condition_niches.py','matched_niche_evidence.py','case_novelty.py'))).hexdigest()
    identity={'checkpoint_sha256':digest,'analysis_code_sha256':code,'control':control,'case':case,'calibration_mode':calibration_mode}
    flag=out/'complete.json'
    if flag.exists() and json.loads(flag.read_text())==identity:return out
    if flag.exists():flag.unlink()
    cache=input_root(source);cells=pd.read_csv(source/'assignments.csv.gz');m=np.load(cache/'members.npy');w=np.load(cache/'weights.npy')
    column=next((v for v in ('library_counts','nCount_Xenium') if v in cells),None)
    if column:cells['local_log_library']=(np.log1p(cells[column].to_numpy()[m])*w).sum(1)/np.maximum(w.sum(1),1e-8)
    arrays=[np.load(source/'gnn'/f'{name}.npy') for name in ('P','activity','H')]
    scale=np.load(cache/'preprocessing.npz')['feature_scale'];comp=np.load(cache/'composition.npy')
    detail,table=discover_condition_niches(cells,*arrays,scale,comp,m,w,out,control,case,calibration_mode=calibration_mode)
    export_matched_evidence(source,detail,control,case)
    flag.write_text(json.dumps(identity,indent=2));return out
