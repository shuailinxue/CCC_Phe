"""Healthy-calibrated, composition-matched descriptive niche novelty.

Condition is used only here, after the condition-blind encoder and K selection.
These scores describe departure from one healthy section, not population p-values.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def purged_background_split(cells,members,control_label='HC',seed=52173,block_um=250.):
    """Whole spatial-block calibration, with no shared physical cell members."""
    valid=cells.analysis_center.to_numpy(bool) if 'analysis_center' in cells else np.ones(len(cells),bool)
    ids=np.flatnonzero((cells.condition.astype(str).to_numpy()==control_label)&valid)
    if len(ids)<200:raise ValueError('Need >=200 control centers')
    xy=cells[['x','y']].to_numpy(float)
    blocks=np.floor((xy[ids]-xy[ids].min(0))/block_um).astype(int)
    keys=np.asarray([f'{a}:{b}' for a,b in blocks]);groups=np.unique(keys)
    if len(groups)<4:raise ValueError('Need >=4 control spatial blocks')
    rng=np.random.default_rng(seed);rng.shuffle(groups)
    calibration=ids[np.isin(keys,groups[:max(1,round(.25*len(groups)))])]
    used=np.unique(members[calibration]);overlap=np.zeros(len(cells),bool);overlap[used]=True
    fit=ids[~overlap[members[ids]].any(1)]
    if min(len(fit),len(calibration))<50:raise ValueError('Too few control centers after physical-cell purge')
    if np.intersect1d(np.unique(members[fit]),np.unique(members[calibration])).size:raise AssertionError('Shared calibration physical cells')
    return fit,calibration


def matched_distance(query, background, query_composition, background_composition,
                     candidates=64, neighbors=5, chunk=1024):
    """Distance in communication space among composition-nearest healthy cells."""
    if len(background)<neighbors:
        raise ValueError('Insufficient healthy background')
    background_sqrt=np.sqrt(np.maximum(background_composition,0))
    tree=cKDTree(background_sqrt);tie_cache={};rng=np.random.default_rng(52173)
    result=np.empty(len(query),np.float32)
    for start in range(0,len(query),chunk):
        stop=min(start+chunk,len(query))
        query_sqrt=np.sqrt(np.maximum(query_composition[start:stop],0))
        comp_distance,ids=tree.query(query_sqrt,
                         k=min(candidates,len(background)))
        ids=np.asarray(ids).reshape(stop-start,-1)
        distance=np.linalg.norm(query[start:stop,None]-background[ids],axis=2)
        result[start:stop]=np.sort(distance,axis=1)[:,:neighbors].mean(1)
        comp_distance=np.asarray(comp_distance).reshape(stop-start,-1)
        # Pure/similar-composition environments can have thousands of tied
        # neighbors. A tree's arbitrary first64 may omit an entire NORMAL
        # communication state. Sample tied candidates across the full stratum,
        # without looking at communication values or case/control identity.
        tied=np.isclose(comp_distance[:,-1],comp_distance[:,0],rtol=0,atol=1e-8)
        for j in np.flatnonzero(tied):
            key=(query_sqrt[j].round(7).tobytes(),round(float(comp_distance[j,-1]),8))
            if key not in tie_cache:
                pool=np.asarray(tree.query_ball_point(query_sqrt[j],float(comp_distance[j,-1])+1e-8),int)
                tie_cache[key]=np.sort(rng.choice(pool,min(512,len(pool)),replace=False))
            pool=tie_cache[key]
            d=np.linalg.norm(query[start+j]-background[pool],axis=1)
            result[start+j]=np.sort(d)[:neighbors].mean()
    return result


def find_case_niches(values, composition, cells, members, weights, seed=52173,
                     block_um=250., quantile=.95, min_cells=60, spatial_support=.5):
    """Use spatially purged healthy blocks to calibrate, then score BOTH sections.

    Candidate clusters are fitted without condition labels; healthy and disease
    use the identical distance threshold and spatial-support rule. No candidate
    is forced to be disease-exclusive, and zero candidates is a valid result.
    """
    values=np.asarray(values,np.float32);composition=np.asarray(composition,np.float32)
    conditions=cells.condition.astype(str).to_numpy();healthy=conditions=='HC'
    ids=np.flatnonzero(healthy)
    if len(ids)<200:raise ValueError('Need >=200 healthy centers for spatial calibration')
    xy=cells[['x','y']].to_numpy(float)
    blocks=np.floor((xy[ids]-xy[ids].min(0))/block_um).astype(int)
    keys=np.asarray([f'{a}:{b}' for a,b in blocks]);groups=np.unique(keys)
    if len(groups)<4:raise ValueError('Need >=4 healthy spatial blocks; reduce block_um')
    rng=np.random.default_rng(seed);rng.shuffle(groups)
    calibration=ids[np.isin(keys,groups[:max(1,round(.25*len(groups)))])]
    used=np.unique(members[calibration]);overlap=np.zeros(len(cells),bool);overlap[used]=True
    fit=ids[~overlap[members[ids]].any(1)]
    if min(len(fit),len(calibration))<50:
        raise ValueError('Spatial purge leaves too few healthy calibration/background centers')
    if np.intersect1d(np.unique(members[fit]),np.unique(members[calibration])).size:
        raise AssertionError('Healthy calibration/background physical cells overlap')
    scores=matched_distance(values,values[fit],composition,composition[fit])
    threshold=float(np.quantile(scores[calibration],quantile))
    # Background cells cannot score their own training neighbors fairly.
    # Score these centers against the held-out calibration half with its own
    # reverse calibration threshold; other centers retain the forward model.
    reverse=matched_distance(values[fit],values[calibration],composition[fit],composition[calibration])
    reverse_threshold=float(np.quantile(reverse,quantile))
    score_ratio=scores/max(threshold,1e-8)
    score_ratio[fit]=reverse/max(reverse_threshold,1e-8)
    raw=score_ratio>1
    support=(raw[members]*weights).sum(1)/np.maximum(weights.sum(1),1e-8)
    candidate=raw&(support>=spatial_support)
    labels=np.zeros(len(cells),np.int16);records=[]
    selected=np.flatnonzero(candidate)
    if len(selected)>=min_cells:
        budget=min(4000,len(selected));fitrows=np.sort(rng.choice(selected,budget,replace=False))
        valid=[]
        for k in range(1,min(5,len(fitrows)//min_cells)+1):
            km=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(values[fitrows])
            predicted=km.predict(values[selected]);counts=np.bincount(predicted,minlength=k)
            score=float(silhouette_score(values[fitrows],km.labels_,sample_size=min(1500,budget),random_state=seed)) if k>1 else -.01
            records.append({'K':k,'silhouette':score,'minimum_cells':int(counts.min()),'eligible':bool(counts.min()>=min_cells)})
            if counts.min()>=min_cells:valid.append((score,k,predicted))
        if valid:
            _,k,predicted=max(valid,key=lambda x:(x[0],-x[1]));labels[selected]=predicted+1
    summary=[]
    for label in np.unique(labels[labels>0]):
        h=float((labels[healthy]==label).mean());d=float((labels[~healthy]==label).mean())
        summary.append({'candidate_niche':f'R{label}','HC_fraction':h,'DM_fraction':d,
                        'log2_DM_vs_HC':float(np.log2((d+.001)/(h+.001))),
                        'cells':int((labels==label).sum()),'DM_enriched':bool(d>=2*h and d>=.005)})
    details=pd.DataFrame({'cell_id':cells.cell_id,'condition':conditions,
                          'novelty_score_ratio':score_ratio,'spatial_support':support,
                          'candidate':candidate,'candidate_niche':labels})
    audit={'healthy_fit_centers':len(fit),'healthy_calibration_centers':len(calibration),
           'healthy_fit_rows':fit.tolist(),'healthy_calibration_rows':calibration.tolist(),
           'healthy_physical_cells_disjoint':True,'threshold_quantile':quantile,
           'forward_threshold':threshold,'reverse_threshold':reverse_threshold,
           'spatial_support':spatial_support,'block_um':block_um,
           'calibration_fraction_above_threshold':float((scores[calibration]>threshold).mean()),
           'interpretation':'Single-section healthy background; descriptive novelty, no population inference; identical rules in HC and DM',
           'candidate_clustering_uses_condition':False,'posthoc_score_uses_condition':True,
           'encoder_uses_condition':False}
    return details,pd.DataFrame(summary,columns=['candidate_niche','HC_fraction','DM_fraction','log2_DM_vs_HC','cells','DM_enriched']),pd.DataFrame(records),audit
