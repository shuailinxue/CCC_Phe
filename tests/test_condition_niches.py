"""Outcome-blind eligibility and local exceptional-region tests."""
import numpy as np,pandas as pd
from phenoniche.cohorts.feature_scope import eligible_ccc_features
from phenoniche.cohorts.condition_niches import spatial_region_filter,discover_condition_niches,composition_local_threshold
from phenoniche.gnn.data import neighborhoods


def test_sample_union_retains_within_section_signal_without_cross_section_LR():
    # A small sample has 20% sender coverage, diluted below 10% when pooled.
    types=np.tile(np.repeat([0,1],100),2);groups=np.repeat(['A','B'],200)
    ligand=np.zeros((400,2));receptor=ligand.copy()
    ligand[200:220,0]=1;receptor[300:,0]=1
    # Second LR: ligand ONLY in A, receptor ONLY in B; no same-slice evidence.
    ligand[:100,1]=1;receptor[300:,1]=1
    opp=np.ones((400,4))
    pooled,_=eligible_ccc_features(ligand,receptor,types,groups,opp,groups,2,.15,.01,'pooled_training')
    union,_=eligible_ccc_features(ligand,receptor,types,groups,opp,groups,2,.15,.01,'sample_union')
    assert not pooled[1,0] and union[1,0]
    assert pooled[1,1] and not union[1,1]


def test_region_rule_cannot_join_sections_or_keep_scattered_points():
    xy=np.vstack([np.column_stack([np.arange(20)*10,np.zeros(20)])]*2)
    section=np.repeat(['A','B'],20);members=np.tile(np.arange(40),(40,1))
    mask=np.zeros(40,bool);mask[:5]=True;mask[20:35]=True
    result=spatial_region_filter(mask,members,xy,section,minimum=10,max_distance=11)
    assert not result[:20].any() and result[20:35].all()


def test_conditional_discovery_recovers_known_same_composition_state(tmp_path):
    grid=np.array([(x,y) for x in range(40) for y in range(25)],float)*30;n=len(grid)
    rng=np.random.default_rng(41);base=rng.uniform(.02,.12,(n,3));base[:,0]+=.6*(grid[:,0]<600);base[:,1]+=.6*(grid[:,0]>=600);base/=base.sum(1,keepdims=True)
    p=np.vstack([base,base]);patch=(grid[:,0]>=240)&(grid[:,0]<=570)&(grid[:,1]>=180)&(grid[:,1]<=450)
    p[n+np.flatnonzero(patch)]=[.1,.1,.8]
    m,w=neighborhoods(grid);members=np.vstack([m,m+n]);weights=np.vstack([w,w])
    cells=pd.DataFrame({'cell_id':[f'c{i}' for i in range(2*n)],'condition':['NL']*n+['AD']*n,'image':['A']*n+['B']*n,
        'x':np.tile(grid[:,0],2),'y':np.tile(grid[:,1],2)})
    comp=np.full((2*n,2),.5)
    detail,summary=discover_condition_niches(cells,p,np.ones(2*n),np.eye(3),np.ones(3),comp,members,weights,tmp_path,minimum_region=30)
    assert (detail.iloc[n:].loc[patch,'novel_niche']>0).mean()>.5
    assert (summary.conditional_communication_fraction>.5).any()
    assert detail.composition_common_support.all()
    assert not detail.iloc[:n].novel_niche.any()
    # The exact same observed states in both groups cannot imply case novelty.
    null_p=np.vstack([base,base]);null,_=discover_condition_niches(cells,null_p,np.ones(2*n),np.eye(3),np.ones(3),comp,members,weights,tmp_path/'null',minimum_region=30)
    assert not null.novel_niche.any()
    local,_=discover_condition_niches(cells,p,np.ones(2*n),np.eye(3),np.ones(3),comp,members,weights,tmp_path/'local',minimum_region=30,calibration_mode='composition_local')
    assert (local.iloc[n:].loc[patch,'novel_niche']>0).mean()>.5
    local_null,_=discover_condition_niches(cells,null_p,np.ones(2*n),np.eye(3),np.ones(3),comp,members,weights,tmp_path/'local_null',minimum_region=30,calibration_mode='composition_local')
    assert not local_null.novel_niche.any()
    cells['analysis_center']=True
    cells.loc[np.r_[np.arange(n-80,n),np.arange(2*n-80,2*n)],'analysis_center']=False
    halo_detail,halo_summary=discover_condition_niches(cells,p,np.ones(2*n),np.eye(3),np.ones(3),comp,members,weights,tmp_path/'halo',minimum_region=30,calibration_mode='composition_local')
    assert not halo_detail.loc[~cells.analysis_center,'novel_niche'].any()
    np.testing.assert_allclose(halo_summary.case_fraction,halo_summary.case_cells/(n-80))


def test_local_threshold_does_not_mix_heterogeneous_healthy_compositions():
    comp=np.vstack([np.tile([1.,0.],(300,1)),np.tile([0.,1.],(300,1))])
    scores=np.r_[np.ones(300),np.full(300,10.)]
    threshold=composition_local_threshold(np.eye(2),comp,scores,.99)
    np.testing.assert_allclose(threshold,[1.,10.])
