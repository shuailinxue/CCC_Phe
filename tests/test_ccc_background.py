import numpy as np
from phenoniche.ccc.background import (FixedPhysicalCCC,depth_strata,profile_permutation,
    nonoverlap_graph,select_features)
from phenoniche.ccc.pairwise import aggregate_pairwise_ccc


def test_fixed_bilinear_CCC_matches_physical_pair_kernel_and_profile_shuffle():
    rng=np.random.default_rng(3);n,t,q=50,3,7
    xy=rng.normal(size=(n,2))*10;types=rng.integers(t,size=n)
    members=np.stack([rng.choice(n,8,replace=False) for _ in range(12)]);members[0,1]=members[0,0]
    weights=rng.random(members.shape).astype('f');l=rng.gamma(2,1,size=(n,q)).astype('f');r=rng.gamma(2,1,size=(n,q)).astype('f')
    mask=rng.random((t*t,q))>.4
    _,expected,opp=aggregate_pairwise_ccc(xy,types,l,r,t,20,members,weights,mask,device='cpu')
    pair,lr=np.divmod(np.flatnonzero(mask.ravel()),q);s,v=np.divmod(pair,t)
    engine=FixedPhysicalCCC(xy,types,members,weights,l,r,lr,s,v,opp)
    np.testing.assert_allclose(engine.evaluate().numpy(),expected,atol=3e-6,rtol=3e-6)
    perm=rng.permutation(n);_,expected,_=aggregate_pairwise_ccc(xy,types,l[perm],r[perm],t,20,members,weights,mask,device='cpu')
    np.testing.assert_allclose(engine.evaluate(perm).numpy(),expected,atol=3e-6,rtol=3e-6)


def test_profile_permutation_is_shared_and_preserves_strata():
    types=np.repeat([0,1],100);totals=np.tile(np.arange(100),2)
    groups=depth_strata(types,totals,5);perm=profile_permutation(groups,np.random.default_rng(4))
    np.testing.assert_array_equal(np.sort(perm),np.arange(len(perm)))
    np.testing.assert_array_equal(groups,groups[perm]);np.testing.assert_array_equal(types,types[perm])
    assert not np.array_equal(perm,np.arange(len(perm)))


def test_statistical_graph_excludes_shared_physical_members():
    from scipy.spatial import cKDTree
    xy=np.array([(i*10.,j*10.) for i in range(10) for j in range(10)])
    members=cKDTree(xy).query(xy,k=5)[1]
    cfg=dict(graph_candidates=100,graph_neighbors=4,graph_max_distance_um=100.,graph_bandwidth_um=40.,composition_bandwidth=.2,spatial_block_um=40.)
    graph,folds=nonoverlap_graph(xy,members,np.tile([.5,.5],(100,1)),cfg)
    u,v=graph.nonzero();assert len(u)>0
    assert all(not set(members[i])&set(members[j]) for i,j in zip(u,v))
    np.testing.assert_allclose(np.asarray(graph.sum(1)).ravel(),1.,atol=2e-7)


def test_screen_requires_both_spatial_folds_and_null_excess():
    rng=np.random.default_rng(8);null=rng.normal(size=(64,3,20));observed=np.zeros((3,20));observed[:,0]=20
    observed[0,1]=20;observed[1,1]=20;observed[2,1]=-10
    cfg=dict(feature_q_threshold=.05,minimum_fold_z=2.)
    keep,z,p,q=select_features(observed,null,cfg)
    assert keep[0] and not keep[1] and keep.sum()==1
