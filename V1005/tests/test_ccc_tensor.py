import numpy as np
from scipy import sparse
from phenoniche.v1005.ccc_tensor import aggregate_pairwise_ccc,training_only,entry_holdout

def test_pairwise_bruteforce_direction_opportunity():
    xy=np.array([[0.,0.],[1.,0.],[2.,1.]],np.float32);ct=np.array([0,0,1])
    l=np.array([[1,9],[4,1],[16,4]],np.float32);r=np.array([[9,1],[1,4],[4,16]],np.float32)
    ids=np.array([[0,1,2],[0,0,2]]);w=np.array([[1,.5,.2],[1,.5,.2]],np.float32)
    p,x,o=aggregate_pairwise_ccc(xy,ct,l,r,2,1.,members=ids,anchor_weights=w,batch_size=1,lr_batch_size=1)
    expected=np.zeros((2,2,2,2));opp=np.zeros((2,2,2))
    for a in range(2):
        for i,u in enumerate(ids[a]):
            for j,v in enumerate(ids[a]):
                if u==v:continue
                weight=w[a,i]*w[a,j]*np.exp(-np.sum((xy[u]-xy[v])**2)/2)
                opp[a,ct[u],ct[v]]+=weight
                expected[a,ct[u],ct[v]]+=weight*np.sqrt(l[u]*r[v])
    expected/=opp[:,:,:,None]+1e-4
    np.testing.assert_allclose(x,expected.reshape(2,-1),rtol=2e-6)
    np.testing.assert_allclose(o,opp.reshape(2,-1),rtol=2e-6)
    np.testing.assert_allclose(p.sum(1),1)
    assert o[0,0]>0 and o[1,0]==0
    assert not np.allclose(x[0,2:4],x[0,4:6])

def test_chunked_mask_and_holdout():
    xy=np.array([[0.,0.],[1.,0.],[2.,1.]]);ct=np.array([0,0,1]);l=np.ones((3,2),np.float32)
    ids=np.tile(np.arange(3),(3,1));mask=np.array([True,False,False,True,True,False,False,True])
    _,full,_=aggregate_pairwise_ccc(xy,ct,l,l,2,1.,members=ids)
    _,sub,_=aggregate_pairwise_ccc(xy,ct,l,l,2,1.,members=ids,feature_mask=mask,batch_size=1,lr_batch_size=1)
    np.testing.assert_allclose(sub,full[:,mask])
    x=sparse.csr_matrix(np.arange(60).reshape(6,10));train=training_only(x)
    m=entry_holdout(np.arange(6)[:,None],np.arange(10)[None])
    assert np.all(train.toarray()[m]==0)
    np.testing.assert_array_equal(train.toarray()[~m],x.toarray()[~m])

def test_complex_expression_row_blocks_identical():
    from phenoniche.v1002.lr_atlas import LRAtlas,LRInteraction
    from phenoniche.v1005.ccc_tensor import _side_expression
    rows=(LRInteraction('a','A+B','C',('A','B'),('C',)),LRInteraction('b','C','A+B',('C',),('A','B')))
    atlas=LRAtlas(rows,{},'fixture');x=np.arange(1,31,dtype=np.float32).reshape(10,3)
    full=_side_expression(x,['A','B','C'],atlas)
    blocks=[_side_expression(x[i:i+3],['A','B','C'],atlas) for i in range(0,len(x),3)]
    for j in range(2):np.testing.assert_array_equal(full[j],np.concatenate([b[j] for b in blocks]))
    np.testing.assert_allclose(full[0][:,0],np.sqrt(x[:,0]*x[:,1]))
