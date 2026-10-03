import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from pynndescent import NNDescent

def knn_graph(embedding,k=15,seed=40700):
    x=np.asarray(embedding,dtype=np.float32);n=len(x);k=min(k,n-1)
    index=NNDescent(x,n_neighbors=k+1,random_state=seed,n_jobs=4,low_memory=True)
    ids,dist=index.neighbor_graph
    rows=np.repeat(np.arange(n),ids.shape[1]);cols=ids.ravel();d=dist.ravel();keep=(rows!=cols)&(cols>=0)
    scale=np.maximum(np.median(dist[:,1:],axis=1),1e-6)
    weight=np.exp(-d[keep]/scale[rows[keep]])
    g=sparse.csr_matrix((weight,(rows[keep],cols[keep])),shape=(n,n));g=g.maximum(g.T);g.setdiag(0);g.eliminate_zeros()
    return g

def fuse_graphs(ccc,comp,alpha=.2):
    if not 0<=alpha<=.3:raise ValueError('Weak composition fusion alpha must be 0..0.3')
    # Common total edge-mass normalization makes alpha meaningful and symmetric.
    ccc=ccc/(ccc.sum()/ccc.shape[0]);comp=comp/(comp.sum()/comp.shape[0])
    return (ccc+alpha*comp).tocsr()
