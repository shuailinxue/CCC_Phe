import numpy as np
import igraph as ig
import leidenalg
from scipy import sparse

def leiden(graph,seed=40700,resolution=1.):
    upper=sparse.triu(graph,k=1).tocoo()
    g=ig.Graph(n=graph.shape[0],edges=list(zip(upper.row.tolist(),upper.col.tolist())),directed=False)
    p=leidenalg.find_partition(g,leidenalg.RBConfigurationVertexPartition,weights=upper.data.tolist(),
                              resolution_parameter=resolution,seed=seed,n_iterations=2)
    return np.asarray(p.membership,dtype=np.int32)
