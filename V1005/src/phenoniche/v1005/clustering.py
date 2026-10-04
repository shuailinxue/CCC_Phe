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

def leiden_sweep(graph,resolutions,seed=40700):
    """Reuse one igraph object for an auditable resolution sweep."""
    upper=sparse.triu(graph,k=1).tocoo()
    edges=zip(upper.row.tolist(),upper.col.tolist())
    g=ig.Graph(n=graph.shape[0],edges=edges,directed=False)
    weights=upper.data.astype(float).tolist();labels=[];qualities=[]
    for resolution in resolutions:
        p=leidenalg.find_partition(g,leidenalg.RBConfigurationVertexPartition,weights=weights,
                                  resolution_parameter=float(resolution),seed=seed,n_iterations=2)
        labels.append(np.asarray(p.membership,dtype=np.int32))
        qualities.append({'leiden_objective':float(p.quality()),
                          'modularity':float(g.modularity(p.membership,weights=weights))})
    return labels,qualities
