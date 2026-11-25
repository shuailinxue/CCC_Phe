"""Post-fit nearest-neighbor audit against all cells, including background."""
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad
from sklearn.neighbors import NearestNeighbors
W=Path('/data1/xueshuailin/CCC_Phe/workspace');O=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/pathway_prior_expansion/v3_comparison')
a=ad.read_h5ad(W/'data/tumor_immune_interface_v2_epithelial_majority/tissue.h5ad',backed='r');y=a.obs.design_region_id.to_numpy();a.file.close();rng=np.random.default_rng(7)
ids=np.concatenate([rng.choice(np.flatnonzero(y==i),min(400,(y==i).sum()),replace=False) for i in [0,1,2,3,4,5]])
rows=[]
for mode,p in [('original',W/'outputs/tumor_immune_interface/v3_full_tissue_trained/h.npy'),('within',O/'reaction_in_pathway/tumor/h.npy'),('anywhere_dedup',O/'reaction_anywhere_dedup/tumor/h.npy'),('strict_dedup',O/'reaction_in_pathway_dedup/tumor/h.npy')]:
 h=np.load(p);ix=NearestNeighbors(n_neighbors=16,n_jobs=3).fit(h).kneighbors(h[ids],return_distance=False);ix=np.array([[v for v in row if v!=i][:15] for row,i in zip(ix,ids)]);same=y[ix]==y[ids,None]
 d={'model':mode};d.update({f'{x}_same':float(same[y[ids]==i].mean()) for x,i in [('BG',0),('Core',1),('A',2),('B',3)]});rows.append(d)
pd.DataFrame(rows).to_csv(O/'tumor_latent_full_knn_diagnostics.csv',index=False);print(pd.DataFrame(rows).to_string(index=False),flush=True)
