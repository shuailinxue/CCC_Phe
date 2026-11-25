"""Compare trained h after fitting, labels used only for diagnostics."""
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
W=Path('/data1/xueshuailin/CCC_Phe/workspace');O=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/pathway_prior_expansion/v3_comparison')
a=ad.read_h5ad(W/'data/tumor_immune_interface_v2_epithelial_majority/tissue.h5ad',backed='r');y=a.obs.design_region_id.to_numpy();a.file.close();ids=np.where(np.isin(y,[1,2,3]))[0];labs=y[ids];rng=np.random.default_rng(17);sub=np.sort(rng.choice(len(ids),min(3500,len(ids)),replace=False));rows=[]
for mode,p in [('original',W/'outputs/tumor_immune_interface/v3_full_tissue_trained/h.npy'),('anywhere',O/'tumor/h.npy'),('within',O/'reaction_in_pathway/tumor/h.npy'),('anywhere_dedup',O/'reaction_anywhere_dedup/tumor/h.npy'),('strict_dedup',O/'reaction_in_pathway_dedup/tumor/h.npy')]:
 h=np.load(p)[ids];neighbors=NearestNeighbors(n_neighbors=16,n_jobs=3).fit(h).kneighbors(return_distance=False)[:,:15];d={'model':mode,'silhouette_core_AB':silhouette_score(h[sub],labs[sub]),'A_same_15nn':float((labs[neighbors[labs==2]]==2).mean()),'B_same_15nn':float((labs[neighbors[labs==3]]==3).mean()),'core_same_15nn':float((labs[neighbors[labs==1]]==1).mean())};rows.append(d)
pd.DataFrame(rows).to_csv(O/'tumor_latent_diagnostics.csv',index=False);print(pd.DataFrame(rows).to_string(index=False),flush=True)
