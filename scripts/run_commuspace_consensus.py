"""Apply CommuSpace's public consensus procedure to its saved embedding."""
import sys,json,time
from pathlib import Path
import numpy as np,anndata as ad
from sklearn.metrics import adjusted_rand_score
root=Path('/data1/xueshuailin/CCC_Phe/workspace');out=root/'outputs/tumor_immune_interface/v2_baseline_k5/commuspace'
sys.path[:0]=[str(root/'artifacts/third_party/python'),str(root/'artifacts/third_party/CommuSpace')]
import CommuSpace as cs
a=ad.read_h5ad(root/'data/tumor_immune_interface_v2_epithelial_majority/tissue.h5ad')
embedding=np.load(out/'embedding.npy');print('native consensus, K=5, cells',len(embedding),flush=True);start=time.time()
_,_,_,_,labels=cs.utils.louvain_clustering_O_CC_large_data(a,embedding,[5],mbk_n_clusters=8000,min_meta_size=1,interval=.01,times_random_running=5)
labels=np.unique(np.asarray(labels[5]),return_inverse=True)[1];np.save(out/'labels.npy',labels)
truth=np.array([0,1,2,3,4,4])[a.obs.design_region_id.to_numpy()];info=json.loads((out/'complete.json').read_text());info.update(clustering='native CommuSpace consensus Louvain, hierarchical cut K=5',global_ARI=float(adjusted_rand_score(truth,labels)),tumor_ARI=float(adjusted_rand_score(truth[truth!=0],labels[truth!=0])),consensus_seconds=time.time()-start,n_clusters=int(len(np.unique(labels))))
(out/'complete.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2),flush=True)
sys.path.insert(0,'/home/xueshuailin/CCC_Phe/scripts');from render_communication_baselines import render;render()
