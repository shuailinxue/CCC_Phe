"""ST1 V2 composition clustering only. User may change K before a new run."""
from pathlib import Path
import json,numpy as np
from sklearn.cluster import KMeans
O=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST1/v2');f=Path(json.loads((O/'target_cache_01.json').read_text())['folder']);x=np.load(f/'composition.npy',mmap_mode='r');model=KMeans(n_clusters=4,n_init=10,random_state=40700);labels=model.fit_predict(np.sqrt(np.maximum(x,0)));np.save(O/'coarse_K4.npy',labels);means=np.stack([np.asarray(x[labels==i]).mean(0) for i in range(4)]);np.save(O/'coarse_K4_centroids.npy',means);print('coarse K=4 n:',np.bincount(labels).tolist(), 'centroids:', means.tolist(),flush=True)
