"""Compute KNN30 only in native coordinates; no distance weights or assumed units."""
from pathlib import Path
import json
import anndata as ad,numpy as np
from scipy.spatial import cKDTree
D=Path('/data1/xueshuailin/CCC_Phe/workspace/data/nsclc_zhencaiwei/processed/ST1');a=ad.read_h5ad(D/'ST1_counts.h5ad',backed='r');xy=np.asarray(a.obsm['spatial_native'],dtype=np.float64);a.file.close()
if not np.isfinite(xy).all():raise ValueError('Spatial coordinates must be finite')
d,i=cKDTree(xy).query(xy,k=30,workers=8);rows=np.arange(len(xy));j=np.argmax(i==rows[:,None],axis=1)
if not np.all(i[rows,j]==rows):raise ValueError('Center absent from 30 nearest cells')
first=i[:,0].copy();i[:,0]=i[rows,j];i[rows,j]=first;dv=d[:,0].copy();d[:,0]=d[rows,j];d[rows,j]=dv
np.save(D/'members_knn30.npy',i.astype('int32'));np.save(D/'distance_knn30_native.npy',d.astype('float32'))
stats={'cells':len(xy),'members_including_center':30,'units':'RDS native; unverified','nearest_nonself_median_native':float(np.median(np.min(np.where(i!=rows[:,None],d,np.inf),axis=1))),'radius30_native_quantiles':np.quantile(d.max(axis=1),[.1,.5,.9]).tolist()};(D/'knn30_audit.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2),flush=True)
