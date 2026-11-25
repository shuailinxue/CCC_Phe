"""Audit detection thresholds on actual ST1 RNA without using disease labels."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,anndata as ad
from scipy import sparse
R=Path('/home/xueshuailin/CCC_Phe');sys.path.insert(0,str(R/'src'))
from phenoniche.ccc.lr_atlas import load_lr_atlas,canonicalize_complex, LRInteraction
D=Path('/data1/xueshuailin/CCC_Phe/workspace/data/nsclc_zhencaiwei/processed/ST1');O=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST1/v2');a=ad.read_h5ad(D/'ST1_counts.h5ad');types=pd.Categorical(a.obs['assign.copyKAT'].astype(str)).codes;counts=np.bincount(types);x=sparse.csr_matrix(a.X);panel=set(a.var_names);pos={g:i for i,g in enumerate(a.var_names)};atlas=load_lr_atlas(R/'data/commuspace_human_lr_atlas.tsv');links=[v for v in atlas.interactions if set(v.ligand_components+v.receptor_components)<=panel];db=pd.read_csv('/data1/xueshuailin/CCC_Phe/workspace/outputs/tumor_immune_interface/v2_cellphonedb_v5/cellphonedb_interaction_input.csv').fillna('');db=db[(db.directionality=='Ligand-Receptor')&(db.classification!='')];paths={}
for row in db.itertuples(index=False):
 try:l,r=row.interactors.split('-',1);paths.setdefault((canonicalize_complex(l),canonicalize_complex(r)),set()).add(row.classification)
 except (ValueError,AttributeError):pass
seen={(v.ligand_components,v.receptor_components) for v in links}
for key in sorted(paths):
 if key not in seen and set(key[0]+key[1])<=panel:
  import hashlib
  lid='CPDBv5_'+hashlib.sha256(repr(key).encode()).hexdigest()[:12]
  links.append(LRInteraction(lid,'&'.join(key[0]),'&'.join(key[1]),key[0],key[1]));seen.add(key)
needed=sorted({g for v in links for g in v.ligand_components+v.receptor_components});idx={g:i for i,g in enumerate(needed)};det=(x[:,[pos[g] for g in needed]]>0).astype(np.uint8).tocsc()
# Cache gene detection vectors, so complex subunits are evaluated on the same cells.
cache={}
def coverage(components):
 if components not in cache:
  mask=np.ones(len(a),bool)
  for gene in components:
   col=det.getcol(idx[gene]);v=np.zeros(len(a),bool);v[col.indices]=True;mask &= v
  cache[components]=np.bincount(types,weights=mask,minlength=len(counts))/counts
 return cache[components]
rows=[]
for j,v in enumerate(links):
 l=coverage(v.ligand_components);r=coverage(v.receptor_components);path=paths.get((v.ligand_components,v.receptor_components),set());rows.append((v.lr_id,v.ligand,v.receptor,float(l.max()),float(r.max()),'|'.join(sorted(path))))
 if (j+1)%1000==0:print('Audited',j+1,'/',len(links),flush=True)
t=pd.DataFrame(rows,columns=['lr_id','ligand','receptor','max_sender_detection','max_receiver_detection','cellphonedb_classification']);t.to_csv(O/'lr_detection_threshold_audit.csv',index=False)
for threshold in [.10,.05,.03,.02,.01,.005]:
 v=t[(t.max_sender_detection>=threshold)&(t.max_receiver_detection>=threshold)];c=v[v.cellphonedb_classification!=''];print(json.dumps({'threshold':threshold,'eligible_lr':len(v),'cellphonedb_lr':len(c),'cellphonedb_classes':len({s for x in c.cellphonedb_classification for s in x.split('|')})}),flush=True)
