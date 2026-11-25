"""Post-fit simulation diagnostics; truth is used only for evaluation."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,anndata as ad
from sklearn.metrics import roc_auc_score
W=Path('/data1/xueshuailin/CCC_Phe/workspace');O=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/pathway_prior_expansion/v3_comparison')
a=ad.read_h5ad(W/'data/tumor_immune_interface_v2_epithelial_majority/tissue.h5ad',backed='r');reg=a.obs.design_region_id.to_numpy();a.file.close()
print('region values',pd.Series(reg).value_counts().sort_index().to_dict(),flush=True)
# Fixed A/B values are evaluation labels, never used for feature selection or training.
for mode,path in [('anywhere',O/'tumor'),('within',O/'reaction_in_pathway/tumor'),('anywhere_dedup',O/'reaction_anywhere_dedup/tumor'),('strict_dedup',O/'reaction_in_pathway_dedup/tumor')]:
 f=np.load(path/'supplementary_view.npz');x=f['x'];m=f['mask'].astype(bool);groups=[]
 for col in range(x.shape[1]):groups.append(hashlib.sha256(x[:,col].tobytes()+m[:,col].tobytes()).hexdigest())
 n=len(groups);unique=len(set(groups));effects=[];aucs=[]
 for j in range(n):
  va=(reg==2)&m[:,j];vb=(reg==3)&m[:,j]
  if va.sum()<25 or vb.sum()<25:continue
  aa=x[va,j];bb=x[vb,j];sd=np.sqrt((aa.var()+bb.var())/2)
  if sd>1e-8:effects.append(abs((aa.mean()-bb.mean())/sd));aucs.append(abs(roc_auc_score(np.r_[np.zeros(len(aa)),np.ones(len(bb))],np.r_[aa,bb])-.5)*2)
 info=pd.read_csv(path/'new_programs.csv');counts=info.mapped_LR.value_counts().sort_index()
 print(json.dumps({'mode':mode,'features':n,'exact_unique_vectors':unique,'one_lr_features':int((info.mapped_LR==1).sum()),'observed_valid_AB_features':len(effects),'median_abs_standardized_AB_difference':float(np.median(effects)) if effects else None,'fraction_AB_difference_gt_1':float(np.mean(np.array(effects)>1)) if effects else None,'mean_AB_auc_separation':float(np.mean(aucs)) if aucs else None}),flush=True)
