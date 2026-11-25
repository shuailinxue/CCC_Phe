"""Prepare official STCase LR inventory for per-cell-type comparison.

This is preparation only; it does not claim to be a trained STCase baseline.
"""
import json,hashlib
from pathlib import Path
import anndata as ad,numpy as np,pandas as pd
root=Path('/data1/xueshuailin/CCC_Phe/workspace')
out=root/'outputs/tumor_immune_interface/v2_baseline_k5/stcase_fair';out.mkdir(parents=True,exist_ok=True)
source=root/'data/tumor_immune_interface_v2_epithelial_majority/tissue.h5ad'
db=root/'artifacts/third_party/STCase/STCaseDB'
a=ad.read_h5ad(source,backed='r');genes=set(a.var_names)
network=pd.read_csv(db/'STCaseDB_Human.csv',index_col=0)
complexes=pd.read_csv(db/'STCaseDB_Human_Complex.csv',index_col=0)
def parts(x):
 if x in complexes.index:
  v=complexes.loc[x];v=v.iloc[0] if isinstance(v,pd.DataFrame) else v
  return [str(y) for y in v.dropna()]
 return [str(x)]
rows=[]
for key,r in network.iterrows():
 lig,rec=parts(r['ligand']),parts(r['receptor'])
 if set(lig+rec)<=genes:
  rows.append({'id':str(key),'ligand':r['ligand'],'receptor':r['receptor'],'ligand_subunits':'|'.join(lig),'receptor_subunits':'|'.join(rec),'annotation':r['ligand_annotation']})
pd.DataFrame(rows).to_csv(out/'panel_covered_official_lr.csv',index=False)
groups=a.obs.cell_type.astype(str).value_counts().to_dict()
# Only after the input inventory is frozen: evaluation support, not training input.
truth=np.array([0,1,2,3,4,4])[a.obs.design_region_id.to_numpy()]
support={}
for ct in groups:
 mask=a.obs.cell_type.astype(str).to_numpy()==ct
 support[ct]={'cells':int(mask.sum()),'truth_counts':np.bincount(truth[mask],minlength=5).tolist()}
manifest={'phase':'input_prepared_training_not_started','n_cells':int(a.n_obs),'n_genes':int(a.n_vars),'official_database_pairs_with_complete_subunits':len(rows),'official_commit':'33f62709da8265c1abc774cc3b0624f3c6c9d12a','cell_type_counts':groups,'evaluation_support':support,'truth_used_for_selection_or_training':False,'required_before_scoring':['upstream CCC background significance and distance rules','type-specific STCase training and native clustering','same-cell subset evaluation'],'truth_support_computed_posthoc_only':True}
(out/'preflight.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
a.file.close()
