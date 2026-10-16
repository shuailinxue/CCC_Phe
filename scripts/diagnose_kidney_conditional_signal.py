"""Post-fit kidney section discrimination within PT-dominated neighborhoods.

Diagnostic only: no patient-level validation, no feedback to GNN or niche K.
Spatially purged probes expose composition/depth confounding explicitly.
"""
import sys,json,numpy as np,pandas as pd,torch
from pathlib import Path
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from phenoniche.gnn.data import spatial_split
from phenoniche.gnn.communication_metric import communication_embedding
torch.set_num_threads(8)
parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);r=parser.parse_args().source;f=pd.read_csv(r/'assignments.csv.gz',low_memory=False);m=np.load(r/'input/members.npy');comp=np.load(r/'input/composition.npy');features=pd.read_csv(r/'input/features.csv');types=json.load(open(r/'input/audit.json'))['cell_types'];valid=f.analysis_center.to_numpy();label=(f.condition=='DKD').to_numpy();pure=(comp[:,types.index('PT')]>=.9)&valid
# Spatially purged diagnostic classification, never niche fitting or K selection.
xy=f[['x','y']].to_numpy();keys=np.array([s+':'+str(int(x//250))+':'+str(int(y//250)) for s,(x,y) in zip(f.image,xy)]);rows,fold,_=spatial_split(xy,m,seed=52173,samples=keys)
rng=np.random.default_rng(52173);use=[]
for role in [0,1]:
 ids=[]
 for group in [0,1]:
  pool=np.flatnonzero(pure&(fold==role)&(label==group));ids.extend(rng.choice(pool,min(1000,len(pool)),replace=False))
 use.append(np.sort(ids))
print('Pure PT counts',pd.crosstab(f.condition,pure).to_string(),[len(x) for x in use],flush=True)
if min([len(x) for x in use])<50:raise ValueError('Insufficient pure PT diagnostic cells')
a,b=use;allids=np.r_[a,b];C=np.load(r/'input/ccc.npy',mmap_mode='r');ff=np.flatnonzero((features.sender=='PT')&(features.receiver=='PT'));raw=np.asarray(C[allids][:,ff]);raw=raw/np.maximum(raw.sum(1,keepdims=True),1e-8);raw=np.log1p(raw*1000)
p=np.load(r/'gnn/P.npy');h=np.load(r/'gnn/H.npy');act=np.load(r/'gnn/activity.npy');v,_,_=communication_embedding(p,act,h,np.load(r/'input/preprocessing.npz')['feature_scale'],np.arange(len(f)),activity_weight=0)
sets={'composition':comp[allids],'RNA_depth':np.log1p(f.library_counts.to_numpy()[allids,None]),'observed_PT_PT_relative_CCC':raw,'GNN_relative_communication':v[allids]};records=[]
for name,x in sets.items():
 scaler=StandardScaler().fit(x[:len(a)]);clf=LogisticRegression(C=.1,class_weight='balanced',max_iter=1000).fit(scaler.transform(x[:len(a)]),label[a]);prob=clf.predict_proba(scaler.transform(x[len(a):]))[:,1];records.append({'diagnostic_representation':name,'section_classification_AUC':roc_auc_score(label[b],prob),'train':len(a),'spatially_purged_probe':len(b),'scope':'One paired-section diagnostic; not patient-level generalization or disease causal evidence'})
out=r/'diagnostics';out.mkdir(exist_ok=True);pd.DataFrame(records).to_csv(out/'pure_PT_signal.csv',index=False);print(pd.DataFrame(records).to_string(index=False),flush=True)
