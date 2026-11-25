"""Prioritize communication splits under a whole-tissue niche budget."""
import os,pickle,tempfile
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from lung3d_v2 import DATA,OUT
from phenoniche.spatial import Settings
from phenoniche.spatial.subclasses import partition_contexts

TARGET=OUT.parent/"v2_bidirectional"

def run():
    data=ad.read_h5ad(DATA/"preprocessed_objects/preprocessed_objects/cosmx.h5ad",backed="r")
    coarse=np.load(TARGET/"coarse_resolution_0.40.npy")
    groups,_=partition_contexts(load_npz(OUT/"contacts.npz"),coarse,Settings())
    paths=list((TARGET/"checkpoints").glob("*/context_*.pkl"))
    mapping={(int(p.stem.split("_")[1]),int(p.stem.split("_")[2])):p for p in paths}
    if len(mapping)!=len(paths): raise ValueError("Multiple incompatible checkpoints")
    labels=np.array([f"C{x}" for x in coarse],dtype=object)
    context=np.empty(len(coarse),dtype=object)
    status=np.full(len(coarse),"pending",dtype=object)
    candidates=[]
    for key,ids in groups.items():
        a,b=key
        parent=f"C{a}" if a==b else f"I{a}_{b}"
        context[ids]=parent
        if key not in mapping: continue
        status[ids]="complete"
        with mapping[key].open("rb") as f: part=pickle.load(f)
        if not np.array_equal(part["ids"],ids): raise ValueError("Checkpoint IDs do not match")
        if not part["feature_scores"] or len(ids)<100: continue
        features=part["feature_scores"][0]
        scale=np.maximum(np.quantile(features,.9,axis=0),1e-7)
        x=np.log1p(features/scale).astype(np.float32)
        rng=np.random.default_rng(73)
        selected=rng.choice(len(x),min(len(x),5000),replace=False)
        model=KMeans(n_clusters=2,n_init=8,random_state=73).fit(x[selected])
        y=model.predict(x)
        sizes=np.bincount(y,minlength=2)
        if sizes.min()<max(30,len(ids)*.02): continue
        validation=rng.choice(len(x),min(len(x),1500),replace=False)
        quality=silhouette_score(x[validation],y[validation]) if len(np.unique(y[validation]))==2 else -1
        if quality<=0: continue
        # The smaller subgroup gets a new label; both use the same area-level CP scores.
        target=int(np.argmin(sizes))
        candidates.append((float(quality),key,ids[y==target],len(ids),int(sizes[target]),part["stability"]))
    budget=20-len(np.unique(coarse))
    candidates.sort(key=lambda r:(-r[0],r[1]))
    details=[]
    for number,(quality,key,ids,total,small,stability) in enumerate(candidates):
        chosen=number<budget
        parent=f"C{key[0]}" if key[0]==key[1] else f"I{key[0]}_{key[1]}"
        if chosen: labels[ids]=f"F{number+1}"
        details.append((parent,total,small,quality,stability,chosen))
    output=TARGET/"limit20"
    output.mkdir(exist_ok=True)
    cells=pd.DataFrame(dict(coarse=coarse,context=context,niche=labels,status=status),index=data.obs_names)
    cells.index.name="cell_id"
    fd,tmp=tempfile.mkstemp(dir=output,prefix="cells_",suffix=".partial")
    with os.fdopen(fd,"w") as f: cells.to_csv(f,sep="\t")
    os.replace(tmp,output/"cells.tsv")
    pd.DataFrame(details,columns=["context","cells","split_cells","silhouette","original_stability","selected"]).to_csv(
        output/"splits.tsv",sep="\t",index=False)
    print(f"{len(mapping)}/{len(groups)} areas complete; {sum(status=='pending'):,} pending; {cells.niche.nunique()} final niches",flush=True)

if __name__=="__main__":run()
