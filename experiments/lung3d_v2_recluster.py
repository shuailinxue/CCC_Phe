"""Merge saved consensus clusters into at most four communication subclasses per context."""
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import load_npz
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from lung3d_v2 import DATA, OUT
import anndata as ad
from phenoniche.spatial import Settings
from phenoniche.spatial.subclasses import partition_contexts

TARGET = OUT.parent / "v2_bidirectional"
CHECKPOINTS = TARGET / "checkpoints"


def merge(part, maximum=4):
    labels = part["labels"]
    feature = np.asarray(part["feature_scores"][0], dtype=np.float32) if part["feature_scores"] else None
    classes, inverse, count = np.unique(labels,return_inverse=True,return_counts=True)
    if feature is None or len(classes) <= maximum:
        return inverse
    # Each centroid is in the exact transformed feature space used by Leiden.
    scale = np.maximum(np.quantile(feature,.9,axis=0),1e-7)
    x = np.log1p(feature/scale)
    centers = np.zeros((len(classes), x.shape[1]),dtype=np.float64)
    np.add.at(centers,inverse,x)
    centers /= count[:,None]
    if not np.any(pdist(centers)):
        return np.zeros(len(labels),dtype=int)
    merged = fcluster(linkage(centers,method="ward"),t=maximum,criterion="maxclust")
    return np.unique(merged[inverse],return_inverse=True)[1]


def run():
    data=ad.read_h5ad(DATA/"preprocessed_objects/preprocessed_objects/cosmx.h5ad",backed="r")
    labels=np.load(TARGET/"coarse_resolution_0.40.npy")
    graph=load_npz(OUT/"contacts.npz")
    groups,_=partition_contexts(graph,labels,Settings())
    roots=list(CHECKPOINTS.glob("*/context_*.pkl"))
    paths={(int(p.stem.split("_")[1]),int(p.stem.split("_")[2])):p for p in roots}
    if len(paths)!=len(roots):
        raise ValueError("Multiple checkpoint protocols found")
    niche=np.empty(len(labels),dtype=object)
    context=np.empty(len(labels),dtype=object)
    status=np.empty(len(labels),dtype=object)
    details=[]
    for key,ids in groups.items():
        a,b=key
        parent=f"C{a}" if a==b else f"I{a}_{b}"
        context[ids]=parent
        if key not in paths:
            niche[ids]=parent+".pending"
            status[ids]="pending"
            continue
        with paths[key].open("rb") as f:
            part=pickle.load(f)
        if not np.array_equal(part["ids"],ids):
            raise ValueError(f"Cell mismatch for {key}")
        merged=merge(part)
        niche[ids]=[f"{parent}.{v+1}" for v in merged]
        status[ids]="completed"
        details.append((parent,len(ids),len(np.unique(part["labels"])),len(np.unique(merged)),part["stability"]))
    output=TARGET/"recluster_fast"
    output.mkdir(exist_ok=True)
    cells=pd.DataFrame({"coarse":labels,"context":context,"niche":niche,"status":status},index=data.obs_names)
    cells.index.name="cell_id"
    fd,tmp=tempfile.mkstemp(dir=output,prefix="cells_",suffix=".partial")
    with os.fdopen(fd,"w") as f:
        cells.to_csv(f,sep="\t")
    os.replace(tmp,output/"cells.tsv")
    table=pd.DataFrame(details,columns=["context","cells","original_classes","merged_classes","original_stability"])
    table.to_csv(output/"contexts.tsv",sep="\t",index=False)
    print(f"{len(details)}/{len(groups)} contexts, {sum(status=='pending'):,} pending cells; {cells.niche.nunique()} labels",flush=True)
    print(table.to_string(index=False),flush=True)


if __name__=="__main__":
    run()
