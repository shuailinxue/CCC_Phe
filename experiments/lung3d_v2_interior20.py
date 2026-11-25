"""Communication programs fitted independently in each complete composition class."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from multiprocessing import get_context
import os,pickle,tempfile
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from lung3d_v2 import DATA,OUT,ROOT
from phenoniche.spatial import Settings
from phenoniche.spatial.channels import Channels
from phenoniche.spatial.cell_programs import CellTensor,pool_programs
from phenoniche.spatial.programs import fit_programs,project

TARGET=OUT.parent/"v2_bidirectional"
RUN=TARGET/"interior20"
CFG=replace(Settings(),fit_windows=512,iterations=40)
def task(key):
    ids=np.flatnonzero(COARSE==key)
    target=RUN/f"class_{key}.pkl"
    if target.exists():
        with target.open('rb') as stream: return key,pickle.load(stream)
    allowed=COARSE==key
    tensor=CellTensor(GRAPH,TYPES,CHANNELS,CFG.sigma_um)
    rng=np.random.default_rng(CFG.seed+key)
    # Sample uniformly across 3D blocks before fitting; no target labels are used.
    _,block=np.unique(np.floor(XYZ[ids]/CFG.block_um),axis=0,return_inverse=True)
    order=rng.permutation(len(ids))
    _,first=np.unique(block[order],return_index=True)
    train=ids[order[np.sort(first)]]
    if len(train)>CFG.fit_windows: train=rng.choice(train,CFG.fit_windows,replace=False)
    elif len(train)<min(CFG.fit_windows,len(ids)):
        remain=np.setdiff1d(ids,train)
        train=np.r_[train,rng.choice(remain,min(CFG.fit_windows-len(train),len(remain)),replace=False)]
    model=fit_programs([tensor.fiber(i,allowed) for i in train],tensor.n_types,CFG,rng,n_senders=1)
    result={"ids":ids,"model":model}
    if model is not None:
        values=np.zeros((len(COARSE),model[2].shape[1]),dtype=np.float32)
        for n,i in enumerate(ids):
            values[i]=np.nan_to_num(project(tensor.fiber(i,allowed),model))
            if n and n%20000==0: print(f"C{key}: {n:,}/{len(ids):,} projected",flush=True)
        features=pool_programs(ids,allowed,GRAPH,TYPES,values,CFG.sigma_um)
        result["activity"]=values[ids]
        result["features"]=features
    fd,tmp=tempfile.mkstemp(dir=RUN,prefix=f"class_{key}_",suffix=".partial")
    with os.fdopen(fd,"wb") as stream:pickle.dump(result,stream,protocol=5)
    os.replace(tmp,target)
    return key,result

def main():
    global COARSE,GRAPH,TYPES,CHANNELS,XYZ
    RUN.mkdir(exist_ok=True)
    data=ad.read_h5ad(DATA/"preprocessed_objects/preprocessed_objects/cosmx.h5ad")
    geom=np.load(OUT/"geometry_v1.npz")
    XYZ=geom["xyz"]
    COARSE=np.load(TARGET/"coarse_resolution_0.40.npy")
    names,TYPES=np.unique(data.obs.celltypes.astype(str),return_inverse=True)
    GRAPH=load_npz(OUT/"contacts.npz")
    CHANNELS=Channels(data.layers["counts"],data.var_names,pd.read_csv(ROOT/"data/commuspace_human_lr_atlas.tsv",sep="\t"))
    results={}
    with ProcessPoolExecutor(max_workers=11,mp_context=get_context("fork")) as pool:
        for future in as_completed([pool.submit(task,int(key)) for key in np.unique(COARSE)]):
            key,res=future.result();results[key]=res
            print(f"完成 {len(results)}/11 粗类 C{key}, {len(res['ids']):,} cells",flush=True)
    labels=np.array([f"C{x}" for x in COARSE],dtype=object)
    candidates=[]
    for key,res in sorted(results.items()):
        if "features" not in res or len(res["ids"])<60:continue
        v=res["features"];scale=np.maximum(np.quantile(v,.9,axis=0),1e-7);x=np.log1p(v/scale)
        rng=np.random.default_rng(73+key)
        sample=rng.choice(len(x),min(5000,len(x)),replace=False)
        km=KMeans(n_clusters=2,n_init=8,random_state=73).fit(x[sample])
        assigned=km.predict(x)
        subset=rng.choice(len(x),min(1200,len(x)),replace=False)
        if len(np.unique(assigned[subset]))<2:continue
        quality=silhouette_score(x[subset],assigned[subset])
        counts=np.bincount(assigned,minlength=2)
        if counts.min()<12:continue
        candidates.append((float(quality),key,res["ids"],assigned,int(np.argmin(counts))))
    candidates.sort(key=lambda x:(-x[0],x[1]))
    if len(candidates)<9:raise ValueError(f"Only {len(candidates)} coarse classes support a split; cannot make 20")
    table=[]
    for index,(quality,key,ids,assigned,small) in enumerate(candidates):
        chosen=index<9
        if chosen: labels[ids[assigned==small]]=f"C{key}.2"
        table.append((key,len(ids),int((assigned==small).sum()),quality,chosen))
    cells=pd.DataFrame(dict(coarse=COARSE,context=[f"C{x}" for x in COARSE],niche=labels,
                            status="complete"),index=data.obs_names)
    cells.index.name="cell_id"
    cells.to_csv(RUN/"cells.tsv",sep="\t")
    pd.DataFrame(table,columns=["coarse","cells","new_subclass_cells","silhouette","selected"]).to_csv(RUN/"splits.tsv",sep="\t",index=False)
    print(f"完成：{len(cells):,} cells, {cells.niche.nunique()} niches",flush=True)

if __name__=="__main__":main()
