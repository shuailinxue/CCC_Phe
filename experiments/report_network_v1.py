"""Read frozen sampled experiment checkpoints and evaluate external labels."""
import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from phenoniche.v1.network_workflow import dictionary_from, explain, niche_from
from phenoniche.v1.probabilistic import Config
from phenoniche.v1.storage import Network, digest, key, save_npz, versions


def enrichment(y, labels, strata):
    names, label = np.unique(labels,return_inverse=True)
    k=4
    groups=[np.flatnonzero(strata==value) for value in np.unique(strata)]
    def counts(v):
        return np.bincount(y*len(names)+v,minlength=k*len(names)).reshape(k,-1)
    observed=counts(label)
    expected=sum(np.outer(np.bincount(y[g],minlength=k),
                          np.bincount(label[g],minlength=len(names)))/len(g) for g in groups)
    rng=np.random.default_rng(73)
    hit=np.ones_like(observed)
    difference=np.abs(observed-expected)
    for _ in range(999):
        v=label.copy()
        for g in groups:
            v[g]=rng.permutation(v[g])
        hit+=np.abs(counts(v)-expected)>=difference
    p=hit.ravel()/1000
    order=np.argsort(p)
    q=np.empty_like(p)
    q[order]=np.minimum.accumulate((len(p)*p[order]/np.arange(1,len(p)+1))[::-1])[::-1].clip(0,1)
    return pd.DataFrame(dict(niche=np.repeat(np.arange(k),len(names)),label=np.tile(names,k),
                            observed=observed.ravel(),expected=expected.ravel(),p=p,q=q))


def main():
    p=argparse.ArgumentParser()
    for name in ("network","seed73","seed74","null","legacy","source"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--dataset",choices=["pt73","ovca"],required=True)
    a=p.parse_args()
    names=("joint","refined","composition","communication","null","legacy","seed74")
    paths=dict(zip(names,[a.seed73/k for k in names[:4]]+[a.null/"joint",a.legacy,a.seed74/"joint"]))
    meta=dict(code=digest(__file__),source=digest(a.source),network=digest(a.network/"manifest.json"),
              inputs={k:digest(v/"cells.npz") for k,v in paths.items()},
              versions=versions(["numpy","anndata","matplotlib","umap-learn","scikit-learn"]))
    dest=a.seed73.parent/("report-"+key(meta))
    dest.mkdir(exist_ok=True)
    (dest/"manifest.json").write_text(json.dumps(meta,indent=2))
    cells={k:dict(np.load(path/"cells.npz")) for k,path in paths.items()}
    reference=cells["joint"]
    ids,xy,test=reference["ids"],reference["xyz"],reference["test"]
    y=reference["niche"]
    obs=ad.read_h5ad(a.source,backed="r")
    ann=obs.obs.loc[ids,["hubType"] if a.dataset=="pt73" else ["major_celltype","minor_celltype","path_region"]].copy()
    obs.file.close()
    block=np.array([f"{i}:{j}" for i,j in np.floor(xy/500).astype(int)])
    rows=[]; enrich=[]
    for name,c in cells.items():
        np.testing.assert_array_equal(c["ids"],ids)
        np.testing.assert_array_equal(c["test"],test)
        label=c["niche"]
        row=dict(model=name,ari_vs_joint=adjusted_rand_score(label,y),
                 test_ari_vs_joint=adjusted_rand_score(label[test],y[test]),
                 sizes=np.bincount(label,minlength=4).tolist())
        if "occurrence_gain" in c:
            pred=(c["composition"]+c["prior"]).argmax(1)
            occ,inten=c["occurrence_gain"],c["intensity_gain"]
            row["communication_assignment_change"]=float(np.mean(pred!=label))
            row["test_communication_assignment_change"]=float(np.mean(pred[test]!=label[test]))
            runner=(c["composition"]+c["prior"]).copy()
            runner[np.arange(len(label)),label]=-np.inf
            runner=runner.argmax(1)
            row["median_margin_components"]={k:float(np.median(c[k][np.arange(len(label)),label]-c[k][np.arange(len(label)),runner]))
                for k in ("composition","prior","occurrence_gain","intensity_gain")}
        if a.dataset=="pt73":
            for scope,scope_mask in [("all",np.ones(len(ids),bool)),("test",test)]:
                selected=scope_mask & ann.hubType.isin(["immune_hub","hybrid_hub"]).to_numpy()
                row[scope+"_hub_n"]=int(selected.sum())
                if selected.any() and ann.hubType[selected].nunique()>1:
                    row[scope+"_hub_ari"]=adjusted_rand_score(ann.hubType[selected],label[selected])
                    row[scope+"_hub_nmi"]=normalized_mutual_info_score(ann.hubType[selected],label[selected])
        else:
            for scope,scope_mask in [("all",np.ones(len(ids),bool)),("test",test)]:
                for ct in ann.major_celltype.unique():
                    keep=scope_mask & ann.major_celltype.eq(ct).to_numpy()
                    L=ann.minor_celltype[keep].to_numpy(dtype=str)
                    if len(L)>=20 and len(np.unique(L))>1:
                        enrich.append(enrichment(label[keep],L,block[keep]).assign(model=name,scope=scope,broad_type=ct))
                keep=scope_mask & ann.path_region.ne("Unassigned").to_numpy()
                strata=np.char.add(block[keep],ann.major_celltype[keep].to_numpy(dtype=str))
                enrich.append(enrichment(label[keep],ann.path_region[keep].to_numpy(dtype=str),strata).assign(
                    model=name,scope=scope,broad_type="Pathology"))
        rows.append(row)
    if enrich:
        pd.concat(enrich,ignore_index=True).to_csv(dest/"state_pathology.tsv",sep="\t",index=False)
    frame=pd.DataFrame(rows)
    frame.to_csv(dest/"assignments.tsv",sep="\t",index=False)
    def vector(path):
        with np.load(path/"programs.npz") as z:
            w=np.column_stack([z["occurrence"],z["intensity"]])
        return w/np.maximum(np.linalg.norm(w,axis=1,keepdims=True),1e-10)
    v=vector(paths["joint"])
    stability=[]
    for name in ("seed74","refined","null"):
        cosine=v@vector(paths[name]).T
        i,j=linear_sum_assignment(-cosine)
        stability.append(pd.DataFrame(dict(model=name,reference=i,program=j,cosine=cosine[i,j])))
    pd.concat(stability).to_csv(dest/"program_matches.tsv",sep="\t",index=False)
    cfg=Config(**json.loads((a.seed73/"manifest.json").read_text())["settings"])
    import torch
    device="cpu"
    data=Network(a.network)
    dictionary=dictionary_from(torch.load(a.seed73/"dictionary.pt",map_location=device,weights_only=True)["state"],cfg)
    niche=niche_from(torch.load(paths["joint"]/"model.pt",map_location=device,weights_only=True),cfg,data.nt,device)
    # Prefer a held-out center whose communication evidence reverses the node+prior winner.
    node=(reference["composition"]+reference["prior"]).argmax(1)
    margin=reference["occurrence_gain"]+reference["intensity_gain"]
    altered=np.flatnonzero(test & (node!=y))
    ix=(altered[np.argmax(margin[altered,y[altered]]-margin[altered,node[altered]])]
        if len(altered) else int(np.flatnonzero(test)[0]))
    info,edges=explain(data,ix,dictionary,niche,reference["activity"][ix],device)
    info["id"]=str(ids[ix])
    (dest/"example_margin.json").write_text(json.dumps(info,indent=2))
    edges.to_csv(dest/"example_network.tsv.gz",sep="\t",index=False)
    if not (dest/"umap.npz").is_file():
        import umap
        emb=umap.UMAP(n_neighbors=20,min_dist=.2,random_state=73,n_jobs=1).fit_transform(
            np.log1p(reference["activity"]))
        save_npz(dest/"umap.npz",ids=ids,xy=emb)
    emb=np.load(dest/"umap.npz")["xy"]
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    palette=np.array(["#4477aa","#cc6677","#228833","#aa8844"])
    def save(fig,name):
        for suffix in ("pdf","png"):
            fig.savefig(dest/f"{name}.{suffix}",dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout="constrained")
    for ax,name in zip(axes.flat,names[:6]):
        label=cells[name]["niche"]
        mat=np.zeros((4,4),int)
        np.add.at(mat,(y,label),1)
        i,j=linear_sum_assignment(-mat)
        match=np.empty(4,int)
        match[j]=i
        ax.scatter(*xy.T,s=4,c=palette[match[label]],rasterized=True)
        ax.set(title=name,aspect="equal",xlabel="X (input units)",ylabel="Y (input units)")
    save(fig,"spatial_niches")
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    axes[0].scatter(*emb.T,c=palette[y],s=6,rasterized=True)
    axes[0].set(title="Independent communication activity",xlabel="UMAP 1",ylabel="UMAP 2")
    axes[1].scatter(*xy.T,c=reference["activity"][:,reference["activity"].std(0).argmax()],cmap="viridis",s=5,rasterized=True)
    axes[1].set(title="Most variable program across tissue",aspect="equal",xlabel="X",ylabel="Y")
    save(fig,"activity")
    fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
    n=np.load(paths["joint"]/"niches.npz")
    im=ax.imshow(n["pi"],vmin=0,vmax=1,aspect="auto",cmap="Blues")
    ax.set(xlabel="Broad cell type",ylabel="Niche",yticks=range(4),
           xticks=range(data.nt),xticklabels=data.names, title="Node mass prototypes")
    ax.tick_params(axis="x",labelrotation=35)
    fig.colorbar(im,ax=ax,label="Proportion")
    save(fig,"composition")
    tab=pd.read_csv(paths["joint"]/"top_edges.tsv",sep="\t")
    r=reference["activity"].std(0).argmax()
    tab=tab[tab.program==r].head(8)
    fig,ax=plt.subplots(figsize=(9,4),layout="constrained")
    for i,e in enumerate(tab.itertuples()):
        ax.add_patch(FancyArrowPatch((.28,i),(.72,i),arrowstyle="-|>",mutation_scale=12,
                                     color="#4477aa",lw=1+2*np.hypot(e.occurrence,e.intensity)))
        ax.text(.25,i,e.sender,ha="right",va="center")
        ax.text(.75,i,e.receiver,va="center")
        ax.text(.5,i+.17,f"{e.ligand} → {e.receptor}",ha="center",fontsize=8)
    ax.set(xlim=(0,1),ylim=(-.4,len(tab)-.2),title=f"Program {r+1} · supported directed edges")
    ax.axis("off")
    save(fig,"directed_program")
    print(dest,flush=True)


if __name__=="__main__":
    main()
