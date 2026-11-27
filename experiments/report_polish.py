"""Post-selection tissue figures, evidence diagnostics and matched external labels."""
import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, squareform, pdist
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch

from phenoniche.v1.polish import Options
from phenoniche.v1.polish_fit import load_dictionary
from phenoniche.v1.probabilistic import Config,batched,batch_data
from phenoniche.v1.storage import Network,digest,key,save_npz,versions
from phenoniche.v1.workflow import runtime


def matched_pairs(indices, labels, xyz, prop, pair, strata, caliper=.30):
    pairs=[]
    for s in np.unique(strata[indices]):
        ix=indices[strata[indices]==s]
        left=ix[labels[ix]==pair[0]]
        right=ix[labels[ix]==pair[1]]
        if not len(left) or not len(right):
            continue
        d=cdist(prop[left],prop[right],metric="cityblock")
        i,j=linear_sum_assignment(np.where(d<=caliper,d,1000.))
        keep=d[i,j]<=caliper
        pairs.extend(zip(left[i[keep]],right[j[keep]]))
    return np.asarray(pairs,dtype=int).reshape(-1,2)


def external(data,ids,ann,labels,pi,activity,setname):
    prop=data.mass/data.mass.sum(1,keepdims=True)
    xyz=data.xyz
    occupied=np.flatnonzero(np.bincount(labels,minlength=len(pi))>=6)
    dist=squareform(pdist(pi,metric="cityblock"))
    rows=[]
    block=np.array([f"{i}:{j}" for i,j in np.floor(xyz/500).astype(int)])
    for where in ("all","test"):
        keep=ids if where=="all" else ids[data.test[ids]]
        pars=[(i,j) for i in occupied for j in occupied if j>i]
        if not pars:
            continue
        for broad,event,s in [
                ("Malignant","minor_celltype",block),
                ("all","path_region",np.char.add(block,ann.major_celltype.to_numpy(dtype=str)))]:
            match_ids=keep[ann.major_celltype[keep].eq(broad).to_numpy()] if broad!="all" else keep[
                ann.path_region[keep].ne("Unassigned").to_numpy()]
            # Choose the closest occupied prototype pair with actual common
            # support in this stratum without using the outcome labels.
            for pair in sorted(pars,key=lambda ab:dist[ab]):
                match=matched_pairs(match_ids,labels,xyz,prop,pair,s)
                if len(match)>=4:
                    break
            else:
                continue
            left,right=match.T
            counts=ann[event].to_numpy(dtype=str)
            test_rows=[]
            for label in sorted(set(counts[left])|set(counts[right])):
                plus=int(((counts[left]==label)&(counts[right]!=label)).sum())
                minus=int(((counts[left]!=label)&(counts[right]==label)).sum())
                # Flip the sign of all paired differences in the same spatial block together.
                diff=(counts[left]==label).astype(int)-(counts[right]==label).astype(int)
                strata_codes=np.unique(block[left],return_inverse=True)[1]
                aggregate=np.bincount(strata_codes,weights=diff)
                if len(aggregate)<2 or not np.any(aggregate):
                    pval=1.
                else:
                    rng=np.random.default_rng(73)
                    draw=rng.choice([-1,1],size=(4095,len(aggregate)))@aggregate
                    pval=(1+(np.abs(draw)>=abs(aggregate.sum())).sum())/4096
                test_rows.append(dict(scope=where,model=setname,niche_a=pair[0],niche_b=pair[1],label=label,
                    broad_type=broad,pairs=len(match),blocks=len(np.unique(block[left])),a_only=plus,b_only=minus,
                    composition_l1=float(dist[pair]),matched_l1=float(np.abs(prop[left]-prop[right]).sum(1).mean()),
                    activity_distance=float(np.linalg.norm(activity[left].mean(0)-activity[right].mean(0))),p=pval))
            p=np.array([r["p"] for r in test_rows])
            order=np.argsort(p)
            q=np.empty_like(p)
            q[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1].clip(0,1)
            rows.extend(dict(r,q=float(value)) for r,value in zip(test_rows,q))
    return rows


def components(data,labels,model,niche,device,annotations):
    ids=np.flatnonzero(data.test)
    support=data.support[ids].sum((1,2,3))
    counts=data.mass[ids].sum(1)
    edges=((data.support[ids]>0)[...,data.group]).sum((1,2,3))
    positive=np.diff(data.ptr)[ids]
    sender=np.stack([data.support[ids].sum((2,3)).max(1),data.support[ids].sum((2,3)).min(1)],1)
    c=labels["composition"][ids]
    g=(labels["occurrence_gain"]+labels["intensity_gain"])[ids]
    prior=labels["prior"][ids]
    y=labels["niche"][ids]
    score=c+g+prior
    runner=score.copy()
    runner[np.arange(len(y)),y]=-np.inf
    other=runner.argmax(1)
    margin=lambda x:x[np.arange(len(y)),y]-x[np.arange(len(y)),other]
    metrics=pd.DataFrame(dict(id=data.ids[ids],node=margin(c),communication=margin(g),prior=margin(prior),
        signed_total=margin(score),counts=counts,eligible_edges=edges,positive_edges=positive,
        sender_support=support,sender_support_max=sender[:,0]))
    sampling=annotations["transcript_counts"] if "transcript_counts" in annotations else annotations.get("ntx")
    if sampling is not None:
        metrics["center_molecule_depth"]=sampling.to_numpy()[ids]
    armgroups={}
    for name,groups in {
        "shared_ligand":[[x] for x in data.lr[:,0]],
        "shared_receptor":[[x] for x in data.lr[:,1]],
        "shared_pathway":[str(s).split(";") if isinstance(s,str) and s else [] for s in annotations.attrs["pathways"]]
    }.items():
        members={}
        for i,rs in enumerate(groups):
            for r in rs:
                members.setdefault(r,[]).append(i)
        degree=np.ones(data.nl)
        for rs in members.values():
            degree[rs]=np.maximum(degree[rs],len(rs))
        armgroups[name]=torch.tensor(degree,device=device)
    for key in armgroups:
        metrics[key+"_retained_after_diagnostic_discount"]=0.
    with torch.no_grad():
        h,m=model.predict(niche["activity"].clamp_min(0))
        for idx in batched(np.arange(len(ids)),16):
            b=batch_data(data,ids[idx],device)
            s,c,n,l=model.features(b)
            i,j=y[idx],other[idx]
            diff=c*(h[i]-h[j])-s*(torch.nn.functional.softplus(h[i])-torch.nn.functional.softplus(h[j]))
            diff+=l*(m[i]-m[j])-.5*n*(m[i].square()-m[j].square())
            magnitudes=diff.reshape(len(idx),-1,data.nl).abs().sum(1)
            den=magnitudes.sum(1).clamp_min(1e-10)
            for name,degree in armgroups.items():
                metrics.loc[idx,name+"_retained_after_diagnostic_discount"]=((magnitudes/degree).sum(1)/den).cpu().numpy()
    return metrics


def main():
    p=argparse.ArgumentParser()
    for name in ("selected","network","source"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--dataset",choices=["pt73","ovca"],required=True)
    a=p.parse_args()
    sel=a.selected
    data=Network(a.network)
    meta=dict(code=digest(__file__),network=digest(a.network/"manifest.json"),source=digest(a.source),
              selection=digest(sel/"selection.json"),metrics=digest(sel/"test_metrics.json"),
              versions=versions(["numpy","pandas","matplotlib","umap-learn","anndata"]))
    out=sel/("report-"+key(meta))
    out.mkdir(exist_ok=True)
    (out/"manifest.json").write_text(json.dumps(meta,indent=2))
    modes=("final73","final74","final75","composition","communication","shuffle","toggle_lr","toggle_diversity")
    pars=json.loads((sel/"selection.json").read_text())
    k=pars["niche"]["k"]
    with np.load(sel/"split.npz") as ss:
        np.testing.assert_array_equal(ss["ids"],data.ids)
    adata=ad.read_h5ad(a.source,backed="r")
    wanted=["hubType","ntx"] if a.dataset=="pt73" else ["major_celltype","minor_celltype","path_region","transcript_counts"]
    ann=adata.obs.loc[data.ids,[x for x in wanted if x in adata.obs]].copy()
    adata.file.close()
    ann.attrs["pathways"]=pd.read_csv(sel/"lr_annotations.tsv",sep="\t",keep_default_na=False).CellChat_pathways.tolist()
    allcells={}
    for mode in modes:
        with np.load(sel/mode/"cells.npz") as c:
            np.testing.assert_array_equal(c["ids"],data.ids)
            allcells[mode]={key:c[key] for key in c.files}
    anchor=allcells["final73"]
    y=anchor["niche"]
    rows=[]; state=[]
    for mode,z in allcells.items():
        label=z["niche"]
        rec=dict(mode=mode,ari=adjusted_rand_score(y,label),test_ari=adjusted_rand_score(y[data.test],label[data.test]),
                 sizes=np.bincount(label,minlength=k).tolist(),
                 test_change=float(np.mean((z["composition"]+z["prior"]).argmax(1)[data.test]!=label[data.test])))
        for i in (74,75):
            if mode in ("final73","final74","final75"):
                ref=allcells[f"final{i}"]["niche"]
                rec[f"ARI_seed{i}"]=adjusted_rand_score(label,ref)
        if "hubType" in ann:
            for scope in ("all","test"):
                ids=ann.hubType.isin(["immune_hub","hybrid_hub"]).to_numpy() & (
                    data.test if scope=="test" else True)
                rec[f"{scope}_hub_n"]=int(ids.sum())
                rec[f"{scope}_hub_ari"]=adjusted_rand_score(ann.hubType[ids],label[ids])
                rec[f"{scope}_hub_nmi"]=normalized_mutual_info_score(ann.hubType[ids],label[ids])
        elif mode in ("final73","composition","shuffle"):
            with np.load(sel/mode/"niches.npz") as n:
                state.extend(external(data,np.arange(len(y)),ann,label,n["pi"],z["activity"],mode))
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out/"assignments.tsv",sep="\t",index=False)
    if a.dataset=="ovca":
        pd.DataFrame(state).to_csv(out/"matched_state_pathology.tsv",sep="\t",index=False)
    w=[]
    for mode in modes[:3]:
        with np.load(sel/mode/"programs.npz") as z:
            v=np.column_stack([z["occurrence"],z["intensity"]])
        w.append(v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12))
    matched=[]
    for i,j in ((0,1),(0,2),(1,2)):
        c=w[i]@w[j].T
        ii,jj=linear_sum_assignment(-c)
        matched.append(pd.DataFrame(dict(seed_a=73+i,seed_b=73+j,program_a=ii,program_b=jj,cosine=c[ii,jj])))
    pd.concat(matched).to_csv(out/"program_matches.tsv",sep="\t",index=False)
    saved=json.loads((sel/"final73/metrics.json").read_text())
    dic=Path(saved["dictionary_fit"])
    dm=json.loads((dic/"manifest.json").read_text())
    model_file=dic/"dictionary.pt"
    if not model_file.is_file():
        model_file=next(p for p in (sel.parent.parent/"phenomap_network").glob("*/dictionary.pt")
                        if digest(p)==dm["reference"])
    device="cpu"
    cfg=Config(**dm["cfg"])
    model=load_dictionary(model_file,cfg,Options(**saved["options"]),device)
    niche={"activity":torch.as_tensor(np.load(sel/"final73/niches.npz")["activity"],device=device)}
    evidence=components(data,anchor,model,niche,device,ann)
    evidence.to_csv(out/"evidence.tsv",sep="\t",index=False)
    evidence.select_dtypes("number").corr(method="spearman").to_csv(out/"evidence_spearman.tsv",sep="\t")
    n=dict(np.load(sel/"final73/niches.npz"))
    dist=squareform(pdist(n["pi"],metric="cityblock"))
    occ=np.bincount(y,minlength=k)
    ii,jj=np.triu_indices(k,1)
    v=n["activity"]
    table=pd.DataFrame(dict(niche_a=ii,niche_b=jj,composition_l1=dist[ii,jj],
        activity_distance=np.linalg.norm(v[ii]-v[jj],axis=1),
        occupancy_a=occ[ii],occupancy_b=occ[jj]))
    table.to_csv(out/"prototype_pairs.tsv",sep="\t",index=False)
    feasible=table.query("occupancy_a>=6 and occupancy_b>=6")
    pair=feasible.sort_values(["composition_l1","activity_distance"],ascending=[True,False]).iloc[0]
    i,j=int(pair.niche_a),int(pair.niche_b)
    from phenoniche.v1.evaluate import decode
    diff=n["log_intensity"][i]-n["log_intensity"][j]
    dp=n["detection"][i]-n["detection"][j]
    ww=np.hypot(diff,dp)
    support=model.support.cpu().numpy()
    best=np.argsort(np.where(support>0,ww,-1))[-30:][::-1]
    edges=decode(best,data.names,data.lr)
    edges["log_intensity_difference"]=diff[best]
    edges["detection_difference"]=dp[best]
    edges["CellChat_pathways"]=np.asarray(ann.attrs["pathways"])[best%data.nl]
    edges.to_csv(out/"closest_pair_edges.tsv",sep="\t",index=False)
    top=pd.read_csv(sel/"final73/top_edges.tsv",sep="\t")
    top["CellChat_pathways"]=np.asarray(ann.attrs["pathways"])[top.edge%data.nl]
    top.to_csv(out/"program_edges_annotated.tsv",sep="\t",index=False)
    if not (out/"embedding.npz").exists():
        import umap
        z=umap.UMAP(n_neighbors=20,min_dist=.2,n_jobs=1,random_state=73).fit_transform(np.log1p(anchor["activity"]))
        save_npz(out/"embedding.npz",ids=data.ids,coordinates=z)
    emb=np.load(out/"embedding.npz")["coordinates"]
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42,"axes.spines.right":False,"axes.spines.top":False})
    cmap=plt.get_cmap("tab20")
    def save(fig,name):
        fig.savefig(out/f"{name}.pdf",dpi=180)
        fig.savefig(out/f"{name}.png",dpi=180)
        plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout="constrained")
    axs[0].scatter(*data.xyz.T,c=cmap(y%20),s=4)
    axs[0].set(title="Sampled niche assignments",aspect="equal",xlabel="X (input units)",ylabel="Y (input units)")
    axs[1].scatter(*emb.T,c=cmap(y%20),s=5)
    axs[1].set(title="Independent program activity",xlabel="UMAP 1",ylabel="UMAP 2")
    axs[2].violinplot([evidence[c] for c in ("node","communication","prior")],showmedians=True)
    axs[2].set(xticks=[1,2,3],xticklabels=["Nodes","Communication","Prior"],title="Test winner vs runner-up",ylabel="Score difference")
    save(fig,"overview")
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout="constrained")
    colors=[cmap(i%20),cmap(j%20)]
    for a1,p1,color in zip((i,j),n["pi"][[i,j]],colors):
        axs[0,0].plot(data.names,p1,"o-",color=color,label=f"Niche {a1}")
        axs[0,1].plot(np.arange(1,len(v[a1])+1),v[a1],"o-",color=color,label=f"Niche {a1}")
    axs[0,0].set(title=f"Closest node prototypes • L1 {pair.composition_l1:.2f}",ylabel="Node mass")
    axs[0,0].tick_params(axis="x",labelrotation=35)
    axs[0,1].set(title="Network prototype activity",xlabel="Program",ylabel="Activity")
    axs[0,1].legend(frameon=False)
    axs[1,0].scatter(*data.xyz.T,color=".87",s=2)
    for value,color in zip((i,j),colors):
        axs[1,0].scatter(*data.xyz[y==value].T,color=color,s=5)
    axs[1,0].set(title="Spatial locations",aspect="equal",xlabel="X",ylabel="Y")
    ax=axs[1,1]
    for q,r in enumerate(edges.head(6).itertuples()):
        color=colors[0 if r.log_intensity_difference>=0 else 1]
        ax.add_patch(FancyArrowPatch((.29,q),(.71,q),arrowstyle="-|>",mutation_scale=12,color=color))
        ax.text(.27,q,r.sender,ha="right",va="center",fontsize=8)
        ax.text(.73,q,r.receiver,va="center",fontsize=8)
        ax.text(.5,q+.18,r.ligand+" → "+r.receptor,ha="center",fontsize=7)
    ax.set(xlim=(0,1),ylim=(-.4,5.5),title="Largest supported edge contrasts")
    ax.axis("off")
    save(fig,"similar_composition_niches")
    print(out,flush=True)


if __name__=="__main__":
    main()
