"""Run the sequential A–E V1 benchmark on frozen observation caches."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from sklearn.metrics import adjusted_rand_score
import torch
from phenoniche.v1.network_workflow import export
from phenoniche.v1.polish import Options
from phenoniche.v1.polish_fit import Experiment, KINDS, match_dictionaries
from phenoniche.v1.polish_metrics import benchmark, gradient_parts
from phenoniche.v1.probabilistic import Config, batched, batch_data
from phenoniche.v1.refit import activities
from phenoniche.v1.storage import Network, digest, key, save_npz


def run(args):
    data=Network(args.network)
    ref=[Path(p) for p in args.reference]
    cfg=Config() if not args.simulation else Config(rank=12,k=4,batch=48,centers=192,epochs=18,min_epochs=12,niche_epochs=16)
    exp=Experiment(data,args.root,args.resources,ref,args.device,cfg)
    train,valid=exp.train,exp.valid
    decisions=[]
    seeds=(73,74)
    def group(rank,opt,baseline=False):
        items=[exp.dictionary(rank,opt,s,ref[i] if baseline and len(ref)>i else None)
               for i,s in enumerate(seeds)]
        sims=match_dictionaries(items[0][0],items[1][0])
        utility=float(np.mean([it[2]["utility"] for it in items]))
        prediction=[np.mean([[x["detection"],x["positive_nll"],x["rmse"]] for x in it[2]["scores"]],0).tolist()
                    for it in items]
        metrics=dict(rank=rank,options=asdict(opt),utility=utility,matched=float(np.mean(sims)),matches=sims,
                 duplicate=float(np.mean([x[2]["diagnostic"]["duplicate_pairs"] for x in items])),
                 prediction=prediction,paths=[str(it[3]) for it in items])
        return items,metrics
    def note(name,metric,chosen):
        decisions.append(dict(stage=name,selected=bool(chosen),**metric))
        print(name,json.dumps(decisions[-1]),flush=True)
    a,am=group(12,Options(),True)
    note("A",am,True)
    b,bm=group(12,Options(lr_strength=2))
    keep_b=(bm["utility"]<am["utility"]-.003 and bm["matched"]>=am["matched"]-.03) or (
        bm["utility"]<=am["utility"]+.003 and bm["matched"]>=am["matched"]+.03)
    note("B",bm,keep_b)
    parent,pm=(b,bm) if keep_b else (a,am)
    opt=Options(**pm["options"])
    copt=replace(opt,diversity=4.)
    candidates=[(parent,pm)]
    for r in (6,8,12):
        candidates.append(group(r,copt))
    eligible=[(x,m) for x,m in candidates[1:] if
              m["utility"]<=pm["utility"]+.005 and m["matched"]>=pm["matched"]-.04 and
              (m["utility"]<pm["utility"]-.003 or m["duplicate"]<pm["duplicate"] or
               (m["rank"]<pm["rank"] and m["utility"]<=pm["utility"]))]
    if eligible:
        parent,pm=min(eligible,key=lambda x: x[1]["utility"]+.001*x[1]["rank"])
    for _,m in candidates[1:]:
        note("C",m,m is pm)
    dopt=replace(Options(**pm["options"]),calibration=True)
    d,dm=group(pm["rank"],dopt)
    keep_d=dm["utility"]<pm["utility"]-.003 and dm["matched"]>=pm["matched"]-.04
    note("D",dm,keep_d)
    if keep_d:
        parent,pm=d,dm
    # Compare communication scales at fixed K, then K at the chosen scale.
    def fits(k,t):
        models=[exp.niche(it,k,t,"joint",s) for it,s in zip(parent,seeds)]
        ms=[x[1] for x in models]
        u=float(np.mean([m["utility"] for m in ms]))
        ari=adjusted_rand_score(ms[0]["labels"],ms[1]["labels"])
        occupancy=np.mean([sum(np.asarray(m["sizes"])<2) for m in ms])
        return models,dict(k=k,temperature=t,utility=u,ari=ari,small=float(occupancy),
            min_prototype_rms=float(min(m["min_prototype_rms"] for m in ms)),
            criterion=u+.06*(1-ari)+.008*occupancy+.002*(k-4),
            paths=[str(x[2]) for x in models])
    temps=[fits(4,t) for t in (.25,.5,1.)]
    _,tm=min(temps,key=lambda x:x[1]["criterion"])
    for _,m in temps:
        note("D_evidence",m,m is tm)
    ks=[x for x in temps if x[1] is tm]+[fits(k,tm["temperature"]) for k in (6,8,12)]
    mixtures,km=min(ks,key=lambda x:x[1]["criterion"])
    for _,m in ks:
        note("E",m,m is km)
    selected=dict(dictionary=pm,niche=km,decisions=decisions)
    dest=exp.root/("selected-"+key(dict(meta=exp.meta,selected=selected)))
    dest.mkdir(exist_ok=True)
    (dest/"selection.json").write_text(json.dumps(selected,indent=2))
    save_npz(dest/"split.npz",ids=data.ids,train=train,validation_scored=valid,test=data.test)
    import pandas as pd
    from scipy import sparse
    sparse.save_npz(dest/"lr_graph.npz",exp.graph)
    pd.DataFrame(dict(ligand=data.lr[:,0],receptor=data.lr[:,1],signaling=data.signaling,
                      CellChat_pathways=exp.pathways)).to_csv(dest/"lr_annotations.tsv",sep="\t",index=False)
    rank,k,t=pm["rank"],km["k"],km["temperature"]
    opt=Options(**pm["options"])
    third=exp.dictionary(rank,opt,75)
    third_n=exp.niche(third,k,t,"joint",75)
    reports=[("final73",parent[0],mixtures[0]),("final74",parent[1],mixtures[1]),("final75",third,third_n)]
    reports += [(mode,parent[0],exp.niche(parent[0],k,t,mode,73)) for mode in ("composition","communication")]
    # If a candidate regularizer was rejected, its addition is still a measured counterfactual.
    no_lr=exp.dictionary(rank,replace(opt,lr_strength=0 if opt.lr_strength else 2),73)
    no_div=exp.dictionary(rank,replace(opt,diversity=0 if opt.diversity else 4),73)
    reports += [("toggle_lr",no_lr,exp.niche(no_lr,k,t,"joint",73)),
                ("toggle_diversity",no_div,exp.niche(no_div,k,t,"joint",73))]
    if args.null:
        null=Network(args.null)
        for name in ("ids","xyz","train","test","support","group","mass","scales","lr"):
            np.testing.assert_array_equal(getattr(null,name),getattr(data,name))
        null_exp=Experiment(null,exp.root/"shuffled",args.resources,[],args.device,cfg)
        item=null_exp.dictionary(rank,opt,73)
        reports.append(("shuffle",item,null_exp.niche(item,k,t,"joint",73)))
    test=np.flatnonzero(data.test)
    metrics={}
    for name,(model,a,_,directory),(n,summary,where) in reports:
        out=dest/name
        out.mkdir(exist_ok=True)
        conf=replace(cfg,rank=model.loading.shape[1],k=len(n.pi))
        if not (out/"complete").exists():
            a=activities(data,model,np.arange(len(data.ids)),conf,args.device)
            diag=export(data,model,n,a,conf,args.device,out)
            save_npz(out/"independent_activity.npz",ids=data.ids,activity=a)
            scores={kind:benchmark(data,test,model,conf,args.device,kind,niche=n)
                    if n.mode!="composition" else benchmark(data,test,model,conf,args.device,kind,baseline=True)
                    for kind in KINDS}
            dictionary={kind:benchmark(data,test,model,conf,args.device,kind) for kind in KINDS}
            record=dict(niche=scores,dictionary=dictionary,diagnostic=diag,
                        options=asdict(model.options),validation=summary,fit=str(where),dictionary_fit=str(directory))
            if name=="final73":
                g=[];loss=[]
                with torch.no_grad():
                    for ix in batched(test,conf.batch):
                        batch=batch_data(data,ix,args.device)
                        gg,ll=gradient_parts(model,batch,model.features(batch),torch.as_tensor(a[ix],device=args.device))
                        g.extend(gg.cpu().tolist());loss.extend(ll.cpu().tolist())
                save_npz(out/"gradient_diagnostics.npz",ids=data.ids[test],gradient=g,loss=loss)
                record["gradient_medians"]=np.median(g,0).tolist()
            (out/"metrics.json").write_text(json.dumps(record,indent=2))
            (out/"complete").touch()
        metrics[name]=json.loads((out/"metrics.json").read_text())
    if ref:
        from phenoniche.v1.network_workflow import niche_from
        for i,s in enumerate(seeds):
            model=exp.dictionary(12,Options(),s,ref[i])[0]
            record=torch.load(ref[i]/"joint/model.pt",map_location=args.device,weights_only=True)
            n=niche_from(record,cfg,data.nt,args.device)
            metrics[f"A{s}"]=dict(niche={kind:benchmark(data,test,model,cfg,args.device,kind,niche=n) for kind in KINDS},
                                  dictionary={kind:benchmark(data,test,model,cfg,args.device,kind) for kind in KINDS})
    (dest/"test_metrics.json").write_text(json.dumps(metrics,indent=2))
    (dest/"complete").touch()
    print(dest,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--network",type=Path,required=True)
    p.add_argument("--null",type=Path)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--resources",type=Path,required=True)
    p.add_argument("--reference",nargs="*",default=[])
    p.add_argument("--device",default="cuda:2")
    p.add_argument("--simulation",action="store_true")
    run(p.parse_args())
