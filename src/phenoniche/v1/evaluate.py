"""Spatially held-out edge prediction and interpretable, candidate-conditioned contrasts."""
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from .model import Batch, local_fit


@torch.no_grad()
def heldout(model,data,device):
    ids = np.flatnonzero(data.test)
    if not len(ids):
        return {}
    rng = np.random.default_rng(813)
    ids = np.sort(rng.choice(ids,min(len(ids),1024),replace=False))
    # Fixed across models/seeds. Local inference sees only the complementary edges.
    hidden = rng.random((data.nt*data.nt*data.nl,)) < .2
    if not hidden.any():
        hidden[0] = True
    mask = torch.as_tensor(hidden,device=device)
    totals = np.zeros(7,np.float64)
    w = model.weights()
    cfg = model.cfg
    for start in range(0,len(ids),cfg.batch):
        b = Batch(data,ids[start:start+cfg.batch],device)
        u = local_fit(model,b,w,b.weights(cfg,~mask),cfg.local_steps)
        if cfg.mode == "composition":
            logits,mu = model.anchor[:,mask].unbind()
        else:
            logits,mu = [v[:,mask] for v in model.predict(u,w)]
        s,c = b.s[:,mask],b.c[:,mask]
        p = logits.sigmoid()
        zeros = s-c
        totals += np.array([v.double().sum().item() for v in (
            s,c,zeros,c*(1-p).square(),zeros*p.square(),
            b.squares[:,mask]-2*mu*b.logs[:,mask]+c*mu.square(),
            s*torch.nn.functional.softplus(logits)-c*logits)])
    n,c,z,pos_brier,zero_brier,sse,nll = totals
    def ratio(x,y):
        return float(x/y) if y else None
    return dict(centers=len(ids),trials=int(n),positive=int(c),effective_zero=int(z),
                brier=ratio(pos_brier+zero_brier,n),positive_brier=ratio(pos_brier,c),
                zero_brier=ratio(zero_brier,z),positive_log_rmse=float(np.sqrt(max(0,sse)/c)) if c else None,
                detection_log_loss=ratio(nll,n),mask_seed=813)


def decode(edges,names,lr):
    n,l = len(names),len(lr)
    return pd.DataFrame(dict(edge=edges,sender=names[edges//(n*l)],receiver=names[edges//l%n],
                             ligand=lr[edges%l,0],receptor=lr[edges%l,1]))


def export(model,data,out,path):
    def arr(x):
        return x.detach().cpu().numpy()
    from .storage import save_npz
    cfg = model.cfg
    w = arr(model.weights())
    eta,pi,prior = arr(model.eta),arr(model.pi.softmax(1)),arr(model.omega.softmax(0))
    support = data.support[data.train].sum(0)[...,data.group].ravel()
    # Dataset-wide observed opportunities (overlapping windows), not independent sample size.
    niche_counts = np.bincount(out["niche"],minlength=cfg.k)
    metrics = dict(niche_sizes=niche_counts.tolist(),empty_niches=int((niche_counts==0).sum()),
                   median_soft_max=float(np.median(out["soft"].max(1))))
    save_npz(path/"niches.npz",pi=pi,eta=eta,omega=prior,names=data.names,
             prototype_activity=np.logaddexp(0,eta),train_support=support)
    if cfg.mode == "composition":
        return metrics
    proto_a = np.logaddexp(0,eta)
    logits,mu = arr(model.base[0])+proto_a@w[0],arr(model.base[1])+proto_a@w[1]
    expected = expit(logits)*np.exp(mu+cfg.sigma_x**2/2)
    save_npz(path/"programs.npz",occurrence=w[0],intensity=w[1],base=arr(model.base),
             lr=data.lr,names=data.names,signaling=data.signaling)
    save_npz(path/"niche_networks.npz",detected=expit(logits),positive_log_mean=mu,
             expected_activity=expected,difference_from_mean=expected-expected.mean(0))
    tables = []
    for r in range(cfg.rank):
        strength = np.hypot(w[0,r],w[1,r])
        edges = np.argsort(np.where(support>0,strength,-1))[::-1][:min(30,(support>0).sum())]
        f = decode(edges,data.names,data.lr)
        f["program"],f["occurrence"],f["intensity"],f["train_support"] = r,w[0,r,edges],w[1,r,edges],support[edges]
        tables.append(f)
    pd.concat(tables).to_csv(path/"top_edges.tsv",sep="\t",index=False)
    vectors = w.transpose(1,0,2).reshape(cfg.rank,-1)
    norm = vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
    distinct = np.triu_indices(cfg.rank,1)
    metrics.update(activity_sd=out["activity"].std(0).tolist(),
                   max_program_cosine=float((norm@norm.T)[distinct].max()) if len(distinct[0]) else None,
                   program_norm=np.linalg.norm(vectors,axis=1).tolist())
    comp = out["composition"]
    if cfg.mode == "joint":
        metrics["different_from_composition_evidence"] = float((comp.argmax(1)!=out["niche"]).mean())
        metrics["different_from_composition_plus_prior"] = float(((comp+np.log(prior)).argmax(1)!=out["niche"]).mean())
    if cfg.k>1:
        a,b = np.triu_indices(cfg.k,1)
        metrics["min_prototype_composition_l1"] = float(np.abs(pi[a]-pi[b]).sum(1).min())
        metrics["min_prototype_u_distance"] = float(np.linalg.norm(eta[a]-eta[b],axis=1).min())
    return metrics


def explain(run,network,cell_id,competitor=None):
    """Observed, predicted and reference-normalized networks; exact soft-score terms."""
    import json
    from pathlib import Path
    from .networks import joint
    run,network = Path(run),Path(network)
    cfg = json.loads((run/"manifest.json").read_text())["settings"]
    if cfg["mode"] == "composition":
        raise ValueError("Communication explanation requires a fitted dictionary")
    with np.load(run/"cells.npz") as z, np.load(network/"schema.npz") as schema:
        if not np.array_equal(z["ids"],schema["ids"]):
            raise ValueError("Network and inferred center IDs disagree")
        i = pd.Index(z["ids"]).get_loc(str(cell_id))
        winner = z["niche"][i]
        comp,latent,soft,activity = (z[k][i] for k in ("composition","latent","soft","activity"))
        mass,group = z["mass"][i],schema["group"]
        names,lr = schema["names"],schema["lr"]
    with np.load(network/f"part-{i//256*256:08d}.npz") as part:
        row = i%256
        sl = slice(*part["ptr"][row:row+2])
        e,stats = part["edge"][sl],part["stats"][sl]
        s = part["support"][row][...,group].ravel()
    with np.load(run/"programs.npz") as z:
        w,base = np.stack([z["occurrence"],z["intensity"]]),z["base"]
    with np.load(run/"niches.npz") as z:
        prior,anchor,train_support = z["omega"],z["prototype_activity"],z["train_support"]
    if competitor is None:
        if len(prior)<2:
            raise ValueError("A comparison requires at least two niches")
        competitor = np.argsort(np.where(np.arange(len(prior))==winner,-np.inf,comp))[-1]
    if competitor == winner or not 0 <= competitor < len(prior):
        raise ValueError("Choose a different existing niche")
    logit,mu = base + activity @ w
    positive = np.zeros_like(s,dtype=float)
    sums,logs,squares = positive.copy(),positive.copy(),positive.copy()
    positive[e],logs[e],squares[e],sums[e] = stats.T
    raw = sums/np.maximum(s,1)
    pred = expit(logit)*np.exp(mu+cfg["sigma_x"]**2/2)
    table = decode(np.arange(len(s)),names,lr)
    table["support"],table["positive"],table["train_support"] = s,positive,train_support
    table["observed_activity"] = np.where(s>0,raw,np.nan)
    table["positive_mean"] = np.where(positive>0,sums/np.maximum(positive,1),np.nan)
    table["predicted_detection"],table["predicted_activity"] = expit(logit),pred
    shape = (len(names),len(names),len(lr))
    tau = json.loads((run/"metrics.json").read_text())["tau"]
    refs = {}
    for label,values in [("observed",raw),("estimated",pred)]:
        edge,ref = joint(mass,values.reshape(shape),s.reshape(shape),tau)
        table[label+"_joint_mass"] = edge.ravel()
        refs[label] = ref.tolist()
    # Candidate-centroid contrast at fixed representative activities, not an
    # additive decomposition of the assignment probability.
    def predictive_nll(k):
        lo,m = base + anchor[k] @ w
        return (s*np.logaddexp(0,lo)-positive*lo +
                (squares-2*m*logs+positive*m*m)/(2*cfg["sigma_x"]**2))
    table["candidate_raw_nll_advantage"] = np.where(s>0,predictive_nll(competitor)-predictive_nll(winner),np.nan)
    for r in range(len(activity)):
        table[f"program_{r}_logit"],table[f"program_{r}_log_intensity"] = activity[r]*w[:,r]
    info = dict(cell_id=str(cell_id),winner=int(winner),competitor=int(competitor),
                composition_margin=float(comp[winner]-comp[competitor]),
                program_margin=float(latent[winner]-latent[competitor]),
                prior_margin=float(np.log(prior[winner]/prior[competitor])),soft=soft.tolist(),
                node_types=names.tolist(),node_proportions=(mass/mass.sum()).tolist(),reference_mass=refs)
    return info,table
