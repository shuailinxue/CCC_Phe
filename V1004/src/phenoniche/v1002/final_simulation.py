"""Fixed microenvironment simulation shared by V1002 and V1003."""
from dataclasses import dataclass, replace
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from phenoniche.v1001.neighborhoods import build_neighborhoods
from phenoniche.v1002.simulation_cells import _layout, _side_expression, aggregate_pairwise_ccc, build_simulation_spec, bulk_potential

ALR_EPS = 1e-6
TRUE_BETA = np.array([.8, 0, -.8, 0, 0], dtype=np.float32)
BACKGROUND = np.full(8, .125, dtype=np.float32)
NICHE_PROTOTYPES = np.array([
    [.24,.20,.16,.10,.09,.08,.07,.06],
    [.22,.21,.17,.10,.09,.08,.07,.06],
    [.12,.12,.13,.10,.10,.08,.28,.07],
    [.12,.12,.13,.10,.10,.08,.07,.28],
    [.08,.10,.12,.24,.18,.10,.10,.08]], dtype=np.float32)


@dataclass
class MicroenvironmentSimulation:
    coordinates: np.ndarray
    labels: np.ndarray
    rho: np.ndarray
    theta: np.ndarray
    cell_types: np.ndarray
    relative_positions: np.ndarray
    expression: np.ndarray
    hc_truth: np.ndarray
    ws_truth: np.ndarray
    neighborhoods: object
    cs: np.ndarray
    communication: np.ndarray
    opportunity: np.ndarray
    pair_support: np.ndarray
    coverage_mask: np.ndarray
    final_mask: np.ndarray


def composition_prototypes(purity):
    if not 0 <= purity <= 1:
        raise ValueError("purity must be in [0,1]")
    return ((1-purity)*BACKGROUND + purity*NICHE_PROTOTYPES).astype(np.float32)


def build_final_spec(full_atlas):
    """Reuse the measurable atlas; set 25% N1/N2 and 80% N3/N4 shared edges."""
    old = build_simulation_spec(full_atlas)
    rng = np.random.default_rng(1402)
    n_lr = len(old.atlas)
    genes = {g: j for j, g in enumerate(old.genes)}
    delta = np.zeros_like(old.expression_delta)
    hi = np.full_like(old.hi_truth, 1e-8)
    active = []
    def pool(types):
        return np.array([(a*8+b)*n_lr+l for a in types for b in types for l in range(n_lr)], dtype=np.int64)
    def pick(source, n, omit=()):
        return rng.choice(np.setdiff1d(source, np.asarray(omit, dtype=np.int64)), n, replace=False)
    p12 = pool((0,1,2,3))
    s12 = pick(p12,25)
    n1 = np.r_[s12,pick(p12,75,s12)]
    n2 = np.r_[s12,pick(p12,75,n1)]
    p34 = pool((0,1,2,3,4,5))
    s34 = pick(p34,80,np.r_[n1,n2])
    n3 = np.r_[s34,pick(p34,20,np.r_[n1,n2,s34])]
    n4 = np.r_[s34,pick(p34,20,np.r_[n1,n2,n3])]
    n5 = pick(pool((3,4,5,6,7)),100,np.r_[n1,n2,n3,n4])
    for niche, features in enumerate((n1,n2,n3,n4,n5)):
        for feature in features:
            pair, lr = divmod(int(feature), n_lr)
            sender, receiver = divmod(pair, 8)
            shared = (niche < 2 and feature in s12) or (niche in (2,3) and feature in s34)
            strength = (.85 if niche in (2,3) else 1.2) * (.3 if niche in (2,3) and not shared else 1)
            active.append((niche,int(feature),strength,"shared" if shared else "specific"))
            hi[niche,feature] = strength
            interaction = old.atlas.interactions[lr]
            for gene in interaction.ligand_components:
                delta[niche,sender,genes[gene]] += strength/len(interaction.ligand_components)
            for gene in interaction.receptor_components:
                delta[niche,receiver,genes[gene]] += strength/len(interaction.receptor_components)
    hi /= hi.sum(1,keepdims=True)
    return replace(old, expression_delta=delta, hi_truth=hi, active_edges=tuple(active))


def _boundary_rho(labels):
    grid = labels.reshape(45,40)
    rho = np.zeros_like(grid,dtype=np.float32)
    for niche in range(1,6):
        distance = distance_transform_edt(grid==niche)
        x = np.clip((distance-.5)/2.5,0,1)
        smooth = x*x*(3-2*x)
        rho[grid==niche] = smooth[grid==niche]
    return rho.ravel()


def simulate_final_spatial(spec,purity=.5,niche_size=100,noise=.05,seed=40700,device=None,n_anchors=1800):
    """Each anchor has 48 local cells. n_anchors limits rows for smoke tests only."""
    if not 1 <= n_anchors <= 1800:
        raise ValueError("n_anchors must be 1..1800")
    coordinates,labels = _layout(niche_size,seed)
    rho = _boundary_rho(labels)
    coordinates,labels,rho = coordinates[:n_anchors],labels[:n_anchors],rho[:n_anchors]
    hc = composition_prototypes(purity)
    center = (1-rho[:,None])*BACKGROUND + rho[:,None]*np.vstack((BACKGROUND,hc))[labels]
    rng = np.random.default_rng(seed+101)
    gamma_draws = rng.gamma(shape=150*center,scale=1.)
    theta = (gamma_draws/gamma_draws.sum(1,keepdims=True)).astype(np.float32)
    cell_types = np.empty((n_anchors,48),dtype=np.int8)
    relative_positions = np.empty((n_anchors,48,2),dtype=np.float32)
    expression = np.empty((n_anchors,48,len(spec.genes)),dtype=np.float32)
    for i in range(n_anchors):
        types = rng.choice(8,48,p=theta[i]).astype(np.int8)
        positions = rng.normal(0,.3,(48,2)).astype(np.float32)
        mean = spec.baseline_mean[types].astype(np.float64)
        if labels[i]:
            mean += rho[i]*spec.expression_delta[labels[i]-1,types]
        if noise:
            mean *= rng.lognormal(0,noise,mean.shape)
        counts = rng.negative_binomial(2.,2./(2.+mean)).astype(np.float32)
        if noise:
            counts[rng.random(counts.shape)<noise] = 0
        cell_types[i],relative_positions[i],expression[i] = types,positions,counts
    ligand,receptor = _side_expression(expression.reshape(-1,len(spec.genes)),spec.genes,spec.atlas)
    ligand = ligand.reshape(n_anchors,48,-1)
    receptor = receptor.reshape(n_anchors,48,-1)
    cs,communication,opportunity = aggregate_pairwise_ccc(
        relative_positions,cell_types,ligand,receptor,8,sigma=.45,
        device=device or 'cpu',
    )
    observed_l = np.stack([(ligand[cell_types==t]>0).sum(0) for t in range(8)])
    observed_r = np.stack([(receptor[cell_types==t]>0).sum(0) for t in range(8)])
    total = np.bincount(cell_types.ravel(),minlength=8)
    coverage_l = observed_l/np.maximum(total[:,None],1)
    coverage_r = observed_r/np.maximum(total[:,None],1)
    coverage = ((coverage_l[:,None,:]>=.10)&(coverage_r[None,:,:]>=.10)).reshape(64,-1)
    pair_support = (opportunity>0).sum(0)
    final_mask = coverage & (pair_support[:,None]>=max(20,math.ceil(.01*n_anchors)))
    ws = np.zeros((n_anchors,6),dtype=np.float32)
    ws[:,0] = 1-rho
    ws[np.arange(n_anchors),labels] += rho
    neighborhoods = build_neighborhoods(coordinates,k=min(15,n_anchors-1),sigma=.8) if n_anchors>1 else None
    return MicroenvironmentSimulation(coordinates,labels,rho,theta,cell_types,relative_positions,
                                      expression,hc,ws,neighborhoods,cs,communication,
                                      opportunity,pair_support,coverage,final_mask)


def true_activity(simulation):
    return simulation.ws_truth


def alr(proportions,eps=ALR_EPS):
    p = np.asarray(proportions)
    return np.log((p[:,1:]+eps)/(p[:,:1]+eps))


def simulate_final_bulk(spec,hc,noise,seed,patients=320):
    rng = np.random.default_rng(seed)
    latent = rng.normal(0,.65,(patients,6))
    latent[:,0] += .2
    positive = np.exp(latent)
    pi = (positive/positive.sum(1,keepdims=True)).astype(np.float32)
    hc6 = np.vstack((BACKGROUND,hc)) if np.shape(hc)==(5,8) else np.asarray(hc)
    cb = np.maximum(pi@hc6+rng.normal(0,.01+noise*.02,(patients,8)),0)
    cb = (cb/cb.sum(1,keepdims=True)).astype(np.float32)
    expression = spec.baseline_mean[None]+np.einsum("pk,kcg->pcg",pi[:,1:],spec.expression_delta,optimize=True)
    expression *= rng.lognormal(0,.03+noise,expression.shape)
    expression = np.maximum(expression,0).astype(np.float32)
    eta = alr(pi)@TRUE_BETA
    failure = rng.exponential(size=patients)/(.05*np.exp(eta))
    censor = rng.exponential(scale=np.median(failure)*1.8,size=patients)
    time = np.maximum(np.minimum(failure,censor),np.finfo(np.float32).tiny).astype(np.float32)
    event = (failure<=censor).astype(np.float32)
    return pi,cb,expression,time,event,TRUE_BETA.copy(),eta.astype(np.float32)
