from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import adjusted_rand_score

from phenoniche.spatial.channels import Channels
from phenoniche.v1 import networks
from phenoniche.v1.fit import Sampler, train, infer
from phenoniche.v1.model import Batch, Settings, hurdle
from phenoniche.v1.storage import Network, prepare


@pytest.fixture
def tissue(tmp_path):
    import anndata as ad
    # Four-cell islands: identical composition, opposite expression patterns.
    n = 32
    xyz = np.tile([[0,0],[1,0],[2,0],[4,0]],(n,1)) + np.repeat(np.c_[np.arange(n)*200,np.zeros(n)],4,0)
    typ = np.tile(["A","A","B","C"],n)
    vals = np.zeros((4*n,5))
    for i in range(n):
        off = i%2*2
        vals[4*i,off] = 1
        vals[4*i+2,off+1] = 1
    vals[:,-1] = 1
    a = ad.AnnData(vals,obs=pd.DataFrame({"type":typ,"s":"one"},index=[str(i) for i in range(4*n)]),
                   var=pd.DataFrame(index=["L","R","L2","R2","BG"]))
    a.obsm["xy"] = xyz
    source = tmp_path/"a.h5ad"
    a.write_h5ad(source)
    atlas = tmp_path/"lr.tsv"
    lr = pd.DataFrame(dict(ligand=["L","L2"],receptor=["R","R2"],signaling_type=["Contact","ECM"]))
    lr.to_csv(atlas,sep="\t",index=False)
    kw = dict(coords="xy",types="type",specimen="s",workers=1)
    path = prepare(source,atlas,tmp_path,**kw)
    assert prepare(source,atlas,tmp_path,**kw) == path
    return Network(path),a,lr


def test_observed_network_and_marginal(tissue):
    d,a,lr = tissue
    ref = Channels(a.X,a.var_names,lr)
    x,left,right,*_ = networks.expression(a.X,a.var_names,lr,d.train)
    np.testing.assert_allclose(x[:,left],ref.expression[:,ref.left].toarray(),atol=1e-6)
    b = Batch(d,np.array([0]),"cpu")
    s = b.s.reshape(3,3,2).numpy()
    c = b.c.reshape(3,3,2).numpy()
    assert s[0,1,0] == 2 and c[0,1,0] == 1
    assert s[1,0,0] == 1 and c[1,0,0] == 0
    assert s[0,0,0] == 2 and s[1,1,0] == 0
    _,edge,v = d.entries(np.array([0]))
    assert len(edge) == 1 and v[0,3] == pytest.approx(1)
    avg = np.zeros(s.size)
    avg[edge] = v[:,3]/s.ravel()[edge]
    j,reference = networks.joint(d.mass[0],avg.reshape(s.shape),s,2.)
    assert np.isnan(j[1,1,0])
    np.testing.assert_allclose(np.nansum(j,(1,2))+reference,[.5,.25,.25])


def test_observed_opportunities_boundaries():
    from scipy import sparse
    xyz = np.array([[0.,0.],[10.,0.],[40.,0.],[55.,0.],[1.,0.]])
    spec = np.array(["a"]*4+["b"])
    ids = networks.window_index(xyz,spec,50)[0]
    exp = np.array([[1.,0],[0,1.],[0,0],[0,1.],[0,1.]],np.float32)
    sp = sparse.csr_matrix(exp)
    mass,s,edge,v = networks.observe(ids,xyz,np.array([0,1,1,1,1]),2,exp,
        np.array([0,0]),np.array([1,1]),np.array([400.,10000.]),np.arange(2),sp.indptr,sp.indices)
    np.testing.assert_array_equal(mass,[1,2])
    np.testing.assert_array_equal(s[1,0],[1,2])
    np.testing.assert_allclose(v[:,3],[1,.5])
    np.testing.assert_allclose(v[:,2],np.log(v[:,3])**2,atol=1e-6)


def test_loss_mask_and_gradients(tissue):
    d,_,_ = tissue
    b = Batch(d,np.array([0]),"cpu")
    cfg = Settings(scale="raw")
    logits = torch.full(b.s.shape,.3,dtype=torch.double,requires_grad=True)
    mu = torch.full_like(logits,-.4,requires_grad=True)
    def objective(l,m):
        a,z = hurdle(b,l,m,b.weights(cfg),1.)
        return (a+z).sum()
    assert torch.autograd.gradcheck(objective,(logits,mu))
    g,m = torch.autograd.grad(objective(logits,mu),(logits,mu))
    torch.testing.assert_close(g,b.s*logits.sigmoid()-b.c)
    torch.testing.assert_close(m,b.c*mu-b.logs)
    assert (g[b.s==0]==0).all()


def test_dynamic_sampling_joint_fit_and_composition(tissue):
    torch.set_num_threads(1)
    d,_,_ = tissue
    sampler = Sampler(d)
    rng = np.random.default_rng(4)
    first = sampler.draw(rng,50,100)
    second = sampler.draw(rng,50,100)
    assert len(first)==32 and not set(first)&set(second)
    cfg = Settings(k=2,rank=2,epochs=5,batch=32,local_steps=8)
    truth = np.arange(len(d.ids))//4%2
    model,_,hist,_ = train(d,cfg,"cpu")
    w = model.weights().detach().numpy()
    assert np.min(w)>=0
    np.testing.assert_allclose(w.max((0,2)),1)
    out = infer(model,d,"cpu")
    assert adjusted_rand_score(truth,out["niche"]) > .95
    np.testing.assert_allclose(out["soft"].sum(1),1,atol=1e-6)
    assert np.isfinite(hist[-1]["total"])
    # Keep all conditional edge observations identical, vary node counts only.
    d.edge = np.tile(d.edge[d.ptr[0]:d.ptr[1]],len(d.ids))
    d.stats = np.tile(d.stats[d.ptr[0]:d.ptr[1]],(len(d.ids),1))
    d.ptr = np.arange(len(d.ids)+1)
    d.mass[truth==0] = [12,2,2]
    d.mass[truth==1] = [2,12,2]
    model,_,_,_ = train(d,replace(cfg,seed=29),"cpu")
    assert adjusted_rand_score(truth,infer(model,d,"cpu")["niche"])>.95


def test_local_gradient_and_spatial_buffer(tissue):
    from phenoniche.v1.model import PhenoMap,local_fit
    d,_,_ = tissue
    train_ids,test = networks.split_region(d.xyz,d.samples,50,.25)
    assert d.xyz[train_ids,0].max()+100 < d.xyz[test,0].min()
    cfg = Settings(k=2,rank=3)
    m = PhenoMap(cfg,d.nt,d.nl).requires_grad_(False)
    b = Batch(d,np.array([0,2,8]),"cpu")
    weights = b.weights(cfg)
    initial = torch.tensor([[.2,-1.,2.],[-.4,.5,.9],[1.,-1.,0.]])
    w = m.weights()
    u = initial.clone().requires_grad_()
    opt = torch.optim.Adam([u],lr=cfg.local_rate)
    for _ in range(4):
        opt.zero_grad()
        m.objective(b,u,w,weights)[0].backward()
        opt.step()
    torch.testing.assert_close(local_fit(m,b,w,weights,4,initial),u.detach(),atol=1e-5,rtol=1e-5)


def test_save_transfer_and_explanation(tissue,tmp_path):
    from phenoniche.v1.workflow import execute,predict
    from phenoniche.v1.evaluate import explain
    d,a,_ = tissue
    cfg = Settings(k=2,rank=2,epochs=2,local_steps=4,batch=32)
    run = execute(d.path,tmp_path,cfg,"cpu")
    assert execute(d.path,tmp_path,cfg,"cpu") == run
    a.obs_names = ["new-"+i for i in a.obs_names]
    a.write_h5ad(tmp_path/"new.h5ad")
    kw = dict(coords="xy",types="type",specimen="s",workers=1)
    transferred = prepare(tmp_path/"new.h5ad",tmp_path/"lr.tsv",tmp_path,
                          **kw,reference=run/"training_schema.npz")
    dest = predict(run,transferred,tmp_path,"cpu")
    with np.load(dest/"cells.npz") as pred,np.load(run/"cells.npz") as original:
        assert np.array_equal(pred["ids"],a.obs_names)
        np.testing.assert_allclose(pred["soft"],original["soft"])
    info,edges = explain(run,d.path,"0")
    x,y = info["winner"],info["competitor"]
    margin = sum(info[name] for name in ("composition_margin","program_margin","prior_margin"))
    assert np.log(info["soft"][x]/info["soft"][y]) == pytest.approx(margin,abs=1e-5)
    assert edges.loc[edges.support==0,"observed_activity"].isna().all()
    assert edges.predicted_activity.notna().all()


def test_opportunity_preserving_null(tissue,tmp_path):
    d,_,_ = tissue
    path = prepare(tmp_path/"a.h5ad",tmp_path/"lr.tsv",tmp_path,coords="xy",
                   types="type",specimen="s",null_seed=31,workers=2)
    null = Network(path)
    np.testing.assert_array_equal(null.ids,d.ids)
    np.testing.assert_array_equal(null.support,d.support)
    np.testing.assert_array_equal(null.mass,d.mass)
    assert not np.array_equal(null.stats,d.stats)


def test_raw_tissue_composition_with_matched_conditional_activity(tmp_path):
    import anndata as ad
    islands = 20
    types = ["A"]*6+["B"]*2+["A"]*2+["B"]*6
    obs = pd.DataFrame(dict(type=types*(islands//2)),index=[str(i) for i in range(islands*8)])
    a = ad.AnnData(np.ones((len(obs),2)),obs=obs,var=pd.DataFrame(index=["L","R"]))
    a.obsm["xy"] = np.c_[np.repeat(np.arange(islands)*200,8)+np.tile(np.arange(8),islands),np.zeros(len(obs))]
    a.write_h5ad(tmp_path/"raw.h5ad")
    lr = tmp_path/"lr.tsv"
    lr.write_text("ligand\treceptor\nL\tR\n")
    d = Network(prepare(tmp_path/"raw.h5ad",lr,tmp_path,coords="xy",types="type",specimen=None))
    b = Batch(d,np.arange(len(obs)),"cpu")
    torch.testing.assert_close(b.c/b.s,torch.ones_like(b.c))
    torch.testing.assert_close(b.logs,torch.zeros_like(b.c))
    cfg = Settings(k=2,rank=2,epochs=3,local_steps=4)
    m,_,_,_ = train(d,cfg,"cpu")
    assert adjusted_rand_score(np.arange(len(obs))//8%2,infer(m,d,"cpu")["niche"]) == 1


def test_full_slice_network_fits_every_center_and_reuses_checkpoints(tissue, tmp_path):
    from phenoniche.v1.polish import Options
    from phenoniche.v1.polish_fit import fit_full_slice
    from phenoniche.v1.probabilistic import Config

    data, _, _ = tissue
    cfg = Config(rank=2, k=2, batch=30, epochs=2, min_epochs=1, niche_epochs=2, local_steps=2)
    options = Options(diversity=4.)
    out = fit_full_slice(data.path, tmp_path, cfg, options, 1., "cpu")
    for stage in ("dictionary", "niche"):
        with np.load(out / f"{stage}_coverage.npz") as coverage:
            np.testing.assert_array_equal(coverage["ids"], data.ids)
            np.testing.assert_array_equal(coverage["visits"], np.full(len(data.ids), 2))
    with np.load(out / "cells.npz") as cells:
        assert cells["train"].all()
        assert not cells["test"].any()
        assert cells["activity"].shape == (len(data.ids), 2)
    checkpoint = out / "dictionary.pt"
    written = checkpoint.stat().st_mtime_ns
    assert fit_full_slice(data.path, tmp_path, cfg, options, 1., "cpu", reuse_only=True) == out
    assert checkpoint.stat().st_mtime_ns == written
