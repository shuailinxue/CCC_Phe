import numpy as np
import pandas as pd
import torch

from phenoniche.v1.polish import Options, StructuredDictionary, biological_graph, grouped_masks
from phenoniche.v1.polish_metrics import benchmark, gradient_parts
from phenoniche.v1.probabilistic import Config, batch_data, pooled
from phenoniche.v1.simulated import simulate


def test_relation_normalization_and_conflicting_annotations():
    lr=np.array([["X","R"],["X","S"],["X","T"],["W","V"]])
    interactions=pd.DataFrame([
        ["X","complex","P1"],["X","complex","P2"],["X","S","P1"],["X","T","Other"]],
        columns=["ligand","receptor","pathway_name"])
    complexes=pd.DataFrame({"name":["complex"],"part1":["R"],"part2":[""]})
    graph,labels=biological_graph(lr,["Contact"]*4,interactions,complexes)
    assert labels==["P1;P2","P1","Other",""]
    assert graph[0,1]>0 and graph[0,2]==0 and graph[3].nnz==0
    np.testing.assert_allclose(graph.toarray(),graph.toarray().T)
    assert np.all(graph.diagonal()==0)
    support=torch.ones(16)
    m=StructuredDictionary(torch.zeros(2,16),support,support,2,Options(2,4),graph)
    w=torch.zeros(2,2,16)
    w[:,0,::4]=1
    w[:,1,::4]=1
    lr_loss,div=m.regularizers(w)
    assert lr_loss>0 and div>0
    w[:,:,1::4]=1
    assert m.regularizers(w)[0].abs()<1e-8


def test_held_out_genes_are_disjoint():
    lr=np.array([["A+B","C"],["A","Z"],["D","E"],["T","U"],["V","W"]])
    target,context=grouped_masks(lr,1,"cpu")
    gene=lambda row:set("+".join(row).split("+"))
    assert target.any() and context.any() and not (target&context).any()
    assert all(not gene(a)&gene(b) for a in lr[target] for b in lr[context])


def test_prediction_groups_and_gradient(tmp_path):
    d=simulate(tmp_path,"sparse")
    cfg=Config(rank=3,k=2)
    m=StructuredDictionary(*pooled(d,np.flatnonzero(d.train),"cpu"),3,Options(calibration=True))
    b=batch_data(d,np.array([2,3]),"cpu")
    ft=m.features(b)
    g,l=gradient_parts(m,b,ft,torch.ones(2,3))
    assert torch.isfinite(g).all() and g.shape==(2,3)
    out=benchmark(d,np.flatnonzero(d.test),m,cfg,"cpu","grouped")
    assert sum(out["groups"][k]["trials"] for k in ("rare_lr","medium_lr","common_lr"))==out["trials"]
    assert 0<=out["auprc"]<=1 and 0<=out["calibration_ece"]<=1


def test_portable_niche_reloads_score(tmp_path):
    from phenoniche.v1.polish_io import exported_niche
    from phenoniche.v1.probabilistic import Niches

    original=Niches(3,2,2,"joint",.5)
    with torch.no_grad():
        original.pi.copy_(torch.tensor([[0.,1.],[-1.,2.],[2.,-.2]]))
        original.omega.copy_(torch.tensor([-1.,2.,.1]))
        original.activity.copy_(torch.tensor([[.3,.1],[1.,1.],[1.1,1.3]]))
        np.savez(tmp_path/"niches.npz",pi=original.pi.softmax(1).numpy(),
                 omega=original.omega.softmax(0).numpy(),activity=original.activity.numpy())
    loaded=exported_niche(tmp_path,"joint",.5,"cpu")
    torch.testing.assert_close(original.pi.log_softmax(1),loaded.pi.log_softmax(1))
    torch.testing.assert_close(original.omega.log_softmax(0),loaded.omega.log_softmax(0))
    torch.testing.assert_close(original.activity,loaded.activity)


def test_stream_group_and_resume(tmp_path):
    from phenoniche.v1.polish_stream import stream
    from phenoniche.v1.probabilistic import Niches

    data=simulate(tmp_path,"both")
    cfg=Config(rank=2,k=2,local_steps=3)
    m=StructuredDictionary(*pooled(data,np.flatnonzero(data.train),"cpu"),2,Options()).requires_grad_(False)
    n=Niches(2,2,2,"joint",1).requires_grad_(False)
    fitted=tmp_path/"fitted"
    fitted.mkdir()
    for name in ("niches.npz","programs.npz","top_edges.tsv"):
        (fitted/name).touch()
    out=tmp_path/"inference"
    out.mkdir()
    first=stream(data.path,out,fitted,m,n,cfg,"cpu")
    target=next((out/"blocks").glob("part-*.npz"))
    saved_mtime=target.stat().st_mtime_ns
    second=stream(data.path,out,fitted,m,n,cfg,"cpu")
    assert first==second and target.stat().st_mtime_ns==saved_mtime
    with np.load(out/"cells.npz") as z:
        np.testing.assert_array_equal(z["ids"],data.ids)
        np.testing.assert_array_equal(z["mass"],data.mass)
        np.testing.assert_allclose(z["soft"].sum(1),1.,atol=1e-6)
