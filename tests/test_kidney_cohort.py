"""Original-row provenance, ROI halo, and reconstructed-support regressions."""
import numpy as np,pandas as pd,scipy.sparse as sp,anndata as ad,torch
from phenoniche.cohorts import kidney_dkd as kidney
from phenoniche.cohorts.colitis import sample_rows
from phenoniche.cohorts.paired_skin import make_ccc_objective
from phenoniche.gnn.data import neighborhoods
from types import SimpleNamespace


def test_parent_labels_preserve_immune_lineage_and_only_merge_injury_states():
    frame=pd.DataFrame({'annotation_updated':['iPT','PT','iTAL','Immune','Immune','Podo'],
        'immune_cell_annotation_combined':['Unknown','Unknown','Unknown','B','CD8+','Unknown']})
    assert kidney.parent_labels(frame).tolist()==['PT','PT','TAL','B','CD8+','Podo']


def test_anatomical_roi_halo_preserves_original_neighbors(tmp_path,monkeypatch):
    root=tmp_path/'data';root.mkdir();out=tmp_path/'out';out.mkdir()
    xy=np.array([(x,y) for x in range(40) for y in range(40)],float)*20
    for sample,condition in [('healthy','Control'),('case','DKD')]:
        obs=pd.DataFrame({'cell_id':[sample+str(i) for i in range(len(xy))],
            'annotation_updated':np.where(np.arange(len(xy))%4==0,'Podo','PT'),
            'immune_cell_annotation_combined':'Unknown','Condition':condition,
            'source_row':np.arange(len(xy)),'patient':sample,'image':sample,'library_counts':80})
        obs.index=obs.cell_id
        a=ad.AnnData(sp.csr_matrix(np.full((len(xy),8),10,np.float32)),obs=obs);a.obsm['spatial']=xy
        a.write_h5ad(root/f'source_{sample}.h5ad')
    monkeypatch.setattr(kidney,'extract_sections',lambda *args:None)
    a=kidney.prepare_tissue({'data_dir':str(root),'sections':['healthy','case'],'healthy_roi_centers':400,'case_roi_centers':800,'seed':17},out)
    healthy=a[a.obs.condition=='HC'];assert healthy.obs.analysis_center.sum()==400 and len(healthy)>400
    original,_=neighborhoods(xy);new=np.load(out/'input/members.npy')[:len(healthy)]
    rr=healthy.obs.source_row.to_numpy();valid=healthy.obs.analysis_center.to_numpy()
    np.testing.assert_array_equal(np.sort(rr[new[valid]],axis=1),np.sort(original[rr[valid]],axis=1))
    rows=sample_rows(a,{'split_unit':'joint_all','seed':1,'evaluation_centers_per_patient':50})
    assert a.obs.analysis_center.iloc[rows[0]].all() and len(rows[0])==1200
    assert set(rows[1])<=set(rows[0])


def test_weighted_objective_uses_training_nonzero_support():
    data=SimpleNamespace(feature_nonzero_fraction=np.array([.1,.8],np.float32))
    config=SimpleNamespace(ccc_nonzero_weight_max=8.)
    prediction=torch.tensor([[.2,.1],[.2,.3]],requires_grad=True);target=torch.tensor([[1.,0.],[0.,2.]])
    loss=make_ccc_objective(data,config)(prediction,target,np.ones(2,np.float32),np.ones(2,bool));loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(prediction.grad).all()


def test_contiguous_NAS_export_chunks_preserve_original_directed_CCC():
    from phenoniche.ccc.pairwise import aggregate_pairwise_ccc
    rng=np.random.default_rng(7);xy=rng.normal(size=(61,2)).astype(np.float32)*30
    members,weights=neighborhoods(xy);types=np.arange(61)%2
    ligand=rng.poisson(2,size=(61,2)).astype(np.float32);receptor=rng.poisson(3,size=(61,2)).astype(np.float32)
    mask=np.array([[True,False],[True,True],[False,True],[True,False]])
    _,expected,_=aggregate_pairwise_ccc(xy,types,ligand,receptor,2,20.,members,weights,mask,device='cpu')
    chunks=[]
    for start in range(0,61,19):
        _,value,_=aggregate_pairwise_ccc(xy,types,ligand,receptor,2,20.,members[start:start+19],weights[start:start+19],mask,device='cpu')
        chunks.append(value)
    np.testing.assert_allclose(np.vstack(chunks),expected,rtol=1e-6,atol=1e-6)
