"""Small exact-ID and healthy-calibration tests; no real-data training."""
import json,ast
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite
import anndata as ad
from phenoniche.cohorts.dermatomyositis import assemble_export,audit_sections
from phenoniche.cohorts.case_novelty import matched_distance,find_case_niches


def test_export_preserves_permuted_cell_alignment(tmp_path):
    p=tmp_path/'export';p.mkdir()
    (p/'r_object_slots.json').write_text('{}')
    cells=['a','b','c'];genes=['L1','R1','L2'];counts=np.array([[1,2,3],[4,5,6],[7,8,9]],np.float32)
    mmwrite(p/'counts_Xenium.mtx',sp.csc_matrix(counts.T))
    (p/'counts_Xenium.genes.txt').write_text('\n'.join(genes)+'\n')
    (p/'counts_Xenium.cells.txt').write_text('\n'.join(cells)+'\n')
    pd.DataFrame({'cell_id':['c','a','b'],'patient_id':['DM','HC','HC'],'sp_type':['C','A','B']}).to_csv(p/'metadata.csv',index=False)
    pd.DataFrame({'cell_id':['b','c','a'],'x':[20,30,10],'y':[2,3,1],'image':['x']*3}).to_csv(p/'coordinates.csv',index=False)
    a=assemble_export(tmp_path)
    np.testing.assert_array_equal(a.X.toarray(),counts)
    np.testing.assert_array_equal(a.obsm['spatial'],[[10,1],[20,2],[30,3]])
    assert a.obs.sp_type.tolist()==['A','B','C']
    assert a.obs.cell_id.tolist()==cells


def test_sample_selection_uses_qc_and_observed_architecture_only():
    rng=np.random.default_rng(9);n=1200
    patients=np.repeat(['patient_H6','patient_H7','patient_DM8','patient_DM9'],n)
    obs=pd.DataFrame({'patient_id':patients,'sp_type':np.tile(['A','B'],2*n),
                      'image':np.repeat(['i1','i2','i3','i4'],n)},index=[f'c{i}' for i in range(n*4)])
    genes=['CXCL10','IL1B','L1','R1'];x=rng.poisson(25,(n*4,4)).astype(np.float32)
    x[n*3:,:2]+=30
    a=ad.AnnData(sp.csr_matrix(x),obs=obs,var=pd.DataFrame(index=genes));a.obsm['spatial']=rng.uniform(0,500,(n*4,2))
    # With only four genes the fixed >=20-gene QC correctly rejects all slices.
    try:audit_sections(a)
    except ValueError as e:assert 'quality' in str(e)
    extra=rng.poisson(4,(n*4,25)).astype(np.float32)
    a=ad.AnnData(sp.hstack([a.X,sp.csr_matrix(extra)]).tocsr(),obs=obs,var=pd.DataFrame(index=genes+[f'G{i}' for i in range(25)]));a.obsm['spatial']=rng.uniform(0,500,(n*4,2))
    table,chosen=audit_sections(a)
    assert len(chosen)==2 and {x['condition'] for x in chosen}=={'HC','DM'}
    assert chosen==audit_sections(a)[1]


def test_healthy_calibration_disjoint_and_reproducible():
    rng=np.random.default_rng(42);grid=np.array([(x,y) for x in range(30) for y in range(30)],float)*12
    n=len(grid);values=rng.normal(0,.03,(n*2,6));values[n:n+300,0]+=2
    comp=np.full((n*2,2),.5);cells=pd.DataFrame({'cell_id':[f'c{i}' for i in range(n*2)],
        'condition':['HC']*n+['DM']*n,'x':np.tile(grid[:,0],2),'y':np.tile(grid[:,1],2)})
    # Identity neighbors isolate calibration logic from model/CCC integration.
    members=np.repeat(np.arange(n*2)[:,None],30,axis=1);weights=np.ones_like(members,dtype=float)
    args=(values,comp,cells,members,weights)
    left,table,_,audit=find_case_niches(*args,block_um=60,min_cells=25)
    right,_,_,_=find_case_niches(*args,block_um=60,min_cells=25)
    np.testing.assert_array_equal(left.candidate_niche,right.candidate_niche)
    assert not set(audit['healthy_fit_rows'])&set(audit['healthy_calibration_rows'])
    assert left.iloc[n:n+300].candidate.mean()>.9
    assert len(table)>0 and table.DM_enriched.any()
    assert not np.isnan(left.novelty_score_ratio).any()


def test_composition_matched_distance_and_notebook_static():
    background=np.array([[0.,0.],[2,2],[3,3]])
    d=matched_distance(np.array([[0.,0.]]),background,np.array([[1.,0.]]),np.array([[1.,0.],[0,1],[0,1]]),candidates=1,neighbors=1)
    assert d[0]==0
    import nbformat
    root=Path(__file__).resolve().parents[1]
    notebook=nbformat.read(root/'Dermatomyositis_Xenium5K_Communication_Niches.ipynb',as_version=4);nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type=='code':ast.parse('\n'.join(line for line in cell.source.splitlines() if not line.startswith('%')))
    config=json.loads((root/'config/dermatomyositis_5k.json').read_text())
    assert config['device']=='cuda:2' and config['split_unit']=='joint_all'


def test_candidate_maps_render_background_and_multiple_candidates(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from phenoniche.cohorts.dermatomyositis_reporting import DermatomyositisReport
    report=DermatomyositisReport.__new__(DermatomyositisReport)
    report.cells=pd.DataFrame({'condition':['HC']*10+['DM']*10,
        'x':np.tile(np.arange(10)*15,2),'y':np.tile([0,1],10)*15,
        'cell_type':['A','B']*10,'niche':[1,2]*10})
    report.novel=pd.DataFrame({'novelty_score_ratio':np.linspace(0,2,20),'candidate_niche':[0,1,2,0,0]*4})
    report.types=['A','B'];report.type_colors=['#3976AC','#D66A58'];report.niche_colors=report.type_colors;report.k=2
    report.save=lambda fig,name:fig
    for name in ('novelty','comparison'):
        fig=getattr(report,name)();fig.savefig(tmp_path/f'{name}.png',dpi=60);plt.close(fig)
        assert (tmp_path/f'{name}.png').stat().st_size>1000


def test_communication_metric_preserves_dictionary_distance_and_activity():
    from phenoniche.gnn.communication_metric import communication_embedding
    h=np.array([[.8,.2,0],[.8,.2,0],[0,0,1]],np.float32)
    p=np.array([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0]],np.float32)
    scale=np.array([2,1,3],np.float32);activity=np.array([2,2,2,4],np.float32)
    values,projection,audit=communication_embedding(p,activity,h,scale,np.arange(4))
    # Swapping activity between identical dictionaries cannot create a new niche.
    np.testing.assert_allclose(values[0],values[1],atol=1e-7)
    assert values[0,-1]!=values[3,-1]
    expected=np.linalg.norm(((p[0]-p[2])@h)/scale)/audit['profile_scale']
    np.testing.assert_allclose(np.linalg.norm(values[0,:-1]-values[2,:-1]),expected,rtol=1e-5)
    assert not audit['condition_labels_used'] and not audit['cell_type_labels_used']


def test_immune_context_is_postfit_and_keeps_shared_niches():
    from phenoniche.cohorts.skin_paper_context import niche_immune_summary
    cells=pd.DataFrame({'condition':['HC']*4+['DM']*4,'niche':[1,1,1,2,1,2,2,2]})
    composition=np.array([[.1,.9]]*4+[[.8,.2]]*4)
    table=niche_immune_summary(cells,composition,['Myeloid','Fibroblasts'])
    assert table.iloc[0].niche==2
    assert (table.HC_cells>0).all() and (table.DM_cells>0).all()
    assert table.HC_cells.sum()==4 and table.DM_cells.sum()==4
    np.testing.assert_allclose(table.HC_fraction.sum(),1)
    try:niche_immune_summary(cells,composition[:3],['Myeloid','Fibroblasts'])
    except ValueError as exc:assert 'dimensions' in str(exc)
    else:raise AssertionError('Misaligned composition accepted')


def test_fine_state_panel_renders_actual_labels_and_keeps_all_cells(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from phenoniche.cohorts.dermatomyositis_reporting import DermatomyositisReport
    report=DermatomyositisReport.__new__(DermatomyositisReport)
    n=40
    report.output=tmp_path;report.k=2;report.niche_colors=['#3976AC','#D66A58']
    report.cells=pd.DataFrame({'cell_id':[f'c{i}' for i in range(n)],'condition':['HC']*20+['DM']*20,
        'patient':['patient_H8']*20+['patient_DM21']*20,'x':np.tile(np.arange(20)*15,2),
        'y':np.tile([0,1],20)*15,'cell_type':['Fibroblasts','Myeloid']*20,
        'cell_type_fine':['reticular_fib','CXCL10_Mac']*20,'niche':[1,2]*20})
    report.immune_fraction=np.linspace(0,1,n);report.save=lambda fig,name:fig
    before=report.cells.copy(deep=True)
    fig=report.paper_context();fig.savefig(tmp_path/'fine_states.png',dpi=60);plt.close(fig)
    pd.testing.assert_frame_equal(report.cells,before)
    display=pd.read_csv(tmp_path/'immune_architecture_display.csv.gz')
    assert display.cell_id.tolist()==before.cell_id.tolist()
    assert len(display)==n
    assert display.immune_rich_display.tolist()==(report.immune_fraction>=.2).tolist()
