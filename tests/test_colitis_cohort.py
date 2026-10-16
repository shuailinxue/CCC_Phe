"""Small cohort adapter tests, independent of any patient outcomes."""
import json,ast
from pathlib import Path
import numpy as np,pandas as pd,scipy.sparse as sp,anndata as ad
from phenoniche.cohorts.colitis import tissue_components,patient_folds,prepare,CONDITIONS,PATIENT,CONDITION,COARSE,FINE
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.training import initialize,epoch
import torch

def test_fragment_and_patient_separation():
 rng=np.random.default_rng(5);xy=np.concatenate([rng.normal(0,2,(120,2)),rng.normal(500,2,(120,2)),rng.normal(0,2,(120,2))])
 patient=np.array(['a']*240+['b']*120)
 core,keep=tissue_components(xy,patient,100,100)
 assert keep.all() and len(set(core))==3
 obs=pd.DataFrame([(f'{c}{j}',c) for c in CONDITIONS for j in range(3)],columns=['patient','condition'])
 folds,mapping=patient_folds(obs,2)
 assert all(len(set(folds[obs.condition==c]))==3 for c in CONDITIONS)
 assert patient_folds(obs,2)[1]==mapping

def test_mini_raw_alignment_ccc_and_backward(tmp_path):
 # All large temporary data remain on NAS via pytest --basetemp.
 rng=np.random.default_rng(11);n=1350;genes=[f'L{i}' for i in range(8)]+[f'R{i}' for i in range(8)]
 patient=np.repeat([f'{c}{j}' for c in CONDITIONS for j in range(3)],150)
 condition=np.repeat(np.repeat(CONDITIONS,3),150);types=np.tile(['A','B','C'],450)
 xy=np.vstack([rng.uniform(0,100,(150,2)) for _ in range(9)]).astype(np.float32)
 obs=pd.DataFrame({PATIENT:patient,CONDITION:condition,COARSE:types,FINE:types},index=[f'c{i}' for i in range(n)])
 counts=sp.csr_matrix(rng.poisson(3,(n,16)).astype(np.float32));raw=ad.AnnData(counts,obs=pd.DataFrame(index=obs.index),var=pd.DataFrame(index=genes));raw.obsm['spatial']=xy
 raw.write_h5ad(tmp_path/'25_11_12_Xenium_Dataset2_5K_Raw.h5ad')
 order=rng.permutation(n);annot=ad.AnnData(counts[order]*.123,obs=obs.iloc[order],var=raw.var.copy());annot.obsm['spatial']=xy[order]
 annot.write_h5ad(tmp_path/'25_11_12_Xenium_Dataset2_5K_Annotated.h5ad')
 atlas=tmp_path/'lr.tsv';pd.DataFrame({'ligand':genes[:8],'receptor':genes[8:]}).to_csv(atlas,sep='\t',index=False)
 out=tmp_path/'result';(out/'input').mkdir(parents=True)
 c={'data_dir':str(tmp_path),'lr_atlas':str(atlas),'seed':5,'core_link_um':100.,'minimum_core_cells':100,
    'train_centers_per_patient':40,'evaluation_centers_per_patient':30,'coverage_fraction':.1,
    'opportunity_fraction':.01,'feature_nonzero_fraction':.005,'encoder_genes':16,'device':'cpu'}
 a,data=prepare(c,out)
 np.testing.assert_array_equal(a.X.toarray(),counts[order].toarray())
 assert data.targets.shape==(n,72) and data.feature_mask.sum()==72
 assert all((data.folds[data.members[r]]==i).all() for i,r in enumerate(data.rows))
 g=GNNConfig(programs=32,hidden=16,encoder_genes=16,max_epochs=2,batch_size=32,device='cpu')
 model=initialize(data,g);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
 before=model.cell_encoder[0].weight.detach().clone();metrics=epoch(model,data,data.rows[0][:32],g,optimizer)
 assert np.isfinite(metrics['total']) and not torch.equal(before,model.cell_encoder[0].weight.detach())
 batch=data.batch(data.rows[1][:4]);r=data.forward(model,batch)
 assert r['Z'].shape==(4,32) and (r['Z']>=0).all() and r['reconstruction'].shape==(4,72)
 # Complete-cohort discovery trains every center; no held-out fragments/test.
 c={**c,'split_unit':'joint_all'};out_all=tmp_path/'all_tissue';(out_all/'input').mkdir(parents=True)
 a_all,d_all=prepare(c,out_all)
 assert len(d_all.rows[0])==n and len(d_all.rows[2])==0
 assert (d_all.folds==0).all() and d_all.audit['monitor_role']=='training_subset'
 assert set(d_all.rows[1])<=set(d_all.rows[0])
 from phenoniche.cohorts.colitis import clustering_representation
 from phenoniche.gnn.training import fit
 values,_=clustering_representation(d_all,{'P':np.full((n,32),1/32),'activity':np.ones(n)},None,
     {'split_unit':'joint_all','clustering_representation':'program_mixture_activity','spatial_mixing':.2})
 assert values.shape==(n,33) and np.isfinite(values).all()
 g_all=GNNConfig(programs=32,hidden=8,encoder_genes=16,max_epochs=2,patience=2,batch_size=128,
     device='cpu',output_dir=str(out_all/'gnn'))
 fitted=fit(d_all,g_all,rescale_activity=True)
 metrics=json.loads((out_all/'gnn/training_monitor_metrics.json').read_text())
 assert metrics['test'] is None and metrics['training_centers']==n
 assert not (out_all/'gnn/heldout_metrics.json').exists()
 assert 'monitor_ccc' in pd.read_csv(out_all/'gnn/training.csv')
 # Interrupted training exports native-scale weights; recover without NMF/refit.
 from phenoniche.cohorts.colitis import load_prepared,recover_training_best,restore_checkpoint
 (out_all/'cohort_config.json').write_text(json.dumps(c))
 a_cached,d_cached=load_prepared(c,out_all)
 np.testing.assert_array_equal(d_cached.gene_mean,d_all.gene_mean)
 np.testing.assert_array_equal(d_cached.targets[:8],d_all.targets[:8])
 batch=d_all.batch(np.arange(4));fitted.eval()
 expected=d_all.forward(fitted,batch)['Z'].detach()
 (out_all/'gnn/checkpoint.pt').unlink()
 recovered=recover_training_best(d_cached,g_all);recovered.eval()
 torch.testing.assert_close(d_cached.forward(recovered,batch)['Z'],expected)
 restored=restore_checkpoint(d_cached,g_all);restored.eval()
 torch.testing.assert_close(d_cached.forward(restored,batch)['Z'],expected)
 assert json.loads((out_all/'gnn/training_monitor_metrics.json').read_text())['additional_training_epochs']==0

def test_joint_discovery_keeps_fragments_separate():
 from phenoniche.cohorts.colitis import fragment_folds,sample_rows
 obs=pd.DataFrame([(f'{c}{p}',c,f'{c}{p}_core{k}') for c in CONDITIONS for p in range(3) for k in range(4) for _ in range(40)],columns=['patient','condition','core'])
 folds,mapping=fragment_folds(obs,5);obs['fold']=folds
 assert obs.loc[folds==0,'patient'].nunique()==9
 assert all(obs.loc[obs.core==core,'fold'].nunique()==1 for core in obs.core.unique())
 a=ad.AnnData(sp.csr_matrix((len(obs),1)),obs=obs)
 rows=sample_rows(a,{'seed':5,'train_centers_per_patient':20,'evaluation_centers_per_patient':10})
 assert [len(r) for r in rows]==[180,90,90]
 assert all((folds[r]==fold).all() for fold,r in enumerate(rows))

def test_clustering_keeps_activity_and_uses_training_scaling():
 from types import SimpleNamespace
 from phenoniche.cohorts.colitis import clustering_representation
 p=np.full((6,32),1/32,np.float32);activity=np.arange(1,7,dtype=np.float32)
 data=SimpleNamespace(rows=[np.array([0,1]),np.array([2,3]),np.array([4,5])],members=np.arange(6)[:,None],weights=np.ones((6,1)))
 c={'clustering_representation':'program_mixture_activity','spatial_mixing':0}
 arrays={'P':p,'activity':activity,'Z':p*activity[:,None]}
 x,_=clustering_representation(data,arrays,None,c)
 assert x.shape==(6,33) and x[0,-1]!=x[1,-1]
 arrays['activity']=activity.copy();arrays['activity'][2:]*=1000
 changed,_=clustering_representation(data,arrays,None,c)
 np.testing.assert_array_equal(x[:2],changed[:2])
 np.testing.assert_array_equal(x[:,:32],changed[:,:32])

def test_all_patient_spatial_uses_identical_cells_and_all_fragments(tmp_path):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from phenoniche.cohorts.colitis_reporting import ColitisReport,packed_coordinates
 report=ColitisReport.__new__(ColitisReport)
 report.cells=pd.DataFrame([{'patient':f'{g}_HS{j}','condition':g,'core':f'{g}_{j}_{fragment}',
     'cell_type':'A','niche':1,'x':1000*fragment+q,'y':q*2} for g in CONDITIONS for j in range(3)
     for fragment in range(2) for q in range(3)])
 report.types=['A'];report.type_colors=['blue'];report.niche_colors=['red'];report.k=1
 report.figures=tmp_path;report.save=lambda fig,name:fig
 original=report.cells[['x','y']].to_numpy().copy()
 packed,_=packed_coordinates(report.cells)
 for core,group in report.cells.groupby('core'):
  ids=group.index.to_numpy()
  np.testing.assert_allclose(packed[ids]-packed[ids[0]],original[ids]-original[ids[0]])
 np.testing.assert_array_equal(report.cells[['x','y']],original)
 figure=report.spatial('HC')
 for column in range(3):
  top=figure.axes[column].collections[0].get_offsets()
  bottom=figure.axes[column+3].collections[0].get_offsets()
  assert len(top)==6
  np.testing.assert_array_equal(top,bottom)
 manifest=pd.read_csv(tmp_path/'all_patient_spatial_niches_HC_fragments.csv')
 assert len(manifest)==6 and manifest.cells.sum()==18
 plt.close(figure)
