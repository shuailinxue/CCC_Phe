"""Small checks for expression-only counterfactual controls, no real training."""
from pathlib import Path
import sys,tempfile,json
import numpy as np
import pandas as pd
import anndata as ad
import pytest
from phenoniche.simulation.config import SimulationConfig,ROOT
from phenoniche.simulation.generation import make_lr,TYPES
from phenoniche.simulation.response_simulation import checksum
from phenoniche.storage import STORAGE_ROOT
sys.path.insert(0,str(ROOT/'scripts'))
from phenoniche.simulation.counterfactual_controls import noise_control,coverage_supervision_mask


def fixture_files(root):
    cfg=SimulationConfig.from_json(ROOT/'config/tumor_edge_gene_rich_simulation.json')
    lr=make_lr(cfg);genes=sorted(set(lr.ligand)|set(lr.receptor))+['unchanged_background']
    types=np.tile(np.resize(np.asarray(TYPES),100),2)
    obs=pd.DataFrame({'cell_type':types,'synthetic_library_factor':np.ones(200)},index=[str(i) for i in range(200)])
    x=np.zeros((200,len(genes)),np.float32);x[:,-1]=7
    a=ad.AnnData(x,obs=obs,var=pd.DataFrame(index=genes))
    profile=json.loads((ROOT/'config/tumor_edge_gene_rich_background.json').read_text())
    a.uns['rich_background_profile']=profile.get('source_background_profile',profile)
    a.obsm['spatial']=np.column_stack((np.arange(200),np.zeros(200))).astype(np.float32)
    for key in ['designed_MI_activity','injected_sender_activity','injected_receiver_activity']:a.obsm[key]=np.ones((200,4),np.float32)
    a.write_h5ad(root/'before_injection.h5ad');a.write_h5ad(root/'injected.h5ad');lr.to_csv(root/'lr.csv',index=False)
    reference=ad.AnnData(np.full((10,len(genes)),1.5,np.float32),obs=pd.DataFrame({'Cell_type':np.tile(np.asarray(TYPES),2)},index=[str(i) for i in range(10)]),var=pd.DataFrame(index=genes))
    reference.write_h5ad(root/'seed_reference.h5ad')
    return a,lr


@pytest.mark.parametrize('mode',['program_sparse','program_coherent'])
def test_independent_mi_sampling_releases_zeros_and_keeps_inputs(mode):
    with tempfile.TemporaryDirectory(dir=STORAGE_ROOT/'tmp',prefix='MI_rate_control_test_') as value:
        root=Path(value);source=root/'source';source.mkdir();original,lr=fixture_files(source)
        digest=checksum(source/'injected.h5ad')
        after,_=noise_control(source,root/'one',mode)
        repeated,_=noise_control(source,root/'two',mode)
        before=ad.read_h5ad(root/'one'/'before_injection.h5ad')
        np.testing.assert_array_equal(after.X,repeated.X)
        np.testing.assert_array_equal(after.X[:,-1],original.X[:,-1])
        np.testing.assert_array_equal(after.obsm['spatial'],original.obsm['spatial'])
        assert checksum(source/'injected.h5ad')==digest
        assert np.equal(after.X,np.floor(after.X)).all() and (after.X>=0).all()
        assert np.any((before.X==0)&(after.X>0)), 'MI activation must not lock preexisting measurement zeros'
        assert np.any(after.X!=before.X)
        assert not (root/'one'/'ccc.npy').exists(), 'Never inject final CCC targets'
        metadata=json.loads((root/'one'/'provenance.json').read_text())
        assert metadata['MI_additive_peak_rate']==3. and metadata['MI_peak_fold'] is None
        if mode=='program_sparse':assert metadata['inactive_unassigned_LR_rate']==.03
        else:assert metadata['unassigned_LR_own_role_rate']==.3
        assert after.obsm['synthetic_MI_cell_state'].shape==(200,4)


def test_noise_control_cannot_silently_fall_back():
    import pytest
    with tempfile.TemporaryDirectory(dir=STORAGE_ROOT/'tmp',prefix='MI_rate_invalid_test_') as value:
        source=Path(value);fixture_files(source)
        with pytest.raises(ValueError,match='Unknown noise control'):noise_control(source,source/'bad','not_a_control')
        with pytest.raises(ValueError,match='overwrite'):noise_control(source,source,'program_sparse')


def test_coverage_mask_is_training_only_and_truth_blind():
    from types import SimpleNamespace
    a=ad.AnnData(np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]],np.float32),var=pd.DataFrame(index=['L','R']))
    features=pd.DataFrame({'sender':['A','B'],'receiver':['B','A'],'ligand':['L','L'],'receptor':['R','R']})
    data=SimpleNamespace(members=np.arange(4)[:,None],rows=[np.array([0,2])],types=np.array([0,0,1,1]),categories=['A','B'],features=features,feature_mask=np.ones(2,bool))
    mask=coverage_supervision_mask(a,data,.1)
    np.testing.assert_array_equal(mask,[True,False])
    a.X[1:4:2]=999 # held-out centers never fit the coverage mask
    a.obs['true_niche']=[9,8,7,6]
    np.testing.assert_array_equal(mask,coverage_supervision_mask(a,data,.1))
