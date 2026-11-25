"""Count-scale invariance and consistent denominators for directed LR inputs."""
import numpy as np
import pytest
from phenoniche.ccc.normalization import normalize_lr_counts
from phenoniche.ccc.pairwise import _side_expression,aggregate_pairwise_ccc
from phenoniche.ccc.lr_atlas import load_lr_atlas


def test_lr_subset_uses_full_panel_total_and_preserves_zero_rows():
    x=np.array([[2,3,95],[4,6,190],[0,0,0]],np.float32)
    result=normalize_lr_counts(x[:,:2],x.sum(1),'panel_total')
    np.testing.assert_allclose(result,[[200,300],[200,300],[0,0]])
    np.testing.assert_array_equal(x,[[2,3,95],[4,6,190],[0,0,0]])
    with pytest.raises(ValueError):normalize_lr_counts(x,[1,2],'panel_total')


def test_complex_directed_targets_are_invariant_to_each_cells_count_multiplier(tmp_path):
    atlas_file=tmp_path/'lr.tsv';atlas_file.write_text('ligand\treceptor\nA_B\tC\nC\tA\n')
    atlas=load_lr_atlas(atlas_file);genes=('A','B','C','OTHER')
    x=np.array([[1,4,2,23],[2,3,0,15],[4,1,5,50],[3,0,1,6]],np.float32)
    y=x*np.array([2,7,3,5],np.float32)[:,None]
    targets=[]
    for raw in [x,y]:
        rna=normalize_lr_counts(raw,raw.sum(1),'panel_total')
        l,r=_side_expression(rna,genes,atlas)
        _,ccc,_=aggregate_pairwise_ccc(np.array([[0,0],[10,0],[0,10],[10,10]],np.float32),
            np.array([0,0,1,1]),l,r,2,20.,members=np.array([[0,1,2],[1,2,3]]),
            anchor_weights=np.array([[1,.8,.7],[1,.9,.5]],np.float32))
        targets.append(ccc)
    np.testing.assert_allclose(targets[0],targets[1],rtol=2e-6,atol=2e-5)
