import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from phenoniche.cohorts.kidney_downstream import (
    unique_physical_contacts, contact_enrichment, gene_effects, panel_enrichment,
)
from phenoniche.gnn.model import physical_weights


def test_overlapping_neighborhoods_count_one_physical_contact():
    xy=np.array([[0.,0.],[3.,0.],[0.,4.]],np.float32)
    members=np.array([[0,1,2],[0,1,2]])
    weights=np.ones((2,3),np.float32)
    u,v,w=unique_physical_contacts(np.array([0,1]),members,weights,xy)
    expected=physical_weights(torch.tensor(xy[None]),torch.ones((1,3)),20.,torch.tensor(members[:1])).numpy()[0]
    assert len(u)==3
    np.testing.assert_allclose(w,expected[u,v],rtol=1e-6)


def test_contact_permutation_large_type_codes_and_reproducibility():
    # Categorical codes are int8; 27*28+27 must not overflow before bincount.
    xy=np.c_[np.arange(40),np.zeros(40)]
    codes=np.tile(np.array([0,27],np.int8),20)
    u=np.arange(39);v=u+1;w=np.ones(39)
    a=contact_enrichment(u,v,w,codes,xy,np.repeat('section',40),28,permutations=19)
    b=contact_enrichment(u,v,w,codes,xy,np.repeat('section',40),28,permutations=19)
    assert a[0].shape==(28,28) and a[0].sum()==39
    for x,y in zip(a,b):
        np.testing.assert_array_equal(x,y)
        assert np.isfinite(x).all()


def test_gene_effects_observed_rna_and_measured_gene_background():
    counts=np.array([[20.,0,2],[20,1,2],[2,5,2],[2,5,2]])
    x=sp.csr_matrix(np.log1p(counts))
    effects=gene_effects(x,np.array(['LIG','OTHER','HOUSEKEEPING']),[0,1],[2,3])
    assert effects.loc[0,'log2FC']>3
    assert abs(effects.loc[2,'log2FC'])<1e-10
    np.testing.assert_allclose(effects.niche_mean_TP10K,[20,.5,2])
    # Only measured genes count towards a pathway's test universe.
    table=pd.DataFrame({'gene':[f'G{i}' for i in range(20)],'universe':True,
                        'log2FC':[2.]*5+[0.]*15,'niche_detection':.8})
    sets={'real_set':{f'G{i}' for i in range(5)}|{'UNMEASURED'},'uncovered':{'ABSENT'}}
    result=panel_enrichment(table,sets)
    assert result.pathway.tolist()==['real_set']
    assert result.iloc[0].covered_genes==5 and result.iloc[0].overlap==5
    assert result.iloc[0].universe_genes==20
    assert 0<result.iloc[0].p_gene_overlap<.001
