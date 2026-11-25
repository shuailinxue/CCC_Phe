import numpy as np
import pytest
from phenoniche.selection.tnbc import patient_id, spread_indices, field_metrics, segmentation_fields, POLICY


def test_patient_ids_not_array_ids():
    assert patient_id('TNBC1_CN1_C1') == 1
    assert patient_id('TNBC2_CN1_E2') == 2
    with pytest.raises(ValueError):
        patient_id('CN1')


def test_native_field_sampling_is_unique_and_reproducible():
    xy = np.random.default_rng(12).normal(size=(100,2))
    first = spread_indices(xy)
    assert len(set(first)) == 4
    assert first == spread_indices(xy)


def test_blank_and_blurred_nuclei_are_not_good_he():
    import cv2
    rgb = np.full((256,256,3),225,dtype=np.uint8)
    for y in range(20,240,20):
        for x in range(20,240,20):
            cv2.circle(rgb,(x,y),4,(60,35,115),-1)
    sharp = field_metrics(rgb)
    blurred = field_metrics(cv2.GaussianBlur(rgb,(41,41),10))
    blank = field_metrics(np.full_like(rgb,255))
    assert sharp['focus'] > blurred['focus']
    assert not blank['usable']
    assert not segmentation_fields(np.zeros((256,256),dtype=np.int32),rgb)['supported']
    assert POLICY['clinical_endpoints_used'] is False


def test_bad_he_cannot_be_compensated_by_more_spots(tmp_path):
    import pandas as pd
    from phenoniche.selection.tnbc import rank_samples
    rows=[]
    for i,name in enumerate(['TNBC1_CN1_C1','TNBC1_CN1_C2','TNBC2_CN1_E2']):
        folder=tmp_path/name;folder.mkdir()
        rng=np.random.default_rng(10+i)
        np.savez_compressed(folder/'spot_summary.npz',activities=rng.normal(size=(150,6)),
                            expression=rng.uniform(size=12),xy=rng.normal(size=(150,2)),good=np.ones(150,bool))
        bad=(i==1)
        rows.append(dict(sample=name,patient_id=1 if i<2 else 2,input_ready=True,
            good_spots=10000 if bad else 150,good_spot_fraction=1.,median_UMI=100000 if bad else 2000,
            median_detected_genes=10000 if bad else 1000,LR_pairs_detected_at_10pct=100,
            he_focus=.01 if bad else 1.,he_nuclear_contrast=.01 if bad else .1,
            he_usable_fraction=.25 if bad else 1.,segmentation_supported_fraction=0. if bad else 1.,
            segmentation_ring_contrast=0. if bad else .1))
    result=rank_samples(pd.DataFrame(rows),tmp_path)
    assert set(result.loc[result.selected,'sample']) == {'TNBC1_CN1_C1','TNBC2_CN1_E2'}
