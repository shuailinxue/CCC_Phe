import numpy as np
import torch
from phenoniche.simulation.structured_ccc import StructuredCCCConfig, decode_feature, feature_index, generate_structured_ccc


def test_structured_ccc_primary_contract():
    result = generate_structured_ccc()
    data, truth, metadata = result.data, result.truth, result.metadata
    assert data.CB.shape == (320, 8)
    assert data.CS.shape == (1200, 8)
    assert data.IB.shape == data.Iraw_B.shape == data.Opp_B.shape == (320, 768)
    assert data.IS.shape == data.Iraw_S.shape == data.Opp_S.shape == (1200, 768)
    assert truth["HI_tensor"].shape == (6, 8, 8, 12)
    assert torch.equal(truth["HI"], truth["HI_tensor"].reshape(6, 768))
    assert torch.equal(truth["HC"][0], truth["HC"][1])
    assert metadata["A_B_truth"]["HI_cosine"] < 0.05
    assert abs(metadata["A_B_truth"]["WB_correlation"]) < 0.10
    assert abs(metadata["A_B_truth"]["WS_correlation"]) < 0.15
    assert metadata["active_edges_per_niche"] == 48
    assert (data.IB >= 0).all() and (data.IS >= 0).all()


def test_feature_mapping_is_invertible_for_all_dimensions():
    for programs in (2, 4, 8, 12):
        config = StructuredCCCConfig(programs=programs)
        for index in range(config.communication_features):
            sender, receiver, program = decode_feature(index, 8, programs)
            assert feature_index(sender, receiver, program, 8, programs) == index
