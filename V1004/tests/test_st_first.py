import inspect
import torch
from phenoniche.simulation.structured_ccc import StructuredCCCConfig, generate_structured_ccc
from phenoniche.training.st_first import STFirstConfig, fit_st_only, infer_spatial_activities


def test_st_only_api_cannot_receive_survival():
    parameters = inspect.signature(fit_st_only).parameters
    assert "time" not in parameters and "event" not in parameters


def test_st_only_shapes_and_spatial_reinference():
    generated = generate_structured_ccc(StructuredCCCConfig(programs=2))
    data = generated.data
    result = fit_st_only(data.CS, data.IS, data.OS, STFirstConfig(iterations=3, seed=31))
    assert result.WS.shape == (1200, 6)
    assert result.HC.shape == (6, 8)
    assert result.HI.shape == (6, 128)
    assert result.HO.shape == (6, 24)
    inferred = infer_spatial_activities(data.CS, data.IS, data.OS, result.HC, result.HI, result.HO, steps=3)
    assert inferred.shape == result.WS.shape
    assert torch.isfinite(inferred).all() and (inferred >= 0).all()
