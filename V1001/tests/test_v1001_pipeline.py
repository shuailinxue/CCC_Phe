import inspect
import pytest
import torch
from phenoniche.evaluation.structured_st_first_benchmark import _prepare
from phenoniche.simulation.structured_ccc import StructuredCCCConfig
from phenoniche.training.st_first import STFirstConfig, fit_st_only
from phenoniche.v1001.simulation import generate_neighborhood_structured
from phenoniche.v1001.spatial_ccc import aggregate_spatial_ccc


@pytest.fixture(scope="module")
def neighborhood_result():
    return generate_neighborhood_structured(StructuredCCCConfig(programs=2))


def test_factorization_shapes(neighborhood_result):
    generated = neighborhood_result.structured
    baseline = _prepare(2)
    scaling = baseline[2]
    cs, is_, os = scaling.spatial(generated.data.CS, generated.data.IS, generated.data.OS)
    result = fit_st_only(cs, is_, os, STFirstConfig(iterations=2, seed=31))
    assert result.WS.shape == (1200, 6)
    assert result.HC.shape == (6, 8)
    assert result.HI.shape == (6, 128)
    assert result.HO.shape == (6, 24)


def test_bulk_feature_axis_alignment(neighborhood_result):
    generated = neighborhood_result.structured
    assert generated.data.IB.shape[1] == generated.data.IS.shape[1]
    assert generated.data.IB.shape[1] == 8 * 8 * 2
    assert generated.metadata["feature_mapping"][37]["index"] == 37


def test_no_phenotype_in_st():
    parameters = set(inspect.signature(aggregate_spatial_ccc).parameters)
    forbidden = {"time", "event", "survival", "beta", "WB", "phenotype"}
    assert not parameters & forbidden
    source = inspect.getsource(aggregate_spatial_ccc)
    assert all(name not in source for name in ("cox", "survival", "event", "phenotype"))
