from dataclasses import replace
import pytest
import torch
from phenoniche.model.niche_factorization import NicheFactorization
from phenoniche.data.schema import CohortData
from phenoniche.data.synthetic import generate_synthetic
from phenoniche.training.config import TrainingConfig, LossWeights
from phenoniche.training.trainer import train


@pytest.mark.parametrize("dimensions", [(7, 11, 3, 5, 9, 2), (13, 6, 4, 7, 3, 5)])
def test_model_shapes_positivity_and_purity(dimensions):
    p, n, c, e, f, k = dimensions
    model = NicheFactorization(*dimensions, composition=torch.rand(p, c), communication=torch.rand(p, f))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    output = model()
    for name, shape in {"CB": (p, c), "IB": (p, f), "CS": (n, c), "OS": (n, e), "IS": (n, f), "risk": (p,)}.items():
        assert output[name].shape == shape
    for name, value in output["factors"].items():
        if name != "gamma":
            assert (value >= 0).all()
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


@pytest.mark.parametrize("change", ["negative", "nonfinite", "patients", "features", "time", "event"])
def test_bad_data_rejected(tiny_data, change):
    if change == "negative":
        kwargs = {"CB": -tiny_data.CB}
    elif change == "nonfinite":
        kwargs = {"CB": tiny_data.CB * float("nan")}
    elif change == "patients":
        kwargs = {"IB": tiny_data.IB[:-1]}
    elif change == "features":
        kwargs = {"CS": tiny_data.CS[:, :-1]}
    elif change == "time":
        kwargs = {"time": torch.zeros_like(tiny_data.time)}
    else:
        kwargs = {"event": torch.full_like(tiny_data.event, 0.5)}
    with pytest.raises(ValueError):
        replace(tiny_data, **kwargs)


@pytest.mark.parametrize("kwargs", [{"number_of_niches": 0}, {"joint_epochs": -1}, {"learning_rate": float("nan")}, {"early_stopping": 0}])
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_eventless_training_guard(tiny_data):
    data = replace(tiny_data, event=torch.zeros_like(tiny_data.event))
    with pytest.raises(ValueError, match="observed event"):
        train(data, TrainingConfig(warmup_epochs=0, joint_epochs=2), LossWeights())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_training(tiny_data):
    result = train(tiny_data, TrainingConfig(number_of_niches=2, warmup_epochs=2, joint_epochs=3, device="cuda"), LossWeights.per_entry(tiny_data))
    assert result.model.bulk_factors().is_cuda
    assert torch.isfinite(result.model()["risk"]).all()
