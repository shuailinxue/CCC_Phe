import pytest
import torch
from phenoniche.data.synthetic import generate_synthetic
from phenoniche.data.scaling import BlockScaler
from phenoniche.training.config import TrainingConfig, LossWeights
from phenoniche.training.trainer import train


@pytest.fixture(scope="session", autouse=True)
def single_thread():
    torch.set_num_threads(1)


@pytest.fixture
def tiny_data():
    return generate_synthetic(patients=24, anchors=20, cell_types=4, contacts=5,
                              communications=6, number_of_niches=2).data


@pytest.fixture(scope="session")
def recovered():
    synthetic = generate_synthetic()
    scaler = BlockScaler.fit(synthetic.data)
    data = scaler.transform(synthetic.data)
    truth = {key: value.clone() for key, value in synthetic.truth.items()}
    truth["HC"] /= scaler.composition
    truth["HI"] /= scaler.communication
    truth["HO"] /= scaler.topology
    result = train(data, TrainingConfig(), LossWeights.per_entry(data, ph=0.01, reg=0.0001))
    return data, truth, result
