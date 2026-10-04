import numpy as np
import torch

from phenoniche.v1003.losses import (
    dictionary_diversity_loss,
    dictionary_entropy_loss,
    information_cluster_loss,
    loss_components,
)
from phenoniche.v1003.config import ExperimentConfig
from phenoniche.v1003.model import InterpretableCCCAutoencoder


def test_encoder_and_dictionary_are_simplex_normalized():
    model = InterpretableCCCAutoencoder(13, niches=8, hidden_1=16, hidden_2=9, dropout=0)
    x = torch.rand(7, 13)
    x = x / x.sum(1, keepdim=True)
    z, reconstructed, raw = model(x, torch.arange(1, 8, dtype=torch.float32))
    assert z.shape == (7, 8)
    assert reconstructed.shape == (7, 13)
    assert raw.shape == (7, 13)
    assert torch.all(z >= 0) and torch.allclose(z.sum(1), torch.ones(7), atol=1e-6)
    assert torch.all(model.dictionary >= 0)
    assert torch.allclose(model.dictionary.sum(1), torch.ones(8), atol=1e-6)
    assert torch.allclose(reconstructed.sum(1), torch.ones(7), atol=1e-6)


def test_information_loss_prefers_balanced_confident_assignments():
    collapsed = torch.zeros(16, 8); collapsed[:, 0] = 1
    balanced = torch.eye(8).repeat(2, 1)
    assert information_cluster_loss(balanced) < information_cluster_loss(collapsed)


def test_entropy_and_diversity_penalties_have_expected_order():
    uniform = torch.full((8, 16), 1 / 16)
    sparse = torch.eye(8, 16)
    duplicate = torch.full((8, 16), 1 / 16)
    distinct = torch.eye(8, 16)
    assert dictionary_entropy_loss(sparse) < dictionary_entropy_loss(uniform)
    assert dictionary_diversity_loss(distinct) < dictionary_diversity_loss(duplicate)


def test_decoder_scale_restores_raw_magnitude():
    model = InterpretableCCCAutoencoder(11, niches=8, hidden_1=12, hidden_2=9, dropout=0)
    p = torch.rand(4, 11); p /= p.sum(1, keepdim=True)
    magnitude = torch.tensor([0.0, 1.0, 2.5, 10.0])
    _, p_hat, x_hat = model(p, magnitude)
    assert torch.allclose(x_hat.sum(1), magnitude, atol=1e-5)
    assert torch.allclose(p_hat.sum(1), torch.ones(4), atol=1e-6)


def test_cluster_loss_can_use_deterministic_encoder_weights():
    config = ExperimentConfig()
    target = torch.full((8, 4), .25)
    reconstructed = target.clone()
    stochastic = torch.zeros(8, 8); stochastic[:, 0] = 1
    deterministic = torch.eye(8)
    dictionary = torch.full((8, 4), .25)
    losses = loss_components(target, reconstructed, stochastic, dictionary, config,
                             cluster_weights=deterministic)
    assert torch.allclose(losses["cluster"], information_cluster_loss(deterministic))
