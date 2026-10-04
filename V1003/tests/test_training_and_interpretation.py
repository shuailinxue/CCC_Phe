from dataclasses import replace
import numpy as np
import pandas as pd

from phenoniche.v1003.config import ExperimentConfig
from phenoniche.v1003.data import CCCProportionDataset
from phenoniche.v1003.interpretation import cosine_matrix, export_results, interpret_dictionary
from phenoniche.v1003.training import deterministic_split, infer_all, train_autoencoder


def test_split_is_disjoint_complete_and_reproducible():
    first = deterministic_split(101, 0.10, 40700)
    second = deterministic_split(101, 0.10, 40700)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    assert len(first[0]) + len(first[1]) == 101
    assert not np.intersect1d(first[0], first[1]).size


def test_tiny_training_checkpoint_and_inference(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.gamma(1.2, 1.0, size=(48, 12)).astype(np.float32)
    magnitude = x.sum(1).astype(np.float32)
    config = replace(ExperimentConfig(), hidden_1=16, hidden_2=10, dropout=0,
                     batch_size=16, max_epochs=2, early_stopping_patience=2,
                     device="cpu", output_dir=str(tmp_path))
    model, summary, dataset, device = train_autoencoder(x, magnitude, config, tmp_path)
    labels, confidence, dictionary = infer_all(model, dataset, config, tmp_path, device)
    assert summary["best_epoch"] in (1, 2)
    assert (tmp_path / "best_checkpoint.pt").is_file()
    assert (tmp_path / "Z.npy").is_file() and (tmp_path / "H.npy").is_file()
    assert labels.shape == confidence.shape == (48,)
    assert dictionary.shape == (8, 12)
    assert np.allclose(dictionary.sum(1), 1, atol=1e-6)


def test_streaming_dataset_normalizes_each_nonzero_row():
    x = np.array([[1, 2, 3], [0, 0, 0]], dtype=np.float32)
    magnitude = x.sum(1)
    dataset = CCCProportionDataset(x, magnitude)
    first, first_magnitude, index = dataset[0]
    second, _, _ = dataset[1]
    assert np.isclose(first.sum().item(), 1)
    assert first_magnitude.item() == 6 and index == 0
    assert second.sum().item() == 0


def test_dictionary_interpretation_and_cosine_shapes():
    dictionary = np.arange(1, 8 * 6 + 1, dtype=float).reshape(8, 6)
    dictionary /= dictionary.sum(1, keepdims=True)
    features = pd.DataFrame({
        "ccc": [f"A → B | L{i}–R{i}" for i in range(6)],
        "sender": ["A"] * 6, "receiver": ["B"] * 6,
        "lr_id": [f"lr{i}" for i in range(6)],
        "ligand": [f"L{i}" for i in range(6)],
        "receptor": [f"R{i}" for i in range(6)],
    })
    ccc, pair, lr = interpret_dictionary(dictionary, features, 3)
    similarity = cosine_matrix(dictionary)
    assert len(ccc) == 24 and len(pair) == 8 and len(lr) == 24
    assert similarity.shape == (8, 8)
    assert np.allclose(np.diag(similarity), 1)


def test_export_summary_is_json_serializable_with_duplicate_pairs(tmp_path):
    config = replace(ExperimentConfig(), output_dir=str(tmp_path), top_ccc=2)
    z = np.full((8, 8), 1 / 8, dtype=np.float32)
    np.save(tmp_path / "Z.npy", z)
    dictionary = np.full((8, 3), 1 / 3, dtype=np.float32)
    features = pd.DataFrame({
        "ccc": [f"A → B | L{i}–R{i}" for i in range(3)],
        "sender": ["A"] * 3, "receiver": ["B"] * 3,
        "lr_id": [f"lr{i}" for i in range(3)],
        "ligand": [f"L{i}" for i in range(3)],
        "receptor": [f"R{i}" for i in range(3)],
    })
    cached = {
        "cells": pd.DataFrame({"cell_id": [f"c{i}" for i in range(8)], "cell_type": ["A"] * 8}),
        "coordinates": np.zeros((8, 2), dtype=np.float32),
        "magnitude": np.ones(8, dtype=np.float32),
        "features": features,
    }
    summary = export_results(np.arange(8), np.ones(8), dictionary, cached,
                             {"best_epoch": 1}, config, tmp_path)
    assert summary["duplicated_H"]
    assert "active_factor" in pd.read_csv(tmp_path / "top_ccc.csv").columns
    assert isinstance(summary["duplicated_H_pairs_at_or_above_0.95"][0]["niche_a"], int)
    assert (tmp_path / "summary.json").is_file()
