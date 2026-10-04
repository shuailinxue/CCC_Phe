from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from phenoniche.v1006.config import ExperimentConfig
from phenoniche.v1006.consolidation import (consolidate_programs, evaluate_cell_latent_k_range, evaluate_k_range,
                                            hierarchical_program_groups)
from phenoniche.v1006.data import PreparedInput, prepare_input
from phenoniche.v1006.interpretation import empirical_profiles, export_results
from phenoniche.v1006.losses import consistency_loss, dictionary_diversity, minimum_usage_loss
from phenoniche.v1006.model import ProgramDecoder, RepresentationAutoencoder
from phenoniche.v1006.pipeline import (NETWORK_REQUIRED, SELECTION_REQUIRED,
                                      network_results_complete, results_complete)
from phenoniche.v1006.training import load_stage1_checkpoint, load_stage2_checkpoint, train_all


def fake_source(tmp_path, x):
    cache = tmp_path / "source/cache"; cache.mkdir(parents=True)
    np.save(cache / "ccc.npy", x.astype(np.float32))
    pd.DataFrame({"feature_id": range(x.shape[1]), "sender": "A", "receiver": "B",
                  "ligand": [f"L{i}" for i in range(x.shape[1])],
                  "receptor": [f"R{i}" for i in range(x.shape[1])],
                  "ccc": [f"A → B | L{i}–R{i}" for i in range(x.shape[1])] }).to_csv(cache / "features.csv", index=False)
    pd.DataFrame({"cell_id": [f"c{i}" for i in range(len(x))], "cell_type": "T"}).to_csv(cache / "cells.csv.gz", index=False)
    np.save(cache / "coordinates.npy", np.arange(len(x) * 2).reshape(len(x), 2).astype(np.float32))
    (cache / "input_manifest.json").write_text("{}")
    return cache.parent


def config_for(tmp_path, source, **kwargs):
    return replace(ExperimentConfig(), v1003_output=str(source), output_dir=str(tmp_path / "out"), device="cpu", **kwargs)


def test_filtering_scaling_and_removed_feature(tmp_path):
    rng = np.random.default_rng(1); x = rng.gamma(1, 1, (100, 12)).astype(np.float32)
    x[:, 0] = 0; x[:, 1] = 3
    source = fake_source(tmp_path, x); cfg = config_for(tmp_path, source)
    prepared = prepare_input(cfg)
    assert 0 not in prepared.retained_indices and 1 not in prepared.retained_indices
    assert len(prepared.retained_indices) == 10
    assert np.all(prepared.scale > 0)
    audit = pd.read_csv(Path(cfg.output_dir) / "feature_filter_audit.csv")
    assert audit.retained.sum() == 10


def test_model_nonnegative_shapes_and_program_identifiability():
    x = torch.rand(7, 13); model = RepresentationAutoencoder(13)
    embedding, reconstruction = model(x); program = ProgramDecoder(32, 32, 13); result = program(embedding)
    assert embedding.shape == (7, 32) and reconstruction.shape == x.shape
    assert result["exposure"].shape == (7, 32) and result["reconstruction"].shape == x.shape
    assert torch.all(embedding >= 0) and torch.all(reconstruction >= 0) and torch.all(result["exposure"] >= 0)
    assert torch.allclose(result["mixture"].sum(1), torch.ones(7), atol=1e-6)
    assert torch.allclose(program.dictionary.sum(1), torch.ones(32), atol=1e-6)
    assert torch.allclose(result["exposure"].sum(1, keepdim=True), result["activity"], atol=1e-6)


def test_program_losses():
    q = torch.softmax(torch.randn(16, 32), dim=1)
    assert consistency_loss(q, q).item() == 0
    assert minimum_usage_loss(torch.full((16, 32), 1 / 32), .005).item() == 0
    assert dictionary_diversity(torch.eye(4), .5).item() == 0


def test_empirical_profiles_are_soft_weighted(tmp_path):
    x = np.asarray([[1, 0], [3, 2], [0, 4]], dtype=np.float32); path = tmp_path / "x.npy"; np.save(path, x)
    q = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    prepared = PreparedInput(path, x.shape, np.arange(2), np.ones(2), np.arange(2), np.array([2]),
                             pd.DataFrame(index=range(2)), pd.DataFrame(index=range(3)), np.zeros((3,2)), tmp_path)
    mean, enrichment = empirical_profiles(prepared, q)
    assert np.allclose(mean, [[2, 1], [0, 4]]) and enrichment.shape == (2, 2)


def test_checkpoint_loading(tmp_path):
    model = RepresentationAutoencoder(12); program = ProgramDecoder(32, 32, 12)
    p1 = tmp_path / "s1.pt"; p2 = tmp_path / "s2.pt"
    torch.save({"features": 12, "model_state": model.state_dict()}, p1)
    torch.save({"features": 12, "model_state": model.state_dict(), "program_state": program.state_dict()}, p2)
    load_stage1_checkpoint(p1, RepresentationAutoencoder(12))
    load_stage2_checkpoint(p2, RepresentationAutoencoder(12), ProgramDecoder(32, 32, 12))


def test_result_cache_detection(tmp_path):
    cfg = replace(ExperimentConfig(), output_dir=str(tmp_path))
    for name in NETWORK_REQUIRED + SELECTION_REQUIRED:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"x")
    (tmp_path / "summary.json").write_text(json.dumps({"revision": cfg.revision}))
    (tmp_path / "model_selection_summary.json").write_text(json.dumps({
        "consolidation_revision": cfg.consolidation_revision}))
    assert results_complete(cfg)
    (tmp_path / "selected_model/H.npy").unlink(); assert not results_complete(cfg)


def test_hierarchical_selection_is_deterministic_and_keeps_k8():
    rng = np.random.default_rng(12)
    h32 = rng.gamma(1, 1, (32, 18)).astype(np.float32)
    h32 /= h32.sum(1, keepdims=True)
    p32 = rng.dirichlet(np.ones(32), size=120).astype(np.float32)
    z32 = p32 * rng.gamma(2, 1, (120, 1)).astype(np.float32)
    cfg = ExperimentConfig()
    first = hierarchical_program_groups(h32, 8)
    second = hierarchical_program_groups(h32, 8)
    assert np.array_equal(first, second) and len(np.unique(first)) == 8
    table, candidates, selected, reason = evaluate_k_range(p32, z32, h32, cfg)
    assert table.K.tolist() == list(range(4, 16))
    assert table.selected.sum() == 1 and selected in table.K.to_numpy()
    assert 8 in candidates and "Q" in candidates[8] and candidates[8]["Q"].shape == (120, 8)
    assert np.allclose(candidates[selected]["Q"].sum(1), 1)
    assert reason


def test_cell_latent_selection_materializes_selected_soft_membership():
    rng = np.random.default_rng(31); n = 160
    p32 = rng.dirichlet(np.ones(32) * .3, size=n).astype(np.float32)
    activity = rng.gamma(2, 1, n).astype(np.float32)
    h32 = rng.dirichlet(np.ones(18), size=32).astype(np.float32)
    coordinates = rng.normal(size=(n, 2)).astype(np.float32)
    cfg = replace(ExperimentConfig(), clustering_sample_size=120, silhouette_sample_size=80)
    table, candidates, selected, reason = evaluate_cell_latent_k_range(
        p32, activity, h32, coordinates, cfg, chunk_size=64)
    chosen = candidates[selected]
    assert table.K.tolist() == list(range(4, 16)) and table.selected.sum() == 1
    assert chosen["Q"].shape == (n, selected) and np.allclose(chosen["Q"].sum(1), 1, atol=1e-5)
    assert chosen["H"].shape == (selected, 18) and reason


def test_tiny_end_to_end_smoke(tmp_path):
    rng = np.random.default_rng(4); x = rng.gamma(1, .2, (64, 12)).astype(np.float32)
    matrix = tmp_path / "x.npy"; np.save(matrix, x)
    features = pd.DataFrame({"sender": "A", "receiver": "B", "ligand": [f"L{i}" for i in range(12)],
                             "receptor": [f"R{i}" for i in range(12)], "ccc": [f"c{i}" for i in range(12)]})
    prepared = PreparedInput(matrix, x.shape, np.arange(12), np.ones(12, np.float32), np.arange(48), np.arange(48,64),
                             features, pd.DataFrame({"cell_id": range(64), "cell_type": "T"}), np.zeros((64,2)), tmp_path)
    cfg = replace(ExperimentConfig(), output_dir=str(tmp_path / "run"), device="cpu", batch_size=16,
                  stage1_max_epochs=2, stage1_patience=2, stage2a_max_epochs=2, stage2a_patience=2,
                  stage2b_max_epochs=2, stage2b_patience=2)
    output = Path(cfg.output_dir); output.mkdir()
    _, _, arrays, h32, checkpoints = train_all(prepared, cfg, output)
    assert arrays["Z32"].shape == (64, 32) and arrays["P32"].shape == (64, 32) and h32.shape == (32, 12)
    assert np.all(arrays["Z32"] >= 0) and np.allclose(np.asarray(arrays["P32"]).sum(1), 1, atol=1e-5)
    assert np.allclose(h32.sum(1), 1, atol=1e-5)
    groups = np.repeat(np.arange(8), 4)
    q8, h8, labels, confidence, mapping = consolidate_programs(arrays["P32"], arrays["Z32"], h32, groups)
    assert q8.shape == (64, 8) and h8.shape == (8, 12) and labels.max() <= 7
    assert np.allclose(q8, arrays["Q8_direct"], atol=1e-5)
    export_results(prepared, cfg, output, arrays, h32, q8, h8, labels, confidence, mapping, checkpoints)
    assert network_results_complete(cfg)


def test_notebook_code_is_syntactically_valid():
    notebook = Path(__file__).resolve().parents[1] / "XeniumPrime5K_Breast_V1006_niche_walkthrough.ipynb"
    payload = json.loads(notebook.read_text())
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{index}", "exec")
