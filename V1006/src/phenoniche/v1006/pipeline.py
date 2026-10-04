from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import default_config
from .consolidation import consolidate_programs, evaluate_cell_latent_k_range
from .data import load_prepared_input, prepare_input
from .interpretation import export_model_selection, export_results
from .training import train_all


NETWORK_REQUIRED = ("summary.json", "E32.npy", "P32.npy", "Z32.npy", "H32.npy", "activity.npy",
                    "Q8_direct.npy", "H8.npy", "niche_labels.npy", "assignment_confidence.npy",
                    "latent32_to_niche8.csv")
SELECTION_REQUIRED = ("model_selection_summary.json", "model_selection/K4_15_metrics.csv",
                      "selected_model/Q.npy", "selected_model/H.npy",
                      "selected_model/niche_labels.npy", "selected_model/niche_assignments.csv.gz",
                      "selected_model/niche_counts.csv", "selected_model/top_ccc.csv",
                      "K8_comparator/Q.npy", "K8_comparator/H.npy")
REQUIRED = NETWORK_REQUIRED + SELECTION_REQUIRED

OBSOLETE_V1 = ("Z8.npy", "Z8_relative.npy", "program32_statistics.csv", "stage2_history.csv",
               "stage2_training_history.csv", "prototypes8.npy", "prototype_cosine_similarity.csv",
               "latent_whitening_center.npy", "latent_whitening_projection.npy")


def network_results_complete(config):
    output = Path(config.output_dir)
    if not all((output / name).is_file() for name in NETWORK_REQUIRED):
        return False
    try:
        return json.loads((output / "summary.json").read_text()).get("revision") == config.revision
    except (OSError, json.JSONDecodeError):
        return False


def selection_results_complete(config):
    output = Path(config.output_dir)
    if not all((output / name).is_file() for name in SELECTION_REQUIRED):
        return False
    try:
        summary = json.loads((output / "model_selection_summary.json").read_text())
        return summary.get("consolidation_revision") == config.consolidation_revision
    except (OSError, json.JSONDecodeError):
        return False


def results_complete(config):
    return network_results_complete(config) and selection_results_complete(config)


def load_artifacts(config=None):
    config = config or default_config()
    output = Path(config.output_dir)
    if not results_complete(config):
        raise FileNotFoundError(f"Complete V1006 selected-K results are absent from {output}; run ensure_results()")
    network_summary = json.loads((output / "summary.json").read_text())
    selection_summary = json.loads((output / "model_selection_summary.json").read_text())
    summary = {**network_summary, **selection_summary}
    selected = output / "selected_model"
    comparator = output / "K8_comparator"
    return {
        "config": config, "output": output, "summary": summary,
        "selected_K": int(selection_summary["selected_K"]),
        "model_selection": pd.read_csv(output / "model_selection/K4_15_metrics.csv"),
        "E32": np.load(output / "E32.npy", mmap_mode="r"),
        "P32": np.load(output / "P32.npy", mmap_mode="r"),
        "Z32": np.load(output / "Z32.npy", mmap_mode="r"),
        "H32": np.load(output / "H32.npy"),
        "Q": np.load(selected / "Q.npy", mmap_mode="r"),
        "H": np.load(selected / "H.npy"),
        "labels": np.load(selected / "niche_labels.npy"),
        "assignments": pd.read_csv(selected / "niche_assignments.csv.gz"),
        "counts": pd.read_csv(selected / "niche_counts.csv"),
        "mapping": pd.read_csv(selected / "program32_to_niche.csv"),
        "top_ccc": pd.read_csv(selected / "top_ccc.csv"),
        "Q8": np.load(comparator / "Q.npy", mmap_mode="r"),
        "H8": np.load(comparator / "H.npy"),
        "labels_K8": np.load(comparator / "niche_labels.npy"),
        "counts_K8": pd.read_csv(comparator / "niche_counts.csv"),
    }


def run_model_selection(config=None, force=False):
    config = config or default_config()
    output = Path(config.output_dir)
    if not network_results_complete(config):
        raise FileNotFoundError("Existing Stage1 AE, P32/Z32 and H32 are required before consolidation")
    if selection_results_complete(config) and not force:
        return load_artifacts(config)
    prepared = load_prepared_input(config)
    p32 = np.load(output / "P32.npy", mmap_mode="r")
    z32 = np.load(output / "Z32.npy", mmap_mode="r")
    h32 = np.load(output / "H32.npy")
    activity = np.load(output / "activity.npy", mmap_mode="r")
    table, candidates, selected_k, reason = evaluate_cell_latent_k_range(
        p32, activity, h32, prepared.coordinates, config)
    comparator = {
        "Q": np.load(output / "Q8_direct.npy", mmap_mode="r"),
        "H": np.load(output / "H8.npy"),
        "labels": np.load(output / "niche_labels.npy"),
        "confidence": np.load(output / "assignment_confidence.npy", mmap_mode="r"),
        "mapping": pd.read_csv(output / "latent32_to_niche8.csv"),
    }
    export_model_selection(prepared, config, output, table, candidates, selected_k, reason, comparator)
    return load_artifacts(config)


def run_pipeline(config=None, force=False):
    """Train only when network artifacts are absent; otherwise run post-training selection."""
    config = config or default_config()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if network_results_complete(config) and not force:
        return run_model_selection(config)
    for name in OBSOLETE_V1:
        path = output / name
        if path.is_file():
            path.unlink()
    prepared = prepare_input(config, force=force)
    _, _, arrays, h32, checkpoints = train_all(prepared, config, output)
    groups = np.repeat(np.arange(config.final_niches), config.latent_dim // config.final_niches)
    q8, h8, labels, confidence, mapping = consolidate_programs(
        arrays["P32"], arrays["Z32"], h32, groups, config.epsilon)
    if not np.allclose(q8, np.asarray(arrays["Q8_direct"]), atol=1e-5):
        raise RuntimeError("Architectural K=8 mixture and consolidated Q8 are inconsistent")
    export_results(prepared, config, output, arrays, h32, q8, h8, labels,
                   confidence, mapping, checkpoints)
    return run_model_selection(config, force=True)


def ensure_results(config=None):
    config = config or default_config()
    if results_complete(config):
        return load_artifacts(config)
    if network_results_complete(config):
        return run_model_selection(config)
    return run_pipeline(config)
