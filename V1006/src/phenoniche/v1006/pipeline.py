from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import default_config
from .consolidation import consolidate_programs
from .data import prepare_input
from .interpretation import export_results
from .training import train_all


REQUIRED = ("summary.json", "E32.npy", "P32.npy", "Z32.npy", "Q8_direct.npy", "H32.npy", "Q8.npy", "H8.npy",
            "niche_labels.npy", "niche_assignments.csv.gz", "top_ccc.csv")

OBSOLETE_V1 = ("Z8.npy", "Z8_relative.npy", "program32_statistics.csv", "stage2_history.csv",
               "stage2_training_history.csv", "prototypes8.npy", "prototype_cosine_similarity.csv",
               "latent_whitening_center.npy", "latent_whitening_projection.npy")


def results_complete(config):
    output = Path(config.output_dir)
    if not all((output / name).is_file() for name in REQUIRED): return False
    try: return json.loads((output / "summary.json").read_text()).get("revision") == config.revision
    except (OSError, json.JSONDecodeError): return False


def load_artifacts(config=None):
    config = config or default_config(); output = Path(config.output_dir)
    if not results_complete(config):
        raise FileNotFoundError(f"Complete V1006 results are absent from {output}; run ensure_results()")
    return {
        "config": config, "output": output, "summary": json.loads((output / "summary.json").read_text()),
        "E32": np.load(output / "E32.npy", mmap_mode="r"), "P32": np.load(output / "P32.npy", mmap_mode="r"),
        "Z32": np.load(output / "Z32.npy", mmap_mode="r"), "H32": np.load(output / "H32.npy"),
        "Q8": np.load(output / "Q8.npy", mmap_mode="r"), "H8": np.load(output / "H8.npy"),
        "labels": np.load(output / "niche_labels.npy"), "assignments": pd.read_csv(output / "niche_assignments.csv.gz"),
        "counts": pd.read_csv(output / "niche_counts.csv"), "mapping": pd.read_csv(output / "latent32_to_niche8.csv"),
        "top_ccc": pd.read_csv(output / "top_ccc.csv"),
    }


def run_pipeline(config=None, force=False):
    config = config or default_config(); output = Path(config.output_dir); output.mkdir(parents=True, exist_ok=True)
    if results_complete(config) and not force: return load_artifacts(config)
    for name in OBSOLETE_V1:
        path = output / name
        if path.is_file(): path.unlink()
    prepared = prepare_input(config, force=force)
    _, _, arrays, h32, checkpoints = train_all(prepared, config, output)
    groups = np.repeat(np.arange(config.final_niches), config.latent_dim // config.final_niches)
    q8, h8, labels, confidence, mapping = consolidate_programs(arrays["P32"], arrays["Z32"], h32, groups, config.epsilon)
    if not np.allclose(q8, np.asarray(arrays["Q8_direct"]), atol=1e-5):
        raise RuntimeError("Hierarchical program mixture and consolidated Q8 are inconsistent")
    export_results(prepared, config, output, arrays, h32, q8, h8, labels, confidence, mapping, checkpoints)
    return load_artifacts(config)


def ensure_results(config=None):
    config = config or default_config()
    return load_artifacts(config) if results_complete(config) else run_pipeline(config)
