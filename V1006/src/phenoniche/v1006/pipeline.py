from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import default_config
from .data import prepare_input
from .interpretation import export_results
from .training import train_all


REQUIRED = ("summary.json", "Z32.npy", "Q8.npy", "prototypes8.npy", "H8.npy",
            "niche_labels.npy", "niche_assignments.csv.gz", "top_ccc.csv")

OBSOLETE_V1 = ("H32.npy", "Z8.npy", "Z8_relative.npy", "H32_cosine_similarity.csv",
               "latent32_to_niche8.csv", "program32_statistics.csv", "stage2A_training_history.csv",
               "stage2B_training_history.csv", "stage2_history.csv", "stage2a_best_checkpoint.pt")


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
        "Z32": np.load(output / "Z32.npy", mmap_mode="r"), "Q8": np.load(output / "Q8.npy", mmap_mode="r"),
        "prototypes": np.load(output / "prototypes8.npy"), "H8": np.load(output / "H8.npy"),
        "labels": np.load(output / "niche_labels.npy"), "assignments": pd.read_csv(output / "niche_assignments.csv.gz"),
        "counts": pd.read_csv(output / "niche_counts.csv"), "top_ccc": pd.read_csv(output / "top_ccc.csv"),
    }


def run_pipeline(config=None, force=False):
    config = config or default_config(); output = Path(config.output_dir); output.mkdir(parents=True, exist_ok=True)
    if results_complete(config) and not force: return load_artifacts(config)
    for name in OBSOLETE_V1:
        path = output / name
        if path.is_file(): path.unlink()
    prepared = prepare_input(config, force=force)
    _, _, z32, q8, prototypes, checkpoints = train_all(prepared, config, output)
    export_results(prepared, config, output, z32, q8, prototypes, checkpoints)
    return load_artifacts(config)


def ensure_results(config=None):
    config = config or default_config()
    return load_artifacts(config) if results_complete(config) else run_pipeline(config)
