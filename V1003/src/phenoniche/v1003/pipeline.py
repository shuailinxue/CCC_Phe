from pathlib import Path
import json

from .config import default_config
from .data import CCCProportionDataset, load_cached_input, prepare_prime5k_ccc
from .interpretation import export_results
from .training import infer_all, load_trained_model, train_autoencoder


def _complete(output_dir, config=None):
    output_dir = Path(output_dir)
    required = ["summary.json", "best_checkpoint.pt", "Z.npy", "H.npy",
                "assignments.csv.gz", "niche_counts.csv", "top_ccc.csv",
                "top_sender_receiver.csv", "top_lr.csv", "H_cosine_similarity.csv"]
    if not all((output_dir / name).is_file() for name in required):
        return False
    if config is None:
        return True
    try:
        summary = json.loads((output_dir / "summary.json").read_text())
        return summary.get("training", {}).get("model_revision") == config.model_revision
    except (OSError, json.JSONDecodeError):
        return False


def run_pipeline(config=None, force=False):
    config = default_config() if config is None else config
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    cached = prepare_prime5k_ccc(config, force=force)
    training_path = output_dir / "training_summary.json"
    checkpoint_current = False
    if (output_dir / "best_checkpoint.pt").is_file() and training_path.is_file():
        try:
            checkpoint_current = json.loads(training_path.read_text()).get("model_revision") == config.model_revision
        except json.JSONDecodeError:
            checkpoint_current = False
    if force or not checkpoint_current:
        model, training_summary, dataset, device = train_autoencoder(
            cached["matrix"], cached["magnitude"], config, output_dir)
    else:
        model, device = load_trained_model(config, cached["matrix"].shape[1], output_dir)
        summary_path = training_path
        if not summary_path.is_file():
            raise FileNotFoundError("Checkpoint exists but training_summary.json is missing")
        training_summary = json.loads(summary_path.read_text())
        dataset = CCCProportionDataset(cached["matrix"], cached["magnitude"], config.epsilon)
    labels, confidence, dictionary = infer_all(model, dataset, config, output_dir, device)
    return export_results(labels, confidence, dictionary, cached, training_summary, config, output_dir)


def ensure_results(config=None):
    config = default_config() if config is None else config
    output_dir = Path(config.output_dir)
    if _complete(output_dir, config):
        return json.loads((output_dir / "summary.json").read_text())
    return run_pipeline(config)


def load_summary(config=None):
    config = default_config() if config is None else config
    path = Path(config.output_dir) / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"V1003 results are absent: {path}")
    return json.loads(path.read_text())
