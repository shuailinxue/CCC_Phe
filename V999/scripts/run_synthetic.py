import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenoniche.data.synthetic import generate_synthetic
from phenoniche.data.scaling import BlockScaler
from phenoniche.evaluation.matching import recovery_report
from phenoniche.evaluation.metrics import concordance_index
from phenoniche.model.normalization import normalize_factors
from phenoniche.training.config import TrainingConfig, LossWeights
from phenoniche.training.trainer import train, joint_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fit-seed", type=int, default=31)
    parser.add_argument("--niches", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=800)
    parser.add_argument("--joint-epochs", type=int, default=800)
    parser.add_argument("--phenotype-weight", type=float, default=0.01)
    parser.add_argument("--regularization-weight", type=float, default=0.0001)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "synthetic_inferred")
    args = parser.parse_args()
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    synthetic = generate_synthetic(seed=args.seed, number_of_niches=args.niches,
                                   cell_types=max(8, args.niches), contacts=max(10, args.niches),
                                   communications=max(16, args.niches))
    scaler = BlockScaler.fit(synthetic.data)
    data = scaler.transform(synthetic.data)
    truth = {key: value.clone() for key, value in synthetic.truth.items()}
    truth["HC"] /= scaler.composition
    truth["HI"] /= scaler.communication
    truth["HO"] /= scaler.topology
    config = TrainingConfig(number_of_niches=args.niches, device=args.device, seed=args.fit_seed,
                            warmup_epochs=args.warmup_epochs, joint_epochs=args.joint_epochs)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"config": asdict(config), "scaling": asdict(scaler), "data_seed": args.seed,
               "torch_version": torch.__version__, "numpy_version": np.__version__,
               "patients": data.CB.shape[0], "anchors": data.CS.shape[0],
               "event_fraction": float(data.event.mean()), "evaluation": "in-sample synthetic recovery"}
    for label, ph in (("unsupervised", 0.0), ("supervised", args.phenotype_weight)):
        weights = LossWeights.per_entry(data, ph=ph, reg=args.regularization_weight)
        result = train(data, config, weights)
        factors = normalize_factors(result.model.factors())
        report = recovery_report(factors, truth)
        risk = (factors["WB"] @ factors["gamma"]).detach().cpu().numpy()
        report["training_c_index"] = concordance_index(data.time.numpy(), data.event.numpy(), risk)
        report["final_losses"] = {key: float(value.detach()) for key, value in joint_loss(
            result.model, data.to(args.device), weights).items()}
        report["weights"] = asdict(weights)
        summary[label] = report
        np.savez_compressed(args.output / f"{label}_factors.npz", **{
            key: value.detach().cpu().numpy() for key, value in factors.items()})
        (args.output / f"{label}_history.json").write_text(json.dumps(result.history, indent=2))
        torch.save(result.model.state_dict(), args.output / f"{label}_state.pt")
    summary["comparison_notes"] = [
        "Unsupervised gamma remains zero; its C-index is a constant-risk baseline, not a separately fitted Cox model.",
        "All C-indices are in-sample and use inferred activities; these scores do not establish generalization.",
        "Supervised recovery improvements are measured, not assumed; a separable simulation can favor unsupervised reconstruction."
    ]
    summary["supervision_delta"] = {
        name: summary["supervised"][name]["cosine"] - summary["unsupervised"][name]["cosine"]
        for name in ("HC", "HI", "HO", "WB", "WS")}
    np.savez_compressed(args.output / "ground_truth.npz", **{key: value.numpy() for key, value in synthetic.truth.items()})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
