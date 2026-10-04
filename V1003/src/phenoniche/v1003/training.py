from pathlib import Path
import copy
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .data import CCCProportionDataset, resolve_device
from .losses import loss_components
from .model import InterpretableCCCAutoencoder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deterministic_split(n_samples, validation_fraction, seed):
    if n_samples < 3:
        raise ValueError("At least three samples are required")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    n_validation = max(1, int(round(validation_fraction * n_samples)))
    if n_validation >= n_samples:
        raise ValueError("Validation split leaves no training samples")
    return np.sort(order[n_validation:]), np.sort(order[:n_validation])


def _loader(dataset, indices, batch_size, shuffle, seed, pin_memory):
    subset = Subset(dataset, np.asarray(indices, dtype=np.int64).tolist())
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=pin_memory, generator=generator, drop_last=False)


def _run_epoch(model, loader, config, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("total", "reconstruction", "cluster", "h_entropy", "h_diversity")}
    observations = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for proportions, magnitudes, _ in loader:
            proportions = proportions.to(device, non_blocking=True)
            magnitudes = magnitudes.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            weights, reconstructed, _ = model(proportions, magnitudes)
            cluster_weights = weights
            if training:
                # Optimize the information-maximization term on the exact
                # representation used at inference. Gradients remain enabled;
                # eval mode only disables Dropout for this second pass.
                model.eval()
                cluster_weights = model.encode(proportions)
                model.train()
            losses = loss_components(
                proportions, reconstructed, weights, model.dictionary, config,
                cluster_weights=cluster_weights,
            )
            if training:
                losses["total"].backward()
                optimizer.step()
            count = len(proportions)
            observations += count
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * count
    if observations == 0:
        raise RuntimeError("Empty data loader")
    return {key: value / observations for key, value in totals.items()}


def train_autoencoder(matrix, magnitude, config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = resolve_device(config.device)
    dataset = CCCProportionDataset(matrix, magnitude, config.epsilon)
    train_index, validation_index = deterministic_split(len(dataset), config.validation_fraction, config.seed)
    np.savez_compressed(output_dir / "split_indices.npz", train=train_index, validation=validation_index)
    pin = device.startswith("cuda")
    train_loader = _loader(dataset, train_index, config.batch_size, True, config.seed, pin)
    validation_loader = _loader(dataset, validation_index, config.batch_size, False, config.seed, pin)
    model = InterpretableCCCAutoencoder(
        matrix.shape[1], config.niches, config.hidden_1, config.hidden_2,
        config.dropout, config.epsilon,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    checkpoint = output_dir / "best_checkpoint.pt"
    history = []
    best_validation = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, config.max_epochs + 1):
        train_metrics = _run_epoch(model, train_loader, config, device, optimizer)
        validation_metrics = _run_epoch(model, validation_loader, config, device)
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        if validation_metrics["total"] < best_validation - 1e-12:
            best_validation = validation_metrics["total"]
            best_epoch = epoch
            stale = 0
            payload = {
                "model_state": copy.deepcopy(model.state_dict()),
                "features": int(matrix.shape[1]),
                "config": config.to_dict(),
                "best_epoch": best_epoch,
                "validation_total": best_validation,
                "model_revision": config.model_revision,
            }
            temporary = checkpoint.with_suffix(".incomplete.pt")
            torch.save(payload, temporary)
            temporary.replace(checkpoint)
        else:
            stale += 1
        print(f"epoch {epoch:03d} train={train_metrics['total']:.6g} "
              f"validation={validation_metrics['total']:.6g} patience={stale}/{config.early_stopping_patience}")
        if stale >= config.early_stopping_patience:
            break
    if not checkpoint.is_file():
        raise RuntimeError("Training did not produce a valid checkpoint")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    final_train = _run_epoch(model, train_loader, config, device)
    final_validation = _run_epoch(model, validation_loader, config, device)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    training_summary = {
        "model_revision": config.model_revision,
        "device": device,
        "n_train": int(len(train_index)),
        "n_validation": int(len(validation_index)),
        "epochs_completed": int(len(history)),
        "best_epoch": int(best_epoch),
        "early_stopped": bool(len(history) < config.max_epochs),
        "best_validation_total": float(best_validation),
        "final_train_total": float(final_train["total"]),
        "final_validation_total": float(final_validation["total"]),
        "final_train_reconstruction": float(final_train["reconstruction"]),
        "final_validation_reconstruction": float(final_validation["reconstruction"]),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(training_summary, indent=2))
    return model, training_summary, dataset, device


def load_trained_model(config, features, output_dir, device=None):
    device = resolve_device(config.device) if device is None else device
    checkpoint = Path(output_dir) / "best_checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Best checkpoint is missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if int(payload["features"]) != int(features):
        raise ValueError("Checkpoint feature count does not match the cached CCC matrix")
    model = InterpretableCCCAutoencoder(
        features, config.niches, config.hidden_1, config.hidden_2,
        config.dropout, config.epsilon,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, device


def infer_all(model, dataset, config, output_dir, device):
    output_dir = Path(output_dir)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.startswith("cuda"))
    temporary = output_dir / "Z.incomplete.npy"
    z_output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                         shape=(len(dataset), config.niches))
    labels = np.empty(len(dataset), dtype=np.int16)
    confidence = np.empty(len(dataset), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for proportions, _, indices in loader:
            weights = model.encode(proportions.to(device, non_blocking=True)).cpu().numpy()
            index = indices.numpy()
            z_output[index] = weights
            labels[index] = weights.argmax(axis=1).astype(np.int16)
            confidence[index] = weights.max(axis=1)
    z_output.flush()
    del z_output
    temporary.replace(output_dir / "Z.npy")
    dictionary = model.dictionary.detach().cpu().numpy().astype(np.float32)
    np.save(output_dir / "H.npy", dictionary)
    return labels, confidence, dictionary
