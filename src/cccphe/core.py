from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def device_from_name(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def cox_breslow_loss(
    risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor
) -> torch.Tensor:
    """Negative Cox partial log likelihood with Breslow handling of ties."""
    order = torch.argsort(time, descending=True)
    ordered_time = time[order]
    ordered_risk = risk[order]
    ordered_event = event[order]
    log_cumulative_risk = torch.logcumsumexp(ordered_risk, dim=0)
    _, counts = torch.unique_consecutive(ordered_time, return_counts=True)
    group_ends = torch.cumsum(counts, dim=0) - 1
    denominators = torch.repeat_interleave(log_cumulative_risk[group_ends], counts)
    events = torch.clamp(ordered_event.sum(), min=1.0)
    return -((ordered_risk - denominators) * ordered_event).sum() / events


def stratified_survival_folds(event: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Deterministic event-stratified fold assignment."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(event), dtype=int)
    for status in (0, 1):
        indices = np.flatnonzero(event == status)
        rng.shuffle(indices)
        assignment[indices] = np.arange(len(indices)) % folds
    return assignment


def structural_feature_indices(mask: np.ndarray) -> np.ndarray:
    """Flatten sender x receiver x LR entries observed in at least one sample."""
    structural = mask.any(axis=0) if mask.ndim == 4 else mask
    return np.flatnonzero(structural.reshape(-1))


def feature_batch(tensor, sample_indices: np.ndarray, feature_indices: np.ndarray) -> np.ndarray:
    values = (
        tensor[sample_indices]
        if isinstance(tensor, np.ndarray)
        else np.asarray(tensor.oindex[sample_indices])
    )
    return values.reshape(len(sample_indices), -1)[:, feature_indices]


def feature_mask(mask: np.ndarray, sample_indices: np.ndarray, feature_indices: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        flat = mask.reshape(-1)[feature_indices]
        return np.broadcast_to(flat, (len(sample_indices), len(feature_indices)))
    if mask.ndim == 4:
        return mask[sample_indices].reshape(len(sample_indices), -1)[:, feature_indices]
    raise ValueError(f"Expected a 3D or 4D mask, got shape {mask.shape}")


def ccc_metadata(
    input_dir: Path,
    feature_indices: np.ndarray,
    tensor_shape: tuple[int, int, int, int],
) -> pd.DataFrame:
    """Map flattened CCC columns back to sender, receiver, and LR identities."""
    cell_types = pd.read_csv(input_dir / "cell_types.tsv", sep="\t")["cell_type"]
    lr = pd.read_csv(input_dir / "lr_pairs.tsv", sep="\t")
    sender_idx, receiver_idx, lr_idx = np.unravel_index(feature_indices, tensor_shape[1:])
    selected_lr = lr.iloc[lr_idx].reset_index(drop=True)
    pathway = selected_lr["pathway"].fillna("unassigned").astype(str)
    receptor = selected_lr["receptor_expression_complex"].fillna(
        selected_lr["receptor_complex"]
    ).astype(str)
    module = pathway.where(pathway.str.lower().ne("unassigned"), "R:" + receptor)
    metadata = pd.DataFrame(
        {
            "feature_index": feature_indices,
            "sender": cell_types.iloc[sender_idx].to_numpy(),
            "receiver": cell_types.iloc[receiver_idx].to_numpy(),
            "lr_id": selected_lr["lr_id"].to_numpy(),
            "pair_key": selected_lr["pair_key"].to_numpy(),
            "ligand": selected_lr["ligand_expression_complex"].to_numpy(),
            "receptor": receptor.to_numpy(),
            "pathway": pathway.to_numpy(),
            "route_module": module.to_numpy(),
        }
    )
    metadata["route"] = (
        metadata["sender"]
        + "->"
        + metadata["receiver"]
        + "|"
        + metadata["route_module"]
    )
    return metadata


def ccc_smoothing_groups(metadata: pd.DataFrame) -> list[np.ndarray]:
    """Groups that define the CCC graph-smoothing structure."""
    route, _ = pd.factorize(metadata["route"], sort=True)
    lr, _ = pd.factorize(metadata["lr_id"], sort=True)
    module, _ = pd.factorize(metadata["route_module"], sort=True)
    return [route.astype(np.int64), lr.astype(np.int64), module.astype(np.int64)]
