from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from .data import resolve_device
from .losses import (assignment_entropy, consistency_loss, huber, minimum_usage_loss,
                     prototype_cluster_loss, prototype_separation)
from .model import PrototypeHead, RepresentationAutoencoder


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class _BatchLoader:
    def __init__(self, prepared, rows, batch_size, shuffle, seed):
        self.matrix = np.load(prepared.matrix_path, mmap_mode="r")
        self.rows = np.asarray(rows, dtype=np.int64); self.columns = prepared.retained_indices
        self.scale = prepared.scale; self.batch_size = int(batch_size); self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        rows = self.rng.permutation(self.rows) if self.shuffle else self.rows
        for start in range(0, len(rows), self.batch_size):
            batch_rows = rows[start:start + self.batch_size]
            block = np.asarray(self.matrix[batch_rows], dtype=np.float32)[:, self.columns]
            yield torch.from_numpy(np.ascontiguousarray(block / self.scale)), torch.from_numpy(batch_rows.copy())

    def __len__(self): return (len(self.rows) + self.batch_size - 1) // self.batch_size


def _loader(prepared, rows, config, shuffle=False, seed=0):
    return _BatchLoader(prepared, rows, config.batch_size, shuffle, seed)


def _stage1_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None; model.train(training); total = count = 0
    with torch.enable_grad() if training else torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            if training: optimizer.zero_grad(set_to_none=True)
            _, xhat = model(x); loss = huber(xhat, x)
            if training: loss.backward(); optimizer.step()
            total += float(loss) * len(x); count += len(x)
    return total / max(count, 1)


def fit_stage1(prepared, config, output):
    device = resolve_device(config.device); set_seed(config.seed)
    model = RepresentationAutoencoder(len(prepared.retained_indices), config.latent_dim,
                                      config.encoder_hidden_1, config.encoder_hidden_2, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.stage1_lr, weight_decay=config.weight_decay)
    train = _loader(prepared, prepared.train_indices, config, True, config.seed)
    validation = _loader(prepared, prepared.validation_indices, config)
    history, best, stale = [], np.inf, 0; checkpoint = output / "stage1_best_checkpoint.pt"
    for epoch in range(1, config.stage1_max_epochs + 1):
        tr = _stage1_epoch(model, train, device, optimizer); va = _stage1_epoch(model, validation, device)
        history.append({"epoch": epoch, "train_reconstruction": tr, "validation_reconstruction": va})
        if va < best - 1e-8:
            best, stale = va, 0
            torch.save({"revision": config.revision, "epoch": epoch, "features": model.features,
                        "model_state": model.state_dict(), "train_reconstruction": tr,
                        "validation_reconstruction": va}, checkpoint)
        else:
            stale += 1
            if stale >= config.stage1_patience: break
    pd.DataFrame(history).to_csv(output / "stage1_training_history.csv", index=False)
    payload = load_stage1_checkpoint(checkpoint, model, device)
    return model, payload


def load_stage1_checkpoint(path, model, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("features") != model.features: raise ValueError("Stage 1 checkpoint feature mismatch")
    model.load_state_dict(payload["model_state"]); return payload


def infer_latent(model, prepared, config, output_path):
    device = resolve_device(config.device); model.eval()
    temporary = output_path.with_name(output_path.stem + ".incomplete.npy")
    out = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                    shape=(prepared.matrix_shape[0], config.latent_dim))
    with torch.no_grad():
        for x, rows in _loader(prepared, np.arange(prepared.matrix_shape[0]), config):
            out[rows.numpy()] = model.encode(x.to(device, non_blocking=True)).cpu().numpy()
    out.flush(); del out; temporary.replace(output_path)
    return np.load(output_path, mmap_mode="r")


def initialize_prototypes(z0, train_indices, config, device):
    rng = np.random.default_rng(config.seed)
    rows = np.asarray(train_indices)
    if len(rows) > config.kmeans_sample_size:
        rows = np.sort(rng.choice(rows, config.kmeans_sample_size, replace=False))
    sample = np.asarray(z0[rows], dtype=np.float32)
    pca = PCA().fit(sample)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    variance_dim = int(np.searchsorted(cumulative, config.pca_variance_fraction) + 1)
    numerical_rank = int(np.sum(pca.explained_variance_ratio_ > 1e-4))
    cluster_dim = max(config.final_niches, min(variance_dim, numerical_rank or variance_dim))
    scale = np.sqrt(np.maximum(pca.explained_variance_[:cluster_dim], 1e-8))
    projection = pca.components_[:cluster_dim] / scale[:, None]
    sample = (sample - pca.mean_) @ projection.T
    norms = np.maximum(np.linalg.norm(sample, axis=1, keepdims=True), config.epsilon); sample /= norms
    kmeans = MiniBatchKMeans(n_clusters=config.final_niches, random_state=config.seed,
                             batch_size=4096, n_init=10, max_iter=100).fit(sample)
    head = PrototypeHead(config.final_niches, config.latent_dim, cluster_dim,
                         config.temperature, config.epsilon).to(device)
    with torch.no_grad():
        head.set_whitening(torch.from_numpy(pca.mean_.astype(np.float32)).to(device),
                           torch.from_numpy(projection.astype(np.float32)).to(device))
        head.prototypes.copy_(torch.from_numpy(kmeans.cluster_centers_).to(device))
    return head


def _augment(x, config):
    mask = (torch.rand_like(x) >= config.feature_mask_fraction).to(x.dtype)
    jitter = (1 + config.multiplicative_jitter * torch.randn_like(x)).clamp_min(0)
    return x * mask * jitter


def _prototype_epoch(model, head, loader, config, device, optimizer=None, freeze_model=False):
    training = optimizer is not None; model.train(training and not freeze_model); head.train(training)
    keys = ("reconstruction", "cluster", "consistency", "usage_penalty", "separation", "entropy", "total")
    sums = {key: 0. for key in keys}; usage = np.zeros(config.final_niches); hard = np.zeros(config.final_niches, dtype=np.int64); count = 0
    with torch.enable_grad() if training else torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            if training: optimizer.zero_grad(set_to_none=True)
            z, xhat = model(x); q = head(z)
            values = {"reconstruction": huber(xhat, x), "cluster": prototype_cluster_loss(q, config.epsilon),
                      "usage_penalty": minimum_usage_loss(q, config.minimum_soft_usage),
                      "separation": prototype_separation(head.prototypes, config.prototype_similarity_margin, config.epsilon),
                      "entropy": assignment_entropy(q, config.epsilon)}
            if training:
                q1 = head(model.encode(_augment(x, config))); q2 = head(model.encode(_augment(x, config)))
                values["consistency"] = consistency_loss(q1, q2)
            else:
                values["consistency"] = torch.zeros((), device=device)
            loss = (values["reconstruction"] + config.lambda_cluster * values["cluster"]
                    + config.lambda_consistency * values["consistency"]
                    + config.lambda_usage * values["usage_penalty"]
                    + config.lambda_separation * values["separation"]
                    + config.lambda_confidence * values["entropy"])
            values["total"] = loss
            if training: loss.backward(); optimizer.step()
            for key in keys: sums[key] += float(values[key]) * len(x)
            q_np = q.detach().cpu().numpy(); usage += q_np.sum(0); hard += np.bincount(q_np.argmax(1), minlength=config.final_niches)
            count += len(x)
    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    soft = usage / max(count, 1); hard_fraction = hard / max(count, 1)
    metrics.update({"min_soft_usage": float(soft.min()), "max_soft_usage": float(soft.max()),
                    "n_hard_active": int((hard > 0).sum()), "max_hard_fraction": float(hard_fraction.max())})
    return metrics


def fit_stage2(model, head, prepared, config, output):
    device = resolve_device(config.device)
    train = _loader(prepared, prepared.train_indices, config, True, config.seed + 1)
    validation = _loader(prepared, prepared.validation_indices, config)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.stage2_lr, weight_decay=config.weight_decay)
    initial = _prototype_epoch(model, head, validation, config, device)
    initial_healthy = (initial["n_hard_active"] == config.final_niches
                       and initial["max_hard_fraction"] < .80
                       and initial["min_soft_usage"] >= config.minimum_soft_usage / 4)
    best_score, stale = (initial["total"] if initial_healthy else np.inf), 0
    best = {"model_state": copy.deepcopy(model.state_dict()), "prototype_state": copy.deepcopy(head.state_dict()),
            "epoch": 0, "validation": initial}
    history = [{"epoch": 0, **{f"validation_{k}": v for k, v in initial.items()}, "accepted": True}]
    for epoch in range(1, config.stage2_max_epochs + 1):
        warmup = epoch <= config.prototype_warmup_epochs
        if epoch == config.prototype_warmup_epochs + 1:
            for parameter in model.parameters(): parameter.requires_grad_(True)
            optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                                          lr=config.stage2_lr, weight_decay=config.weight_decay)
        tr = _prototype_epoch(model, head, train, config, device, optimizer, freeze_model=warmup)
        va = _prototype_epoch(model, head, validation, config, device)
        healthy = (va["reconstruction"] <= initial["reconstruction"] * config.reconstruction_degradation_limit
                   and va["n_hard_active"] == config.final_niches and va["max_hard_fraction"] < .80
                   and va["min_soft_usage"] >= config.minimum_soft_usage / 4)
        improved = healthy and va["total"] < best_score - 1e-8
        history.append({"epoch": epoch, "phase": "prototype_warmup" if warmup else "joint", **{f"train_{k}": v for k, v in tr.items()},
                        **{f"validation_{k}": v for k, v in va.items()}, "accepted": improved})
        if improved:
            best_score, stale = va["total"], 0
            best = {"model_state": copy.deepcopy(model.state_dict()), "prototype_state": copy.deepcopy(head.state_dict()),
                    "epoch": epoch, "validation": va}
        else:
            stale += 1
            if stale >= config.stage2_patience: break
    for parameter in model.parameters(): parameter.requires_grad_(True)
    best.update({"revision": config.revision, "features": model.features, "cluster_dim": head.cluster_dim})
    checkpoint = output / "stage2_best_checkpoint.pt"; torch.save(best, checkpoint)
    pd.DataFrame(history).to_csv(output / "stage2_training_history.csv", index=False)
    load_stage2_checkpoint(checkpoint, model, head, device)
    best["final_train"] = _prototype_epoch(model, head, train, config, device)
    best["final_validation"] = _prototype_epoch(model, head, validation, config, device)
    torch.save(best, checkpoint); return best


def load_stage2_checkpoint(path, model, head, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("features") != model.features: raise ValueError("Stage 2 checkpoint feature mismatch")
    model.load_state_dict(payload["model_state"]); head.load_state_dict(payload["prototype_state"]); return payload


def infer_assignments(model, head, prepared, config, z_path, q_path):
    device = resolve_device(config.device); model.eval(); head.eval(); n = prepared.matrix_shape[0]
    z_tmp = z_path.with_name(z_path.stem + ".incomplete.npy"); q_tmp = q_path.with_name(q_path.stem + ".incomplete.npy")
    z_out = np.lib.format.open_memmap(z_tmp, mode="w+", dtype=np.float32, shape=(n, config.latent_dim))
    q_out = np.lib.format.open_memmap(q_tmp, mode="w+", dtype=np.float32, shape=(n, config.final_niches))
    with torch.no_grad():
        for x, rows in _loader(prepared, np.arange(n), config):
            z = model.encode(x.to(device, non_blocking=True)); q = head(z)
            z_out[rows.numpy()] = z.cpu().numpy(); q_out[rows.numpy()] = q.cpu().numpy()
    z_out.flush(); q_out.flush(); del z_out, q_out; z_tmp.replace(z_path); q_tmp.replace(q_path)
    return np.load(z_path, mmap_mode="r"), np.load(q_path, mmap_mode="r")


def train_all(prepared, config, output):
    output.mkdir(parents=True, exist_ok=True)
    model, stage1 = fit_stage1(prepared, config, output)
    z0 = infer_latent(model, prepared, config, output / "Z0.npy")
    device = resolve_device(config.device); head = initialize_prototypes(z0, prepared.train_indices, config, device)
    stage2 = fit_stage2(model, head, prepared, config, output)
    z32, q8 = infer_assignments(model, head, prepared, config, output / "Z32.npy", output / "Q8.npy")
    prototypes = head.normalized_prototypes.detach().cpu().numpy().astype(np.float32)
    np.save(output / "prototypes8.npy", prototypes)
    np.save(output / "latent_whitening_center.npy", head.latent_center.detach().cpu().numpy().astype(np.float32))
    np.save(output / "latent_whitening_projection.npy", head.whitening_projection.detach().cpu().numpy().astype(np.float32))
    return model, head, z32, q8, prototypes, {"stage1": stage1, "stage2": stage2}
