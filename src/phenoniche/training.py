from pathlib import Path
import copy
import math
import random

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans

from .data import resolve_device
from .losses import (consistency_loss, dictionary_diversity, dictionary_entropy, huber,
                     latent_preservation, minimum_usage_loss, normalized_entropy)
from .model import ProgramDecoder, RepresentationAutoencoder


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class _BatchLoader:
    def __init__(self, prepared, rows, batch_size, shuffle, seed):
        self.matrix = np.load(prepared.matrix_path, mmap_mode="r"); self.rows = np.asarray(rows, dtype=np.int64)
        self.columns = prepared.retained_indices; self.scale = prepared.scale
        self.batch_size, self.shuffle = int(batch_size), shuffle; self.rng = np.random.default_rng(seed)

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
    payload = load_stage1_checkpoint(checkpoint, model, device); return model, payload


def load_stage1_checkpoint(path, model, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("features") != model.features: raise ValueError("Stage 1 checkpoint feature mismatch")
    model.load_state_dict(payload["model_state"]); return payload


def infer_latent(model, prepared, config, output_path):
    device = resolve_device(config.device); model.eval(); temporary = output_path.with_name(output_path.stem + ".incomplete.npy")
    out = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                    shape=(prepared.matrix_shape[0], config.latent_dim))
    with torch.no_grad():
        for x, rows in _loader(prepared, np.arange(prepared.matrix_shape[0]), config):
            out[rows.numpy()] = model.encode(x.to(device, non_blocking=True)).cpu().numpy()
    out.flush(); del out; temporary.replace(output_path); return np.load(output_path, mmap_mode="r")


def initialize_program_decoder(prepared, config, device):
    rng = np.random.default_rng(config.seed); rows = np.asarray(prepared.train_indices)
    selected = rng.choice(rows, min(max(config.latent_dim * 16, 20000), len(rows)), replace=False)
    matrix = np.load(prepared.matrix_path, mmap_mode="r")
    sample = np.asarray(matrix[selected], dtype=np.float32)[:, prepared.retained_indices] / prepared.scale
    nonzero = sample.sum(1) > config.epsilon
    sample = sample[nonzero]
    if len(sample) < config.latent_dim: raise RuntimeError("Too few nonzero CCC profiles to initialize 32 programs")
    centroids = MiniBatchKMeans(n_clusters=config.latent_dim, random_state=config.seed,
                                batch_size=2048, n_init=5, max_iter=100).fit(sample).cluster_centers_
    normalized = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), config.epsilon)
    coarse = KMeans(n_clusters=config.final_niches, random_state=config.seed, n_init=20).fit(normalized).cluster_centers_
    coarse /= np.maximum(np.linalg.norm(coarse, axis=1, keepdims=True), config.epsilon)
    capacity = config.latent_dim // config.final_niches
    for _ in range(20):
        _, columns = linear_sum_assignment(1 - normalized @ np.repeat(coarse, capacity, axis=0).T)
        groups = columns // capacity
        updated = np.vstack([normalized[groups == niche].mean(0) for niche in range(config.final_niches)])
        updated /= np.maximum(np.linalg.norm(updated, axis=1, keepdims=True), config.epsilon)
        if np.allclose(updated, coarse): break
        coarse = updated
    group_order = sorted(range(config.final_niches), key=lambda niche: int(np.flatnonzero(groups == niche).min()))
    chosen = np.vstack([centroids[np.flatnonzero(groups == niche)] for niche in group_order])
    program = ProgramDecoder(config.latent_dim, config.latent_dim, len(prepared.retained_indices),
                             config.final_niches, config.program_temperature, config.epsilon).to(device)
    with torch.no_grad():
        program.initialize_dictionary(torch.from_numpy(chosen).to(device))
        mean_total = float(np.median(sample.sum(1)))
        bias = mean_total if mean_total > 20 else math.log(max(math.expm1(mean_total), config.epsilon))
        program.activity_head.weight.zero_(); program.activity_head.bias.fill_(bias)
    return program


def _augment(x, config):
    mask = (torch.rand_like(x) >= config.feature_mask_fraction).to(x.dtype)
    jitter = (1 + config.multiplicative_jitter * torch.randn_like(x)).clamp_min(0)
    return x * mask * jitter


def _pseudo_program_loss(x, dictionary, mixture, epsilon):
    x_profile = x / x.sum(1, keepdim=True).clamp_min(epsilon)
    x_normalized = torch.nn.functional.normalize(x_profile, p=2, dim=1, eps=epsilon)
    h_normalized = torch.nn.functional.normalize(dictionary.detach(), p=2, dim=1, eps=epsilon)
    target = (x_normalized @ h_normalized.T).argmax(1)
    return torch.nn.functional.nll_loss(mixture.clamp_min(epsilon).log(), target)


def _summary_metrics(sums, usage, hard, niche_usage, niche_hard, count, dictionary, config):
    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    soft = usage / max(count, 1); positive = soft[soft > 0]
    effective = float(np.exp(-(positive * np.log(positive)).sum()))
    h = dictionary.detach().cpu().numpy(); h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), config.epsilon)
    off = (h @ h.T)[np.triu_indices(len(h), 1)]
    metrics.update({"min_program_usage": float(soft.min()), "max_program_usage": float(soft.max()),
                    "effective_programs": effective, "hard_active_programs": int((hard > 0).sum()),
                    "max_hard_program_fraction": float(hard.max() / max(count, 1)),
                    "min_niche_usage": float((niche_usage / max(count, 1)).min()),
                    "max_niche_usage": float((niche_usage / max(count, 1)).max()),
                    "hard_active_niches": int((niche_hard > 0).sum()),
                    "max_hard_niche_fraction": float(niche_hard.max() / max(count, 1)),
                    "max_H_cosine": float(off.max(initial=-1))})
    return metrics


def _program_epoch(model, program, loader, config, device, optimizer=None):
    training = optimizer is not None; model.eval(); program.train(training)
    keys = ("linear", "pseudo_program", "program_entropy", "usage_penalty", "niche_usage_penalty",
            "h_entropy", "h_diversity", "consistency", "total")
    sums = {k: 0. for k in keys}; usage = np.zeros(config.latent_dim); hard = np.zeros(config.latent_dim, dtype=np.int64)
    niche_usage = np.zeros(config.final_niches); niche_hard = np.zeros(config.final_niches, dtype=np.int64); count = 0
    with torch.enable_grad() if training else torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            if training: optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(): embedding = model.encode(x)
            result = program(embedding); h = program.dictionary
            values = {"linear": huber(result["reconstruction"], x),
                      "pseudo_program": _pseudo_program_loss(x, h, result["mixture"], config.epsilon),
                      "program_entropy": normalized_entropy(result["mixture"], config.epsilon),
                      "usage_penalty": minimum_usage_loss(result["mixture"], config.minimum_program_usage),
                      "niche_usage_penalty": minimum_usage_loss(result["niche_mixture"], config.minimum_niche_usage),
                      "h_entropy": dictionary_entropy(h, config.epsilon),
                      "h_diversity": dictionary_diversity(h, config.h_similarity_margin, config.epsilon)}
            if training:
                with torch.no_grad(): augmented_embedding = model.encode(_augment(x, config))
                augmented_result = program(augmented_embedding)
                values["consistency"] = consistency_loss(result["niche_mixture"], augmented_result["niche_mixture"])
            else: values["consistency"] = torch.zeros((), device=device)
            loss = (values["linear"] + config.lambda_pseudo_stage2a * values["pseudo_program"]
                    + config.lambda_program_sparsity * values["program_entropy"]
                    + config.lambda_usage * values["usage_penalty"] + config.lambda_niche_usage * values["niche_usage_penalty"]
                    + config.lambda_h_sparsity * values["h_entropy"]
                    + config.lambda_h_diversity * values["h_diversity"]
                    + config.lambda_consistency * values["consistency"])
            values["total"] = loss
            if training: loss.backward(); optimizer.step()
            for key in keys: sums[key] += float(values[key]) * len(x)
            p = result["mixture"].detach().cpu().numpy(); usage += p.sum(0); hard += np.bincount(p.argmax(1), minlength=config.latent_dim); count += len(x)
            q = result["niche_mixture"].detach().cpu().numpy(); niche_usage += q.sum(0); niche_hard += np.bincount(q.argmax(1), minlength=config.final_niches)
    return _summary_metrics(sums, usage, hard, niche_usage, niche_hard, count, program.dictionary, config)


def fit_stage2a(model, program, prepared, config, output):
    device = resolve_device(config.device); program.raw_dictionary.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in program.parameters() if p.requires_grad), lr=config.stage2a_lr, weight_decay=config.weight_decay)
    train = _loader(prepared, prepared.train_indices, config, True, config.seed + 1); validation = _loader(prepared, prepared.validation_indices, config)
    history, best, stale = [], np.inf, 0; checkpoint = output / "stage2a_best_checkpoint.pt"
    for epoch in range(1, config.stage2a_max_epochs + 1):
        warmup = epoch <= config.stage2a_head_warmup_epochs
        if epoch == config.stage2a_head_warmup_epochs + 1:
            program.raw_dictionary.requires_grad_(True)
            optimizer = torch.optim.AdamW(program.parameters(), lr=config.stage2a_lr, weight_decay=config.weight_decay)
        tr = _program_epoch(model, program, train, config, device, optimizer); va = _program_epoch(model, program, validation, config, device)
        history.append({"epoch": epoch, "phase": "head_warmup" if warmup else "dictionary_joint",
                        **{f"train_{k}": v for k, v in tr.items()}, **{f"validation_{k}": v for k, v in va.items()}})
        if va["total"] < best - 1e-8:
            best, stale = va["total"], 0
            torch.save({"revision": config.revision, "epoch": epoch, "features": model.features,
                        "program_state": program.state_dict(), "validation": va}, checkpoint)
        else:
            stale += 1
            if stale >= config.stage2a_patience: break
    pd.DataFrame(history).to_csv(output / "stage2A_training_history.csv", index=False)
    program.raw_dictionary.requires_grad_(True)
    payload = torch.load(checkpoint, map_location=device, weights_only=False); program.load_state_dict(payload["program_state"])
    return payload


def _joint_epoch(model, program, loader, reference, config, device, optimizer=None):
    training = optimizer is not None; model.train(training); program.train(training)
    keys = ("ae", "linear", "preserve", "pseudo_program", "program_entropy", "usage_penalty",
            "niche_usage_penalty", "h_entropy", "h_diversity", "consistency", "total")
    sums = {k: 0. for k in keys}; usage = np.zeros(config.latent_dim); hard = np.zeros(config.latent_dim, dtype=np.int64)
    niche_usage = np.zeros(config.final_niches); niche_hard = np.zeros(config.final_niches, dtype=np.int64); count = 0
    with torch.enable_grad() if training else torch.no_grad():
        for x, rows in loader:
            x = x.to(device, non_blocking=True); ref = torch.from_numpy(np.asarray(reference[rows.numpy()]).copy()).to(device)
            if training: optimizer.zero_grad(set_to_none=True)
            embedding, xhat = model(x); result = program(embedding); h = program.dictionary
            values = {"ae": huber(xhat, x), "linear": huber(result["reconstruction"], x),
                      "preserve": latent_preservation(embedding, ref, config.epsilon),
                      "pseudo_program": _pseudo_program_loss(x, h, result["mixture"], config.epsilon),
                      "program_entropy": normalized_entropy(result["mixture"], config.epsilon),
                      "usage_penalty": minimum_usage_loss(result["mixture"], config.minimum_program_usage),
                      "niche_usage_penalty": minimum_usage_loss(result["niche_mixture"], config.minimum_niche_usage),
                      "h_entropy": dictionary_entropy(h, config.epsilon),
                      "h_diversity": dictionary_diversity(h, config.h_similarity_margin, config.epsilon)}
            if training:
                augmented = model.encode(_augment(x, config)); augmented_result = program(augmented)
                values["consistency"] = consistency_loss(result["niche_mixture"], augmented_result["niche_mixture"])
            else: values["consistency"] = torch.zeros((), device=device)
            loss = (values["ae"] + config.lambda_linear * values["linear"] + config.lambda_preserve * values["preserve"]
                    + config.lambda_pseudo_stage2b * values["pseudo_program"]
                    + config.lambda_program_sparsity * values["program_entropy"] + config.lambda_usage * values["usage_penalty"]
                    + config.lambda_niche_usage * values["niche_usage_penalty"]
                    + config.lambda_h_sparsity * values["h_entropy"] + config.lambda_h_diversity * values["h_diversity"]
                    + config.lambda_consistency * values["consistency"])
            values["total"] = loss
            if training: loss.backward(); optimizer.step()
            for key in keys: sums[key] += float(values[key]) * len(x)
            p = result["mixture"].detach().cpu().numpy(); usage += p.sum(0); hard += np.bincount(p.argmax(1), minlength=config.latent_dim); count += len(x)
            q = result["niche_mixture"].detach().cpu().numpy(); niche_usage += q.sum(0); niche_hard += np.bincount(q.argmax(1), minlength=config.final_niches)
    return _summary_metrics(sums, usage, hard, niche_usage, niche_hard, count, program.dictionary, config)


def fit_stage2b(model, program, prepared, reference, config, output):
    device = resolve_device(config.device); train = _loader(prepared, prepared.train_indices, config, True, config.seed + 2)
    validation = _loader(prepared, prepared.validation_indices, config)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(program.parameters()), lr=config.stage2b_lr, weight_decay=config.weight_decay)
    initial = _joint_epoch(model, program, validation, reference, config, device); best_score, stale = initial["total"], 0
    best = {"model_state": copy.deepcopy(model.state_dict()), "program_state": copy.deepcopy(program.state_dict()),
            "epoch": 0, "validation": initial, "rollback_to_stage2a": True}
    history = [{"epoch": 0, **{f"validation_{k}": v for k, v in initial.items()}, "accepted": True}]
    for epoch in range(1, config.stage2b_max_epochs + 1):
        tr = _joint_epoch(model, program, train, reference, config, device, optimizer)
        va = _joint_epoch(model, program, validation, reference, config, device)
        healthy = (va["ae"] <= initial["ae"] * config.reconstruction_degradation_limit
                   and va["preserve"] <= config.latent_drift_limit and va["effective_programs"] >= config.final_niches
                   and va["max_program_usage"] < .5 and va["hard_active_niches"] == config.final_niches
                   and va["max_hard_niche_fraction"] < .75)
        improved = healthy and va["total"] < best_score - 1e-8
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                        **{f"validation_{k}": v for k, v in va.items()}, "accepted": improved})
        if improved:
            best_score, stale = va["total"], 0
            best = {"model_state": copy.deepcopy(model.state_dict()), "program_state": copy.deepcopy(program.state_dict()),
                    "epoch": epoch, "validation": va, "rollback_to_stage2a": False}
        else:
            stale += 1
            if stale >= config.stage2b_patience: break
    best.update({"revision": config.revision, "features": model.features})
    checkpoint = output / "stage2_best_checkpoint.pt"; torch.save(best, checkpoint)
    pd.DataFrame(history).to_csv(output / "stage2B_training_history.csv", index=False)
    load_stage2_checkpoint(checkpoint, model, program, device)
    best["final_train"] = _joint_epoch(model, program, train, reference, config, device)
    best["final_validation"] = _joint_epoch(model, program, validation, reference, config, device)
    torch.save(best, checkpoint); return best


def load_stage2_checkpoint(path, model, program, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("features") != model.features: raise ValueError("Stage 2 checkpoint feature mismatch")
    model.load_state_dict(payload["model_state"]); program.load_state_dict(payload["program_state"]); return payload


def infer_programs(model, program, prepared, config, paths):
    device = resolve_device(config.device); model.eval(); program.eval(); n = prepared.matrix_shape[0]
    shapes = {"E32": (n, config.latent_dim), "P32": (n, config.latent_dim), "Z32": (n, config.latent_dim),
              "Q8_direct": (n, config.final_niches), "activity": (n,)}
    temporary = {}; outputs = {}
    for key, shape in shapes.items():
        temporary[key] = paths[key].with_name(paths[key].stem + ".incomplete.npy")
        outputs[key] = np.lib.format.open_memmap(temporary[key], mode="w+", dtype=np.float32, shape=shape)
    with torch.no_grad():
        for x, rows in _loader(prepared, np.arange(n), config):
            embedding = model.encode(x.to(device, non_blocking=True)); result = program(embedding); idx = rows.numpy()
            outputs["E32"][idx] = embedding.cpu().numpy(); outputs["P32"][idx] = result["mixture"].cpu().numpy()
            outputs["Z32"][idx] = result["exposure"].cpu().numpy(); outputs["activity"][idx] = result["activity"].squeeze(1).cpu().numpy()
            outputs["Q8_direct"][idx] = result["niche_mixture"].cpu().numpy()
    for key in shapes: outputs[key].flush(); del outputs[key]; temporary[key].replace(paths[key])
    return {key: np.load(paths[key], mmap_mode="r") for key in shapes}


def train_all(prepared, config, output):
    output.mkdir(parents=True, exist_ok=True); model, stage1 = fit_stage1(prepared, config, output)
    reference = infer_latent(model, prepared, config, output / "Z0.npy")
    device = resolve_device(config.device); program = initialize_program_decoder(prepared, config, device)
    stage2a = fit_stage2a(model, program, prepared, config, output)
    stage2b = fit_stage2b(model, program, prepared, reference, config, output)
    arrays = infer_programs(model, program, prepared, config,
                            {key: output / f"{key}.npy" for key in ("E32", "P32", "Z32", "Q8_direct", "activity")})
    h32 = program.dictionary.detach().cpu().numpy().astype(np.float32); np.save(output / "H32.npy", h32)
    return model, program, arrays, h32, {"stage1": stage1, "stage2a": stage2a, "stage2b": stage2b}
