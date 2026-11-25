"""Streamed, masked nonnegative CP on observed cell–neighbor-type fibers."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse
import torch
from torch.nn.functional import normalize


@dataclass
class CPFit:
    factors: tuple[np.ndarray, np.ndarray, np.ndarray]
    history: list[dict]
    metrics: dict


def cp_device(name):
    if name == "auto":
        count = torch.cuda.device_count()
        name = f"cuda:{2 if count >= 3 else 0}" if count else "cpu"
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("CP device must be auto, cpu or cuda[:index]")
    if device.type == "cuda" and (
        not torch.cuda.is_available() or
        (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        raise ValueError(f"CP device {device} is unavailable")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _gram(factor):
    unit = normalize(factor, dim=0, eps=1e-8)
    return unit.T @ unit


def _offdiag(gram):
    indices = torch.triu_indices(*gram.shape, offset=1, device=gram.device)
    return gram[indices[0], indices[1]]


def orthogonality(a, b, c):
    """Squared off-diagonal sums of the normalized A and joint BC Grams."""
    return 2 * _offdiag(_gram(a)).square().sum(), 2 * _offdiag(
        _gram(b) * _gram(c)
    ).square().sum()


def masked_squared_error(a, b, c, rows, neighbor, values, energy):
    """Compute error across all LR columns of observed fibers, including zeros."""
    design = a[rows] * b[neighbor]
    cross = (design * torch.sparse.mm(values, c)).sum(dtype=torch.float64)
    gram = c.T @ c
    predicted = ((design @ gram) * design).sum(dtype=torch.float64)
    return energy - 2 * cross + predicted


def _observed_batch(ids, fiber):
    rows, types, matrices = [], [], []
    for row, cell in enumerate(ids):
        neighbor, values, _ = fiber(cell)
        if len(neighbor):
            rows.extend([row] * len(neighbor))
            types.append(neighbor)
            matrices.append(sparse.csr_matrix(values))
    if not rows:
        return None
    return np.asarray(rows), np.concatenate(types), sparse.vstack(matrices, format="csr")


def _batch_size(settings, device, n_types, n_lr, factor_bytes):
    size = settings.cp_batch_size
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        # Reserve headroom for optimizer states, gradients, and sparse kernels.
        bytes_per_cell = n_types * (24 * n_lr + 32 * settings.rank)
        size = min(size, max(1, (free // 3 - 8 * factor_bytes) // bytes_per_cell))
    return size


def fit_programs(ids, fiber, n_types, n_lr, settings, rng):
    """Optimize shared factors with one accumulated gradient step per full pass."""
    if not len(ids):
        return None
    started = perf_counter()
    device = cp_device(settings.cp_device)
    factor_bytes = 4 * (len(ids) + n_types + n_lr) * settings.rank
    size = min(len(ids), _batch_size(settings, device, n_types, n_lr, factor_bytes))
    batches = [(start, min(start + size, len(ids))) for start in range(0, len(ids), size)]
    observed = np.zeros(len(ids), dtype=bool)
    energy, entries = 0.0, 0
    for start, stop in batches:
        batch = _observed_batch(ids[start:stop], fiber)
        if batch is not None:
            rows, _, values = batch
            observed[start + rows] = True
            data = values.data.astype(np.float64)
            energy += np.dot(data, data)
            entries += values.shape[0] * n_lr
    if not np.isfinite(energy):
        raise FloatingPointError("Non-finite CCC energy")
    if not energy:
        return None

    scale = float(np.sqrt(energy / entries))
    rank = min(settings.rank, int(observed.sum()), n_lr)
    initial = [
        (rng.uniform(0.3, 1, (rows, rank)) * rank ** (-1 / 3)).astype(np.float32)
        for rows in (len(ids), n_types, n_lr)
    ]
    initial[0][~observed] = 0
    a, b, c = [
        torch.from_numpy(matrix).to(device).requires_grad_() for matrix in initial
    ]
    factors = (a, b, c)
    optimizer = torch.optim.Adam(factors, lr=settings.cp_learning_rate)
    history = []
    best_loss, reference, stale = float("inf"), float("inf"), 0

    for epoch in range(settings.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        a.grad = torch.zeros_like(a)
        reconstruction = torch.zeros((), dtype=torch.float64, device=device)
        for start, stop in batches:
            batch = _observed_batch(ids[start:stop], fiber)
            if batch is None:
                continue
            rows, neighbor, values = batch
            data = torch.from_numpy(values.data).to(device) / scale
            tensor = torch.sparse_csr_tensor(
                values.indptr.astype(np.int64), values.indices.astype(np.int64),
                data, size=values.shape, device=device,
            )
            local = a[start:stop].detach().requires_grad_()
            loss = masked_squared_error(
                local, b, c, torch.as_tensor(rows, device=device),
                torch.as_tensor(neighbor, device=device), tensor,
                data.square().sum(dtype=torch.float64),
            ) / entries
            loss.backward()
            a.grad[start:stop].copy_(local.grad)
            reconstruction += loss.detach()

        loss_a, loss_bc = orthogonality(a, b, c)
        rec = max(0.0, reconstruction.item())
        qa, qbc = loss_a.item(), loss_bc.item()
        if epoch == 0:
            alpha = settings.cp_alpha * rec / max(qa, 1e-8) if qa > 1e-8 else 0.0
            beta = settings.cp_beta * rec / max(qbc, 1e-8) if qbc > 1e-8 else 0.0
        penalty = alpha * loss_a + beta * loss_bc
        total = rec + alpha * qa + beta * qbc
        if not np.isfinite(total):
            raise FloatingPointError("Non-finite CP objective")
        history.append(dict(
            epoch=epoch, reconstruction_loss=rec, a_orthogonality_loss=qa,
            bc_orthogonality_loss=qbc, orthogonality_loss=alpha * qa + beta * qbc,
            total_loss=total, elapsed_seconds=perf_counter() - started,
        ))
        if total < best_loss:
            best_loss = total
            best_epoch = epoch
            best = tuple(factor.detach().cpu().numpy().copy() for factor in factors)
        if reference - total > settings.cp_tol * max(abs(total), 1e-12):
            reference, stale = total, 0
        else:
            stale += 1
        if stale >= settings.cp_patience or epoch == settings.iterations:
            break

        penalty.backward()
        optimizer.step()
        with torch.no_grad():
            for factor in factors:
                factor.clamp_(min=0)

    with torch.no_grad():
        aa, bb, cc = map(torch.from_numpy, best)
        a_cos = _offdiag(_gram(aa))
        bc_cos = _offdiag(_gram(bb) * _gram(cc))
        metrics = dict(
            device=str(device), batch_size=size, centers=len(ids),
            centers_with_observations=int(observed.sum()), observed_entries=entries,
            rms_scale=scale, calibrated_alpha=alpha, calibrated_beta=beta,
            best_epoch=best_epoch, epochs_completed=epoch,
            stopped_early=epoch < settings.iterations,
            runtime_seconds=history[-1]["elapsed_seconds"],
            a_offdiag_max_cosine=float(a_cos.max()) if a_cos.numel() else 0.0,
            bc_offdiag_max_cosine=float(bc_cos.max()) if bc_cos.numel() else 0.0,
            **history[best_epoch],
        )
    activity, neighbor, lr = best
    activity *= scale
    for factor in (neighbor, lr):
        norms = np.linalg.norm(factor, axis=0)
        safe_norms = np.where(norms > 0, norms, 1)
        factor /= safe_norms
        activity *= safe_norms
    return CPFit(best, history, metrics)
