import torch


def matrix(value, name, nonnegative=True):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or min(value.shape) < 1:
        raise ValueError(f"{name} must be a nonempty rank-2 tensor")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point values")
    if nonnegative and (value < 0).any():
        raise ValueError(f"{name} must be nonnegative")


def same_context(values):
    first = values[0]
    if any(v.device != first.device or v.dtype != first.dtype for v in values[1:]):
        raise ValueError("Tensors must have identical device and dtype")


def survival_vectors(risk, time, event):
    if risk.ndim != 1 or risk.numel() == 0 or time.shape != risk.shape or event.shape != risk.shape:
        raise ValueError("risk_score, time and event must have matching nonempty [P] shapes")
    if not risk.is_floating_point() or not time.is_floating_point():
        raise ValueError("risk_score and time must be floating-point tensors")
    if any(v.device != risk.device for v in (time, event)):
        raise ValueError("Survival tensors must share a device")
    if not all(torch.isfinite(v).all() for v in (risk, time, event)):
        raise ValueError("Survival tensors must be finite")
    if (time <= 0).any() or not ((event == 0) | (event == 1)).all():
        raise ValueError("time must be positive and event must be binary")
