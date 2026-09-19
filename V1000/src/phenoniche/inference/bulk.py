import math
import torch
from phenoniche.utils.validation import matrix, same_context


def infer_bulk_activities(composition, communication, hc, hi, lambda_bc=1.0, lambda_bi=1.0,
                          steps=100, learning_rate=1.0, create_graph=True):
    for name, value in (("composition", composition), ("communication", communication), ("hc", hc), ("hi", hi)):
        matrix(value, name)
    same_context((composition, communication, hc, hi))
    if composition.shape[0] != communication.shape[0] or hc.shape[0] != hi.shape[0]:
        raise ValueError("Patient counts and dictionary niche counts must agree")
    if composition.shape[1] != hc.shape[1] or communication.shape[1] != hi.shape[1]:
        raise ValueError("Bulk features must align with fixed dictionary columns")
    if not all(math.isfinite(v) and v >= 0 for v in (lambda_bc, lambda_bi)) or lambda_bc + lambda_bi == 0:
        raise ValueError("Inference weights must be finite, nonnegative and not both zero")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must be in (0, 1] relative to the Lipschitz bound")
    with torch.set_grad_enabled(torch.is_grad_enabled() and create_graph):
        gram = lambda_bc * (hc @ hc.T) + lambda_bi * (hi @ hi.T)
        target = lambda_bc * (composition @ hc.T) + lambda_bi * (communication @ hi.T)
        lipschitz = gram.abs().sum(1).max().clamp_min(torch.finfo(gram.dtype).tiny)
        step = learning_rate / lipschitz
        activity = torch.zeros_like(target)
        extrapolated = activity
        momentum = 1.0
        for _ in range(steps):
            updated = torch.relu(extrapolated - step * (extrapolated @ gram - target))
            next_momentum = (1 + math.sqrt(1 + 4 * momentum ** 2)) / 2
            extrapolated = updated + ((momentum - 1) / next_momentum) * (updated - activity)
            activity, momentum = updated, next_momentum
    if not torch.isfinite(activity).all():
        raise FloatingPointError("Nonfinite differentiable bulk inference")
    return activity
