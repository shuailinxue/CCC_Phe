import math
import numpy as np
import torch
from scipy.optimize import nnls
from phenoniche.utils.validation import matrix, same_context


def nnls_reference(composition, communication, hc, hi, lambda_bc=1.0, lambda_bi=1.0, max_iterations=1000):
    for name, value in (("composition", composition), ("communication", communication), ("hc", hc), ("hi", hi)):
        matrix(value, name)
    same_context((composition, communication, hc, hi))
    if composition.shape[0] != communication.shape[0] or hc.shape[0] != hi.shape[0]:
        raise ValueError("Patient counts and dictionary niche counts must agree")
    if composition.shape[1] != hc.shape[1] or communication.shape[1] != hi.shape[1]:
        raise ValueError("Bulk features must align with fixed dictionary columns")
    if not all(math.isfinite(v) and v >= 0 for v in (lambda_bc, lambda_bi)) or lambda_bc + lambda_bi == 0:
        raise ValueError("Inference weights must be finite, nonnegative and not both zero")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    arrays = [v.detach().cpu().double().numpy() for v in (composition, communication, hc, hi)]
    cb, ib, hc_array, hi_array = arrays
    design = np.concatenate((math.sqrt(lambda_bc) * hc_array.T, math.sqrt(lambda_bi) * hi_array.T))
    targets = np.concatenate((math.sqrt(lambda_bc) * cb, math.sqrt(lambda_bi) * ib), axis=1)
    solution = np.stack([nnls(design, target, maxiter=max_iterations)[0] for target in targets])
    if not np.isfinite(solution).all():
        raise FloatingPointError("Nonfinite held-out activity inference")
    return torch.tensor(solution, device=composition.device, dtype=composition.dtype)
