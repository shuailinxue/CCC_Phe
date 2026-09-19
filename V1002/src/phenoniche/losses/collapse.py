import math
import torch
from phenoniche.utils.validation import matrix, same_context


def pairwise_view_similarities(hc, ho, hi, eps=1e-8):
    for name, value in (("HC", hc), ("HO", ho), ("HI", hi)):
        matrix(value, name)
    same_context((hc, ho, hi))
    if len({value.shape[0] for value in (hc, ho, hi)}) != 1:
        raise ValueError("HC, HO and HI must share the niche count")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    result = {}
    for name, value in (("HC", hc), ("HO", ho), ("HI", hi)):
        normalized = value / (torch.linalg.vector_norm(value, dim=1, keepdim=True) + eps)
        result[name] = normalized @ normalized.T
    return result


def cross_view_collapse_loss(hc, ho, hi, eps=1e-8):
    similarities = pairwise_view_similarities(hc, ho, hi, eps)
    pairs = torch.triu_indices(hc.shape[0], hc.shape[0], offset=1, device=hc.device)
    if pairs.shape[1] == 0:
        return (hc.sum() + ho.sum() + hi.sum()) * 0
    joint = similarities["HC"] * similarities["HO"] * similarities["HI"]
    return joint[pairs[0], pairs[1]].mean()
