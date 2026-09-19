import math
import torch
from torch.nn import functional as F
from phenoniche.inference.bulk import infer_bulk_activities
from phenoniche.utils.validation import matrix, same_context


def infer_background_activities(composition, communication, hc_background, hi_background,
                                lambda_bc=1.0, lambda_bi=1.0, steps=100,
                                learning_rate=1.0, create_graph=True):
    return infer_bulk_activities(composition, communication, hc_background, hi_background,
                                 lambda_bc=lambda_bc, lambda_bi=lambda_bi, steps=steps,
                                 learning_rate=learning_rate, create_graph=create_graph)


def infer_phenotype_activities(composition, communication, background_activities,
                               hc_background, hi_background, hc_phenotype, hi_phenotype,
                               lambda_bc=1.0, lambda_bi=1.0, steps=100,
                               learning_rate=1.0, create_graph=True):
    values = (("composition", composition), ("communication", communication),
              ("background_activities", background_activities),
              ("hc_background", hc_background), ("hi_background", hi_background),
              ("hc_phenotype", hc_phenotype), ("hi_phenotype", hi_phenotype))
    for name, value in values:
        matrix(value, name)
    same_context(tuple(value for _, value in values))
    patients = composition.shape[0]
    if communication.shape[0] != patients or background_activities.shape[0] != patients:
        raise ValueError("Bulk blocks and background activities must share patient count")
    if background_activities.shape[1] != hc_background.shape[0] or hc_background.shape[0] != hi_background.shape[0]:
        raise ValueError("Background activity and dictionary niche counts must agree")
    if hc_phenotype.shape[0] != hi_phenotype.shape[0]:
        raise ValueError("Phenotype dictionaries must share niche count")
    if composition.shape[1] != hc_background.shape[1] or composition.shape[1] != hc_phenotype.shape[1]:
        raise ValueError("Composition features must align with both dictionary banks")
    if communication.shape[1] != hi_background.shape[1] or communication.shape[1] != hi_phenotype.shape[1]:
        raise ValueError("Communication features must align with both dictionary banks")
    if not all(math.isfinite(v) and v >= 0 for v in (lambda_bc, lambda_bi)) or lambda_bc + lambda_bi == 0:
        raise ValueError("Inference weights must be finite, nonnegative and not both zero")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must be in (0, 1] relative to the Lipschitz bound")
    with torch.set_grad_enabled(torch.is_grad_enabled() and create_graph):
        residual_c = composition - background_activities @ hc_background
        residual_i = communication - background_activities @ hi_background
        gram = lambda_bc * (hc_phenotype @ hc_phenotype.T) + lambda_bi * (hi_phenotype @ hi_phenotype.T)
        target = lambda_bc * (residual_c @ hc_phenotype.T) + lambda_bi * (residual_i @ hi_phenotype.T)
        lipschitz = gram.abs().sum(1).max().clamp_min(torch.finfo(gram.dtype).tiny)
        step = learning_rate / lipschitz
        initial = torch.full_like(target, -2.0)
        raw = initial
        for _ in range(steps):
            activity = F.softplus(raw)
            gradient = (activity @ gram - target) * torch.sigmoid(raw)
            raw = raw - step * gradient
        activity = F.softplus(raw)
    if not torch.isfinite(activity).all():
        raise FloatingPointError("Nonfinite phenotype residual inference")
    return activity


def infer_dual_bank_activities(composition, communication, hc_background, hi_background,
                               hc_phenotype, hi_phenotype, lambda_bc=1.0, lambda_bi=1.0,
                               steps=100, learning_rate=1.0, create_graph=True):
    background = infer_background_activities(
        composition, communication, hc_background, hi_background,
        lambda_bc=lambda_bc, lambda_bi=lambda_bi, steps=steps,
        learning_rate=learning_rate, create_graph=create_graph)
    phenotype = infer_phenotype_activities(
        composition, communication, background, hc_background, hi_background,
        hc_phenotype, hi_phenotype, lambda_bc=lambda_bc, lambda_bi=lambda_bi,
        steps=steps, learning_rate=learning_rate, create_graph=create_graph)
    return background, phenotype
