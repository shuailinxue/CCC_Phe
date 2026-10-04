import torch
from torch.nn import functional as F


def huber(a, b): return F.smooth_l1_loss(a, b, reduction="mean")


def dec_target(q, epsilon=1e-8):
    weight = q.square() / q.sum(0, keepdim=True).clamp_min(epsilon)
    return weight / weight.sum(1, keepdim=True).clamp_min(epsilon)


def prototype_cluster_loss(q, epsilon=1e-8):
    pseudo_label = q.detach().argmax(1)
    return F.nll_loss(q.clamp_min(epsilon).log(), pseudo_label)


def assignment_entropy(q, epsilon=1e-8):
    return -(q * q.clamp_min(epsilon).log()).sum(1).mean()


def minimum_usage_loss(q, minimum=.02):
    deficit = torch.relu(minimum - q.mean(0)) / max(minimum, 1e-8)
    return deficit.square().mean()


def prototype_separation(prototypes, margin=.2, epsilon=1e-8):
    p = F.normalize(prototypes, p=2, dim=1, eps=epsilon)
    similarity = p @ p.T
    mask = torch.triu(torch.ones_like(similarity, dtype=torch.bool), 1)
    return torch.relu(similarity[mask] - margin).square().mean()


def consistency_loss(q1, q2):
    return F.mse_loss(q1, q2)
