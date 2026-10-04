import math
import torch
from torch.nn import functional as F


def huber(a, b): return F.smooth_l1_loss(a, b, reduction="mean")


def normalized_entropy(probabilities, epsilon=1e-8):
    p = probabilities.clamp_min(epsilon)
    return (-(p * p.log()).sum(1) / math.log(probabilities.shape[1])).mean()


def minimum_usage_loss(probabilities, minimum=.005):
    deficit = torch.relu(minimum - probabilities.mean(0)) / max(minimum, 1e-8)
    return deficit.square().mean()


def dictionary_entropy(dictionary, epsilon=1e-8):
    return normalized_entropy(dictionary, epsilon)


def dictionary_diversity(dictionary, margin=.5, epsilon=1e-8):
    h = F.normalize(dictionary, p=2, dim=1, eps=epsilon)
    similarity = h @ h.T
    mask = torch.triu(torch.ones_like(similarity, dtype=torch.bool), 1)
    return torch.relu(similarity[mask] - margin).square().mean()


def stable_embedding(embedding, epsilon=1e-8):
    return F.normalize(torch.log1p(embedding), p=2, dim=1, eps=epsilon)


def latent_preservation(embedding, reference, epsilon=1e-8):
    return F.mse_loss(stable_embedding(embedding, epsilon), stable_embedding(reference, epsilon))


def consistency_loss(p1, p2): return F.mse_loss(p1, p2)
