import torch
from torch.nn import functional as F


def entropy(probabilities, epsilon=1e-8):
    probabilities = probabilities.clamp_min(epsilon)
    return -(probabilities * probabilities.log()).sum(dim=-1)


def information_cluster_loss(weights, gamma=1.0, epsilon=1e-8):
    sample_entropy = entropy(weights, epsilon).mean()
    marginal_entropy = entropy(weights.mean(dim=0, keepdim=True), epsilon).mean()
    return sample_entropy - float(gamma) * marginal_entropy


def dictionary_entropy_loss(dictionary, epsilon=1e-8):
    return entropy(dictionary, epsilon).mean()


def dictionary_diversity_loss(dictionary, epsilon=1e-8):
    normalized = F.normalize(dictionary, p=2, dim=1, eps=epsilon)
    similarity = normalized @ normalized.T
    mask = torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)
    return similarity[mask].mean() if mask.any() else similarity.new_zeros(())


def loss_components(proportions, reconstructed, weights, dictionary, config, cluster_weights=None):
    # Sum featurewise Huber costs per microenvironment so the loss scale does
    # not shrink merely because the retained CCC space is wider.
    reconstruction = F.smooth_l1_loss(reconstructed, proportions, reduction="none").sum(dim=1).mean()
    # During training cluster_weights is the deterministic (dropout-disabled)
    # encoder output. This prevents the encoder from satisfying information
    # maximization only through dropout noise while its inference-time output
    # leaves dead components.
    cluster = information_cluster_loss(
        weights if cluster_weights is None else cluster_weights,
        config.gamma, config.epsilon,
    )
    h_sparsity = dictionary_entropy_loss(dictionary, config.epsilon)
    diversity = dictionary_diversity_loss(dictionary, config.epsilon)
    total = (reconstruction + config.lambda_cluster * cluster
             + config.lambda_h * h_sparsity + config.lambda_div * diversity)
    return {
        "total": total,
        "reconstruction": reconstruction,
        "cluster": cluster,
        "h_entropy": h_sparsity,
        "h_diversity": diversity,
    }
