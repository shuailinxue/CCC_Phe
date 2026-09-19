import torch
from phenoniche.utils.validation import matrix


def validate_laplacian(laplacian, anchors):
    if laplacian.shape != (anchors, anchors) or not laplacian.is_floating_point():
        raise ValueError("Laplacian must be floating point with shape [N, N]")
    sparse = laplacian.layout != torch.strided
    if sparse:
        value = laplacian.to_sparse_coo().coalesce()
        if not torch.isfinite(value.values()).all():
            raise ValueError("Laplacian must be finite")
        difference = (value - value.transpose(0, 1)).coalesce().values()
        rows, columns = value.indices()
        off_diagonal = value.values()[rows != columns]
        sums = torch.sparse.sum(value, dim=1).to_dense()
    else:
        if not torch.isfinite(laplacian).all():
            raise ValueError("Laplacian must be finite")
        difference = laplacian - laplacian.T
        off_diagonal = laplacian[~torch.eye(anchors, dtype=torch.bool, device=laplacian.device)]
        sums = laplacian.sum(dim=1)
    if (difference.abs() > 1e-5).any() or (off_diagonal > 1e-6).any() or (sums.abs() > 1e-5).any():
        raise ValueError("Laplacian must be symmetric with zero row sums and nonpositive off-diagonals")


def spatial_loss(activity, laplacian):
    matrix(activity, "spatial activity")
    validate_laplacian(laplacian, activity.shape[0])
    if laplacian.device != activity.device or laplacian.dtype != activity.dtype:
        raise ValueError("Laplacian and spatial activity must share device and dtype")
    product = laplacian @ activity
    return (activity * product).sum()
