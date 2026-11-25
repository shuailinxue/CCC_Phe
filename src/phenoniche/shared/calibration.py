"""Training-only input calibration for observed CCC/composition models."""
import torch


@torch.no_grad()
def fit_input_calibration(data, cfg):
    """Center each input block and give each one fixed training RMS radius 1."""
    features = int(data.feature_mask.sum())
    sums = torch.zeros(features + len(data.categories), dtype=torch.float64, device=cfg.device)
    squares = torch.zeros(2, dtype=torch.float64, device=cfg.device)
    mask = torch.as_tensor(data.feature_mask, device=cfg.device)
    scale = torch.as_tensor(data.feature_scale, device=cfg.device)[mask]
    rows = data.rows[0]
    for start in range(0, len(rows), 2048):
        batch = data.batch(rows[start:start + 2048], cfg.device)
        communication = torch.log1p(batch['target'][:, mask] / scale)
        composition = batch['composition_target'].clamp_min(0)
        composition = (composition / composition.sum(1, keepdim=True).clamp_min(1e-12)).sqrt()
        sums += torch.cat((communication, composition), 1).double().sum(0)
        squares += torch.stack((communication.double().square().sum(), composition.double().square().sum()))
    center = sums / len(rows)
    variance = squares / len(rows) - torch.stack((center[:features].square().sum(), center[features:].square().sum()))
    return center.float(), variance.clamp_min(1e-12).sqrt().float()
