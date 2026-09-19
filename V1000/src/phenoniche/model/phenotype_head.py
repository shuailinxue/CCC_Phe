import torch
from torch import nn
from phenoniche.utils.validation import matrix


class CoxHead(nn.Module):
    def __init__(self, number_of_niches, device=None, dtype=torch.float32):
        super().__init__()
        if not isinstance(number_of_niches, int) or number_of_niches < 1:
            raise ValueError("number_of_niches must be a positive integer")
        self.gamma = nn.Parameter(torch.zeros(number_of_niches, device=device, dtype=dtype))

    def forward(self, activity):
        matrix(activity, "bulk activity")
        if activity.shape[1] != self.gamma.numel():
            raise ValueError("Bulk activity niche count does not match gamma")
        if activity.device != self.gamma.device or activity.dtype != self.gamma.dtype:
            raise ValueError("Bulk activity and gamma must share device and dtype")
        return activity @ self.gamma
