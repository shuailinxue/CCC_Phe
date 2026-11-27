"""Reload a selected mixture from its portable probability and activity export."""
from pathlib import Path

import numpy as np
import torch

from .probabilistic import Niches


def exported_niche(path,mode,temperature,device):
    with np.load(Path(path)/"niches.npz",allow_pickle=False) as record:
        pi,omega,activity=(record[name] for name in ("pi","omega","activity"))
    model=Niches(len(pi),pi.shape[1],activity.shape[1],mode,temperature).to(device)
    with torch.no_grad():
        model.pi.copy_(torch.as_tensor(np.log(np.maximum(pi,1e-35)),device=device))
        model.omega.copy_(torch.as_tensor(np.log(np.maximum(omega,1e-35)),device=device))
        model.activity.copy_(torch.as_tensor(activity,device=device))
    return model.requires_grad_(False)
