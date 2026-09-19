import math
import torch
from torch import nn


class BoundedDictionaryAdapter(nn.Module):
    def __init__(self, hc_anchor, hi_anchor, ho_anchor, ws_anchor, delta_max=math.log(1.25)):
        super().__init__()
        if hc_anchor.ndim != 2 or hi_anchor.ndim != 2 or ho_anchor.ndim != 2 or ws_anchor.ndim != 2:
            raise ValueError("Adapter anchors must be rank-2 tensors")
        if len({hc_anchor.shape[0], hi_anchor.shape[0], ho_anchor.shape[0], ws_anchor.shape[1]}) != 1:
            raise ValueError("Adapter anchors must share niche count")
        if not math.isfinite(delta_max) or delta_max <= 0:
            raise ValueError("delta_max must be positive and finite")
        self.delta_max = delta_max
        self.register_buffer("HC0", hc_anchor.detach().clone())
        self.register_buffer("HI0", hi_anchor.detach().clone())
        self.register_buffer("HO0", ho_anchor.detach().clone())
        self.register_buffer("WS0", ws_anchor.detach().clone())
        self.raw_delta_c = nn.Parameter(torch.zeros_like(hc_anchor))
        self.raw_delta_i = nn.Parameter(torch.zeros_like(hi_anchor))
        self.gamma = nn.Parameter(torch.zeros(hc_anchor.shape[0], dtype=hc_anchor.dtype, device=hc_anchor.device))

    def deltas(self):
        return self.delta_max * torch.tanh(self.raw_delta_c), self.delta_max * torch.tanh(self.raw_delta_i)

    def dictionaries(self):
        delta_c, delta_i = self.deltas()
        hc = self.HC0 * torch.exp(delta_c)
        hi = self.HI0 * torch.exp(delta_i)
        hc = hc / hc.sum(1, keepdim=True).clamp_min(torch.finfo(hc.dtype).tiny) * self.HC0.sum(1, keepdim=True)
        hi = hi / hi.sum(1, keepdim=True).clamp_min(torch.finfo(hi.dtype).tiny) * self.HI0.sum(1, keepdim=True)
        return hc, hi, self.HO0
