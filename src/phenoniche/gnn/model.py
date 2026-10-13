"""One directed local graph -> one intensity-preserving CCC program vector."""
import torch
from torch import nn
from torch.nn import functional as F


def physical_weights(coordinates, anchor_weights, sigma, cell_ids=None):
    """Same w_anchor(s) w_anchor(r) exp(-d_sr²/2σ²) as ccc.pairwise."""
    distance2 = (coordinates[:, :, None] - coordinates[:, None, :]).square().sum(-1)
    weight = anchor_weights[:, :, None] * anchor_weights[:, None, :] * torch.exp(-distance2 / (2 * sigma**2))
    k = coordinates.shape[1]
    distinct = ~torch.eye(k, dtype=torch.bool, device=coordinates.device)[None]
    if cell_ids is not None:
        distinct = distinct & (cell_ids[:, :, None] != cell_ids[:, None, :])
    return weight * distinct


class MicroenvironmentGNN(nn.Module):
    def __init__(self, genes, types, features, programs=32, hidden=64, sigma=20., feature_mask=None):
        super().__init__()
        self.genes, self.types, self.features = genes, types, features
        self.programs, self.hidden, self.sigma = programs, hidden, sigma
        self.type_embedding = nn.Embedding(types, min(16, hidden))
        self.cell_encoder = nn.Sequential(nn.Linear(genes + min(16, hidden), hidden), nn.GELU(),
                                          nn.LayerNorm(hidden))
        self.sender = nn.Linear(hidden, hidden, bias=False)
        self.receiver = nn.Linear(hidden, hidden, bias=False)
        self.sender_value = nn.Linear(hidden, hidden, bias=False)
        self.receiver_value = nn.Linear(hidden, hidden, bias=False)
        self.node_update = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU())
        self.pool = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU())
        self.composition_head = nn.Linear(hidden, programs)
        self.activity_head = nn.Linear(hidden, 1)
        self.raw_dictionary = nn.Parameter(torch.randn(programs, features) * .15 - 2)
        mask = torch.ones(features, dtype=torch.bool) if feature_mask is None else torch.as_tensor(feature_mask, dtype=torch.bool)
        if mask.shape != (features,) or not mask.any(): raise ValueError("Empty/invalid CCC feature mask")
        self.register_buffer("feature_mask", mask)

    @property
    def dictionary(self):
        h = F.softplus(self.raw_dictionary) * self.feature_mask
        return h / h.sum(1, keepdim=True).clamp_min(1e-8)

    def forward(self, expression, types, local_index, coordinates, anchor_weights, cell_ids=None, pair_override=None):
        # Encode each unique physical cell once in the batch, reuse overlapping members.
        unique_h = self.cell_encoder(torch.cat((expression, self.type_embedding(types)), dim=1))
        h = unique_h[local_index]
        weight = physical_weights(coordinates, anchor_weights, self.sigma, cell_ids)
        if pair_override is not None: weight = pair_override
        valid = weight > 0
        scores = self.sender(h) @ self.receiver(h).transpose(1, 2) / self.hidden**.5
        scores = scores + weight.clamp_min(1e-30).log()
        # Incoming attention normalizes over senders; outgoing over receivers.
        bounded = scores.masked_fill(~valid, -1e9)
        incoming = torch.softmax(bounded, dim=1) * valid
        outgoing = torch.softmax(bounded, dim=2) * valid
        received = incoming.transpose(1, 2) @ self.sender_value(h)
        sent = outgoing @ self.receiver_value(h)
        nodes = self.node_update(torch.cat((h, sent, received), dim=-1))
        a = anchor_weights / anchor_weights.sum(1, keepdim=True).clamp_min(1e-8)
        sender_mass = weight.sum(2); receiver_mass = weight.sum(1)
        sender_mass = sender_mass / sender_mass.sum(1, keepdim=True).clamp_min(1e-8)
        receiver_mass = receiver_mass / receiver_mass.sum(1, keepdim=True).clamp_min(1e-8)
        pooled = self.pool(torch.cat(((nodes * a[..., None]).sum(1),
                                     (sent * sender_mass[..., None]).sum(1),
                                     (received * receiver_mass[..., None]).sum(1)), dim=1))
        mixture = self.composition_head(pooled).softmax(1)
        activity = F.softplus(self.activity_head(pooled)) + 1e-8
        z = activity * mixture
        return {"Z": z, "P": mixture, "activity": activity.squeeze(1), "H": self.dictionary,
                "reconstruction": z @ self.dictionary}
