import torch
from torch import nn
from torch.nn import functional as F


class InterpretableCCCAutoencoder(nn.Module):
    """Nonlinear simplex encoder with a normalized nonnegative linear dictionary."""

    def __init__(self, features, niches=8, hidden_1=256, hidden_2=64, dropout=0.1, epsilon=1e-8):
        super().__init__()
        if features < 1 or niches < 2:
            raise ValueError("features and niches must be positive")
        self.features = int(features)
        self.niches = int(niches)
        self.epsilon = float(epsilon)
        self.encoder = nn.Sequential(
            nn.Linear(features, hidden_1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, niches),
        )
        self.raw_dictionary = nn.Parameter(torch.empty(niches, features))
        nn.init.normal_(self.raw_dictionary, mean=-2.0, std=0.05)

    @property
    def dictionary(self):
        positive = F.softplus(self.raw_dictionary)
        return positive / positive.sum(dim=1, keepdim=True).clamp_min(self.epsilon)

    def encode(self, proportions):
        return torch.softmax(self.encoder(proportions), dim=-1)

    def decode(self, weights):
        return weights @ self.dictionary

    def forward(self, proportions, magnitudes=None):
        weights = self.encode(proportions)
        reconstructed_proportions = self.decode(weights)
        reconstructed = None if magnitudes is None else magnitudes.reshape(-1, 1) * reconstructed_proportions
        return weights, reconstructed_proportions, reconstructed

