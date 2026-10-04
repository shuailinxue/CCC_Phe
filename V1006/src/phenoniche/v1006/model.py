import torch
from torch import nn
from torch.nn import functional as F


class RepresentationAutoencoder(nn.Module):
    def __init__(self, features, latent_dim=32, hidden1=256, hidden2=64, dropout=.1):
        super().__init__()
        self.features, self.latent_dim = int(features), int(latent_dim)
        self.encoder = nn.Sequential(nn.Linear(features, hidden1), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden1, hidden2), nn.GELU(),
                                     nn.Linear(hidden2, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden2), nn.GELU(),
                                     nn.Linear(hidden2, hidden1), nn.GELU(),
                                     nn.Linear(hidden1, features), nn.Softplus())

    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)
    def forward(self, x):
        z = self.encode(x)
        return z, self.decode(z)


class PrototypeHead(nn.Module):
    def __init__(self, niches, latent_dim, cluster_dim=None, temperature=.2, epsilon=1e-8):
        super().__init__()
        self.niches, self.latent_dim = int(niches), int(latent_dim)
        self.cluster_dim = int(cluster_dim or latent_dim)
        self.temperature, self.epsilon = float(temperature), float(epsilon)
        self.prototypes = nn.Parameter(torch.randn(niches, self.cluster_dim))
        self.register_buffer("latent_center", torch.zeros(self.latent_dim))
        self.register_buffer("whitening_projection", torch.eye(self.cluster_dim, self.latent_dim))

    def set_whitening(self, center, projection):
        if tuple(center.shape) != (self.latent_dim,) or tuple(projection.shape) != (self.cluster_dim, self.latent_dim):
            raise ValueError("Invalid latent whitening shapes")
        self.latent_center.copy_(center); self.whitening_projection.copy_(projection)

    def transform(self, z):
        return (z - self.latent_center) @ self.whitening_projection.T

    @property
    def normalized_prototypes(self):
        return F.normalize(self.prototypes, p=2, dim=1, eps=self.epsilon)

    def similarities(self, z):
        transformed = self.transform(z)
        return F.normalize(transformed, p=2, dim=1, eps=self.epsilon) @ self.normalized_prototypes.T

    def forward(self, z):
        return F.softmax(self.similarities(z) / self.temperature, dim=1)
