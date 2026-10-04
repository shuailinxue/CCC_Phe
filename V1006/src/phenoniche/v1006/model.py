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


class ProgramDecoder(nn.Module):
    """Magnitude-separated simplex exposures with an interpretable nonnegative dictionary."""
    def __init__(self, latent_dim, programs, features, niches=8, temperature=.7, epsilon=1e-8):
        super().__init__()
        self.latent_dim, self.programs, self.features = int(latent_dim), int(programs), int(features)
        self.niches = int(niches)
        if self.programs % self.niches: raise ValueError("program count must be divisible by niche count")
        self.programs_per_niche = self.programs // self.niches
        self.temperature, self.epsilon = float(temperature), float(epsilon)
        self.niche_head = nn.Linear(latent_dim, niches)
        self.subprogram_head = nn.Linear(latent_dim, programs)
        self.activity_head = nn.Linear(latent_dim, 1)
        self.raw_dictionary = nn.Parameter(torch.empty(programs, features))
        nn.init.normal_(self.raw_dictionary, mean=-2., std=.1)

    @property
    def dictionary(self):
        positive = F.softplus(self.raw_dictionary)
        return positive / positive.sum(1, keepdim=True).clamp_min(self.epsilon)

    def initialize_dictionary(self, profiles):
        profiles = profiles / profiles.sum(1, keepdim=True).clamp_min(self.epsilon)
        target = profiles.clamp_min(1e-6)
        with torch.no_grad():
            self.raw_dictionary.copy_(torch.log(torch.expm1(target)))

    def mixtures(self, embedding):
        niche = F.softmax(self.niche_head(embedding) / self.temperature, dim=1)
        conditional = F.softmax(
            self.subprogram_head(embedding).reshape(-1, self.niches, self.programs_per_niche) / self.temperature,
            dim=2)
        program = (niche[:, :, None] * conditional).reshape(-1, self.programs)
        return niche, program

    def mixture(self, embedding): return self.mixtures(embedding)[1]

    def forward(self, embedding):
        niche_mixture, mixture = self.mixtures(embedding)
        activity = F.softplus(self.activity_head(embedding)) + self.epsilon
        exposure = activity * mixture
        reconstruction = exposure @ self.dictionary
        return {"activity": activity, "niche_mixture": niche_mixture, "mixture": mixture, "exposure": exposure,
                "reconstruction": reconstruction}
