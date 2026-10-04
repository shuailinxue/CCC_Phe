from dataclasses import asdict, dataclass
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT.parent


@dataclass(frozen=True)
class ExperimentConfig:
    revision: str = "v1006_whitened_prototype_v3"
    seed: int = 40700
    latent_dim: int = 32
    final_niches: int = 8
    validation_fraction: float = .10
    nonzero_fraction_min: float = .005
    encoder_hidden_1: int = 256
    encoder_hidden_2: int = 64
    dropout: float = .1
    batch_size: int = 1024
    stage1_lr: float = 1e-3
    stage1_max_epochs: int = 50
    stage1_patience: int = 7
    stage2_lr: float = 1e-4
    stage2_max_epochs: int = 20
    stage2_patience: int = 5
    weight_decay: float = 1e-4
    temperature: float = .2
    pca_variance_fraction: float = .99
    prototype_warmup_epochs: int = 3
    feature_mask_fraction: float = .05
    multiplicative_jitter: float = .05
    lambda_cluster: float = .10
    lambda_consistency: float = .05
    lambda_usage: float = .10
    lambda_separation: float = .10
    lambda_confidence: float = .005
    minimum_soft_usage: float = .02
    prototype_similarity_margin: float = .2
    reconstruction_degradation_limit: float = 1.15
    kmeans_sample_size: int = 200000
    epsilon: float = 1e-8
    top_ccc: int = 15
    device: str = "auto"
    v1003_source: str = str(REPO / "V1003/src")
    v1003_output: str = str(REPO / "V1003/outputs/xenium_prime_5k")
    output_dir: str = str(ROOT / "outputs/xenium_prime_5k")

    def __post_init__(self):
        if self.seed != 40700 or self.latent_dim != 32 or self.final_niches != 8:
            raise ValueError("V1006 primary settings require seed=40700, D=32 and K=8")
        if not 0 < self.validation_fraction < 1 or self.batch_size < 2:
            raise ValueError("invalid split or batch size")

    def to_dict(self): return asdict(self)

    @classmethod
    def from_json(cls, path): return cls(**json.loads(Path(path).read_text()))


def default_config():
    return ExperimentConfig.from_json(ROOT / "config/prime5k.json")
