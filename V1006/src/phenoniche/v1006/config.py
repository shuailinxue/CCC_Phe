from dataclasses import asdict, dataclass
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT.parent


@dataclass(frozen=True)
class ExperimentConfig:
    revision: str = "v1006_dual_decoder_program_v4"
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
    stage2a_lr: float = 1e-3
    stage2a_max_epochs: int = 15
    stage2a_patience: int = 5
    stage2a_head_warmup_epochs: int = 5
    stage2b_lr: float = 1e-4
    stage2b_max_epochs: int = 20
    stage2b_patience: int = 5
    weight_decay: float = 1e-4
    program_temperature: float = .7
    feature_mask_fraction: float = .05
    multiplicative_jitter: float = .05
    lambda_linear: float = .30
    lambda_preserve: float = .10
    lambda_consistency: float = .05
    lambda_program_sparsity: float = .01
    lambda_usage: float = .05
    lambda_niche_usage: float = .20
    lambda_pseudo_stage2a: float = .20
    lambda_pseudo_stage2b: float = .05
    lambda_h_sparsity: float = .01
    lambda_h_diversity: float = .05
    minimum_program_usage: float = .005
    minimum_niche_usage: float = .02
    h_similarity_margin: float = .5
    latent_drift_limit: float = .02
    reconstruction_degradation_limit: float = 1.10
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
