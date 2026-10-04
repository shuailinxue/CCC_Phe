from dataclasses import asdict, dataclass
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = PROJECT_ROOT.parent


@dataclass(frozen=True)
class ExperimentConfig:
    model_revision: str = "clean_info_max_v2"
    seed: int = 40700
    niches: int = 8
    hidden_1: int = 256
    hidden_2: int = 64
    dropout: float = 0.1
    optimizer: str = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 100
    validation_fraction: float = 0.10
    early_stopping_patience: int = 10
    lambda_cluster: float = 0.1
    gamma: float = 1.0
    lambda_h: float = 0.01
    lambda_div: float = 0.01
    epsilon: float = 1e-8
    n_neighbors: int = 30
    sigma_um: float = 20.0
    ccc_batch_size: int = 256
    lr_batch_size: int = 32
    expression_coverage_fraction: float = 0.10
    opportunity_support_fraction: float = 0.01
    opportunity_support_minimum: int = 20
    top_ccc: int = 15
    duplicate_h_cosine_threshold: float = 0.95
    collapsed_usage_threshold: float = 0.01
    device: str = "auto"
    processed_h5ad: str = "/data1/xueshuailin/CCC_Phe_Niche/data/xenium_prime_5k_breast/processed/XeniumPrime5K_Breast_V1002_processed.h5ad"
    lr_atlas: str = str(REPOSITORY_ROOT / "V1002/data/commuspace_human_lr_atlas.tsv")
    output_dir: str = str(PROJECT_ROOT / "outputs/xenium_prime_5k")

    def __post_init__(self):
        if self.model_revision != "clean_info_max_v2":
            raise ValueError("Unsupported V1003 model revision")
        if self.niches != 8 or self.seed != 40700:
            raise ValueError("V1003 primary analysis requires K=8 and seed=40700")
        if self.optimizer != "AdamW":
            raise ValueError("V1003 primary analysis requires AdamW")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")
        if self.batch_size < 2 or self.max_epochs < 1 or self.early_stopping_patience < 1:
            raise ValueError("invalid training schedule")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def default_config():
    path = PROJECT_ROOT / "config/prime5k.json"
    return ExperimentConfig.from_json(path) if path.is_file() else ExperimentConfig()
