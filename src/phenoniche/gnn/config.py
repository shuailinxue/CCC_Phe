"""Configuration for the independent downstream microenvironment model."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
from ..storage import STORAGE_ROOT, OUTPUT_DIR


@dataclass(frozen=True)
class GNNConfig:
    seed: int = 40700
    split_seed: int = 40700
    programs: int = 32
    hidden: int = 64
    encoder_genes: int = 512
    batch_size: int = 128
    max_epochs: int = 60
    patience: int = 8
    lr: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 5.0
    n_neighbors: int = 30
    sigma_um: float = 20.0
    block_um: float = 400.0
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    sample_column: str = ""
    mode: str = "ccc_only"
    lambda_bio: float = 0.1
    alpha: float = 1.0
    beta: float = 1.0
    lambda_reg: float = 0.01
    bio_warmup_epochs: int = 5
    min_targets: int = 5
    min_target_coverage: float = 0.2
    holdout_bio_genes: bool = True
    prior_control: str = "none"
    contrastive: bool = False
    lambda_contrast: float = 0.01
    device: str = "auto"
    upstream_prior: str = str(STORAGE_ROOT / "data/priors/upstream.tsv")
    downstream_prior: str = str(STORAGE_ROOT / "data/priors/downstream.tsv")
    prior_manifest: str = str(STORAGE_ROOT / "data/priors/manifest.json")
    audit_file: str = ""
    output_dir: str = str(STORAGE_ROOT / "outputs/microenvironment_gnn")
    legacy_output: str = str(OUTPUT_DIR)
    use_legacy_feature_mask: bool = True  # Missing in historical checkpoints: preserve their exact input.
    k_min: int = 4
    k_max: int = 15
    clustering_sample_size: int = 50000
    silhouette_sample_size: int = 5000
    activity_weight: float = 0.25
    near_optimal_score_tolerance: float = 0.05

    def __post_init__(self):
        if self.prior_control not in ("none", "shuffle_links", "shuffle_activity"):
            raise ValueError("Invalid prior negative control")
        if self.mode not in ("ccc_only", "upstream", "downstream", "both"):
            raise ValueError("Unknown biological ablation")
        if self.prior_control != "none" and self.mode == "ccc_only":
            raise ValueError("Biological negative controls require an enabled biological branch")
        if min(self.programs, self.hidden, self.batch_size, self.max_epochs, self.patience) < 2:
            raise ValueError("Invalid model/training dimensions")
        if self.n_neighbors != 30 or self.sigma_um != 20.0:
            raise ValueError("Real CCC neighborhood definition is fixed at KNN=30, sigma=20 µm")
        if not 0 < self.validation_fraction < 1 or not 0 < self.test_fraction < 1:
            raise ValueError("Validation and test fractions must be positive")
        if self.validation_fraction + self.test_fraction >= 1 or self.block_um <= 0:
            raise ValueError("Invalid spatial split")
        if not 2 <= self.k_min <= self.k_max or self.min_targets < 2:
            raise ValueError("Invalid niche/prior settings")
        if not 0 < self.min_target_coverage <= 1:
            raise ValueError("Invalid regulatory target coverage")
        if min(self.lambda_bio, self.alpha, self.beta, self.lambda_reg, self.lambda_contrast) < 0:
            raise ValueError("Loss weights must be nonnegative")

    @property
    def use_upstream(self): return self.mode in ("upstream", "both")

    @property
    def use_downstream(self): return self.mode in ("downstream", "both")

    def to_dict(self): return asdict(self)

    @classmethod
    def from_json(cls, path): return cls(**json.loads(Path(path).read_text()))
