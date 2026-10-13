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
    architecture: str = "expression_gnn_composition_v1"
    lambda_comp: float = 0.0
    lambda_reg: float = 0.005
    device: str = "auto"
    audit_file: str = ""
    output_dir: str = str(STORAGE_ROOT / "outputs/microenvironment_gnn_ccc_only_expr05")
    legacy_output: str = str(OUTPUT_DIR)
    use_legacy_feature_mask: bool = False  # Optional fixed feature-universe comparison with the archived AE.
    k_min: int = 4
    k_max: int = 15
    clustering_sample_size: int = 50000
    silhouette_sample_size: int = 5000
    activity_weight: float = 0.25
    near_optimal_score_tolerance: float = 0.05

    def __post_init__(self):
        if self.architecture != "expression_gnn_composition_v1":
            raise ValueError("Checkpoint architecture is not the expression-only composition model")
        if min(self.programs, self.hidden, self.batch_size, self.max_epochs, self.patience) < 2:
            raise ValueError("Invalid model/training dimensions")
        if self.n_neighbors != 30 or self.sigma_um != 20.0:
            raise ValueError("Real CCC neighborhood definition is fixed at KNN=30, sigma=20 µm")
        if not 0 < self.validation_fraction < 1 or not 0 < self.test_fraction < 1:
            raise ValueError("Validation and test fractions must be positive")
        if self.validation_fraction + self.test_fraction >= 1 or self.block_um <= 0:
            raise ValueError("Invalid spatial split")
        if not 2 <= self.k_min <= self.k_max:
            raise ValueError("Invalid niche range")
        if min(self.lambda_comp,self.lambda_reg) < 0:
            raise ValueError("Loss weights must be nonnegative")

    def to_dict(self): return asdict(self)

    @classmethod
    def from_json(cls, path): return cls(**json.loads(Path(path).read_text()))
