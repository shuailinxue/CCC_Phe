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
    encoder_gene_selection: str = "variance"
    encoder_depth: int = 1
    message_layers: int = 1
    dropout: float = 0.0
    ccc_nonzero_weight_max: float = 1.0
    dictionary_parameterization: str = "softplus"
    dictionary_coordinates: str = "native"
    dictionary_initialization: str = "rms_kmeans"
    nmf_initialization_sample_size: int = 8192
    nmf_initialization_max_iter: int = 500
    dictionary_lr: float = 0.00005
    program_pair_pooling: bool = False
    lr_pair_pooling: bool = False
    expression_normalization: str = "library_log"
    standardize_library_size: bool = False
    activity_parameterization: str = 'separated'
    cell_state_components: int = 0
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
    output_dir: str = str(STORAGE_ROOT / "outputs/microenvironment_gnn_ccc_only_expr10")
    legacy_output: str = str(OUTPUT_DIR)
    use_legacy_feature_mask: bool = False  # Optional fixed feature-universe comparison with the archived AE.
    k_min: int = 4
    k_max: int = 15
    clustering_sample_size: int = 50000
    silhouette_sample_size: int = 5000
    activity_weight: float = 0.25
    near_optimal_score_tolerance: float = 0.05

    def __post_init__(self):
        if self.cell_state_components not in (0,32,64,128):raise ValueError('Cell-state PCA dimensions must be 0, 32, 64 or 128')
        if self.activity_parameterization not in ('separated','independent'):
            raise ValueError('Invalid program activity parameterization')
        if self.activity_parameterization=='independent' and self.program_pair_pooling:
            raise ValueError('Independent activity and residual pair pooling are separate experimental ablations')
        if self.lr_pair_pooling and not self.program_pair_pooling:
            raise ValueError('LR pair pooling requires program pair pooling')
        if self.expression_normalization not in ('library_log','log_counts'):
            raise ValueError('Invalid expression normalization')
        if self.dictionary_parameterization not in ('softplus','projected') or self.dictionary_lr<=0:
            raise ValueError('Invalid dictionary optimization')
        if self.dictionary_coordinates not in ('native','feature_scaled'):
            raise ValueError('Invalid dictionary coordinates')
        if self.dictionary_initialization not in ('rms_kmeans','log1p_active_kmeans','nmf'):
            raise ValueError('Invalid dictionary initialization')
        if self.nmf_initialization_sample_size < self.programs or self.nmf_initialization_max_iter < 1:
            raise ValueError('NMF initialization needs enough training centers and a positive iteration budget')
        if self.architecture not in ("expression_gnn_composition_v1","expression_gnn_residual_v2"):
            raise ValueError("Checkpoint architecture is not the expression-only composition model")
        if self.encoder_gene_selection not in ('variance','lr_plus_variance') or self.encoder_genes<1:
            raise ValueError('Invalid expression gene selection')
        if not 1<=self.encoder_depth<=3 or not 1<=self.message_layers<=3 or not 0<=self.dropout<.5:
            raise ValueError('Invalid residual encoder configuration')
        if self.ccc_nonzero_weight_max<1 or self.ccc_nonzero_weight_max>20:
            raise ValueError('CCC nonzero weight cap must be between 1 and 20')
        if self.architecture=='expression_gnn_composition_v1' and (self.encoder_depth!=1 or self.message_layers!=1 or self.dropout!=0):
            raise ValueError('Residual/deep encoders require the new architecture identifier')
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
