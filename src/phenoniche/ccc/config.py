from dataclasses import asdict, dataclass
from pathlib import Path
import json

from ..storage import INPUT_DIR, PROCESSED_DIR, STORAGE_ROOT

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class CCCInputConfig:
    minimum_features: int = 8
    n_neighbors: int = 30
    sigma_um: float = 20.0
    ccc_batch_size: int = 256
    lr_batch_size: int = 32
    expression_coverage_fraction: float = 0.10
    opportunity_support_fraction: float = 0.01
    opportunity_support_minimum: int = 20
    device: str = "auto"
    processed_h5ad: str = str(PROCESSED_DIR / "xenium_prime_5k_breast.h5ad")
    lr_atlas: str = str(ROOT / "data/commuspace_human_lr_atlas.tsv")
    output_dir: str = str(INPUT_DIR)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path):
        values = json.loads(Path(path).read_text())
        for key in ("processed_h5ad", "lr_atlas", "output_dir"):
            if key in values and not Path(values[key]).is_absolute():
                base = ROOT if key == "lr_atlas" else STORAGE_ROOT
                values[key] = str(base / values[key])
        return cls(**values)


def default_config():
    return CCCInputConfig.from_json(ROOT / "config/ccc_input.json")
