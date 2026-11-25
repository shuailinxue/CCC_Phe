import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    window_um: float = 50
    contact_um: float = 50
    sigma_um: float = 20
    contact_radius_um: float = 20
    secreted_radius_um: float = 100
    ecm_radius_um: float = 20
    contact_sigma_um: float = 20
    secreted_sigma_um: float = 40
    ecm_sigma_um: float = 20
    composition_neighbors: int = 20
    resolution: float = 0.4
    rank: int = 12
    iterations: int = 40
    repeats: int = 5
    cp_device: str = "auto"
    cp_batch_size: int = 128
    cp_learning_rate: float = 0.03
    cp_alpha: float = 0.02
    cp_beta: float = 0.02
    cp_tol: float = 1e-4
    cp_patience: int = 5
    seed: int = 73
    interface_um: float = 25
    delaunay_max_um: float | None = None
    enrichment_permutations: int = 1000
    aggregation: str = "mean-mean"
    communication_neighbors: int = 20

    def __post_init__(self):
        if self.aggregation not in {"sum-sum", "mean-sum", "mean-mean"}:
            raise ValueError("Invalid aggregation mode")
        for key, value in vars(self).items():
            if key in {"aggregation", "cp_device"} or (key == "delaunay_max_um" and value is None):
                continue
            allow_zero = key in {"seed", "cp_alpha", "cp_beta"}
            if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
                raise ValueError(f"Invalid {key}: {value}")
        for key in (
            "composition_neighbors",
            "rank",
            "iterations",
            "cp_batch_size",
            "cp_patience",
            "repeats",
            "seed",
            "communication_neighbors",
            "enrichment_permutations",
        ):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if self.seed < 0 or self.repeats < 2:
            raise ValueError("Use a nonnegative seed and at least two repeats")
