import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    window_um: float = 50
    contact_um: float = 50
    sigma_um: float = 20
    composition_neighbors: int = 20
    resolution: float = 0.4
    rank: int = 12
    iterations: int = 40
    fit_windows: int = 256
    repeats: int = 5
    block_um: float = 100
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
            if key == "aggregation" or (key == "delaunay_max_um" and value is None):
                continue
            if not math.isfinite(value) or (key != "seed" and value <= 0):
                raise ValueError(f"Invalid {key}: {value}")
        for key in (
            "composition_neighbors",
            "rank",
            "iterations",
            "fit_windows",
            "repeats",
            "seed",
            "communication_neighbors",
            "enrichment_permutations",
        ):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if self.seed < 0 or self.repeats < 2:
            raise ValueError("Use a nonnegative seed and at least two repeats")
