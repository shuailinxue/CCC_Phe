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
    min_cells: int = 12
    seed: int = 73
    interface_fraction: float = 0.05
    communication_neighbors: int = 20

    def __post_init__(self):
        for key, value in vars(self).items():
            if not math.isfinite(value) or (key != "seed" and value <= 0):
                raise ValueError(f"Invalid {key}: {value}")
        for key in (
            "composition_neighbors",
            "rank",
            "iterations",
            "fit_windows",
            "repeats",
            "min_cells",
            "seed",
            "communication_neighbors",
        ):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if self.seed < 0 or self.interface_fraction > 1 or self.repeats < 2:
            raise ValueError("Use a nonnegative seed, an interface fraction at most 1, and at least two repeats")
