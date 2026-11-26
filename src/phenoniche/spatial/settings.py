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
    composition_neighbors: int = 20
    resolution: float = 0.4
    rank: int = 12
    rank_mode: str = "manual"
    rank_candidates: tuple[int, ...] = (4, 8, 12, 16, 24)
    iterations: int = 500
    repeats: int = 5
    cp_device: str = "auto"
    cp_batch_size: int = 128
    cp_learning_rate: float = 0.03
    cp_beta: float = 0.02
    cp_tol: float = 1e-4
    cp_patience: int = 5
    seed: int = 73
    interface_um: float = 25
    delaunay_max_um: float | None = None
    enrichment_permutations: int = 1000
    expression_preprocessing: str = "commuspace"
    expression_quantile: float = 0.95
    tensor_log1p: bool = False
    representation: str = "H1"
    communication_neighbors: int = 20

    def __post_init__(self):
        if self.expression_preprocessing not in {"commuspace", "cp10k"}:
            raise ValueError("expression_preprocessing must be commuspace or cp10k")
        if not 0 < self.expression_quantile <= 1:
            raise ValueError("expression_quantile must be in (0, 1]")
        if self.representation not in {"A", "H1", "H2", "H3"}:
            raise ValueError("representation must be A, H1, H2 or H3")
        if self.rank_mode not in {"manual", "auto"}:
            raise ValueError("rank_mode must be manual or auto")
        if len(self.rank_candidates) < 3 or any(type(x) is not int or x < 1 for x in self.rank_candidates) or tuple(sorted(set(self.rank_candidates))) != tuple(self.rank_candidates):
            raise ValueError("rank_candidates must contain at least three increasing positive integers")
        if type(self.tensor_log1p) is not bool:
            raise ValueError("tensor_log1p must be boolean")
        for key, value in vars(self).items():
            if key in {"expression_preprocessing", "representation", "rank_mode", "rank_candidates", "tensor_log1p", "cp_device"} or (key == "delaunay_max_um" and value is None):
                continue
            allow_zero = key in {"seed", "cp_beta"}
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
