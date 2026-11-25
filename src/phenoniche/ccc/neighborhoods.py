from dataclasses import asdict, dataclass
import math
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class NeighborhoodConfig:
    radius: float = 1.6
    k: int | None = None
    kernel: str = "gaussian"
    sigma: float = 0.8
    cells_per_anchor: int = 12

    def __post_init__(self):
        if not math.isfinite(self.sigma) or self.sigma <= 0 or self.kernel != "gaussian":
            raise ValueError("Primary neighborhoods require a positive sigma and Gaussian kernel")
        if self.k is None and (not math.isfinite(self.radius) or self.radius <= 0):
            raise ValueError("A positive radius is required when k is not set")
        if self.k is not None and (not isinstance(self.k, int) or self.k < 2):
            raise ValueError("k must be an integer of at least two")
        if not isinstance(self.cells_per_anchor, int) or self.cells_per_anchor < 2:
            raise ValueError("cells_per_anchor must be at least two")


@dataclass(frozen=True)
class Neighborhoods:
    offsets: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    distances: np.ndarray
    anchor_coordinates: np.ndarray
    cell_coordinates: np.ndarray
    config: dict

    def members(self, anchor):
        if not isinstance(anchor, (int, np.integer)) or not 0 <= anchor < len(self.offsets) - 1:
            raise IndexError("anchor is outside the neighborhood collection")
        section = slice(self.offsets[anchor], self.offsets[anchor + 1])
        return self.indices[section], self.weights[section], self.distances[section]


def _coordinates(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite N by 2 matrix")
    return array


def build_neighborhoods(coordinates, cell_coordinates=None, radius=None, k=None, kernel="gaussian", sigma=0.8):
    anchors = _coordinates(coordinates, "coordinates")
    cells = anchors if cell_coordinates is None else _coordinates(cell_coordinates, "cell_coordinates")
    config = NeighborhoodConfig(radius=1.6 if radius is None and k is None else radius, k=k, kernel=kernel, sigma=sigma)
    tree = cKDTree(cells)
    if config.k is None:
        selections = tree.query_ball_point(anchors, config.radius)
    else:
        _, selected = tree.query(anchors, k=min(config.k, len(cells)))
        selections = [np.atleast_1d(row).tolist() for row in selected]
    offsets = [0]
    indices = []
    weights = []
    distances = []
    for anchor, selected in zip(anchors, selections):
        current = np.asarray(sorted(set(int(value) for value in selected)), dtype=np.int64)
        if current.size < 2:
            raise ValueError("Every neighborhood must contain at least two cells")
        distance = np.linalg.norm(cells[current] - anchor, axis=1)
        weight = np.exp(-(distance ** 2) / (2 * config.sigma ** 2))
        indices.append(current)
        weights.append(weight)
        distances.append(distance)
        offsets.append(offsets[-1] + len(current))
    return Neighborhoods(np.asarray(offsets, dtype=np.int64), np.concatenate(indices), np.concatenate(weights),
                         np.concatenate(distances), anchors, cells, asdict(config))
