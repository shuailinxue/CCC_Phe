from dataclasses import asdict, dataclass, replace
import time
import tracemalloc
import numpy as np
import torch
from phenoniche.simulation.structured_ccc import StructuredCCCConfig, StructuredCCCResult, generate_structured_ccc
from phenoniche.v1001.neighborhoods import NeighborhoodConfig, build_neighborhoods
from phenoniche.v1001.spatial_ccc import aggregate_spatial_ccc


@dataclass(frozen=True)
class SpatialCells:
    cell_id: np.ndarray
    coordinates: np.ndarray
    cell_type: np.ndarray
    parent_anchor: np.ndarray
    niche_activity: np.ndarray


@dataclass(frozen=True)
class NeighborhoodStructuredResult:
    structured: StructuredCCCResult
    cells: SpatialCells
    neighborhoods: object
    audit: dict


def _systematic_types(probability, count, rng):
    offset = rng.random() / count
    quantiles = offset + np.arange(count) / count
    values = np.searchsorted(np.cumsum(probability), quantiles, side="right")
    return values[rng.permutation(count)]


def generate_spatial_cells(coordinates, ws, hc, config, seed):
    rng = np.random.default_rng(seed)
    expected = np.asarray(ws) @ np.asarray(hc)
    expected /= expected.sum(1, keepdims=True)
    count = config.cells_per_anchor
    parent = np.repeat(np.arange(len(coordinates)), count)
    cell_coordinates = np.repeat(np.asarray(coordinates), count, axis=0) + rng.uniform(-0.28, 0.28, (len(parent), 2))
    types = np.concatenate([_systematic_types(row, count, rng) for row in expected])
    niche = np.repeat(np.asarray(ws), count, axis=0)
    return SpatialCells(np.arange(len(parent), dtype=np.int64), cell_coordinates, types, parent, niche), expected


def _observe(rng, values, fraction):
    scale = fraction * np.sqrt(np.mean(values ** 2))
    return np.maximum(values + rng.normal(0, scale, values.shape), 0)


def generate_neighborhood_structured(structured_config=None, neighborhood_config=None):
    structured_config = structured_config or StructuredCCCConfig()
    neighborhood_config = neighborhood_config or NeighborhoodConfig()
    base = generate_structured_ccc(structured_config)
    coordinates = base.data.coordinates.numpy()
    ws = base.truth["WS"].numpy()
    hc = base.truth["HC"].numpy()
    cells, expected = generate_spatial_cells(coordinates, ws, hc, neighborhood_config, structured_config.seed + 1101)
    neighborhoods = build_neighborhoods(coordinates, cells.coordinates, radius=neighborhood_config.radius,
                                        k=neighborhood_config.k, kernel=neighborhood_config.kernel,
                                        sigma=neighborhood_config.sigma)
    tracemalloc.start()
    started = time.perf_counter()
    aggregation = aggregate_spatial_ccc(neighborhoods, cells.cell_type, cells.parent_anchor, cells.niche_activity,
                                        base.truth["HI_tensor"].numpy(), tau=structured_config.opportunity_tau,
                                        baseline=structured_config.communication_baseline,
                                        pair_sigma=neighborhood_config.sigma)
    construction_seconds = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    noise_rng = np.random.default_rng(structured_config.seed + 2202)
    normalized = _observe(noise_rng, aggregation.normalized, structured_config.observation_noise)
    raw = normalized * (aggregation.opportunity + structured_config.opportunity_tau)
    tensor = lambda value: torch.tensor(value, dtype=torch.float32)
    data = replace(base.data, CS=tensor(aggregation.composition), IS=tensor(normalized),
                   Iraw_S=tensor(raw), Opp_S=tensor(aggregation.opportunity))
    metadata = dict(base.metadata)
    metadata["spatial_CCC_semantics"] = "all ordered non-self cell pairs inside the anchor-centered neighborhood"
    metadata["neighborhood"] = asdict(neighborhood_config)
    structured = StructuredCCCResult(data, base.truth, metadata)
    flat_expected = expected.reshape(-1)
    flat_observed = aggregation.composition.reshape(-1)
    audit = dict(aggregation.audit)
    audit.update({"construction_seconds": construction_seconds, "peak_python_memory_bytes": int(peak),
                  "composition_expected_correlation": float(np.corrcoef(flat_observed, flat_expected)[0, 1]),
                  "composition_row_sum_max_error": float(np.abs(aggregation.composition.sum(1) - 1).max()),
                  "feature_axis": [structured_config.cell_types, structured_config.cell_types, structured_config.programs],
                  "neighborhood": asdict(neighborhood_config)})
    return NeighborhoodStructuredResult(structured, cells, neighborhoods, audit)
