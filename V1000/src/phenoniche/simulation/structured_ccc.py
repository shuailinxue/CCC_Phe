from dataclasses import asdict, dataclass
import numpy as np
import torch
from phenoniche.evaluation.metrics import concordance_index


@dataclass(frozen=True)
class StructuredCCCConfig:
    patients: int = 320
    rows: int = 30
    columns: int = 40
    cell_types: int = 8
    programs: int = 12
    topology_features: int = 24
    niches: int = 6
    active_fraction: float = 0.0625
    opportunity_tau: float = 0.0001
    communication_baseline: float = 0.001
    observation_noise: float = 0.005
    beta_risk: float = 10.0
    beta_protective: float = -10.0
    seed: int = 20260918

    def __post_init__(self):
        if self.patients != 320 or self.rows * self.columns != 1200 or self.cell_types != 8:
            raise ValueError("Primary structured dimensions require P=320, N=1200 and C=8")
        if self.programs not in (2, 4, 8, 12) or self.topology_features != 24 or self.niches != 6:
            raise ValueError("Structured experiment requires Q in {2,4,8,12}, E=24 and K=6")
        if self.active_fraction != 0.0625 or self.opportunity_tau <= 0 or self.observation_noise < 0:
            raise ValueError("Structured simulation constants are fixed and nonnegative")

    @property
    def anchors(self):
        return self.rows * self.columns

    @property
    def communication_features(self):
        return self.cell_types * self.cell_types * self.programs


@dataclass(frozen=True)
class StructuredCCCData:
    CB: torch.Tensor
    IB: torch.Tensor
    Iraw_B: torch.Tensor
    Opp_B: torch.Tensor
    CS: torch.Tensor
    IS: torch.Tensor
    Iraw_S: torch.Tensor
    Opp_S: torch.Tensor
    OS: torch.Tensor
    coordinates: torch.Tensor
    time: torch.Tensor
    event: torch.Tensor


@dataclass(frozen=True)
class StructuredCCCResult:
    data: StructuredCCCData
    truth: dict[str, torch.Tensor]
    metadata: dict


def feature_index(sender, receiver, program, cell_types, programs):
    if not (0 <= sender < cell_types and 0 <= receiver < cell_types and 0 <= program < programs):
        raise ValueError("Structured CCC coordinates are out of range")
    return (sender * cell_types + receiver) * programs + program


def decode_feature(index, cell_types, programs):
    features = cell_types * cell_types * programs
    if not isinstance(index, int) or not 0 <= index < features:
        raise ValueError("Structured CCC feature index is out of range")
    pair, program = divmod(index, programs)
    sender, receiver = divmod(pair, cell_types)
    return sender, receiver, program


def _composition_dictionary():
    values = np.array([
        [0.34, 0.29, 0.18, 0.11, 0.04, 0.02, 0.01, 0.01],
        [0.34, 0.29, 0.18, 0.11, 0.04, 0.02, 0.01, 0.01],
        [0.04, 0.08, 0.31, 0.28, 0.18, 0.07, 0.02, 0.02],
        [0.03, 0.05, 0.08, 0.12, 0.34, 0.25, 0.09, 0.04],
        [0.06, 0.03, 0.04, 0.08, 0.09, 0.18, 0.31, 0.21],
        [0.18, 0.08, 0.04, 0.04, 0.07, 0.10, 0.20, 0.29],
    ], dtype=np.float64)
    return values / values.sum(1, keepdims=True)


def _communication_dictionary(rng, config):
    k, f = config.niches, config.communication_features
    active = int(round(f * config.active_fraction))
    shared = max(1, active // 4)
    specific = active - shared
    order = rng.permutation(f)
    cursor = 0
    specific_sets = []
    for _ in range(k):
        specific_sets.append(order[cursor:cursor + specific])
        cursor += specific
    pair_sets = {}
    for pair in ((0, 2), (1, 3), (4, 5)):
        pair_sets[pair] = order[cursor:cursor + shared]
        cursor += shared
    tensor = np.full((k, config.cell_types, config.cell_types, config.programs), 1e-8, dtype=np.float64)
    active_records = []
    partner = {0: (0, 2), 2: (0, 2), 1: (1, 3), 3: (1, 3), 4: (4, 5), 5: (4, 5)}
    for niche in range(k):
        indices = np.concatenate((specific_sets[niche], pair_sets[partner[niche]]))
        strengths = np.concatenate((rng.uniform(1.0, 1.5, specific), rng.uniform(0.30, 0.45, shared)))
        flat = tensor[niche].reshape(-1)
        flat[indices] = strengths
        flat /= flat.sum()
        for index in indices:
            sender, receiver, program = decode_feature(int(index), config.cell_types, config.programs)
            active_records.append({"niche": niche, "index": int(index), "sender": sender,
                                   "receiver": receiver, "program": program,
                                   "strength": float(flat[index])})
    return tensor, active_records


def _topology_dictionary(rng, config):
    values = np.full((config.niches, config.topology_features), 0.002, dtype=np.float64)
    for niche in range(config.niches):
        indices = (np.arange(4) * config.niches + niche) % config.topology_features
        values[niche, indices] += rng.uniform(0.7, 1.2, len(indices))
    return values / values.sum(1, keepdims=True)


def _bulk_activities(rng, config):
    for _ in range(1000):
        raw = rng.lognormal(-0.1, 0.45, (config.patients, config.niches))
        raw[:, 4] = rng.lognormal(0.1, 1.15, config.patients)
        values = raw / raw.sum(1, keepdims=True)
        if abs(np.corrcoef(values[:, 0], values[:, 1])[0, 1]) < 0.10:
            return values
    raise RuntimeError("Unable to generate near-independent A/B bulk activities")


def _spatial_activities(rng, config):
    y, x = np.meshgrid(np.arange(config.rows), np.arange(config.columns), indexing="ij")
    coordinates = np.column_stack((y.reshape(-1), x.reshape(-1))).astype(np.float64)
    for _ in range(1000):
        fields = []
        for niche in range(config.niches):
            field = np.full(config.anchors, 0.03)
            for _ in range(3):
                center = np.array([rng.uniform(2, config.rows - 3), rng.uniform(2, config.columns - 3)])
                sigma = rng.uniform(4.5, 8.0)
                distance = ((coordinates - center) ** 2).sum(1)
                field += rng.uniform(0.7, 1.3) * np.exp(-distance / (2 * sigma ** 2))
            fields.append(field)
        values = np.column_stack(fields)
        values /= values.sum(1, keepdims=True)
        correlation = np.corrcoef(values[:, 0], values[:, 1])[0, 1]
        overlap = np.dot(values[:, 0], values[:, 1]) / (np.linalg.norm(values[:, 0]) * np.linalg.norm(values[:, 1]))
        if abs(correlation) < 0.15 and overlap > 0.20:
            return coordinates, values
    raise RuntimeError("Unable to generate overlapping near-independent A/B spatial fields")


def _opportunity(composition, config):
    sender = np.repeat(np.arange(config.cell_types), config.cell_types * config.programs)
    receiver = np.tile(np.repeat(np.arange(config.cell_types), config.programs), config.cell_types)
    return composition[:, sender] * composition[:, receiver]


def _observe(rng, values, noise_fraction):
    scale = noise_fraction * np.sqrt(np.mean(values ** 2))
    return np.maximum(values + rng.normal(0, scale, values.shape), 0)


def generate_structured_ccc(config=None):
    config = config or StructuredCCCConfig()
    sequences = np.random.SeedSequence(config.seed).spawn(6)
    dictionary_rng, activity_rng, spatial_rng, noise_rng, survival_rng, censor_rng = [np.random.default_rng(value) for value in sequences]
    hc = _composition_dictionary()
    hi_tensor, active_records = _communication_dictionary(dictionary_rng, config)
    hi = hi_tensor.reshape(config.niches, -1)
    ho = _topology_dictionary(dictionary_rng, config)
    wb = _bulk_activities(activity_rng, config)
    coordinates, ws = _spatial_activities(spatial_rng, config)
    cb = _observe(noise_rng, wb @ hc, config.observation_noise)
    cs = _observe(noise_rng, ws @ hc, config.observation_noise)
    cb /= cb.sum(1, keepdims=True)
    cs /= cs.sum(1, keepdims=True)
    opportunity_b = _opportunity(cb, config)
    opportunity_s = _opportunity(cs, config)
    interaction_b = wb @ hi
    interaction_s = ws @ hi
    raw_b_mean = opportunity_b * (config.communication_baseline + interaction_b)
    raw_s_mean = opportunity_s * (config.communication_baseline + interaction_s)
    iraw_b = _observe(noise_rng, raw_b_mean, config.observation_noise)
    iraw_s = _observe(noise_rng, raw_s_mean, config.observation_noise)
    ib = iraw_b / (opportunity_b + config.opportunity_tau)
    is_ = iraw_s / (opportunity_s + config.opportunity_tau)
    os = _observe(noise_rng, ws @ ho, config.observation_noise)
    beta = np.array([config.beta_risk, 0.0, config.beta_protective, 0.0, 0.0, 0.0])
    eta = wb @ beta
    failure = survival_rng.exponential(size=config.patients) / (0.04 * np.exp(eta))
    censoring = censor_rng.exponential(scale=np.median(failure) * 1.8, size=config.patients)
    time = np.maximum(np.minimum(failure, censoring), np.finfo(np.float32).tiny)
    event = (failure <= censoring).astype(np.float32)
    tensor = lambda value: torch.tensor(value, dtype=torch.float32)
    data = StructuredCCCData(tensor(cb), tensor(ib), tensor(iraw_b), tensor(opportunity_b),
                             tensor(cs), tensor(is_), tensor(iraw_s), tensor(opportunity_s),
                             tensor(os), tensor(coordinates), tensor(time), tensor(event))
    truth = {name: tensor(value) for name, value in {
        "HC": hc, "HI": hi, "HI_tensor": hi_tensor, "HO": ho, "WB": wb,
        "WS": ws, "beta": beta, "eta": eta,
    }.items()}
    ab_hi = float(np.dot(hi[0], hi[1]) / (np.linalg.norm(hi[0]) * np.linalg.norm(hi[1])))
    wb_ab = float(np.corrcoef(wb[:, 0], wb[:, 1])[0, 1])
    ws_ab = float(np.corrcoef(ws[:, 0], ws[:, 1])[0, 1])
    if not np.array_equal(hc[0], hc[1]) or ab_hi >= 0.05 or abs(wb_ab) >= 0.10 or abs(ws_ab) >= 0.15:
        raise AssertionError("Structured A/B truth constraints failed")
    mapping = [{"index": index, "sender": decode_feature(index, config.cell_types, config.programs)[0],
                "receiver": decode_feature(index, config.cell_types, config.programs)[1],
                "program": decode_feature(index, config.cell_types, config.programs)[2]}
               for index in range(config.communication_features)]
    metadata = {"config": asdict(config), "feature_mapping": mapping,
                "active_edges": active_records, "active_edges_per_niche": int(round(config.communication_features * config.active_fraction)),
                "roles": ["risk", "neutral", "protective", "neutral", "nuisance", "spatial"],
                "A_B_truth": {"HC_cosine": 1.0, "HI_cosine": ab_hi,
                              "WB_correlation": wb_ab, "WS_correlation": ws_ab},
                "true_survival_c_index": concordance_index(time, event, eta),
                "event_fraction": float(event.mean()),
                "opportunity_normalization": {"tau": config.opportunity_tau, "uses_survival": False}}
    return StructuredCCCResult(data, truth, metadata)
