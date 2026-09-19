from dataclasses import dataclass
import math
from phenoniche.data.schema import CohortData


@dataclass(frozen=True)
class BlockScaler:
    composition: float
    communication: float
    topology: float

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in (self.composition, self.communication, self.topology)):
            raise ValueError("Scaling factors must be finite and positive")

    @classmethod
    def fit(cls, data):
        def scale(*blocks):
            value = sum(float(block.square().sum()) for block in blocks)
            rows = sum(block.shape[0] for block in blocks)
            return math.sqrt(value / rows) if value > 0 else 1.0
        return cls(scale(data.CB, data.CS), scale(data.IB, data.IS), scale(data.OS))

    def transform(self, data):
        return CohortData(
            data.CB / self.composition, data.IB / self.communication,
            data.CS / self.composition, data.OS / self.topology,
            data.IS / self.communication, data.time, data.event, data.laplacian,
        )

    def inverse_transform(self, data):
        return BlockScaler(1 / self.composition, 1 / self.communication, 1 / self.topology).transform(data)
