from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class LossWeights:
    bc: float = 1.0
    bi: float = 1.0
    sc: float = 1.0
    so: float = 1.0
    si: float = 1.0
    ph: float = 0.01
    sp: float = 0.0
    reg: float = 0.0
    lambda_collapse: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(getattr(self, f.name)) and getattr(self, f.name) >= 0 for f in fields(self)):
            raise ValueError("Loss weights must be finite and nonnegative")

    @classmethod
    def per_entry(cls, data, **kwargs):
        return cls(bc=1 / data.CB.numel(), bi=1 / data.IB.numel(), sc=1 / data.CS.numel(),
                   so=1 / data.OS.numel(), si=1 / data.IS.numel(), **kwargs)


@dataclass(frozen=True)
class TrainingConfig:
    number_of_niches: int = 4
    warmup_epochs: int = 800
    joint_epochs: int = 800
    learning_rate: float = 0.025
    weight_decay: float = 0.0
    gradient_clip: float | None = 10.0
    early_stopping: int | None = None
    min_delta: float = 1e-7
    seed: int = 31
    device: str = "cpu"
    log_every: int = 100
    inner_steps: int = 100
    inner_lr: float = 1.0

    def __post_init__(self):
        if not isinstance(self.inner_steps, int) or self.inner_steps < 1:
            raise ValueError("inner_steps must be a positive integer")
        if not math.isfinite(self.inner_lr) or not 0 < self.inner_lr <= 1:
            raise ValueError("inner_lr must be in (0, 1]")
        if not isinstance(self.number_of_niches, int) or self.number_of_niches < 1:
            raise ValueError("number_of_niches must be a positive integer")
        if any(not isinstance(v, int) or v < 0 for v in (self.warmup_epochs, self.joint_epochs)) or self.warmup_epochs + self.joint_epochs == 0:
            raise ValueError("Epoch counts must be nonnegative integers with a positive total")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if any(not math.isfinite(v) or v < 0 for v in (self.weight_decay, self.min_delta)):
            raise ValueError("weight_decay and min_delta must be finite and nonnegative")
        if self.gradient_clip is not None and (not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0):
            raise ValueError("gradient_clip must be finite and positive or None")
        if self.early_stopping is not None and (not isinstance(self.early_stopping, int) or self.early_stopping < 1):
            raise ValueError("early_stopping must be a positive patience integer or None")
        if not isinstance(self.log_every, int) or self.log_every < 1:
            raise ValueError("log_every must be a positive integer")
