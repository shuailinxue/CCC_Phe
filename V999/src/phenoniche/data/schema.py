from dataclasses import dataclass, fields
import torch
from phenoniche.utils.validation import matrix, same_context, survival_vectors
from phenoniche.losses.spatial import validate_laplacian


@dataclass(frozen=True)
class CohortData:
    CB: torch.Tensor
    IB: torch.Tensor
    CS: torch.Tensor
    OS: torch.Tensor
    IS: torch.Tensor
    time: torch.Tensor
    event: torch.Tensor
    laplacian: torch.Tensor | None = None

    def __post_init__(self):
        blocks = self.blocks()
        for name, value in blocks.items():
            matrix(value, name)
        same_context(tuple(blocks.values()))
        if self.CB.shape[0] != self.IB.shape[0]:
            raise ValueError("CB and IB must have the same patient count")
        if len({v.shape[0] for v in (self.CS, self.OS, self.IS)}) != 1:
            raise ValueError("CS, OS and IS must have the same anchor count")
        if self.CB.shape[1] != self.CS.shape[1] or self.IB.shape[1] != self.IS.shape[1]:
            raise ValueError("Bulk and spatial composition/CCC feature counts must agree")
        survival_vectors(self.CB[:, 0], self.time, self.event)
        if self.laplacian is not None:
            validate_laplacian(self.laplacian, self.CS.shape[0])
            same_context((self.CS, self.laplacian))

    def blocks(self):
        return {name: getattr(self, name) for name in ("CB", "IB", "CS", "OS", "IS")}

    def to(self, device, dtype=None):
        return CohortData(**{
            field.name: None if (value := getattr(self, field.name)) is None else value.to(device=device, dtype=dtype)
            for field in fields(self)
        })
