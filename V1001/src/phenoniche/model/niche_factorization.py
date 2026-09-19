import torch
from torch import nn
from torch.nn import functional as F
from phenoniche.inference import bulk
from phenoniche.model.phenotype_head import CoxHead


class NicheFactorization(nn.Module):
    def __init__(self, patients, anchors, cell_types, contacts, communications, number_of_niches,
                 device=None, dtype=torch.float32, composition=None, communication=None,
                 inner_steps=100, inner_lr=1.0, lambda_bc=1.0, lambda_bi=1.0):
        super().__init__()
        dimensions = (patients, anchors, cell_types, contacts, communications, number_of_niches)
        if not all(isinstance(v, int) and v > 0 for v in dimensions):
            raise ValueError("All model dimensions must be positive integers")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Model dtype must be float32 or float64")
        if (composition is None) != (communication is None):
            raise ValueError("Both bulk input blocks must be supplied together")
        if composition is not None and (composition.shape != (patients, cell_types) or communication.shape != (patients, communications)):
            raise ValueError("Bound bulk observations do not match constructor dimensions")
        self.register_buffer("composition", None if composition is None else composition.to(device=device, dtype=dtype), persistent=False)
        self.register_buffer("communication", None if communication is None else communication.to(device=device, dtype=dtype), persistent=False)
        self.inference_options = {"steps": inner_steps, "learning_rate": inner_lr, "lambda_bc": lambda_bc, "lambda_bi": lambda_bi}
        k = number_of_niches
        for name, shape in {"WS": (anchors, k), "HC": (k, cell_types), "HO": (k, contacts), "HI": (k, communications)}.items():
            initial = torch.randn(shape, device=device, dtype=dtype) * 0.5 - 1.2
            self.register_parameter(f"raw_{name}", nn.Parameter(initial))
        self.phenotype_head = CoxHead(k, device=device, dtype=dtype)

    @property
    def gamma(self):
        return self.phenotype_head.gamma

    def bulk_factors(self, composition=None, communication=None):
        if (composition is None) != (communication is None):
            raise ValueError("Both bulk input blocks must be supplied together")
        composition = self.composition if composition is None else composition
        communication = self.communication if communication is None else communication
        if composition is None or communication is None:
            raise ValueError("Bulk inference requires composition and communication observations")
        dictionaries = self.niche_dictionaries()
        return bulk.infer_bulk_activities(composition, communication, dictionaries["HC"], dictionaries["HI"], **self.inference_options)

    def spatial_factors(self):
        return F.softplus(self.raw_WS)

    def niche_dictionaries(self):
        return {name: F.softplus(getattr(self, f"raw_{name}")) for name in ("HC", "HO", "HI")}

    def factors(self, composition=None, communication=None):
        return {"WB": self.bulk_factors(composition, communication), "WS": self.spatial_factors(),
                **self.niche_dictionaries(), "gamma": self.gamma}

    def forward(self, composition=None, communication=None):
        factors = self.factors(composition, communication)
        wb, ws = factors["WB"], factors["WS"]
        return {"CB": wb @ factors["HC"], "IB": wb @ factors["HI"],
                "CS": ws @ factors["HC"], "OS": ws @ factors["HO"],
                "IS": ws @ factors["HI"], "risk": self.phenotype_head(wb), "factors": factors}
