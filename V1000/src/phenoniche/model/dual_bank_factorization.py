import torch
from torch import nn
from torch.nn import functional as F
from phenoniche.inference.dual_bank import infer_background_activities, infer_dual_bank_activities
from phenoniche.model.phenotype_head import CoxHead


class DualBankFactorization(nn.Module):
    def __init__(self, patients, anchors, cell_types, contacts, communications,
                 phenotype_niches, background_niches, device=None, dtype=torch.float32,
                 composition=None, communication=None, inner_steps=100, inner_lr=1.0,
                 lambda_bc=1.0, lambda_bi=1.0):
        super().__init__()
        dimensions = (patients, anchors, cell_types, contacts, communications,
                      phenotype_niches, background_niches)
        if not all(isinstance(value, int) and value > 0 for value in dimensions):
            raise ValueError("All model dimensions must be positive integers")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Model dtype must be float32 or float64")
        if (composition is None) != (communication is None):
            raise ValueError("Both bulk input blocks must be supplied together")
        if composition is not None and (composition.shape != (patients, cell_types) or
                                        communication.shape != (patients, communications)):
            raise ValueError("Bound bulk observations do not match constructor dimensions")
        self.phenotype_niches = phenotype_niches
        self.background_niches = background_niches
        self.register_buffer("composition", None if composition is None else composition.to(device=device, dtype=dtype), persistent=False)
        self.register_buffer("communication", None if communication is None else communication.to(device=device, dtype=dtype), persistent=False)
        self.inference_options = {"steps": inner_steps, "learning_rate": inner_lr,
                                  "lambda_bc": lambda_bc, "lambda_bi": lambda_bi}
        shapes = {
            "WS0": (anchors, background_niches), "HC0": (background_niches, cell_types),
            "HO0": (background_niches, contacts), "HI0": (background_niches, communications),
            "WSp": (anchors, phenotype_niches), "HCp": (phenotype_niches, cell_types),
            "HOp": (phenotype_niches, contacts), "HIp": (phenotype_niches, communications),
        }
        for name, shape in shapes.items():
            self.register_parameter(f"raw_{name}", nn.Parameter(torch.randn(shape, device=device, dtype=dtype) * 0.5 - 1.2))
        self.phenotype_head = CoxHead(phenotype_niches, device=device, dtype=dtype)

    @property
    def gamma(self):
        return self.phenotype_head.gamma

    def dictionaries(self):
        return {name: F.softplus(getattr(self, f"raw_{name}")) for name in ("HC0", "HO0", "HI0", "HCp", "HOp", "HIp")}

    def spatial_factors(self):
        return {name: F.softplus(getattr(self, f"raw_{name}")) for name in ("WS0", "WSp")}

    def _bulk_inputs(self, composition, communication):
        if (composition is None) != (communication is None):
            raise ValueError("Both bulk input blocks must be supplied together")
        composition = self.composition if composition is None else composition
        communication = self.communication if communication is None else communication
        if composition is None or communication is None:
            raise ValueError("Bulk inference requires composition and communication observations")
        return composition, communication

    def background_activities(self, composition=None, communication=None, create_graph=True):
        composition, communication = self._bulk_inputs(composition, communication)
        dictionaries = self.dictionaries()
        return infer_background_activities(composition, communication, dictionaries["HC0"], dictionaries["HI0"],
                                           create_graph=create_graph, **self.inference_options)

    def bulk_factors(self, composition=None, communication=None, create_graph=True):
        composition, communication = self._bulk_inputs(composition, communication)
        dictionaries = self.dictionaries()
        return infer_dual_bank_activities(
            composition, communication, dictionaries["HC0"], dictionaries["HI0"],
            dictionaries["HCp"], dictionaries["HIp"], create_graph=create_graph,
            **self.inference_options)

    def background_forward(self, composition=None, communication=None):
        composition, communication = self._bulk_inputs(composition, communication)
        dictionaries = self.dictionaries()
        spatial = self.spatial_factors()
        wb0 = infer_background_activities(composition, communication, dictionaries["HC0"], dictionaries["HI0"],
                                          **self.inference_options)
        return {"CB": wb0 @ dictionaries["HC0"], "IB": wb0 @ dictionaries["HI0"],
                "CS": spatial["WS0"] @ dictionaries["HC0"],
                "OS": spatial["WS0"] @ dictionaries["HO0"],
                "IS": spatial["WS0"] @ dictionaries["HI0"],
                "factors": {"WB0": wb0, "WS0": spatial["WS0"],
                            **{name: dictionaries[name] for name in ("HC0", "HO0", "HI0")}}}

    def factors(self, composition=None, communication=None, create_graph=True):
        wb0, wbp = self.bulk_factors(composition, communication, create_graph=create_graph)
        return {"WB0": wb0, "WBp": wbp, **self.spatial_factors(), **self.dictionaries(), "gamma": self.gamma}

    def forward(self, composition=None, communication=None):
        factors = self.factors(composition, communication)
        return {
            "CB0": factors["WB0"] @ factors["HC0"], "CBp": factors["WBp"] @ factors["HCp"],
            "IB0": factors["WB0"] @ factors["HI0"], "IBp": factors["WBp"] @ factors["HIp"],
            "CS0": factors["WS0"] @ factors["HC0"], "CSp": factors["WSp"] @ factors["HCp"],
            "OS0": factors["WS0"] @ factors["HO0"], "OSp": factors["WSp"] @ factors["HOp"],
            "IS0": factors["WS0"] @ factors["HI0"], "ISp": factors["WSp"] @ factors["HIp"],
            "risk": self.phenotype_head(factors["WBp"]), "factors": factors,
        }


def set_trainable_bank(model, background, phenotype):
    for name, parameter in model.named_parameters():
        if name == "phenotype_head.gamma" or name.endswith(("HCp", "HOp", "HIp", "WSp")):
            parameter.requires_grad_(phenotype)
        else:
            parameter.requires_grad_(background)
