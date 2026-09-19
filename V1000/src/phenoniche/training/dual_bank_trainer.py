from copy import deepcopy
from dataclasses import dataclass, field
import math
import torch
from phenoniche.losses.reconstruction import squared_frobenius
from phenoniche.losses.spatial import spatial_loss
from phenoniche.losses.survival import cox_breslow_loss
from phenoniche.model.dual_bank_factorization import DualBankFactorization, set_trainable_bank
from phenoniche.utils.seed import set_seed


@dataclass(frozen=True)
class DualBankTrainingConfig:
    phenotype_niches: int
    background_niches: int
    background_epochs: int = 800
    phenotype_epochs: int = 800
    jointft_epochs: int = 200
    scratch_epochs: int = 1600
    learning_rate: float = 0.025
    jointft_learning_rate_factor: float = 0.1
    weight_decay: float = 0.0
    gradient_clip: float | None = 10.0
    seed: int = 31
    device: str = "cpu"
    inner_steps: int = 100
    inner_lr: float = 1.0

    def __post_init__(self):
        if not all(isinstance(value, int) and value > 0 for value in (self.phenotype_niches, self.background_niches)):
            raise ValueError("Both bank sizes must be positive integers")
        if not all(isinstance(value, int) and value >= 0 for value in
                   (self.background_epochs, self.phenotype_epochs, self.jointft_epochs, self.scratch_epochs)):
            raise ValueError("Epoch counts must be nonnegative integers")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.jointft_learning_rate_factor) or not 0 < self.jointft_learning_rate_factor <= 1:
            raise ValueError("jointft_learning_rate_factor must be in (0, 1]")
        if self.gradient_clip is not None and (not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0):
            raise ValueError("gradient_clip must be finite and positive or None")
        if not isinstance(self.inner_steps, int) or self.inner_steps < 1:
            raise ValueError("inner_steps must be a positive integer")
        if not math.isfinite(self.inner_lr) or not 0 < self.inner_lr <= 1:
            raise ValueError("inner_lr must be in (0, 1]")

    @property
    def total_niches(self):
        return self.phenotype_niches + self.background_niches


@dataclass
class DualBankTrainingResult:
    model: DualBankFactorization
    history: list[dict]
    diagnostics: dict = field(default_factory=dict)


def build_dual_bank_model(data, config, weights):
    return DualBankFactorization(
        data.CB.shape[0], data.CS.shape[0], data.CB.shape[1], data.OS.shape[1],
        data.IB.shape[1], config.phenotype_niches, config.background_niches,
        device=config.device, composition=data.CB, communication=data.IB,
        inner_steps=config.inner_steps, inner_lr=config.inner_lr,
        lambda_bc=weights.bc, lambda_bi=weights.bi)


def background_loss(model, data, weights):
    output = model.background_forward(data.CB, data.IB)
    coefficients = {"CB": weights.bc, "IB": weights.bi, "CS": weights.sc,
                    "OS": weights.so, "IS": weights.si}
    reconstruction = {name: coefficients[name] * squared_frobenius(output[name], getattr(data, name))
                      for name in coefficients}
    regularization = sum(output["factors"][name].square().sum()
                         for name in ("WB0", "WS0", "HC0", "HO0", "HI0"))
    bulk = reconstruction["CB"] + reconstruction["IB"]
    spatial = reconstruction["CS"] + reconstruction["OS"] + reconstruction["IS"]
    return {"total_loss": bulk + spatial + weights.reg * regularization,
            "bulk_recon": bulk, "spatial_recon": spatial,
            "cox_loss": bulk * 0, "spatial_loss": bulk * 0,
            "regularization": regularization, "collapse_loss": bulk * 0}


def dual_bank_loss(model, data, weights):
    if weights.lambda_collapse != 0:
        raise ValueError("Dual-bank experiments require lambda_collapse=0")
    output = model(data.CB, data.IB)
    totals = {name: output[f"{name}0"] + output[f"{name}p"] for name in ("CB", "IB", "CS", "OS", "IS")}
    coefficients = {"CB": weights.bc, "IB": weights.bi, "CS": weights.sc,
                    "OS": weights.so, "IS": weights.si}
    reconstruction = {name: coefficients[name] * squared_frobenius(totals[name], getattr(data, name))
                      for name in totals}
    bulk = reconstruction["CB"] + reconstruction["IB"]
    spatial = reconstruction["CS"] + reconstruction["OS"] + reconstruction["IS"]
    cox = cox_breslow_loss(output["risk"], data.time, data.event)
    smooth = output["risk"].sum() * 0
    if weights.sp > 0:
        if data.laplacian is None:
            raise ValueError("Positive spatial weight requires a graph Laplacian")
        smooth = spatial_loss(output["factors"]["WS0"], data.laplacian) + spatial_loss(output["factors"]["WSp"], data.laplacian)
    regularization = sum(output["factors"][name].square().sum() for name in
                         ("WB0", "WBp", "WS0", "WSp", "HC0", "HO0", "HI0",
                          "HCp", "HOp", "HIp", "gamma"))
    total = bulk + spatial + weights.ph * cox + weights.sp * smooth + weights.reg * regularization
    return {"total_loss": total, "bulk_recon": bulk, "spatial_recon": spatial,
            "cox_loss": cox, "spatial_loss": smooth,
            "regularization": regularization, "collapse_loss": total * 0}


def _run_stage(model, data, weights, epochs, learning_rate, stage, loss_function, config):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Training stage has no trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=config.weight_decay)
    history = []
    diagnostics = {"gradient_clip_count": 0, "max_gradient_norm": 0.0,
                   "nonfinite_loss": False, "nonfinite_gradient": False}
    for stage_epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses = loss_function(model, data, weights)
        total = losses["total_loss"]
        if not torch.isfinite(total):
            diagnostics["nonfinite_loss"] = True
            raise FloatingPointError(f"Nonfinite objective at {stage} epoch {stage_epoch}")
        total.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in parameters):
            diagnostics["nonfinite_gradient"] = True
            raise FloatingPointError(f"Nonfinite gradient at {stage} epoch {stage_epoch}")
        if config.gradient_clip is None:
            norm = torch.linalg.vector_norm(torch.stack([parameter.grad.norm() for parameter in parameters if parameter.grad is not None]))
        else:
            norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip, error_if_nonfinite=True)
        norm_value = float(norm)
        clipped = config.gradient_clip is not None and norm_value > config.gradient_clip
        diagnostics["gradient_clip_count"] += int(clipped)
        diagnostics["max_gradient_norm"] = max(diagnostics["max_gradient_norm"], norm_value)
        optimizer.step()
        history.append({"epoch": stage_epoch + 1, "stage": stage,
                        **{name: float(value.detach()) for name, value in losses.items()},
                        "gradient_norm": norm_value, "gradient_clipped": clipped})
    with torch.no_grad():
        diagnostics["final_losses"] = {name: float(value) for name, value in loss_function(model, data, weights).items()}
    diagnostics["optimizer_steps"] = epochs
    return history, diagnostics


def train_background(data, config, weights):
    if weights.lambda_collapse != 0:
        raise ValueError("Dual-bank experiments require lambda_collapse=0")
    set_seed(config.seed)
    data = data.to(config.device, torch.float32)
    model = build_dual_bank_model(data, config, weights)
    set_trainable_bank(model, background=True, phenotype=False)
    history, diagnostics = _run_stage(model, data, weights, config.background_epochs,
                                      config.learning_rate, "background", background_loss, config)
    return DualBankTrainingResult(model, history, diagnostics)


def train_residual_from_background(data, config, weights, background_state, jointft=True):
    if weights.lambda_collapse != 0:
        raise ValueError("Dual-bank experiments require lambda_collapse=0")
    data = data.to(config.device, torch.float32)
    model = build_dual_bank_model(data, config, weights)
    model.load_state_dict(background_state)
    set_trainable_bank(model, background=False, phenotype=True)
    history, stage2 = _run_stage(model, data, weights, config.phenotype_epochs,
                                 config.learning_rate, "phenotype_residual", dual_bank_loss, config)
    frozen_state = deepcopy(model.state_dict())
    frozen_diagnostics = {"stages": {"phenotype_residual": stage2},
                          "gradient_clip_count": stage2["gradient_clip_count"],
                          "max_gradient_norm": stage2["max_gradient_norm"],
                          "final_losses": stage2["final_losses"]}
    frozen = DualBankTrainingResult(deepcopy(model), list(history), frozen_diagnostics)
    if not jointft or config.jointft_epochs == 0:
        return frozen, None
    set_trainable_bank(model, background=True, phenotype=True)
    ft_history, ft = _run_stage(model, data, weights, config.jointft_epochs,
                                config.learning_rate * config.jointft_learning_rate_factor,
                                "joint_finetune", dual_bank_loss, config)
    joint_diagnostics = {"stages": {"phenotype_residual": stage2, "joint_finetune": ft},
                         "gradient_clip_count": stage2["gradient_clip_count"] + ft["gradient_clip_count"],
                         "max_gradient_norm": max(stage2["max_gradient_norm"], ft["max_gradient_norm"]),
                         "final_losses": ft["final_losses"],
                         "frozen_state_preserved": all(torch.equal(frozen_state[key], frozen.model.state_dict()[key]) for key in frozen_state)}
    return frozen, DualBankTrainingResult(model, history + ft_history, joint_diagnostics)


def train_joint_scratch(data, config, weights):
    if weights.lambda_collapse != 0:
        raise ValueError("Dual-bank experiments require lambda_collapse=0")
    set_seed(config.seed)
    data = data.to(config.device, torch.float32)
    model = build_dual_bank_model(data, config, weights)
    set_trainable_bank(model, background=True, phenotype=True)
    history, diagnostics = _run_stage(model, data, weights, config.scratch_epochs,
                                      config.learning_rate, "joint_scratch", dual_bank_loss, config)
    return DualBankTrainingResult(model, history, diagnostics)
