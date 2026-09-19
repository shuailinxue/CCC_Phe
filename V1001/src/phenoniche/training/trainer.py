from copy import deepcopy
from dataclasses import dataclass, field
import logging
import torch
from phenoniche.model.niche_factorization import NicheFactorization
from phenoniche.losses.reconstruction import squared_frobenius
from phenoniche.losses.survival import cox_breslow_loss
from phenoniche.losses.spatial import spatial_loss
from phenoniche.losses.collapse import cross_view_collapse_loss
from phenoniche.utils.seed import set_seed


@dataclass
class TrainingResult:
    model: NicheFactorization
    history: list[dict]
    diagnostics: dict = field(default_factory=dict)


def joint_loss(model, data, weights, phenotype=True):
    output = model(data.CB, data.IB)
    factors = output["factors"]
    coefficients = {"CB": weights.bc, "IB": weights.bi, "CS": weights.sc, "OS": weights.so, "IS": weights.si}
    reconstruction = {name: coefficients[name] * squared_frobenius(output[name], value)
                      for name, value in data.blocks().items()}
    bulk = reconstruction["CB"] + reconstruction["IB"]
    spatial = reconstruction["CS"] + reconstruction["OS"] + reconstruction["IS"]
    cox = cox_breslow_loss(output["risk"], data.time, data.event)
    smooth = output["risk"].sum() * 0
    if weights.sp > 0:
        if data.laplacian is None:
            raise ValueError("Positive spatial weight requires a graph Laplacian")
        smooth = spatial_loss(factors["WS"], data.laplacian)
    active_names = ["WB", "WS", "HI", "gamma"]
    if weights.bc > 0 or weights.sc > 0:
        active_names.append("HC")
    if weights.so > 0:
        active_names.append("HO")
    regularization = sum(factors[name].square().sum() for name in active_names)
    total = bulk + spatial + (weights.ph if phenotype else 0) * cox + weights.sp * smooth + weights.reg * regularization
    collapse = cross_view_collapse_loss(factors["HC"], factors["HO"], factors["HI"])
    if weights.lambda_collapse > 0:
        total = total + weights.lambda_collapse * collapse
    return {"total_loss": total, "bulk_recon": bulk, "spatial_recon": spatial,
            "cox_loss": cox, "spatial_loss": smooth, "regularization": regularization, "collapse_loss": collapse}


def train(data, config, weights):
    set_seed(config.seed)
    data = data.to(config.device, torch.float32)
    if weights.ph > 0 and config.joint_epochs > 0 and not data.event.any():
        raise ValueError("Phenotype-supervised training requires at least one observed event")
    model = NicheFactorization(data.CB.shape[0], data.CS.shape[0], data.CB.shape[1],
                               data.OS.shape[1], data.IB.shape[1], config.number_of_niches,
                               device=config.device, composition=data.CB, communication=data.IB,
                               inner_steps=config.inner_steps, inner_lr=config.inner_lr, lambda_bc=weights.bc, lambda_bi=weights.bi)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = []
    diagnostics = {"gradient_clip_count": 0, "max_gradient_norm": 0.0, "nonfinite_loss": False, "nonfinite_gradient": False}
    logger = logging.getLogger(__name__)
    for stage, epochs in (("warmup", config.warmup_epochs), ("joint", config.joint_epochs)):
        best_loss, stale, best_state = float("inf"), 0, None
        for stage_epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            losses = joint_loss(model, data, weights, phenotype=stage == "joint")
            total = losses["total_loss"]
            if not torch.isfinite(total):
                raise FloatingPointError(f"Nonfinite objective at {stage} epoch {stage_epoch}")
            record = {"epoch": len(history) + 1, "stage": stage,
                      **{name: float(value.detach()) for name, value in losses.items()}}
            history.append(record)
            improved = record["total_loss"] < best_loss - config.min_delta
            if improved:
                best_loss, stale = record["total_loss"], 0
                if config.early_stopping is not None:
                    best_state = deepcopy(model.state_dict())
            else:
                stale += 1
            if stage_epoch % config.log_every == 0 or stage_epoch == epochs - 1:
                logger.info("%s", record)
            if config.early_stopping is not None and stale >= config.early_stopping:
                break
            total.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("Nonfinite parameter gradient")
            if config.gradient_clip is not None:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            else:
                norm = torch.linalg.vector_norm(torch.stack([parameter.grad.norm() for parameter in model.parameters() if parameter.grad is not None]))
            norm_value = float(norm)
            clipped = config.gradient_clip is not None and norm_value > config.gradient_clip
            record["gradient_norm"] = norm_value
            record["gradient_clipped"] = clipped
            diagnostics["gradient_clip_count"] += int(clipped)
            diagnostics["max_gradient_norm"] = max(diagnostics["max_gradient_norm"], norm_value)
            optimizer.step()
        if config.early_stopping is not None and best_state is not None:
            with torch.no_grad():
                final_loss = float(joint_loss(model, data, weights, phenotype=stage == "joint")["total_loss"])
            if final_loss >= best_loss:
                model.load_state_dict(best_state)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    with torch.no_grad():
        if not all(torch.isfinite(v).all() for v in model.factors().values()):
            raise FloatingPointError("Training produced nonfinite factors")
        diagnostics["final_losses"] = {key: float(value) for key, value in joint_loss(model, data, weights, phenotype=config.joint_epochs > 0).items()}
    diagnostics["optimizer_steps"] = sum("gradient_norm" in row for row in history)
    return TrainingResult(model, history, diagnostics)
