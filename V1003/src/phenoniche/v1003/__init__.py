"""Interpretable CCC autoencoder for the Prime 5K V1003 experiment."""
from .config import ExperimentConfig, default_config
from .model import InterpretableCCCAutoencoder
from .pipeline import ensure_results, run_pipeline

__all__ = [
    "ExperimentConfig",
    "InterpretableCCCAutoencoder",
    "default_config",
    "ensure_results",
    "run_pipeline",
]

