"""Independent downstream GNN niche analysis; no connection to clinical Cox screening."""
from .config import GNNConfig
from .model import MicroenvironmentGNN

__all__ = ["GNNConfig", "MicroenvironmentGNN"]
