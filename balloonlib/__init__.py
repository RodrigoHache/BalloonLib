"""
BalloonLib — Physics-Informed Neural Network library for the Balloon haemodynamic model.
"""

# Runtime configuration (must come first — submodules consume it)
from . import config
from .config import set_device, set_dtype  # noqa: F401

# Core submodules (import order respects dependency graph)
from . import balloonmodellib
from . import utils
from . import layers
from . import physics
from . import data
from . import metrics
from . import model
from . import plotting
from . import training
from . import balloon_analysis

# Backward-compat shim (keeps `from balloonlib import balloonpinnlib` working)
from . import balloonpinnlib

# Convenience top-level re-exports
from .model import Multihead  # noqa: F401
from .training import loss, train  # noqa: F401
from .plotting import plotSignals  # noqa: F401

# Package metadata
__version__ = "0.1.0"
__author__ = "Rodrigo H. Avaria"
__license__ = "MIT"
__email__ = "rodrigo.avaria@uv.cl"
__url__ = "https://github.com/errehache/BalloonLib"

__all__ = [
    # submodules
    "config",
    "balloonmodellib",
    "utils",
    "layers",
    "physics",
    "data",
    "metrics",
    "model",
    "plotting",
    "training",
    "balloonpinnlib",
    "balloon_analysis",
    # top-level symbols
    "set_device",
    "set_dtype",
    "Multihead",
    "loss",
    "train",
    "plotSignals",
]
