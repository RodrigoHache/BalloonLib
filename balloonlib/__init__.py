"""
BalloonLib — Physics-Informed Neural Network library for the Balloon haemodynamic model.
"""

# Runtime configuration (must come first — submodules consume it)
# Core submodules (import order respects dependency graph)
# Backward-compat shim (keeps `from balloonlib import balloonpinnlib` working)
from . import (
    balloon_analysis,
    balloonmodellib,
    balloonpinnlib,
    config,
    data,
    layers,
    metrics,
    model,
    physics,
    plotting,
    training,
    utils,
)
from .config import set_device, set_dtype

# Convenience top-level re-exports
from .model import Multihead  
from .plotting import plotSignals 
from .training import loss, train 

# Package metadata
__version__ = "0.1.0"
__author__ = "Rodrigo H. Avaria"
__license__ = "MIT"
__email__ = "rodrigo.avaria@uv.cl"
__url__ = "https://github.com/errehache/BalloonLib"

# submodules
__all__ = [

    "balloon_analysis",
    "balloonmodellib",
    "balloonpinnlib",
    "config",
    "data",
    "layers",
    "metrics",
    "model",
    "physics",
    "plotting",
    "training",
    "utils",
]
# top-level symbols
__all__ += [
    "Multihead",
    "loss",
    "plotSignals",
    "set_device",
    "set_dtype",
    "train",
]
