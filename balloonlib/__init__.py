"""
BalloonLib — Physics-Informed Neural Network library for the Balloon haemodynamic model.
"""

# Runtime configuration (must come first — submodules consume it)
# Core submodules (import order respects dependency graph)
# Backward-compat shim (keeps `from balloonlib import balloonpinnlib` working)
from . import (
    MOO_Loss,
    ar1_whitening,
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
from .ar1_whitening import AR1Whitening
from .MOO_Loss import SoftAdapt, ReLoBRaLo, InverseGradNorm, LRAnnealing 
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
    "ar1_whitening",
    "MOO_Loss",
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
    "AR1Whitening",
    "SoftAdapt",
    "ReLoBRaLo",
    "InverseGradNorm",
    "LRAnnealing",
    "Multihead",
    "loss",
    "plotSignals",
    "set_device",
    "set_dtype",
    "train",
]
