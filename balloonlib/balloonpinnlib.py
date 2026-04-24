"""
balloonpinnlib — backward-compatibility shim.

All symbols have been moved to focused submodules.
Import directly from those for new code:

    from balloonlib.model    import Multihead
    from balloonlib.training import loss, train
    from balloonlib.utils    import scale_domains, pytorch_convolve
    ...

This module re-exports every public symbol so that existing notebooks
which use ``from balloonlib.balloonpinnlib import *`` continue to work
without modification.
"""

# Layers
from balloonlib.layers import (  # noqa: F401
    FourierFeatureMapping,
    FactorizedLinear,
    replace_linear_with_factorized,
)

# Utilities
from balloonlib.utils import (  # noqa: F401
    tensor2np,
    np2tensor,
    scale_domains,
    DoubleGamma,
    timeBall,
    pytorch_convolve,
    tofit,
)

# Physics
from balloonlib.physics import (  # noqa: F401
    dfdt,
    segment_temporal_residuals,
    compute_temporal_weights,
    weighted_temporal_ode_loss,
)

# Data
from balloonlib.data import (  # noqa: F401
    training_data,
    normFn,
    segmentData,
    experimental_stims,
    load_pickle,
)

# Metrics
from balloonlib.metrics import (  # noqa: F401
    kge_stat,
    hrf_description,
)

# Model
from balloonlib.model import Multihead  # noqa: F401

# Plotting
from balloonlib.plotting import (  # noqa: F401
    plot_trace,
    plot_balloon_fitting,
    plot_weights,
    plotHRFs,
    plotSignals,
)

# Training
from balloonlib.training import (  # noqa: F401
    loss_reweight_paranoid,
    compute_per_loss_gradients,
    loss,
    train,
)

__all__ = [
    # layers
    "FourierFeatureMapping", "FactorizedLinear", "replace_linear_with_factorized",
    # utils
    "tensor2np", "np2tensor", "scale_domains", "DoubleGamma",
    "timeBall", "pytorch_convolve", "tofit",
    # physics
    "dfdt", "segment_temporal_residuals", "compute_temporal_weights",
    "weighted_temporal_ode_loss",
    # data
    "training_data", "normFn", "segmentData", "experimental_stims", "load_pickle",
    # metrics
    "kge_stat", "hrf_description",
    # model
    "Multihead",
    # plotting
    "plot_trace", "plot_balloon_fitting", "plot_weights", "plotHRFs", "plotSignals",
    # training
    "loss_reweight_paranoid", "compute_per_loss_gradients", "loss", "train",
]