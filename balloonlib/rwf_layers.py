"""
rwf_layers — backward-compatibility shim.

All symbols have moved to :mod:`balloonlib.layers`, which is now the single
canonical source for RWF layers.  This module re-exports everything so that
existing imports continue to work:

    from balloonlib.rwf_layers import FactorizedLinear  # still works
"""

from balloonlib.layers import (  # noqa: F401
    FactorizedLinear,
    FourierFeatureMapping,
    replace_linear_with_factorized,
)

__all__ = ["FactorizedLinear", "FourierFeatureMapping", "replace_linear_with_factorized"]
