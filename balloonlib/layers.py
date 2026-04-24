"""
Neural network layers for the BalloonLib PINN framework.

Provides Fourier Feature Mapping and Random Weight Factorization (RWF) layers.
This is the single canonical source; ``rwf_layers.py`` re-exports from here
for backward compatibility.

References
----------
Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions
in Low Dimensional Domains", NeurIPS 2020.

Wang et al., "Random Weight Factorization Improves the Training of
Continuous Neural Representations", arXiv:2210.01274, 2022.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Fourier Feature Mapping
# ---------------------------------------------------------------------------

class FourierFeatureMapping(nn.Module):
    """Fourier Feature Mapping for coordinate-based neural networks.

    Maps input coordinates through random Fourier features to enable networks
    to learn high-frequency functions. Essential for PINNs and coordinate-based
    tasks like image fitting, SDFs, and neural radiance fields.

    The mapping transforms input ``x ∈ R^d`` to:

    .. math::

        \\gamma(x) = [\\cos(2\\pi B x),\\; \\sin(2\\pi B x)]

    where ``B ∈ R^{m × d}`` is a random matrix sampled from a Gaussian.

    References
    ----------
    Tancik et al., "Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains", NeurIPS 2020.
    """

    def __init__(
        self,
        input_dim: int = 1,
        mapping_size: int = 1,
        scale: float = 1.0,
        learnable: bool = True,
        use_2pi: bool = False,
    ):
        """Initialise a Fourier Feature Mapping layer.

        Parameters
        ----------
        input_dim : int
            Dimension of input coordinates (e.g., 1 for time, 2 for 2-D).
        mapping_size : int
            Number of random Fourier features (*m*).
        scale : float
            Standard deviation of the Gaussian used to sample the ``B`` matrix.
            Higher values capture higher frequencies (typical range 1.0–100.0).
        learnable : bool
            If ``True``, ``B`` is a trainable parameter.
        use_2pi : bool
            If ``True``, multiply projections by ``2π``.
        """
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.output_dim = 2 * mapping_size  # cos and sin features
        self.mult = 2 * torch.pi if use_2pi else 1.0

        B = torch.randn(mapping_size, input_dim) * scale

        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature mapping.

        Computes ``γ(x) = [cos(B x), sin(B x)]``
        (optionally with a ``2π`` prefactor).

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates, shape ``(..., input_dim)``.

        Returns
        -------
        torch.Tensor
            Fourier features, shape ``(..., 2 * mapping_size)``.
        """
        x_proj = self.mult * (x @ self.B.T)
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


# ---------------------------------------------------------------------------
# Random Weight Factorization linear layer
# ---------------------------------------------------------------------------

class FactorizedLinear(nn.Module):
    """Linear layer with Random Weight Factorization (RWF).

    Factorises weights as ``W = diag(exp(s)) * V`` where *s* are
    learnable scale factors and *V* are value weights.  This improves
    gradient flow and conditioning for PINNs and other continuous
    neural representations.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool
        If ``True``, adds a learnable bias.  Default ``True``.
    dtype : torch.dtype
        Parameter data type.
    scale_mean : float
        Mean for scale-factor initialisation (``s ~ N(mu, sigma)``).
    scale_std : float
        Standard deviation for scale-factor initialisation.

    Notes
    -----
    Initialisation follows Wang et al. (2022):

    1. Glorot-initialise a full weight matrix ``W``.
    2. Sample ``s ~ N(scale_mean, scale_std)``.
    3. Set ``V = W / exp(s)`` to maintain proper initialisation scale.

    References
    ----------
    Wang et al., "Random Weight Factorization Improves the Training
    of Continuous Neural Representations", arXiv:2210.01274, 2022.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        scale_mean: float = 1.0,
        scale_std: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale_mean = scale_mean
        self.scale_std = scale_std

        # Step 1: Glorot (Xavier) initialisation
        weight_init = torch.empty((in_features, out_features), dtype=dtype)
        nn.init.xavier_normal_(weight_init)

        # Step 2: Scale factors s ~ N(μ, σ²)
        s_init = torch.empty(out_features, dtype=dtype)
        nn.init.normal_(s_init, scale_mean, scale_std)

        # Step 3: V = W / exp(s)
        exp_s_init = torch.exp(s_init)
        v_init = weight_init / exp_s_init.unsqueeze(0)

        # Learnable parameters (store s, not exp(s))
        self.scale = nn.Parameter(s_init)
        self.value = nn.Parameter(v_init)

        if bias:
            self.bias = nn.Parameter(torch.ones(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly weight reconstruction.

        Reconstructs ``W = exp(s) * V`` and computes ``y = x @ W^T + b``.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(..., in_features)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(..., out_features)``.
        """
        exp_scale = torch.exp(self.scale)
        kernel = self.value * exp_scale.unsqueeze(0)
        return F.linear(x, kernel.T, self.bias)

    def get_effective_weight(self) -> torch.Tensor:
        """Return the reconstructed weight matrix ``W = exp(s) * V``.

        Returns
        -------
        torch.Tensor
            Shape ``(in_features, out_features)``.
        """
        return self.value * torch.exp(self.scale).unsqueeze(0)

    @property
    def weight(self) -> torch.Tensor:
        """Effective weight matrix (``nn.Linear``-compatible property)."""
        return self.get_effective_weight()

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"scale_mean={self.scale_mean}, "
            f"scale_std={self.scale_std}"
        )


# ---------------------------------------------------------------------------
# Utility: swap nn.Linear → FactorizedLinear in an existing model
# ---------------------------------------------------------------------------

def replace_linear_with_factorized(
    module: nn.Module,
    scale_mean: float = 1.0,
    scale_std: float = 0.1,
    inplace: bool = True,
) -> nn.Module:
    """Recursively replace all ``nn.Linear`` layers with :class:`FactorizedLinear`.

    Parameters
    ----------
    module : nn.Module
        PyTorch module to convert.
    scale_mean : float
        Mean for scale-factor initialisation.
    scale_std : float
        Std dev for scale-factor initialisation.
    inplace : bool
        If ``True``, modifies *module* in place; otherwise deep-copies first.

    Returns
    -------
    nn.Module
        Modified module with :class:`FactorizedLinear` layers.
    """
    if not inplace:
        module = copy.deepcopy(module)

    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            factorized = FactorizedLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=(child.bias is not None),
                dtype=child.weight.dtype,
                scale_mean=scale_mean,
                scale_std=scale_std,
            )
            setattr(module, name, factorized)
        else:
            replace_linear_with_factorized(
                child,
                scale_mean=scale_mean,
                scale_std=scale_std,
                inplace=True,
            )

    return module
