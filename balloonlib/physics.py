"""
Physics utilities for the BalloonLib PINN framework.

Provides automatic-differentiation helpers, temporal segmentation, and
causal ODE loss weighting functions used during PINN training.
"""

import torch
import torch.nn as nn
from typing import List


# ---------------------------------------------------------------------------
# Automatic differentiation
# ---------------------------------------------------------------------------

def dfdt(signal: torch.Tensor, arg: torch.Tensor) -> torch.Tensor:
    """Compute the derivative of ``signal`` with respect to ``arg`` via autograd.

    Parameters
    ----------
    signal : torch.Tensor
        Input tensor of shape ``(N,)`` or ``(N, 1)``.
    arg : torch.Tensor
        Tensor with respect to which the derivative is computed;
        must have ``requires_grad=True``.

    Returns
    -------
    torch.Tensor
        Derivative ``d(signal)/d(arg)``, shape ``(N, 1)``.

    Raises
    ------
    ValueError
        If ``signal`` has shape incompatible with ``(N, 1)``.
    ValueError
        If autograd returns ``None`` (disconnected graph).
    """
    if signal.dim() == 1:
        signal = signal.unsqueeze(1)
    elif signal.dim() == 2 and signal.size(1) != 1:
        raise ValueError("signal must have shape [N] or [N, 1]")

    sig = signal

    ds_dt = torch.autograd.grad(
        outputs=sig,
        inputs=arg,
        grad_outputs=torch.ones_like(sig),
        create_graph=True,   # required for higher-order derivatives
        allow_unused=True,
    )[0]

    if ds_dt is None:
        raise ValueError(
            "d(signal)/d(arg) is None — check that arg is in the computation graph."
        )
    return ds_dt.requires_grad_()


# ---------------------------------------------------------------------------
# Temporal segmentation
# ---------------------------------------------------------------------------

def segment_temporal_residuals(residual: torch.Tensor, n_segments: int) -> List[torch.Tensor]:
    """Split ODE residuals into temporal segments.

    Divides the residual tensor along its largest (temporal) dimension
    into ``n_segments`` contiguous chunks.  Remainder samples are
    spread across the first segments to keep sizes uniform.

    Parameters
    ----------
    residual : torch.Tensor
        ODE residual tensor, e.g. shape ``(sample_size, n_equations)``.
    n_segments : int
        Number of temporal segments to create.

    Returns
    -------
    list of torch.Tensor
        One tensor per segment.
    """
    time_dim = residual.shape.index(max(residual.shape))
    sample_size = residual.shape[time_dim]

    base_size = sample_size // n_segments
    remainder = sample_size % n_segments

    segments = []
    current_idx = 0

    for i in range(n_segments):
        segment_size = base_size + (1 if i < remainder else 0)
        segment = torch.narrow(residual, time_dim, current_idx, segment_size)
        segments.append(segment)
        current_idx += segment_size

    return segments


def compute_temporal_weights(
    segment_losses: torch.Tensor,
    epsilon: float = 1.0,
    device: str | None = None,
) -> torch.Tensor:
    """Compute causal exponential weights for temporal segments.

    Weight for segment *i* is ``exp(-epsilon * sum(losses[0:i]))``,
    ensuring earlier segments with high loss down-weight later segments.
    Segment 0 always receives weight 1.0.

    Parameters
    ----------
    segment_losses : torch.Tensor
        Per-segment loss values, shape ``(n_segments,)``.
    epsilon : float
        Decay rate.  Larger values produce more aggressive down-weighting.
    device : str or None
        Computation device.  ``None`` infers the device from ``segment_losses``.

    Returns
    -------
    torch.Tensor
        Weights, shape ``(n_segments,)``.
    """
    dev = segment_losses.device if device is None else device
    
    cumsum_losses = torch.cumsum(segment_losses, dim=0)

    # Shift right by one: weights[i] = exp(-ε * cumsum[i-1]), weights[0] = 1                                 
    exponents = torch.cat([
        torch.zeros(1, device=dev, dtype=segment_losses.dtype),                                              
        cumsum_losses[:-1]
      ])                                                     # (n_segments,)                                   
   
    return torch.exp(-epsilon * exponents) 


def weighted_temporal_ode_loss(
    residual: torch.Tensor,
    meFn,
    n_segments: int = 10,
    epsilon: float = 1.0,
    normalize_weights: bool = True,
    device: str | None = None,
) -> torch.Tensor:
    """Compute ODE loss with causal temporal weighting.

    Segments the residual tensor into ``n_segments`` temporal chunks,
    computes the loss for each, and returns their weighted sum using
    exponential causal weights from :func:`compute_temporal_weights`.

    Parameters
    ----------
    residual : torch.Tensor
        ODE residual, shape ``(sample_size, n_equations)``.
    meFn : callable
        Loss function, e.g. ``nn.MSELoss()``.
    n_segments : int
        Number of temporal segments.
    epsilon : float
        Decay rate for the causal weights.
    normalize_weights : bool
        If ``True``, divide weights by ``n_segments``.
    device : str or None
        Computation device.  ``None`` infers from ``residual``.

    Returns
    -------
    torch.Tensor
        Scalar weighted ODE loss.
    """
    segments = segment_temporal_residuals(residual, n_segments)

    segment_losses = torch.stack([
        meFn(seg, torch.zeros_like(seg)) for seg in segments
    ])

    weights = compute_temporal_weights(segment_losses, epsilon, device)

    if normalize_weights:
        weights = weights / weights.shape[0]

    return torch.sum(weights * segment_losses)
