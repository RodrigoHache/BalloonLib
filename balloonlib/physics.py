"""
Physics utilities for the BalloonLib PINN framework.

Provides automatic-differentiation helpers, temporal segmentation, and
causal ODE loss weighting functions used during PINN training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        create_graph=True,  # required for higher-order derivatives
        allow_unused=True,
    )[0]

    if ds_dt is None:
        raise ValueError("d(signal)/d(arg) is None — check that arg is in the computation graph.")
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

    The returned weights are detached from the autograd graph.  They are a
    function of ``segment_losses`` but are held constant within a step; see
    Wang et al., arXiv:2203.07404, Alg. 1.

    Parameters
    ----------
    segment_losses : torch.Tensor
        Per-segment loss values, shape ``(n_segments,)``.
    epsilon : float
        Decay rate.  Larger values produce more aggressive down-weighting.

    Returns
    -------
    torch.Tensor
        Weights, shape ``(n_segments,)``.
    """
    # Weights are treated as constants of the current step: the cumulative sum is
    # detached so that autograd does not differentiate w_i through the preceding
    # segment losses.  Differentiating through w_i would reward inflating early
    # residuals in order to suppress the weights of later segments
    # (Wang et al. 2024, arXiv:2203.07404, use lax.stop_gradient here).
    cumsum_losses = torch.cumsum(segment_losses.detach(), dim=0)

    # Shift right by one: weights[i] = exp(-ε * cumsum[i-1]), weights[0] = 1
    exponents = torch.cat([cumsum_losses.new_zeros(1), cumsum_losses[:-1]])   # (n_segments,)

    return torch.exp(-epsilon * exponents)


# Loss modules that are an elementwise loss followed by a mean/sum: they can be
# evaluated once on the whole residual and reduced per segment (segment_losses).
_ELEMENTWISE_LOSSES = {
    nn.MSELoss:      lambda f, r: F.mse_loss(r, torch.zeros_like(r), reduction="none"),
    nn.L1Loss:       lambda f, r: F.l1_loss(r, torch.zeros_like(r), reduction="none"),
    nn.SmoothL1Loss: lambda f, r: F.smooth_l1_loss(r, torch.zeros_like(r), reduction="none", beta=f.beta),
    nn.HuberLoss:    lambda f, r: F.huber_loss(r, torch.zeros_like(r), reduction="none", delta=f.delta),
}


def segment_sizes(sample_size: int, n_segments: int) -> List[int]:
    """Segment lengths used by :func:`segment_temporal_residuals` (remainder to the first segments)."""
    base, rem = divmod(sample_size, n_segments)
    return [base + (1 if i < rem else 0) for i in range(n_segments)]


def segment_losses(residual: torch.Tensor, meFn, n_segments: int) -> torch.Tensor:
    """Per-segment loss, shape ``(n_segments,)``.

    Equal in value (to float32 rounding) to
    ``torch.stack([meFn(s, 0*s) for s in segment_temporal_residuals(residual, n_segments)])``.
    For ``nn.MSELoss``, ``nn.L1Loss``, ``nn.SmoothL1Loss`` and ``nn.HuberLoss`` with
    ``reduction`` 'mean' or 'sum', the elementwise loss is computed once and reduced per
    segment (one reshape for equal segments, one scatter-add otherwise).  Any other
    callable falls back to one ``meFn`` call per segment.
    """
    elem = _ELEMENTWISE_LOSSES.get(type(meFn))
    if elem is None or getattr(meFn, "reduction", None) not in ("mean", "sum"):
        segs = segment_temporal_residuals(residual, n_segments)
        return torch.stack([meFn(seg, torch.zeros_like(seg)) for seg in segs])

    time_dim = residual.shape.index(max(residual.shape))     # same rule as segment_temporal_residuals
    e = elem(meFn, residual).movedim(time_dim, 0)
    N = e.shape[0]
    e = e.reshape(N, -1)                                      # (time points, elements per time point)
    R = e.shape[1]
    mean = meFn.reduction == "mean"
    if N % n_segments == 0:
        e = e.reshape(n_segments, -1)
        return e.mean(1) if mean else e.sum(1)
    sizes = torch.tensor(segment_sizes(N, n_segments), device=e.device)
    ids = torch.repeat_interleave(torch.arange(n_segments, device=e.device), sizes)
    s = e.new_zeros(n_segments).index_add_(0, ids, e.sum(1))
    return s / (sizes * R).to(s.dtype) if mean else s


def weighted_temporal_ode_loss(
    residual: torch.Tensor,
    meFn=nn.MSELoss(),
    n_segments: int = 10,
    epsilon: float = 1.0,
    tolerance: float = 1e-6,
    normalise_weights: bool = True,
    device: str | None = None,
    return_weights=False,
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
    tolerance : float
        How above nought are we going to consider active segment
    normalise_weights : bool
        If ``True``, divide weights by the number of active segments
        (weight > ``tolerance``).  Note: this is not the paper's 1/N_t
        (divide by ``n_segments``) unless all segments are active.
    device : str or None
        Computation device.  ``None`` infers from ``residual``.

    Returns
    -------
    torch.Tensor
        Scalar weighted ODE loss.
    """
    seg_losses = segment_losses(residual, meFn, n_segments)

    weights = compute_temporal_weights(seg_losses, epsilon, device)

    if normalise_weights:
        # count of active segments as a tensor: no host-device sync on GPU
        weights = weights / (weights > tolerance).sum()

    total = torch.dot(weights, seg_losses)
    if return_weights:
        return total, weights.detach(), seg_losses.detach()
    return total
