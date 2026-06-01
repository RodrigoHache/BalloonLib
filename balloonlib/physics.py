"""
Physics utilities for the BalloonLib PINN framework.

Provides automatic-differentiation helpers, temporal segmentation, and
causal ODE loss weighting functions used during PINN training.
"""

import numpy as np
import torch
import torch.nn as nn
from numba import njit
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


def segment_temporal_residuals(
    residual: torch.Tensor, n_segments: int, time_dim: int = 0
) -> List[torch.Tensor]:
    """Split ODE residuals into temporal segments.

    Divides the residual tensor along ``time_dim`` into ``n_segments``
    contiguous chunks.  Remainder samples are spread across the first
    segments to keep sizes uniform.

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
    exponents = torch.cat(
        [torch.zeros(1, device=dev, dtype=segment_losses.dtype), cumsum_losses[:-1]]
    )  # (n_segments,)

    return torch.exp(-epsilon * exponents)


def weighted_temporal_ode_loss(
    residual: torch.Tensor,
    meFn=nn.MSELoss(),
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
    N = residual.shape[0]
    seg_size = N // n_segments
    # Reshape into (n_segments, seg_size, n_equations) — one kernel replaces n_segments MSE calls.
    # Truncation drops at most n_segments-1 trailing samples, negligible for large N.
    r = residual[: seg_size * n_segments].reshape(n_segments, seg_size, -1)
    segment_losses = r.pow(2).mean(dim=(1, 2))  # (n_segments,)

    weights = compute_temporal_weights(segment_losses, epsilon, device)

    if normalize_weights:
        weights = weights / n_segments

    return torch.sum(weights * segment_losses)

@torch.no_grad()
def neurovascular_rk4_torch(
    I: "torch.Tensor",
    dt: float = 0.01,
    AmpI_f=0.2,
    AmpI_m=0.05,
    kappa=1 / 1.54,
    gamma=1 / 2.46,
    f0: float = 1.0,
    m0: float = 1.0,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Neurovascular coupling ODE solved with fixed-step RK4 in PyTorch.

    Implements the Stephan 2007 / Friston 2000 flow-inducing signal model for
    both CBF (f) and CMRO2 (m) in a single integration pass.

    State vector: ``[s_f, f, s_m, m]``

    .. code-block:: none

        ds_f/dt = I(t)*AmpI_f - kappa*s_f - gamma*(f - 1)
        df/dt   = s_f
        ds_m/dt = I(t)*AmpI_m - kappa*s_m - gamma*(m - 1)
        dm/dt   = s_m

    Parameters
    ----------
    I : torch.Tensor
        Stimulus tensor, shape ``(N,)`` or ``(N, 1)``. Piecewise-constant.
        Midpoint interpolation is used between adjacent samples for RK4
        intermediate stages.
    dt : float
        Time step in seconds (must match the resolution of ``I``).
    AmpI_f : float or torch.Tensor
        Stimulus amplitude scaling for CBF (λ_r for the f head). Default 0.2.
    AmpI_m : float or torch.Tensor
        Stimulus amplitude scaling for CMRO2 (λ_r for the m head). Default 0.05.
    kappa : float or torch.Tensor
        Flow-signal decay rate (κ).
    gamma : float or torch.Tensor
        Autoregulatory feedback gain (γ).
    f0, m0 : float
        Baseline initial conditions (default 1.0).

    Returns
    -------
    f : torch.Tensor
        Normalised cerebral blood flow, shape ``(N,)``, same device as ``I``.
    m : torch.Tensor
        Normalised CMRO2, shape ``(N,)``, same device as ``I``.

    Notes
    -----
    ``kappa``, ``gamma``, ``AmpI_f``, and ``AmpI_m`` may be scalar
    ``torch.Tensor`` values with ``requires_grad=True``.  Gradients flow
    through the full integration, so this function can be used as a
    differentiable reference inside a training loss.
    """
    I = I.squeeze()
    device, dtype = I.device, I.dtype
    N = len(I)
    h = torch.tensor(dt, device=device, dtype=dtype)

    state = torch.tensor([0.0, float(f0), 0.0, float(m0)], device=device, dtype=dtype)

    def ode(st, I_t):
        s_f, f, s_m, m = st[0], st[1], st[2], st[3]
        return torch.stack([
            I_t * AmpI_f - kappa * s_f - gamma * (f - 1),
            s_f,
            I_t * AmpI_m - kappa * s_m - gamma * (m - 1),
            s_m,
        ])

    f_out = torch.empty(N, device=device, dtype=dtype)
    m_out = torch.empty(N, device=device, dtype=dtype)
    f_out[0] = state[1]
    m_out[0] = state[3]

    for i in range(N - 1):
        I0, I1 = I[i], I[i + 1]
        Ih = 0.5 * (I0 + I1)

        k1 = ode(state,                I0)
        k2 = ode(state + h / 2 * k1,  Ih)
        k3 = ode(state + h / 2 * k2,  Ih)
        k4 = ode(state + h      * k3, I1)

        state = state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        f_out[i + 1] = state[1]
        m_out[i + 1] = state[3]

    return f_out.reshape(-1, 1), m_out.reshape(-1, 1)

@torch.no_grad()
def balloon_rk4_torch(
    f_in: torch.Tensor,
    mt: torch.Tensor,
    dt: float = 0.01,
    y0=(1.0, 1.0),
    tau_MTT: float = 3.0,
    alpha: float = 0.4,
    tau_m: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-step RK4 Balloon ODE in PyTorch. Runs on the same device as f_in."""
    device, dtype = f_in.device, f_in.dtype
    N = f_in.shape[0]
    h = torch.tensor(dt, device=device, dtype=dtype)

    def fout(v, f):
        raw = (tau_MTT * v.clamp(min=1e-8) ** (1 / alpha) + tau_m * f) / (tau_MTT + tau_m)
        return raw.clamp(min=0.0)

    def ode(v, q, f, m):
        v_s = v.clamp(min=1e-8)
        fo = fout(v_s, f)
        dvdt = (f - v_s ** (1 / alpha)) / (tau_MTT + tau_m)
        dqdt = (m - (q / v_s) * fo) / tau_MTT
        return dvdt, dqdt

    v = torch.tensor(float(y0[0]), device=device, dtype=dtype)
    q = torch.tensor(float(y0[1]), device=device, dtype=dtype)

    v_out = torch.empty(N, device=device, dtype=dtype)
    q_out = torch.empty(N, device=device, dtype=dtype)
    v_out[0] = v
    q_out[0] = q

    for i in range(N - 1):
        f0, m0 = f_in[i],   mt[i]
        f1, m1 = f_in[i+1], mt[i+1]
        fh, mh = 0.5 * (f0 + f1), 0.5 * (m0 + m1)

        k1v, k1q = ode(v, q, f0, m0)
        k2v, k2q = ode(v + h/2 * k1v, q + h/2 * k1q, fh, mh)
        k3v, k3q = ode(v + h/2 * k2v, q + h/2 * k2q, fh, mh)
        k4v, k4q = ode(v + h    * k3v, q + h    * k3q, f1, m1)

        v = v + (h / 6) * (k1v + 2*k2v + 2*k3v + k4v)
        q = q + (h / 6) * (k1q + 2*k2q + 2*k3q + k4q)
        v_out[i + 1] = v
        q_out[i + 1] = q

    return v_out.reshape(-1, 1), q_out.reshape(-1, 1)


@njit(cache=True)
def _balloon_full_rk4_njit(
    I, AmpI_f, AmpI_m, kappa, gamma, tau_MTT, alpha, tau_m,
    dt, f0, m0, v0, q0,
):
    N = len(I)
    tau_sum = tau_MTT + tau_m
    inv_a = 1.0 / alpha
    h2 = dt * 0.5
    h6 = dt / 6.0
    state = np.array([0.0, f0, 0.0, m0, v0, q0])
    out = np.empty((N, 4))
    out[0, 0] = state[1]; out[0, 1] = state[3]
    out[0, 2] = state[4]; out[0, 3] = state[5]

    for i in range(N - 1):
        I0 = I[i]; I1 = I[i + 1]; Ih = 0.5 * (I0 + I1)

        def ode(st, I_t):
            sf, f, sm, m, v, q = st[0], st[1], st[2], st[3], st[4], st[5]
            vs = max(v, 1e-8)
            vs_a = vs ** inv_a          # cached — used in both fo and dv/dt
            fo = max((tau_MTT * vs_a + tau_m * f) / tau_sum, 0.0)
            r = np.empty(6)
            r[0] = I_t * AmpI_f - kappa * sf - gamma * (f - 1.0)
            r[1] = sf
            r[2] = I_t * AmpI_m - kappa * sm - gamma * (m - 1.0)
            r[3] = sm
            r[4] = (f - vs_a) / tau_sum
            r[5] = (m - (q / vs) * fo) / tau_MTT
            return r

        k1 = ode(state, I0)
        k2 = ode(state + h2 * k1, Ih)
        k3 = ode(state + h2 * k2, Ih)
        k4 = ode(state + dt * k3, I1)
        state = state + h6 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[i + 1, 0] = state[1]; out[i + 1, 1] = state[3]
        out[i + 1, 2] = state[4]; out[i + 1, 3] = state[5]

    return out


@torch.no_grad()
def balloon_full_rk4_torch(
    I: torch.Tensor,
    dt: float = 0.01,
    AmpI_f=0.2,
    AmpI_m=0.05,
    kappa=1 / 1.54,
    gamma=1 / 2.46,
    tau_MTT: float = 3.0,
    alpha: float = 0.4,
    tau_m: float = 20.0,
    f0: float = 1.0,
    m0: float = 1.0,
    v0: float = 1.0,
    q0: float = 1.0,
) -> torch.Tensor:
    """Full Balloon model solved with fixed-step RK4 in a single pass.

    Integrates the neurovascular coupling (s_f, f, s_m, m) and the Balloon
    haemodynamic ODEs (v, q) simultaneously over a 6-variable state vector,
    replacing the sequential ``neurovascular_rk4_torch`` + ``balloon_rk4_torch``
    calls with one loop of the same length.

    State vector: ``[s_f, f, s_m, m, v, q]``

    .. code-block:: none

        ds_f/dt = I(t)*AmpI_f - kappa*s_f - gamma*(f - 1)
        df/dt   = s_f
        ds_m/dt = I(t)*AmpI_m - kappa*s_m - gamma*(m - 1)
        dm/dt   = s_m
        dv/dt   = (f - v^(1/α)) / (τ_MTT + τ_m)
        dq/dt   = (m - (q/v)·f_out(v,f)) / τ_MTT

    where ``f_out(v, f) = (τ_MTT·v^(1/α) + τ_m·f) / (τ_MTT + τ_m)``.

    Parameters
    ----------
    I : torch.Tensor
        Stimulus, shape ``(N,)`` or ``(N, 1)``. Piecewise-constant;
        midpoint interpolation is used for RK4 intermediate stages.
    dt : float
        Time step in seconds.
    AmpI_f, AmpI_m : float or torch.Tensor
        Stimulus amplitude scalings for CBF and CMRO2.
    kappa : float or torch.Tensor
        Flow-signal decay rate (κ).
    gamma : float or torch.Tensor
        Autoregulatory feedback gain (γ).
    tau_MTT : float or torch.Tensor
        Mean transit time (τ_MTT).
    alpha : float or torch.Tensor
        Grubb's stiffness exponent (α).
    tau_m : float or torch.Tensor
        Viscoelastic time constant (τ_m).
    f0, m0, v0, q0 : float
        Baseline initial conditions (default 1.0; s_f and s_m start at 0).

    Returns
    -------
    torch.Tensor
        Shape ``(N, 4)`` with columns ``[f, m, v, q]``, on the same device as ``I``.
    """
    device, dtype = I.device, I.dtype
    I_np = I.detach().cpu().numpy().ravel().astype(np.float64)

    # Scalar params may arrive as torch.Tensor from ELBO sampling — unwrap.
    def _f(x):
        return x.item() if isinstance(x, torch.Tensor) else float(x)

    fmvq = _balloon_full_rk4_njit(
        I_np,
        _f(AmpI_f), _f(AmpI_m),
        _f(kappa),  _f(gamma),
        _f(tau_MTT), _f(alpha), _f(tau_m),
        float(dt), float(f0), float(m0), float(v0), float(q0),
    )
    return torch.from_numpy(fmvq).to(device=device, dtype=dtype)