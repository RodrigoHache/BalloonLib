"""
Balloon haemodynamic model — numerical reference implementation.

Provides scipy-based ODE solvers for the full Balloon model pipeline:
  stimulus → neurovascular coupling → blood volume / deoxyhemoglobin → BOLD

Used to generate reference trajectories for PINN training and for standalone
exploration in example notebooks.

Changes vs. balloonmodellib.py
-------------------------------
A  ODE integration is now a single solver call over the full time axis,
   with piecewise-constant inputs fed through np.interp.  The original
   step-by-step loop (one odeint call per sample) was O(N) overhead and
   O(N²) in memory due to np.append inside the loop.

B  balloon_ivp merged into balloon_odeint via the ``method`` parameter.
   Pass ``method="LSODA"`` (default, odeint backend) or any solve_ivp
   method string (``"DOP853"``, ``"RK45"``, …).

C  vol_func and q_func removed — dead code; balloon_odeint covers both.

D  _p() helper eliminates the repeated param-extraction boilerplate.

E  ODE parameters are captured in the closure before the inner function is
   defined; f_out is no longer re-called (and re-parsed) at every integrator
   function evaluation inside balloon_odeint.

F  Dead import of Balloon_odeint / NeurovascularCoupling from training.py
   should be removed (not in scope of this file, noted here for reference).

G  cartesian(): unused ``out`` parameter removed.

H  scale_fun(): zero-division guard added for constant arrays.

I  time_segment() is no longer needed after the loop refactor; it has been
   made private (_time_segment) and is only kept for the convolution branch
   of neurovascular_coupling, which still needs a time axis for gamma().
"""

import math

import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.signal import convolve


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _p(params, key, default):
    """Return params[key] if it exists, else default."""
    return params[key] if (params is not None and key in params) else default


def _time_grid(n: int, dt: float) -> np.ndarray:
    """Return n evenly-spaced time points starting at 0 with resolution dt."""
    return np.arange(n, dtype=np.float64) * dt


# ---------------------------------------------------------------------------
# Neural response (inhibition ODE — Buxton 2004 eq. 14)
# ---------------------------------------------------------------------------


def neural_response(
    stimulus: np.ndarray,
    dt: float = 0.01,
    N_0: bool = False,
    scale: bool = True,
    params=None,
):
    """Solve the neural inhibition ODE and return the neural response.

    Parameters
    ----------
    stimulus : np.ndarray
        Array of zeros and ones at resolution ``dt`` (seconds).
    dt : float
        Integration step in seconds.
    N_0 : bool
        If True, the output includes non-zero basal neural activity.
    scale : bool
        If True, normalise the output to [0, 1].
    params : dict, optional
        Override ``k`` (inhibitory gain) and ``tau_i`` (inhibitory time
        constant).

    Returns
    -------
    response : np.ndarray
        Neural response, same length as ``stimulus``.
    time : np.ndarray
        Time axis in seconds, same length as ``stimulus``.
    """
    k = _p(params, "k", 1)
    tau_i = _p(params, "tau_i", 2)
    n_0 = 0.316

    stim = np.asarray(stimulus, dtype=np.float64)
    t = _time_grid(len(stim), dt)
    stim_fn = lambda t_: float(np.interp(float(t_), t, stim))

    def didt(i_arr, t_):
        return [(k * stim_fn(t_) - (k + 1) * float(i_arr[0])) / tau_i]

    # Single solver call over the full time axis — replaces the step loop.
    impulse = odeint(didt, y0=[0.0], t=t)[:, 0]  # shape (N,)

    if not N_0:
        response = stim * (1 - impulse)
    else:
        response = n_0 + stim - impulse
        response[response <= 0.0] = 0.0

    if scale:
        rng = np.max(response) - np.min(response)
        if rng > 0.0:
            response = (response - np.min(response)) / rng

    return response, t


# ---------------------------------------------------------------------------
# Neurovascular coupling (Stephan 2007 differential / Buxton 2004 convolution)
# ---------------------------------------------------------------------------


def neurovascular_coupling(
    stimulus: np.ndarray,
    version: str = "differential",
    params=None,
    dt: float = 0.01,
    mode: str = "full",
    method: str = "direct",
    y0=(1, 0),
    AmpI: float = 0.2,
):
    """Solve the neurovascular coupling equations.

    Parameters
    ----------
    stimulus : np.ndarray
        Array of zeros and ones at resolution ``dt``.
    version : str
        ``"differential"`` (Stephan 2007 / Friston 2000) or
        ``"convolution"`` (Buxton 2004).
    params : dict, optional
        Version-specific parameter overrides (see Notes).
    dt : float
        Integration step in seconds.
    mode, method : str
        Forwarded to ``scipy.signal.convolve`` (convolution version only).
    y0 : tuple
        Initial conditions ``(f0, s0)`` for the differential version.
    AmpI : float
        Stimulus amplitude scaling.  Use 0.2 for CBF and 0.05 for CMRO2.

    Returns
    -------
    Differential version
        ``(fm, s)`` — normalised flow (or CMRO2) and flow-inducing signal,
        each of length ``len(stimulus)``.
    Convolution version
        ``(NVC, h)`` — normalised flow and the gamma impulse response.

    Notes
    -----
    Differential params: ``kappa``, ``gamma``.
    Convolution params: ``tau_f``, ``delta_tf``, ``scale``, ``f1``.
    """
    if version == "convolution":
        tau_f = _p(params, "tau_f", 4)
        delta_tf = _p(params, "delta_tf", 1)
        do_scale = _p(params, "scale", True)
        f1 = _p(params, "f1", 1.009)
        tau_h = 0.242 * tau_f

        Nt, time = neural_response(stimulus, dt, N_0=False, scale=do_scale, params=None)

        def gamma_fn(tau, t):
            k = 3
            return (1 / (tau * math.factorial(k))) * (t / tau) ** k * np.exp(-(t / tau))

        h = gamma_fn(tau_h, time - delta_tf)
        h[h <= 1e-4] = 0.0

        NVC = 1 + (f1 - 1) * convolve(Nt, h, mode=mode, method=method)
        return NVC, h

    elif version == "differential":
        k = _p(params, "kappa", 1 / 1.54)
        g = _p(params, "gamma", 1 / 2.46)

        Nt = np.asarray(stimulus, dtype=np.float64) * AmpI
        t = _time_grid(len(Nt), dt)
        Nt_fn = lambda t_: float(np.interp(float(t_), t, Nt))

        # State: [s, fm].  y0 convention matches original: (f0, s0).
        def dNC_dt(NC, t_):
            s, fm = float(NC[0]), float(NC[1])
            nt = Nt_fn(t_)
            return [nt - k * s - g * (fm - 1), s]

        sol = odeint(dNC_dt, y0=[y0[1], y0[0]], t=t)  # shape (N, 2)
        return sol[:, 1].astype(np.float32), sol[:, 0].astype(np.float32)  # fm, s

    else:
        raise ValueError(f"Unknown version {version!r}. Use 'differential' or 'convolution'.")


def neurovascular_coupling_fm(
    stimulus: np.ndarray,
    params=None,
    dt: float = 0.01,
    AmpI_f: float = 0.2,
    AmpI_m: float = 0.05,
    y0_f=(1, 0),
    y0_m=(1, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the neurovascular coupling for CBF and CMRO2 in a single ODE call.

    Combines the two separate ``neurovascular_coupling`` calls (one for f,
    one for m) into one ``odeint`` over the 4D state ``[s_f, f, s_m, m]``.
    Both signals share κ and γ; only the stimulus amplitude differs.

    Parameters
    ----------
    stimulus : np.ndarray
        Array of zeros and ones at resolution ``dt``.
    params : dict, optional
        Override ``kappa`` and ``gamma``.
    dt : float
        Integration step in seconds.
    AmpI_f : float
        Stimulus amplitude scaling for CBF (default 0.2).
    AmpI_m : float
        Stimulus amplitude scaling for CMRO2 (default 0.05).
    y0_f, y0_m : tuple
        Initial conditions ``(f0, s0)`` for each signal (default ``(1, 0)``).

    Returns
    -------
    f : np.ndarray
        Normalised CBF, length ``len(stimulus)``.
    m : np.ndarray
        Normalised CMRO2, length ``len(stimulus)``.
    """
    k = _p(params, "kappa", 1 / 1.54)
    g = _p(params, "gamma", 1 / 2.46)

    stim = np.asarray(stimulus, dtype=np.float64)
    t = _time_grid(len(stim), dt)
    stim_fn = lambda t_: float(np.interp(float(t_), t, stim))

    # state: [s_f, f, s_m, m]
    def ddt(state, t_):
        s_f, f, s_m, m = state
        u = stim_fn(t_)
        return [
            u * AmpI_f - k * s_f - g * (f - 1),
            s_f,
            u * AmpI_m - k * s_m - g * (m - 1),
            s_m,
        ]

    ic = [float(y0_f[1]), float(y0_f[0]), float(y0_m[1]), float(y0_m[0])]
    sol = odeint(ddt, y0=ic, t=t)  # (N, 4)
    return sol[:, 1].astype(np.float32), sol[:, 3].astype(np.float32)


# ---------------------------------------------------------------------------
# Simple array utilities
# ---------------------------------------------------------------------------


def array_extend(arr: np.ndarray, dt: float) -> np.ndarray:
    """Upsample a stimulus array from 1 sample/s to 1/dt samples/s.

    Parameters
    ----------
    arr : np.ndarray
        Stimulus of zeros and ones (one element per second).
    dt : float
        Target resolution in seconds (e.g. dt=0.5 maps [0,1] → [0,0,1,1]).

    Returns
    -------
    new_arr : np.ndarray
    """
    n = int(np.ceil(1 / dt))
    return np.repeat(np.asarray(arr, dtype=np.float32), n)


def scale_fun(arr: np.ndarray, factor: float) -> np.ndarray:
    """Rescale ``arr`` so its range spans ``[min(arr), min(arr) + factor]``.

    Returns a copy of ``arr`` unchanged when it is constant (zero range).
    """
    lo, hi = np.min(arr), np.max(arr)
    rng = hi - lo
    if rng == 0.0:
        return arr.copy()
    return lo + (factor / rng) * (arr - lo)


# ---------------------------------------------------------------------------
# Oxygen / CMRO2 helpers
# ---------------------------------------------------------------------------


def efun(f_in: np.ndarray, E0: float = 0.32) -> np.ndarray:
    """Compute oxygen extraction fraction E (Friston 2000, Buxton 1998).

    Parameters
    ----------
    f_in : np.ndarray
        Normalised inflow.
    E0 : float
        Baseline oxygen extraction fraction.

    Returns
    -------
    E : np.ndarray
    """
    return 1 - (1 - E0) ** (1 / f_in)


def m_t_E(f_in: np.ndarray, E0: float = 0.32) -> np.ndarray:
    """Compute normalised CMRO2 (Buxton 2004 eq. 2, Friston 2000).

    Parameters
    ----------
    f_in : np.ndarray
        Normalised inflow.
    E0 : float
        Baseline oxygen extraction fraction.

    Returns
    -------
    mE : np.ndarray
    """
    return f_in * (efun(f_in, E0) / E0)


# ---------------------------------------------------------------------------
# Venous outflow (closed-form, fully vectorised)
# ---------------------------------------------------------------------------


def f_out(vol: np.ndarray, f_in: np.ndarray, viscoelastic: bool = False, params=None):
    """Compute venous outflow (Buxton 2004 eq. 11).

    Parameters
    ----------
    vol : np.ndarray
        Blood volume time series.
    f_in : np.ndarray
        Normalised inflow.
    viscoelastic : bool
        Include the viscoelastic correction (tau_m > 0).
    params : dict, optional
        Override ``tau_MTT``, ``alpha``, ``tau_m``.

    Returns
    -------
    fout : np.ndarray
    """
    tau_MTT = _p(params, "tau_MTT", 3.0)
    alpha = _p(params, "alpha", 0.4)
    tau_m = _p(params, "tau_m", 10)
    taum = tau_m if viscoelastic else 0

    fout = (tau_MTT * vol ** (1 / alpha) + taum * f_in) / (tau_MTT + taum)
    return np.maximum(fout, 0.0)


# ---------------------------------------------------------------------------
# Balloon ODE solver — unified (replaces balloon_odeint + balloon_ivp)
# ---------------------------------------------------------------------------


def balloon_odeint(
    f_in: np.ndarray,
    mt: np.ndarray,
    params=None,
    dt: float = 0.01,
    y0=(1, 1),
    viscoelastic: bool = False,
    method: str = "LSODA",
    rtol: float = 1.49e-8,
    atol: float = 1.49e-8,
):
    """Solve the Balloon model ODE system (Buxton 2004 eqs. 10–11).

    Parameters
    ----------
    f_in : np.ndarray
        Normalised inflow (eq. 13).
    mt : np.ndarray
        Normalised CMRO2 time series.
    params : dict, optional
        Override ``tau_MTT``, ``alpha``, ``tau_m``.
    dt : float
        Time resolution of the input arrays in seconds.
    y0 : tuple
        Initial conditions ``(v0, q0)``.
    viscoelastic : bool
        Include the viscoelastic outflow correction.
    method : str
        ODE solver backend.  ``"LSODA"`` (default) uses
        ``scipy.integrate.odeint``; any other string (``"DOP853"``,
        ``"RK45"``, ``"Radau"``, …) is forwarded to
        ``scipy.integrate.solve_ivp``.  To match the behaviour of the
        former ``balloon_ivp`` function, pass ``method="DOP853"`` with
        ``rtol=1e-9, atol=1e-9``.
    rtol, atol : float
        Relative and absolute solver tolerances.

    Returns
    -------
    v : np.ndarray
        Blood volume time series, length ``len(f_in)``.
    q : np.ndarray
        Deoxyhemoglobin time series, length ``len(f_in)``.
    """
    # Extract params once — not inside the ODE closure.
    tau_MTT = _p(params, "tau_MTT", 3.0)
    alpha = _p(params, "alpha", 0.4)
    tau_m = _p(params, "tau_m", 10)
    taum = tau_m if viscoelastic else 0.0

    f_arr = np.asarray(f_in, dtype=np.float64)
    m_arr = np.asarray(mt, dtype=np.float64)
    t = _time_grid(len(f_arr), dt)
    f_fn = lambda t_: float(np.interp(float(t_), t, f_arr))
    m_fn = lambda t_: float(np.interp(float(t_), t, m_arr))

    # Inline outflow — avoids re-parsing params at every integrator step.
    def _fout(v, f):
        v_safe = max(v, 1e-8)
        raw = (tau_MTT * v_safe ** (1 / alpha) + taum * f) / (tau_MTT + taum)
        return max(raw, 0.0)

    def dB_dt(B, t_):
        v, q = float(B[0]), float(B[1])
        f = f_fn(t_)
        m = m_fn(t_)
        v_safe = max(v, 1e-8)
        fout = _fout(v_safe, f)
        return [
            (f - v_safe ** (1 / alpha)) / (tau_MTT + taum),
            (m - (q / v_safe) * fout) / tau_MTT,
        ]

    if method == "LSODA":
        sol = odeint(dB_dt, y0=list(y0), t=t, rtol=rtol, atol=atol)
    else:
        def dB_dt_ivp(t_, B):
            return dB_dt(B, t_)

        result = solve_ivp(
            dB_dt_ivp,
            t_span=(t[0], t[-1]),
            y0=list(y0),
            method=method,
            t_eval=t,
            dense_output=False,
            vectorized=False,
            rtol=rtol,
            atol=atol,
        )
        sol = result.y.T  # shape (N, 2)

    return sol[:, 0].astype(np.float32), sol[:, 1].astype(np.float32)


# ---------------------------------------------------------------------------
# Cartesian product helper (out parameter removed — it was never used)
# ---------------------------------------------------------------------------


def cartesian(arrays) -> np.ndarray:
    """Return the Cartesian product of the input 1-D arrays.

    Parameters
    ----------
    arrays : sequence of array-like
        1-D arrays to combine.

    Returns
    -------
    out : np.ndarray
        Shape ``(prod(len(a) for a in arrays), len(arrays))``.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           ...
           [3, 5, 7]])
    """
    mesh = np.meshgrid(*arrays)
    return np.concatenate([x.reshape(-1, 1) for x in mesh], axis=1)


# ---------------------------------------------------------------------------
# BOLD signal models
# ---------------------------------------------------------------------------


def bold_func(vt: np.ndarray, qt: np.ndarray, params=None, BM: str = "classic") -> np.ndarray:
    """Compute BOLD signal from volume and deoxyhemoglobin (Stephan 2007).

    Parameters
    ----------
    vt : np.ndarray
        Blood volume time series.
    qt : np.ndarray
        Deoxyhemoglobin time series.
    params : dict, optional
        Override ``E_0``, ``V_0``, ``TE``, ``O_0``, ``r_0``, ``epsilon``.
    BM : str
        ``"classic"`` or ``"revised"`` balloon model coefficients
        (Obata 2004 / Buxton 2000).

    Returns
    -------
    bold : np.ndarray
    """
    E_0 = _p(params, "E_0", 0.32)
    V_0 = _p(params, "V_0", 0.03)
    TE = _p(params, "TE", 0.04)
    eps = _p(params, "epsilon", 1.43)
    r_0 = _p(params, "r_0", 25)
    omega_0 = _p(params, "O_0", 40.3)

    if BM == "classic":
        k_1 = (1 - V_0) * 4.3 * omega_0 * E_0 * TE
        k_2 = 2 * E_0
    elif BM == "revised":
        k_1 = 4.3 * omega_0 * E_0 * TE
        k_2 = eps * r_0 * E_0 * TE
    else:
        raise ValueError(f"Unknown BM {BM!r}. Use 'classic' or 'revised'.")

    k_3 = 1.0 - eps
    return V_0 * (k_1 * (1.0 - qt) + k_2 * (1.0 - qt / vt) + k_3 * (1.0 - vt))


def bold_davis(f: np.ndarray, m: np.ndarray, author: str = "Davis1998") -> np.ndarray:
    """Compute BOLD signal using the Davis model.

    Parameters
    ----------
    f : np.ndarray
        Normalised cerebral blood flow.
    m : np.ndarray
        Normalised CMRO2.
    author : str
        Parameter set: ``"Davis1998"`` (A=0.075, α=0.4, β=1.5) or
        ``"Maith2022"`` (A=140.9, α=0.14, β=0.91).

    Returns
    -------
    bold : np.ndarray
        ``A * (1 - f^(alpha - beta) * m^beta)``
    """
    if author == "Davis1998":
        A, alpha, beta = 0.075, 0.4, 1.5
    elif author == "Maith2022":
        A, alpha, beta = 140.9, 0.14, 0.91
    else:
        raise ValueError(f"Unknown author {author!r}. Use 'Davis1998' or 'Maith2022'.")

    return A * (1 - f ** (alpha - beta) * m**beta)


# ---------------------------------------------------------------------------
# Backward-compatible PascalCase aliases
# ---------------------------------------------------------------------------
NeurovascularCoupling = neurovascular_coupling
Efun = efun
Balloon_odeint = balloon_odeint
BOLD_func = bold_func
BOLD_Davis = bold_davis


def Balloon_ivp(f, m, params=None, y0=(1, 1), viscoelastic=False, method="DOP853"):
    """Backward-compatible alias for balloon_odeint with solve_ivp backend.

    Use ``balloon_odeint(method="DOP853", rtol=1e-9, atol=1e-9)`` directly.
    """
    return balloon_odeint(
        f, m, params=params, y0=y0, viscoelastic=viscoelastic,
        method=method, rtol=1e-9, atol=1e-9,
    )
