import math

import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.signal import convolve


def neural_response(
    stimulus: np.ndarray,
    dt: np.float32 = 0.01,
    N_0: bool = False,
    scale: bool = True,
    params=None,
):
    """
    neural_response

    Solves the inhibition equation and returns the neural response according to the equation system eq.14.

    **Inputs:**

        - ``stimulus``: An array of zeros and ones called stimulus it already must have the time resolution included.
        - ``dt``: np.float32, equivalent to the integration step, which defaults to one hundredth of a second.
        - ``N_0``: bool, default is False, when True, the function's output has a basal condition different from zero.
        - ``scale``: bool, determines if the output is scaled, so the output would have a maximum of 1, independent of N_0 and param.
        - ``param``: dict, containing the parameters ``[k=1, tau_I=2]`` from equation 14 according to buxton2004.

    These values can be found in ``buxton2004modeling``.

    **Outputs:**

        - ``response``: array, neural response
        - ``time``: array, the time over which the response takes place

    """
    k = 1  # "Inhibitory gain factor" )
    tau_i = 2  # "Inhibitory time constant" )
    n_0 = 0.316  # "Basal Neural Activity")

    if params is not None:
        if "k" in params:
            k = params["k"]
        if "tau_i" in params:
            tau_i = params["tau_i"]

    def didt(t, i, s):
        return (k * s - (k + 1) * i) / tau_i

    time = time_segment(stimulus, dt=dt)
    impulse = np.zeros(1, dtype=np.float32)
    for ts in zip(time, stimulus):
        y = odeint(didt, y0=impulse[-1], t=ts[0], args=(ts[1],), tfirst=True)
        impulse = np.append(impulse, [y.T[0][1]])

    if not N_0:
        response = stimulus * (1 - impulse)
    else:
        response = n_0 + stimulus - impulse
        response[response <= 0.0] = 0.0

    if scale:
        response = (response - np.min(response)) / (np.max(response) - np.min(response))

    return response, np.append(time[:, 0], time[-1, 1])


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
    """
    NeurovascularCoupling

    Solves the neurovascular coupling equations diferencital or convlolutional

    **Input:**

        - ``stimulus``= np.ndarray A list of zeros and ones called 'stimulus', where each digit corresponds to one second.
        - ``version`` = str 'differential' for Maith 2022 or 'convolution' for Buxton 2004.
        - ``dt`` = np.float6 equivalent to the integration time step, which by default is one-hundredth of a second.
        - ``params``: dict, 3 parameters whose name depend on whether we use "differential" version or "convolution".

    - if version == 'differential'

            This function calculates the neurovascular coupling according to Stephen 2007 (therefore a modification of Friston 2000)
            where s is some ﬂow inducing signal deﬁned, operationally, in units corresponding to the rate of change of normalised ﬂow
        (i.e., s^{-1}) and the stimulus*0.2 corresponds with the Impulse I_{CBF}, and  stimulus*0.05 with I_{CMRO2}

    **Inputs:**

        - ``stimulus``= np.ndarray, A list of zeros and ones called "stimulus", where each digit corresponds to one second.
        - ``params`` = dict, kappa and gamma acoording to Maith 2022, or tau_s and tau_f according to Friston 2000
        - ``dt`` = np.float32(0.01) equivalent to the integration time step, which by default is one-hundredth of a second.
        - ``y0`` =(1,0) Initual contiditions for (f,s)
        - ``AmpI``= np.float32, Scales Stimulus to be used for f_{in}(t) [app = 0.2] or m(t)[app = 0.05]

    **Outputs:**

        - ``fm``= np.ndarray this could be either m or f depending on AmpI.
        - ``s``= np.ndarray  some ﬂow inducing signal deﬁned, operationally.

    - elif version == 'convolution'

            Solves the neurovascular coupling equations described for the normalized baseline cerebral blood flow (CBF) 'f_in'.
        We then assume that both CBF and CMRO2 are linear convolutions of an impulse response function h(t) with the appropriate
        measurement of neuronal activity N(t). According to Buxton 2004. The 'h(t)' shape is then scaled to provide the desired
        amplitude and duration of the impulse response. For this shape and a desired FWHM of tau_f, the time constant in Eq. (12)
        is given by the empirical expression tau_h = 0.242 * tau_f.

    **Input:**

        - ``params``: dict, 3 parameters {'tau_h':4, 'delta_tf':1, 'f_1':1.009].

    The values for tau and delta were extracted from Buxton 2004, while the value for f_1=1.009 was obtained heuristically, such
    that max(f_in(t)) approx 1.5 (note that if f_1=1.00 then f_in(t)=0).

    mode and method are parameters inherited from the scipy.signal.convolve function, used for the convolution between N(t) and h(t),
    with options including:

        - ``mode``: str {'full', 'valid', 'same'}, optional. A string indicating the size of the output:

            * 'full': The output is the full discrete linear convolution of the inputs. (Default)
            * 'valid': The output consists only of those elements that do not depend on the padding with zeros.
            * 'same': The output is the same size as the first signal, in our case N(t), centered with respect to the 'full' output.

        - ``method``: str {'auto', 'direct', 'fft'}, optional. A string indicating which method to use for convolution.

            * 'direct': Convolution is determined directly from sums, the convolution definition.
            * 'fft': Fourier transform is used to perform convolution by calling fftconvolved.
            * 'auto': Automatically chooses between direct or Fourier method based on an estimation of which is faster (default).

    **Outputs:**

        - ``h(t)``: A plausible form for h(t) is a gamma-variant function with a full-width at half-maximum (FWHM) of approximately
         4 s, corresponding to equation 12.
        - ``f_in(t)``: corresponds to equation 13.
        - ``m(t)``: corresponds to equation 13.
    """
    if version == "convolution":
        tau_f = 4
        delta_tf = 1
        scale = True
        f1 = 1.009

        if params is not None:
            if "tau_f" in params:
                tau_f = params["tau_f"]
            if "delta_tf" in params:
                delta_tf = params["delta_tf"]
            if "scale" in params:
                scale = params["scale"]
            if "f1" in params:
                f1 = params["f1"]

        tau_h = 0.242 * tau_f
        Nt, time = neural_response(stimulus, dt, N_0=False, scale=scale, params=None)

        def gamma(tau_h, t):
            k = 3
            return (1 / (tau_h * math.factorial(k))) * ((t / tau_h) ** k) * np.exp(-(t / tau_h))

        h = gamma(tau_h, time - delta_tf)
        h[h <= 1e-4] = 0.0

        NVC = 1 + (f1 - 1) * convolve(Nt, h, mode=mode, method=method)

        return NVC, h

    elif version == "differential":
        k = 1 / 1.54
        g = 1 / 2.46

        if params is not None:
            if "kappa" in params:
                k = params["kappa"]
            if "gamma" in params:
                g = params["gamma"]

        Nt = stimulus * AmpI

        def dNC_dt(t, NC, Nt):
            s, fm = NC
            return [Nt - k * s - g * (fm - 1), s]

        time = time_segment(Nt, dt=dt)
        s = np.ones(1, dtype=np.float32) * y0[1]
        fm = np.ones(1, dtype=np.float32) * y0[0]

        for tfm in zip(time, Nt):
            sol = odeint(dNC_dt, y0=(s[-1], fm[-1]), t=tfm[0], args=(tfm[1],), tfirst=True)
            s = np.append(s, [sol.T[0, 1]])
            fm = np.append(fm, [sol.T[1, 1]])

        return fm, s

    else:
        raise ValueError(f"Unknown version {version!r}. Use 'differential' or 'convolution'.")


def array_extend(arr: np.ndarray, dt: float) -> np.ndarray:
    """Upsample a stimulus array from 1 sample/s to 1/dt samples/s.

    Parameters
    ----------
    arr : np.ndarray
        Stimulus of zeros and ones (one element per second).
    dt : float
        Target resolution in seconds. e.g. dt=0.5 maps [0,1] → [0,0,1,1].

    Returns
    -------
    new_arr : np.ndarray
    """
    n = int(np.ceil(1 / dt))
    return np.repeat(np.asarray(arr, dtype=np.float32), n)


def scale_fun(arr: np.ndarray, factor: float) -> np.ndarray:
    """Rescale ``arr`` so its range spans ``[min(arr), factor]``."""
    return np.min(arr) + (factor / (np.max(arr) - np.min(arr))) * (arr - np.min(arr))


def efun(f_in: np.ndarray, E0: float = 0.32) -> np.ndarray:
    """Compute oxygen extraction fraction E (Friston 2000, Buxton 1998).

    Parameters
    ----------
    f_in : np.ndarray
        Normalised inflow (eq. 13).
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
        Normalised inflow (eq. 13).
    E0 : float
        Baseline oxygen extraction fraction.

    Returns
    -------
    mE : np.ndarray
        Normalised CMRO2 relative to baseline.
    """
    return f_in * (efun(f_in, E0) / E0)


# @njit
def vol_func(
    f_in: np.ndarray,
    params: np.ndarray,
    vol0: float = 1,
    dt: float = 0.01,
    viscoelastic: bool = False,
):
    """
    **vol_func**

    vol_func gives the solution for the differential equation for volume, according to a combination of equations
    10 and 11 on Buxton`s 2004 paper

    **Inputs:**

        - ``f_in``:  np.ndarray, corresponds to equation 13, i.e. the income flow.
        - ``params``: dict, includes the constants from equations 10 and 11, which are {tau_MTT = 3.0, alpha = 0.4,
        tau_m in [0, 30]}.
        - ``dt``: float, dt refers to the integration step.
        - ``viscoelastic``: bool, determines whether the output accounts for the viscoelastic effect, i.e.
        tau = 0 or 0 < tau <= 30.

    **Outputs:**

        - ``v(t)``: np.ndarray, time series of the volume.
        - ``time``: np.ndarray, time series in which v(t) transpires.

    """

    tau_MTT = 3.0
    alpha = 0.4
    tau_m = 10

    if params is not None:
        if "tau_MTT" in params:
            tau_MTT = params["tau_MTT"]
        if "alpha" in params:
            alpha = params["alpha"]
        if "tau_m" in params:
            tau_m = params["tau_m"]

    taum = tau_m if viscoelastic else 0

    def dvdt(t, v, f) -> float:
        return (f - v ** (1 / alpha)) / (tau_MTT + taum)

    v = np.empty(0, dtype=np.float32)
    time = time_segment(f_in, dt=dt)

    for ts in zip(time, f_in):
        y0 = vol0 if 0 in ts[0] else v[-1]
        y = odeint(dvdt, y0=y0, t=ts[0], args=(ts[1],), tfirst=True)
        v = np.append(v, [y.T[0][1]])

    return v, time[:, 0]


# @njit
def f_out(vol: np.ndarray, f_in: np.ndarray, viscoelastic: bool = False, params=None):
    """
    **f_out**

        Integral to the balloon model, the outflow is obtained using our 'f_out' function by solving the rate of
        change of volume in the outflow equation, i.e., equation 11 in the Buxton 2004 article.

    **Inputs:**

        - ``vol``: np.ndarray, time series of volume.
        - ``f_in``: np.ndarray, corresponds to equation 13, i.e. the inflow.
        - ``viscoelastic``: bool, determines whether the outflow accounts for the viscoelastic effect, i.e.
         tau = 0 or 0 < tau <= 30.
        - ``params``: dict, includes the constants from equations 10 and 11, which are
        {tau_MTT = 3.0, alpha = 0.4, tau_m in [0, 30]}.

    **Outputs:**

        - ``fout``: np.ndarray, time series of the outflow.

    """

    tau_MTT = 3.0
    alpha = 0.4
    tau_m = 10

    if params is not None:
        if "tau_MTT" in params:
            tau_MTT = params["tau_MTT"]
        if "alpha" in params:
            alpha = params["alpha"]
        if "tau_m" in params:
            tau_m = params["tau_m"]

    taum = tau_m if viscoelastic else 0

    fout = (tau_MTT * vol ** (1 / alpha) + taum * f_in) / (tau_MTT + taum)
    return np.maximum(fout, 0.0)


def time_segment(time: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """Generate consecutive time intervals of length ``dt`` covering ``len(time)`` steps.

    Parameters
    ----------
    time : np.ndarray
        Array whose length determines the number of intervals.
    dt : float
        Interval length in seconds.

    Returns
    -------
    new_time : np.ndarray
        Shape ``(len(time) - 1, 2)`` — each row is ``[t_start, t_end]``.
    """
    t = np.arange(len(time), dtype=np.float32) * dt
    return np.column_stack([t[:-1], t[1:]])


def q_func(vol: np.ndarray, mt: np.ndarray, f_out: np.ndarray, params=None, dt: float = 0.01):
    """Solve the deoxyhemoglobin ODE (Buxton 2004 eqs. 10–11).

    Parameters
    ----------
    vol : np.ndarray
        Blood volume time series.
    mt : np.ndarray
        Normalised CMRO2 time series.
    f_out : np.ndarray
        Venous outflow time series.
    params : dict, optional
        Override ``tau_MTT`` (venous time constant).
    dt : float
        Integration step in seconds.

    Returns
    -------
    q : np.ndarray
    time : np.ndarray
    """
    tau_MTT = 3.0

    if params is not None:
        if "tau_MTT" in params:
            tau_MTT = params["tau_MTT"]

    def dqdt(t, q, V, CMRO, fout) -> float:
        return (CMRO - (q / V) * fout) / tau_MTT

    q = np.ones(1, dtype=np.float32)
    time = time_segment(vol, dt=dt)

    for ts in zip(time, vol, mt, f_out):
        y = odeint(dqdt, y0=q[-1], t=ts[0], args=(ts[1], ts[2], ts[3]), tfirst=True)
        q = np.append(q, [y.T[0][1]])

    return q, np.append(time[:, 0], time[-1, 1])


def balloon_odeint(
    f_in: np.ndarray,
    mt: np.ndarray,
    params=None,
    dt: float = 0.01,
    y0=(1, 1),
    viscoelastic: bool = False,
):
    """Solve the Balloon model ODE system with odeint (Buxton 2004 eqs. 10–11).

    Parameters
    ----------
    f_in : np.ndarray
        Normalised inflow (eq. 13).
    mt : np.ndarray
        Normalised CMRO2 time series.
    params : dict, optional
        Override defaults: ``tau_MTT``, ``alpha``, ``tau_m``.
    dt : float
        Integration step in seconds.
    y0 : tuple
        Initial conditions ``(v0, q0)``.
    viscoelastic : bool
        If True, include viscoelastic outflow.

    Returns
    -------
    v : np.ndarray
        Blood volume time series.
    q : np.ndarray
        Deoxyhemoglobin time series.
    """
    tau_MTT = 3.0
    alpha = 0.4
    tau_m = 10

    if params is not None:
        if "tau_MTT" in params:
            tau_MTT = params["tau_MTT"]
        if "alpha" in params:
            alpha = params["alpha"]
        if "tau_m" in params:
            tau_m = params["tau_m"]

    taum = tau_m if viscoelastic else 0

    def dB_dt(t, B, f, m):
        v, q = B
        fout = f_out(v, f, viscoelastic=viscoelastic, params=params)
        return [
            (f - v ** (1 / alpha)) / (tau_MTT + taum),
            (m - (q / v) * fout) / tau_MTT,
        ]

    time = time_segment(f_in, dt=dt)
    v = np.ones(1, dtype=np.float32) * y0[0]
    q = np.ones(1, dtype=np.float32) * y0[1]

    for tfm in zip(time, f_in, mt):
        sol = odeint(dB_dt, y0=(v[-1], q[-1]), t=tfm[0], args=(tfm[1], tfm[2]), tfirst=True)
        v = np.append(v, sol[1][0])
        q = np.append(q, sol[1][1])

    return v, q


def balloon_ivp(
    f: np.ndarray,
    m: np.ndarray,
    params=None,
    y0=(1, 1),
    viscoelastic: bool = False,
    method: str = "DOP853",
):
    """Solve the Balloon model ODE system with solve_ivp (Buxton 2004 eqs. 10–11).

    Parameters
    ----------
    f : np.ndarray
        Normalised inflow.
    m : np.ndarray
        Normalised CMRO2.
    params : dict, optional
        Override defaults: ``tau_MTT``, ``alpha``, ``tau_m``.
    y0 : tuple
        Initial conditions ``(v0, q0)``.
    viscoelastic : bool
        If True, include viscoelastic outflow.
    method : str
        ODE solver method passed to ``solve_ivp`` (default ``"DOP853"``).

    Returns
    -------
    v : np.ndarray
        Blood volume time series.
    q : np.ndarray
        Deoxyhemoglobin time series.
    """
    tau_MTT = 3.0
    alpha = 0.4
    tau_m = 10

    if params is not None:
        if "tau_MTT" in params:
            tau_MTT = params["tau_MTT"]
        if "alpha" in params:
            alpha = params["alpha"]
        if "tau_m" in params:
            tau_m = params["tau_m"]

    taum = tau_m if viscoelastic else 0
    time = time_segment(f, dt=0.01)

    def dB_dt(t, B, f, m):
        v, q = B
        fout = f_out(v, f, viscoelastic=viscoelastic, params=params)
        return [
            (f - v ** (1 / alpha)) / (tau_MTT + taum),
            (m - (q / v) * fout) / tau_MTT,
        ]

    v = np.ones(1, dtype=np.float32) * y0[0]
    q = np.ones(1, dtype=np.float32) * y0[1]

    for tfm in zip(time, f, m):
        sol_ivp = solve_ivp(
            dB_dt,
            t_span=tfm[0],
            y0=(v[-1], q[-1]),
            method=method,
            t_eval=[tfm[0][1]],
            dense_output=False,
            vectorized=False,
            rtol=1e-9,
            atol=1e-9,
            args=(tfm[1], tfm[2]),
        )
        v = np.append(v, sol_ivp.y[0])
        q = np.append(q, sol_ivp.y[1])

    return v, q


# @njit
def cartesian(arrays, out=None):
    """
    **Cartesian**

        Generates the Cartesian product between the input arrays.

    **Inputs:**

        - ``arrays``: tuple of array-like, 1-D arrays among which to compute the Cartesian product.

    **Output:**

        - ``out``: np.ndarray, a 2-D array with a format/shape of (M, len(arrays)) containing the Cartesian product
        formed from the inputs.

    **Example:**

    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
        array([[1, 4, 6],
            [1, 4, 7],
            [1, 5, 6],
            [1, 5, 7],
            [2, 4, 6],
            [2, 4, 7],
            [2, 5, 6],
            [2, 5, 7],
            [3, 4, 6],
            [3, 4, 7],
            [3, 5, 6],
            [3, 5, 7]])

    """
    mesh = np.meshgrid(*arrays)
    # Concatenar los resultados en una matriz de
    result = np.concatenate([x.reshape(-1, 1) for x in mesh], axis=1)
    return result


def bold_func(vt: np.ndarray, qt: np.ndarray, params=None, BM: str = "classic") -> np.ndarray:
    """Compute BOLD signal from volume and deoxyhemoglobin (Stephan 2007).

    Parameters
    ----------
    vt : np.ndarray
        Blood volume time series.
    qt : np.ndarray
        Deoxyhemoglobin time series.
    params : dict, optional
        Override defaults: ``E_0``, ``V_0``, ``TE``, ``O_0``, ``r_0``, ``epsilon``.
    BM : str
        ``"classic"`` or ``"revised"`` balloon model coefficients (Obata 2004 / Buxton 2000).

    Returns
    -------
    bold : np.ndarray
    """
    E_0 = 0.32
    V_0 = 0.03
    TE = 0.04
    eps = 1.43
    r_0 = 25
    omega_0 = 40.3  # frequency offset at vessel surface for fully deoxygenated blood at 1.5T

    if params is not None:
        if "E_0" in params:
            E_0 = params["E_0"]
        if "V_0" in params:
            V_0 = params["V_0"]
        if "TE" in params:
            TE = params["TE"]
        if "O_0" in params:
            omega_0 = params["O_0"]
        if "r_0" in params:
            r_0 = params["r_0"]
        if "epsilon" in params:
            eps = params["epsilon"]

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


# Backward-compatible PascalCase aliases
NeurovascularCoupling = neurovascular_coupling
Efun = efun
Balloon_odeint = balloon_odeint
Balloon_ivp = balloon_ivp
BOLD_func = bold_func
BOLD_Davis = bold_davis
