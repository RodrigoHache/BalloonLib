# %%
import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.signal import convolve

# from numba import jit, njit, cfunc
# from random import choices as rch


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

    if params != None:
        if "k" in params:
            k = params["k"]
        if "tau_i" in params:
            tau_i = params["tau_i"]

    # stimulus_array= np.asarray(stimulus, dtype=np.float32)
    ext_stim = stimulus  # array_extend(stimulus_array, dt=dt)

    # ext_stim[100] = 1 ###### ESTO ES SOLO PARA PROBAR EL DELTA EN LA VERSIÖN FINAL
    # @njit
    def didt(t, i, s):
        # retornamos lo que sale de ((k*(s-i)-i)/taui)
        return (k * s - (k + 1) * i) / tau_i

    time = time_segment(ext_stim, dt=dt)

    def solver(t, stim):
        I = np.zeros(1, dtype=np.float32)  # preallocate the output
        y = np.zeros(1, dtype=np.float32)
        for ts in zip(t, stim):
            y = odeint(didt, y0=I[-1], t=ts[0], args=(ts[1],), tfirst=True)
            I = np.append(I, [y.T[0][1]])
        return I

    I = solver(time, ext_stim)

    if N_0 == False:
        response = ext_stim * (
            1 - I
        )  # Im using ext_stim like indicator function otherwise we would have a rebound
    else:
        response = n_0.random() + ext_stim - I
        response[np.where(response <= 0.0)] = 0.0

    if scale:
        response = (response - np.min(response)) / (np.max(response) - np.min(response))

    return response, np.append(time[:, 0], time[-1, 1])


def NeurovascularCoupling(
    stimulus: np.ndarray,
    version: str = "differential",
    params=None,
    dt: np.float32 = 0.01,
    mode="full",
    method="direct",
    y0=(1, 0),
    AmpI: np.float32 = 0.2,
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

        if params != None:
            if "tau_f" in params:
                tau_f = params["tau_f"]  # "Width of CBF impulse response"
            if "delta_tf" in params:
                # "the delay after the start of the stimulus before the CBF response begins"
                delta_tf = params["delta_tf"]
            if "scale" in params:
                # "represents the normalized flow increase on the plateau of the CBF response to a sustained neural 
                # activity with unit amplitude"
                scale = params["scale"]
            if "f1" in params:
                f1 = params["f1"]
        else:
            tau_f = 4  # "Width of CBF impulse response"
            delta_tf = 1  # "the delay after the start of the stimulus before the CBF response begins"
            # "represents the normalized flow increase on the plateau of the CBF response to a sustained neural activity 
            # with unit amplitude"
            scale = True

        TOLERANCE = 1.0e-04

        tau_h = 0.242 * tau_f
        # ``Nt``: array, obtained neuronal response.
        # ``time``: array,internally used to implement a gamma function
        # using time (plus a delay) as its domain.the time over which this unfolds.
        Nt, time = neural_response(stimulus, dt, N_0=False, scale=True, params=None)

        def gamma(tau_h, t):
            k = 3  # "Nameless constant, revisit Buxton 2004 eq 12" )
            return (
                (1 / (tau_h * np.math.factorial(k)))
                * ((t / tau_h) ** k)
                * np.exp(-(t / tau_h))
            )

        # h corresponds to h(t) the impulse function gamma
        h = gamma(tau_h, (time - delta_tf))
        # fix for convergence
        h[np.where(h <= TOLERANCE)] = 0.0

        NVC = 1 + (f1 - 1) * convolve(Nt, h, mode=mode, method=method)

        return NVC, h

    elif version == "differential":

        k = 1 / 1.54  ## Maith 2022
        # kappa = 0.8 ## Friston 2000
        g = 1 / 2.46  ## Maith 2022
        # gamma = 0.4 ## Friston 2000

        if params != None:
            if "kappa" in params:
                k = params["kappa"]
            if "gamma" in params:
                g = params["gamma"]

        # Nt= array_extend(np.asanyarray(stimulus), dt=dt)*AmpI
        Nt = (
            stimulus * AmpI
        )  # now we supposed that the stim comes with the time resolution included from the start

        def dNC_dt(t, NC, Nt):  # NC stands for Neurovascular coupling

            s, fm = NC

            return [Nt - k * s - g * (fm - 1), s]

        time = time_segment(Nt, dt=dt)
        s = np.ones(1, dtype=np.float32) * y0[1]
        fm = np.ones(1, dtype=np.float32) * y0[0]

        for tfm in zip(time, Nt):

            sol = odeint(
                dNC_dt, y0=(s[-1], fm[-1]), t=tfm[0], args=(tfm[1],), tfirst=True
            )
            s = np.append(s, [sol.T[0, 1]])
            fm = np.append(fm, [sol.T[1, 1]])

        return fm, s


def array_extend(arr: np.ndarray, dt: np.float32):
    """
    array_extend

    Takes the stimulus function (only ones and zeros) and extends it increasing the sampling rate by 1/dt

    **Inputs**
        - arr: np.ndarray, stimulus function made of zeros and ones, each digit es equivalent to a second
        - dt: np.float32,  the time differential by which each digit/second will be expanded e.g. if dt=0.5 then 
        [0,0,1]->[0,0,0,0,1,1]

    **output**
        - new_arr: np.ndarray, stimulus function made of zeros and ones, each digit es equivalent to a dt of a 
        second
    """

    new_arr = np.empty(0, np.float32)  # output preallocation

    for a in arr:
        tmp = np.ones(int(np.ceil(1 / dt)), dtype=np.float32) * a
        new_arr = np.append(new_arr, tmp)

    return new_arr


def scale_fun(arr: np.ndarray, factor: np.float32):
    """
    Escala funciones usando la forma [a,b]->[a,d]
    """
    return np.min(arr) + (factor / (np.max(arr) - np.min(arr))) * (arr - np.min(arr))


# @njit
def Efun(f_in: np.ndarray, E0: float = 0.32) -> float:
    """
    Efun

    Resuelve la ecuación para la proporción de oxígeno extraído de la sangre 'E'. 'Efun', definido aquí, proviene 
    de (Friston et al., 2000: Nonlinear Responses in fMRI:...), que a su vez cita a (Buxton et al., 1998)

    **Inputs:**

        -``f_in``: array, corresponde a la ecuación 13 i.e. el flujo de ingreso.
        -``E0``: float, oxígeno extraído de la sangre, 0.32 por defecto. Valor en Buxton 2004

    **Output:**

        -``E``: array, la proporción de oxígeno extraído de la sangre

    """
    E_0 = 0.32  # "baseline value of oxygen extraction fraction"

    if E0 != None:
        E_0 = E0

    try:
        E = 1 - (1 - E_0) ** (1 / f_in)

    except ZeroDivisionError:
        E = 1

    return E


# @njit
def m_t_E(f_in: np.ndarray, E0: float = 0.32):
    """
    Compute normalized CMRO2 (cerebral metabolic rate of oxygen).

    At steady state, CBF and CMRO2 are related by arterial oxygen concentration
    and the net oxygen extraction fraction E. This corresponds to the normalized
    form from Buxton 2004 eq. (2), with E from Friston 2000.

    Parameters
    ----------
    f_in : np.ndarray
        Input flow (equation 13).
    E0 : float, optional
        Baseline oxygen extraction fraction, by default 0.32 (Buxton 2004).

    Returns
    -------
    mE : np.ndarray
        Normalized CMRO2 relative to baseline, as a function of oxygen
        extraction fraction E.

    # '''

    E_0 = 0.32  # "baseline value of oxygen extraction fraction"

    if E0 != None:
        E_0 = E0

    mE = f_in * (Efun(f_in, E_0) / E_0)

    return mE


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
    10 and 11 on Buxton's 2004

    **Inputs:**

        - ``f_in``:  np.ndarray, corresponds to equation 13, i.e. the income flow.
        - ``params``: dict, includes the constants from equations 10 and 11, which are {tau_MTT = 3.0, alpha = 0.4, 
        tau_m ∈ [0, 30]}.
        - ``dt``: float, dt refers to the integration step.
        - ``viscoelastic``: bool, determines whether the output accounts for the viscoelastic effect, i.e. 
        tau = 0 or 0 < tau ≤ 30.

    **Outputs:**

        - ``v(t)``: np.ndarray, time series of the volume.
        - ``time``: np.ndarray, time series in which v(t) transpires.

    """

    tau_MTT = 3.0  # venous time constant
    alpha = 0.4  # Grubb's exponent (stiffness)
    tau_m = 10  # Viscoelastic time constant (deflation)
    tau_p = 15  # Viscoelastic time constant (inflation)

    if params != None:
        if "tau_MTT" in params:
            tau_MTT = params["tau_MTT"]
        if "alpha" in params:
            alpha = params["alpha"]
        if "tau_m" in params:
            tau_m = params["tau_m"]

    if viscoelastic:
        taum = tau_m
    else:
        taum = 0

    tauMTT, a = tau_MTT, alpha

    def dvdt(t, v, f) -> float:
        return (f - v ** (1 / a)) / (tauMTT + taum)

    v = np.empty(0, dtype=np.float32)  # preallocate the output
    time = time_segment(f_in, dt=dt)

    for ts in zip(time, f_in):
        if 0 in ts[0]:
            tmp = vol0
        else:
            tmp = v[-1]

        y = odeint(dvdt, y0=tmp, t=ts[0], args=(ts[1],), tfirst=True)
        v = np.append(v, [y.T[0][1]])

    time = time[:, 0]

    return [v, time]


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
         tau = 0 or 0 < tau ≤ 30.
        - ``params``: dict, includes the constants from equations 10 and 11, which are 
        {tau_MTT = 3.0, alpha = 0.4, tau_m ∈ [0, 30]}.

    **Outputs:**

        - ``fout``: np.ndarray, time series of the outflow.

    """

    tau_MTT = 3.0  # "venous time constant"
    alpha = 0.4  # "Grubb's exponent (stiffness)"
    tau_m = 10  # "Viscoelastic time constant (deflation)"
    tau_p = 15  # "Viscoelastic time constant (inflation)"

    if params != None:
        if "tau_MTT" in params:
            tau_MTT = params["tau_MTT"]
        if "alpha" in params:
            alpha = params["alpha"]
        if "tau_m" in params:
            tau_m = params["tau_m"]

    if viscoelastic:
        taum = tau_m
    else:
        taum = 0

    tauMTT, a = tau_MTT, alpha

    fout = ((tauMTT * vol ** (1 / a)) + taum * f_in) / (tauMTT + taum)
    fout[fout < 0.0] = 0

    return fout


# @njit
def time_segment(time: np.ndarray, dt: np.float32 = 0.01):
    """
    **time_segment**

        This function generates time intervals of length 'dt' that segment a time range similar to 'time'.

    **Inputs:**

        - ``time``: np.ndarray or list, whose length is used for duplication.
        - ``dt``: float, integration step used as the length of the intervals.

    **Output:**

        - ``new_time``: np.ndarray, such that 'array.shape = (-1,2)' with -1 time intervals of length 'dt'.

    """

    newt = np.arange(start=0, stop=len(time) * dt, step=dt, dtype=np.float32)

    # segments the nwet[a,b] in n intervals [a_i,b_i]
    # whith a_0 = 0 and b_n = newt[-1]
    if len(newt) % 2 == 0:  # if len(newt) is even, no problem
        tmp = [[newt[i], newt[i + 1]] for i in range(len(newt) - 1)]
    else:  # if len(newt) is odd, just add an item
        newt = np.append(newt, newt[-1] + dt)
        tmp = [[newt[i], newt[i + 1]] for i in range(len(newt) - 1)]

    # turns tmp into an array cause <3 (heart) numpy
    new_time = np.array(tmp)

    return new_time


# @njit
def q_func(
    vol: np.ndarray, mt: np.ndarray, f_out: np.ndarray, params, dt: float = 0.01
):
    """
    **q_fun**

        q_fun provides the solution to the differential equation for deoxyhemoglobin content, based on a 
        combination of equations 10 and 11 from the Buxton 2004 article.

    **Inputs:**

        - ``vol``: np.ndarray, time series of volume according to equation 10 of Buxton 2004.
        - ``mt``: np.ndarray, time series of normalized cerebral oxygen metabolic rate (CMRO2) to its resting value.
        - ``f_out``: np.ndarray, time series of the outflow according to equation 11 of Buxton 2004.
        - ``params``: list, includes the constants from equations 10 and 11, which are [tau_MTT = 3.0, alpha = 0.4, 
        tau_m ∈ [0, 30]].
        - ``dt``: float, dt refers to the integration step.
        - ``viscoelastic``: bool, determines whether the output accounts for the viscoelastic effect, i.e.
         tau = 0 or 0 < tau ≤ 30.

    **Outputs:**

        - ``q``: np.ndarray, time series of deoxyhemoglobin content.
        - ``time``: np.ndarray, time series in which q(t) transpires.
    """

    tau_MTT = 3.0  # "venous time constant")

    if params != None:
        if "tau_MTT" in params:
            tau_MTT = params["tau_MTT"]

    tauMTT = tau_MTT

    def dqdt(t, q, V=vol, CMRO=mt, fout=f_out) -> float:
        return (CMRO - (q / V) * fout) / tauMTT

    q = np.ones(1, dtype=np.float32)  # preallocate the output
    time = time_segment(vol, dt=dt)

    for ts in zip(time, vol, mt, f_out):
        y = odeint(
            dqdt,
            y0=q[-1],
            t=ts[0],
            args=(
                ts[1],
                ts[2],
                ts[3],
            ),
            tfirst=True,
        )
        q = np.append(q, [y.T[0][1]])

    time = np.append(time[:, 0], time[-1, 1])

    return q, time


def Balloon_odeint(
    f_in: np.ndarray,
    mt: np.ndarray,
    params=None,
    dt: float = 0.01,
    y0=(1, 1),
    viscoelastic=False,
):
    """
    **Balloon_odeint**

        Balloon_odeint solves the system of equations for the balloon model (equations 10 and 11 from the Buxton 2004 article).

    **Inputs:**

        - ``f_in``: np.ndarray, corresponds to equation 13, i.e., the inflow.
        - ``mt``: np.ndarray, time series of cerebral oxygen metabolic rate (CMRO2) normalized to its resting value.
        - ``params``: dict, includes the constants from equations 10 and 11, which are [tau_MTT = 3.0, alpha = 0.4, 
        tau_m ∈ [0, 30]].
        - ``dt``: float, dt refers to the integration step.
        - ``y0``: tuple, initial coordinates for both vt and qt.
        - ``viscoelastic``: bool, determines whether the output accounts for the viscoelastic effect, i.e.
         tau = 0 or 0 < tau ≤ 30.

    **Outputs:**

        - ``v``: np.ndarray, time series of blood volume.
        - ``q``: np.ndarray, time series of deoxyhemoglobin content.
    """

    tau_MTT = 3.0  # "venous time constant"
    alpha = 0.4  # "Grubb's exponent (stiffness)"
    tau_m = 10  # "Viscoelastic time constant (deflation)"

    if params != None:
        if "tau_MTT" in params:
            tau_MTT = params["tau_MTT"]
        if "alpha" in params:
            alpha = params["alpha"]
        if "tau_m" in params:
            tau_m = params["tau_m"]

    if viscoelastic:
        taum = tau_m
    else:
        taum = 0

    tauMTT, a = tau_MTT, alpha

    # @njit
    def dB_dt(t, B, f, m):

        v, q = B

        foutF = lambda v, f: ((tauMTT * v ** (1 / a)) + (taum * f)) / (tauMTT + taum)

        return [
            (f - v ** (1 / a)) / (tauMTT + taum),
            ((m - (q / v) * foutF(v, f)) / tauMTT),
        ]

    time = time_segment(f_in, dt=dt)
    v = np.ones(1, dtype=np.float32) * y0[0]
    q = np.ones(1, dtype=np.float32) * y0[1]

    for tfm in zip(time, f_in, mt):
        sol = odeint(
            dB_dt,
            y0=(v[-1], q[-1]),
            t=tfm[0],
            args=(
                tfm[1],
                tfm[2],
            ),
            tfirst=True,
        )
        v = np.append(v, sol[1][0])
        q = np.append(q, sol[1][1])

    return v, q


def Balloon_ivp(
    f: np.ndarray,
    m: np.ndarray,
    params=None,
    y0=(1, 1),
    viscoelastic: bool = False,
    method: str = "DOP853",
):
    """
    **Balloon_ivp**

        Balloon_ivp solves the system of equations for the balloon model (equations 10 and 11 from the Buxton 2004 article).

    **Inputs:**

        - ``f_in``: np.ndarray, corresponds to equation 13, i.e., the inflow.
        - ``mt``: np.ndarray, time series of cerebral oxygen metabolic rate (CMRO2) normalized to its resting value.
        - ``params``: dict, includes the constants from equations 10 and 11, which are [tau_MTT = 3.0, alpha = 0.4,
         tau_m ∈ [0, 30]].
        - ``dt``: float, dt refers to the integration step.
        - ``y0``: tuple, initial coordinates for both vt and qt.
        - ``viscoelastic``: bool, determines whether the output accounts for the viscoelastic effect, i.e.
         tau = 0 or 0 < tau ≤ 30.

    **Outputs:**

        - ``v``: np.ndarray, time series of blood volume.
        - ``q``: np.ndarray, time series of deoxyhemoglobin content.
    """

    if params != None:
        if "tau_MTT" in params:
            tau_MTT = params["tau_MTT"]
        if "alpha" in params:
            alpha = params["alpha"]
        if "tau_m" in params:
            tau_m = params["tau_m"]

    if viscoelastic:
        taum = tau_m
    else:
        taum = 0

    tauMTT, a = tau_MTT, alpha

    time = time_segment(f, dt=0.01)

    # @njit
    def dB_dt(t, B, f, m):
        v, q = B

        foutF = lambda v, f: ((tauMTT * v ** (1 / a)) + (taum * f)) / (tauMTT + taum)

        return [
            (f - v ** (1 / a)) / (tauMTT + taum),
            ((m - (q / v) * foutF(v, f)) / tauMTT),
        ]

    def solver(t, f, m, y0):
        v = np.ones(1, dtype=np.float32) * y0[0]
        q = np.ones(1, dtype=np.float32) * y0[1]

        for tfm in zip(t, f, m):

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
                args=(
                    tfm[1],
                    tfm[2],
                ),
            )
            v = np.append(v, sol_ivp.y[0])
            q = np.append(q, sol_ivp.y[1])

        return v, q

    v, q = solver(time, f, m, y0)
    return (v, q)


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


def BOLD_func(vt: np.ndarray, qt: np.ndarray, params=None, BM: str = "classic"):
    """
    **BOLD_func**

    Calculates the blood oxygen level-dependent (BOLD) signal using volume (vt) and deoxyhemoglobin (qt) values 
    according to the estimates of Obata (2004) and Buxton (2000) as presented in Stephen (2007).

    **Inputs:**

    - ``vt``: np.ndarray, a 1D array of volume (in arbitrary units) over time.
    - ``qt``: np.ndarray, a 1D array of cerebral deoxyhemoglobin (in arbitrary units) over time.
    - ``params``: dict, or None, a list of model parameters (optional, defaults to None) E_0, V_0, v_0, TE, epsilon, and r_0.
    - ``BM``: str, kind of Balloon Model (optional, default is "classic," while the alternative is "revised") 
    according to Stephen 2007.

    **Outputs:**

    - ``bold``: np.ndarray, a 1D array of simulated BOLD signals over time.

    """

    E_0 = 0.32  # "Baseline value of oxygen extraction fraction")
    V_0 = 0.03  # "Baseline blood volume")
    TE = 0.04  # "Echo time in miliseconds")
    eps = 1.43  # "Ratio of intra- to extravascular BOLD signal at rest")
    r_0 = 25  # "The slope of the relation between the intravascular relaxation rate and oxygen saturation. 
    # For a field strength of 1.5[T], r0=25 s^{-1}")
    O_o = 40.3  # "The frequency offset at the outer surface of the magnetised vessel for fully deoxygenated blood. 
    # For a field strength of 1.5[T], v0=40.3 s^{-1}")

    if params != None:
        if "E_0" in params:
            E_0 = params["E_0"]
        if "V_0" in params:
            V_0 = params["V_0"]
        if "TE" in params:
            TE = params["TE"]
        if "O_0" in params:
            O_o = params["O_0"]
        if "r_0" in params:
            r_0 = params["r_0"]
        if "epsilon" in params:
            eps = params["epsilon"]

    if BM == "classic":
        # Classic coefficients
        k_1 = (1 - V_0) * 4.3 * O_o * E_0 * TE
        k_2 = 2 * E_0
    elif BM == "revised":
        # Revised coefficients
        k_1 = 4.3 * O_o * E_0 * TE
        k_2 = eps * r_0 * E_0 * TE
    else:
        print(
            "So far we only have known 2 kinds of Balloons, as sujested by Stephan 2007"
        )

    k_3 = 1.0 - eps

    return V_0 * (k_1 * (1.0 - qt) + k_2 * (1.0 - qt / vt) + k_3 * (1.0 - vt))


# @njit
def BOLD_Davis(f: np.ndarray, m: np.ndarray, author: str = "Davis198"):
    """
    Compute BOLD signal using the Davis model.

    Parameters
    ----------
    f : np.ndarray
        Normalized cerebral blood flow.
    m : np.ndarray
        Normalized CMRO2 (cerebral metabolic rate of oxygen).
    author : str, optional
        Model parameters to use. Either "Davis1998" or "Maith2022",
        by default "Davis198".

    Returns
    -------
    bold : np.ndarray
        BOLD signal computed as: A * (1 - f^(alpha - beta) * m^beta)
        where A, alpha, beta depend on the chosen author.

    Notes
    -----
    Davis1998: A=0.075, alpha=0.4, beta=1.5
    Maith2022: A=140.9, alpha=0.14, beta=0.91
    """
    if author == "Davis1998":
        A = 0.075  #   "Amplitud constant"
        alpha = 0.4  #   "TODO"
        beta = 1.5  #   "TODO"

    elif author == "Maith2022":
        A = 140.9  #   "Amplitud constant"
        alpha = 0.14  #   "TODO"
        beta = 0.91  #   "TODO"

    else:
        print("So far we only have known 2 authors, Davis1998, and Maith2022")

    A, a, b = A, alpha, beta
    bold = A * (1 - f ** (a - b) * m**b)

    return bold


# %%
