"""
Evaluation metrics for the BalloonLib PINN framework.

Provides quality-of-fit statistics and HRF descriptor extraction.
"""

import numpy as np
import torch
from scipy.stats import pearsonr

from balloonlib.utils import tensor2np


# ---------------------------------------------------------------------------
# Kling–Gupta Efficiency
# ---------------------------------------------------------------------------

def kge_stat(y_obs, y_sim):
    """Compute the Kling--Gupta Efficiency (KGE) between two signals.

    Parameters
    ----------
    y_obs : array-like or torch.Tensor
        Observed (ground-truth) signal.
    y_sim : array-like or torch.Tensor
        Simulated (predicted) signal.

    Returns
    -------
    float
        KGE value; 1.0 indicates a perfect match.
    """
    def t2np(y):
        return np.copy(tensor2np(y)) if isinstance(y, torch.Tensor) else y

    y_0 = t2np(y_obs)
    y_1 = t2np(y_sim)

    r = pearsonr(y_1, y_0)[0]
    mu_rate = np.mean(y_1) / np.mean(y_0)
    std_rate = np.std(y_1) / np.std(y_0)

    return 1 - ((r - 1) ** 2 + (mu_rate - 1) ** 2 + (std_rate - 1) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# HRF descriptor extraction
# ---------------------------------------------------------------------------

def hrf_description(
    hrf_data,
    max_time: float = 30,
    first_non_zero_t: float = 0.05,
    integration_rule: str = "rectangle",
) -> dict:
    """Extract shape descriptors from one or more fitted HRF waveforms.

    Computes the following descriptors per signal:

    * **HP**    — height (amplitude) of the main peak.
    * **TTP**   — time to peak (seconds).
    * **FWHM**  — full width at half maximum of the main peak (seconds).
    * **TO**    — time to onset, defined as time to a 10 % rise from baseline.
    * **AUC**   — area under the curve of the first (positive) peak.
    * **MU**    — minimum undershoot amplitude (``NaN`` if no undershoot).
    * **TTU**   — time to undershoot minimum (seconds).
    * **TT0**   — time to return to baseline after undershoot (seconds).

    Parameters
    ----------
    hrf_data : numpy.ndarray or torch.Tensor
        HRF waveform(s).  Shape ``(n_signals, length)`` or ``(length,)`` for
        a single waveform.
    max_time : float
        Time span (seconds) of the HRF window.
    first_non_zero_t : float
        Minimum time threshold for a valid onset detection (seconds).
    integration_rule : {``'rectangle'``, ``'trapezoidal'``}
        Numerical integration method for AUC.

    Returns
    -------
    dict
        Dictionary with keys ``HP``, ``TTP[s]``, ``FWHM[s]``, ``TO[s]``,
        ``AUC``, ``MU``, ``TTU[s]``, ``TT0[s]``.  Each value is a NumPy
        array of length ``n_signals``.
    """
    tmp = (
        hrf_data.squeeze()
        if isinstance(hrf_data, torch.Tensor)
        else np.squeeze(hrf_data)
    )
    hrf = np.copy(tensor2np(tmp)) if isinstance(tmp, torch.Tensor) else np.copy(tmp)

    # Threshold below which values are treated as zero
    zero = 1e-4

    # Normalise to 2-D (n_signals, length)
    if hrf.ndim == 2:
        hrf = hrf if hrf.shape[0] < hrf.shape[1] else hrf.T
    elif hrf.ndim == 1:
        hrf = hrf.reshape(1, -1)
    else:
        raise TypeError(
            "Several signals should be delivered using a order-2 tensor (matrix)"
        )

    # Truncate to 6 decimal places to avoid floating-point precision issues
    hrf = np.trunc(hrf * 1e6) * 1e-6
    n_signals, signal_length = hrf.shape

    time = np.arange(0, max_time, max_time / signal_length)

    output = {
        "HP":     np.empty(n_signals),
        "TTP[s]": np.empty(n_signals),
        "FWHM[s]": np.empty(n_signals),
        "TO[s]":  np.empty(n_signals),
        "AUC":    np.empty(n_signals),
        "MU":     np.empty(n_signals),
        "TTU[s]": np.empty(n_signals),
        "TT0[s]": np.empty(n_signals),
    }

    output["HP"] = np.max(hrf, axis=1)
    TTPi = np.argmax(hrf, axis=1)
    output["TTP[s]"] = time[TTPi]

    output["MU"] = np.min(hrf, axis=1)
    TTUi = np.argmin(hrf, axis=1)
    output["TTU[s]"] = time[TTUi]

    for j in range(n_signals):
        before_max = hrf[j, :TTPi[j]]
        after_max = hrf[j, TTPi[j]:]
        half_max = output["HP"][j] / 2
        half_max_min = half_max <= zero

        exclusion = [
            before_max.shape == (0,),
            after_max.shape == (0,),
            half_max_min and output["TTP[s]"][j] > max_time / 2,
            output["TTP[s]"][j] <= 2,
        ]

        if any(exclusion):
            for key in output.keys():
                output[key][j] = np.nan
        else:
            FWHM_l = np.argmax(before_max >= half_max)
            FWHM_r = TTPi[j] + np.argmax(after_max <= half_max)
            output["FWHM[s]"][j] = time[FWHM_r] - time[FWHM_l]

            if half_max_min and output["FWHM[s]"][j] > max_time / 2:
                for key in output.keys():
                    output[key][j] = np.nan

            # Time to onset (TO)
            tmp_to = time[np.argmax(before_max > 0.1 * output["HP"][j])]
            if (tmp_to <= first_non_zero_t) or (half_max <= zero):
                output["TO[s]"][j] = np.nan
            else:
                output["TO[s]"][j] = tmp_to

            # Area under the curve (AUC)
            root_l = TTPi[j] - np.argmax(before_max[::-1] <= zero)
            root_r = TTPi[j] + np.argmax(after_max <= zero)
            dt = time[1] - time[0]

            if integration_rule == "trapezoidal":
                output["AUC"][j] = (
                    0.5 * dt * np.sum(
                        hrf[j, root_l:root_r - 1] + hrf[j, root_l + 1:root_r]
                    )
                )
            elif integration_rule == "rectangle":
                output["AUC"][j] = dt * np.sum(hrf[j, root_l:root_r])

            # Undershoot descriptors (MU, TTU, TT0)
            output["MU"][j] = np.nan if output["MU"][j] >= -1*zero else output["MU"][j]

            if output["MU"][j] < -1*zero:
                TTUi_j = TTPi[j] + np.argmin(after_max)
                output["TTU[s]"][j] = time[TTUi_j]
                tmp_tt0 = np.argmax(hrf[j, TTUi_j:] >= -1*zero)
                if tmp_tt0 == 0:
                    output["TT0[s]"][j] = np.nan
                else:
                    TT0i = TTUi_j + tmp_tt0
                    output["TT0[s]"][j] = time[TT0i] if TT0i < signal_length - 1 else np.nan
            else:
                output["TTU[s]"][j] = np.nan
                output["TT0[s]"][j] = np.nan

    return output
