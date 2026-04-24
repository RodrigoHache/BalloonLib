"""
Data utilities for the BalloonLib PINN framework.

Provides functions for sampling training indices, normalising BOLD signals,
segmenting fMRI epochs, generating stimulus patterns, and loading pickle files.
"""

import os
import pickle

import numpy as np
import torch
from typing import List

from balloonlib.utils import tensor2np


# ---------------------------------------------------------------------------
# Training index sampler
# ---------------------------------------------------------------------------

def training_data(input_data: dict, num_points: int, random: bool = False):
    """Sample indices from a dataset dictionary.

    Parameters
    ----------
    input_data : dict
        Dictionary whose values are all sequences of the same length.
    num_points : int
        Number of indices to sample.  Must not exceed the dataset length.
    random : bool
        If ``True``, draws a sorted random (equiprobable) sample.
        If ``False`` (default), uses uniform stride-based sampling.

    Returns
    -------
    torch.Tensor
        Sorted index tensor of length ``num_points``.

    Raises
    ------
    ValueError
        If ``num_points`` exceeds the dataset length.
    """
    max_len = len(list(input_data.values())[0])

    if num_points is not None:
        if num_points > max_len:
            raise ValueError(
                "num_points must not exceed the length of input_data."
            )

    if random:
        index = torch.randint(low=1, high=max_len, size=(num_points,), dtype=int)
        index = torch.sort(index)[0]
    else:
        subs = max_len // num_points
        index = torch.arange(max_len, dtype=int)[::subs]

    return index


# ---------------------------------------------------------------------------
# BOLD normalisation
# ---------------------------------------------------------------------------

def normFn(data, ref=None, step=1.75, first_stim=None, alt_ref=None):
    """Normalise BOLD signal data as a percentage change from baseline.

    Computes ``(data - B0) * 100 / B0`` where *B0* is a baseline value
    determined by the ``ref`` strategy.

    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        BOLD signal time series.
    ref : {``'stim'``, ``'mean'``, ``None``}, optional
        Baseline method:

        * ``'stim'``  — mean of samples before the first stimulus onset.
        * ``'mean'``  — mean of the entire signal.
        * ``None``    — use ``alt_ref`` (must be provided).
    step : float
        Time step between measurements (TR), in seconds.
    first_stim : float or None
        Time of first stimulus onset in seconds.  Defaults to ``4 * step``.
    alt_ref : array-like or None
        Alternative reference signal; used when ``ref`` is ``None``.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Percentage-change normalised signal, same type as ``data``.

    Raises
    ------
    TypeError
        If no valid baseline reference can be computed.
    """
    signal = torch.clone(data) if isinstance(data, torch.Tensor) else np.copy(data)
    f_stim = first_stim if first_stim is not None else 4 * step
    mean_fn = torch.mean if isinstance(signal, torch.Tensor) else np.mean
    size = signal.size()[0] if isinstance(signal, torch.Tensor) else signal.shape[0]

    time = (
        torch.arange(0, size * step, step)
        if isinstance(signal, torch.Tensor)
        else np.arange(0, size * step, step)
    )

    baseline_methods = {
        "stim": lambda: mean_fn(signal[time < f_stim]),
        "mean": lambda: mean_fn(signal),
        None:   lambda: mean_fn(alt_ref) if alt_ref is not None else None,
    }

    B0 = baseline_methods.get(ref, baseline_methods[None])()

    if B0 is None:
        raise TypeError("Reference for baseline is needed")

    return (signal - B0) * (100 / B0)


# ---------------------------------------------------------------------------
# Epoch segmentation
# ---------------------------------------------------------------------------

def segmentData(
    normData, Sti_Onsets: list, time_bf_stim=1, t0s=0, TR=1.75, dtype=torch.float32
):
    """Segment fMRI BOLD signal data into stimulus-locked epochs.

    Extracts contiguous time windows of a normalised BOLD time series
    centred around each stimulus onset.  Output tensors retain gradient
    tracking for PINN training.

    Parameters
    ----------
    normData : numpy.ndarray
        Normalised fMRI BOLD time series, shape ``(timepoints, features)``.
    Sti_Onsets : list of float
        Stimulus onset times in seconds.
    time_bf_stim : float
        Time before each onset to include in the epoch (seconds).
    t0s : float
        Offset added to the constructed time vector (seconds).
    TR : float
        Repetition time of fMRI acquisition (seconds).
    dtype : torch.dtype
        Data type for output tensors.

    Returns
    -------
    Bold_segments : list of torch.Tensor
        1-D BOLD tensors per epoch (``requires_grad=True``).
    time_corrected : list of torch.Tensor
        Time vectors relative to stimulus onset (``requires_grad=True``).

    Notes
    -----
    All epochs are truncated to the length of the shortest segment to
    maintain uniform tensor sizes.
    """
    Bold_time = np.arange(0, normData.shape[0] * TR, TR) + t0s

    Bold_segments = []
    time_corrected = []
    temp_len = []

    for i, onset in enumerate(Sti_Onsets):
        start_time = onset - time_bf_stim

        if i == len(Sti_Onsets) - 1:
            end_time = Bold_time[-1] + TR
        else:
            end_time = Sti_Onsets[i + 1]

        mask = (Bold_time >= start_time) & (Bold_time < end_time)

        if np.any(mask):
            segment_data = normData[mask]
            segment_time = Bold_time[mask]

            if not isinstance(segment_data, torch.Tensor):
                segment_tensor = torch.tensor(
                    segment_data.reshape(-1), requires_grad=True, dtype=dtype
                )
            else:
                segment_tensor = segment_data.reshape(-1).requires_grad_(True).to(dtype)
            
            corrected_time = torch.tensor(
                np.round(segment_time - onset + time_bf_stim, 5),
                requires_grad=True,
                dtype=dtype,
            )

            Bold_segments.append(segment_tensor)
            time_corrected.append(corrected_time)
            temp_len.append(len(segment_tensor))

    # Truncate all epochs to the shortest length
    Bold_segments = [s[:temp_len[0]] for s in Bold_segments]
    time_corrected = [t[:temp_len[0]] for t in time_corrected]
    return Bold_segments, time_corrected


# ---------------------------------------------------------------------------
# Stimulus pattern generation
# ---------------------------------------------------------------------------

def experimental_stims(
    normDataSize: int = 84,
    Sti_Onsets=None,
    TR=1.75,
    block_len=3,
    stmxblck: int = 1,
    Hz=100,
    dtype=torch.float32,
    device="cuda",
):
    """Generate a binary stimulus pattern tensor at high temporal resolution.

    Creates a stimulus timecourse upsampled to ``Hz`` samples/second for
    use in PINN training (convolution with an HRF requires micro-time resolution).

    Parameters
    ----------
    normDataSize : int
        Length of the original fMRI signal in samples.
    Sti_Onsets : list of float
        Stimulus onset times in seconds.
    TR : float
        Repetition time (seconds).
    block_len : int or float
        Stimulus duration in seconds.
    stmxblck : int
        Number of stimuli per block (used to scale amplitude).
    Hz : float
        Sampling frequency for the output stimulus (samples/second).
    dtype : torch.dtype
        Output tensor dtype.
    device : str
        Computation device (``'cuda'`` or ``'cpu'``).

    Returns
    -------
    stim_pattern : torch.Tensor
        Upsampled stimulus, shape ``(total_samples, 1)``.
    stim_time : torch.Tensor
        Corresponding time vector, shape ``(total_samples,)``.
    """
    total_time = normDataSize * TR
    stim_time = torch.arange(0, total_time, 1 / Hz, dtype=dtype, device=device)
    stim_pattern = torch.zeros(stim_time.size(0), dtype=dtype, device=device)

    for onset in Sti_Onsets:
        start_idx = int(onset * Hz)
        end_idx = int((onset + block_len) * Hz)
        stim_pattern[start_idx:end_idx] = 1 / stmxblck

    return stim_pattern.unsqueeze(1).to(device), stim_time.to(device)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_pickle(file_path: str):
    """Load a Python object from a pickle file.

    Parameters
    ----------
    file_path : str
        Path to the ``.pkl`` file.

    Returns
    -------
    object
        De-serialised Python object.

    Raises
    ------
    TypeError
        If the file does not exist or is empty.
    """
    print(f"File exists:{os.path.exists(file_path)}")
    print(f"File size:{os.path.getsize(file_path)} bytes")

    if os.path.getsize(file_path) != 0:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
    else:
        raise TypeError("File doesn't exist or was corrupted")

    if isinstance(obj, dict):
        print(obj.keys())
    return obj
