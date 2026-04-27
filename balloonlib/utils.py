"""
Utility functions for the BalloonLib PINN framework.

Provides tensor/numpy converters, domain scaling, temporal matching,
1-D convolution, the Double-Gamma HRF factory, and stimulus-HRF convolution.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import gamma

# ---------------------------------------------------------------------------
# Module-level device / dtype (mirrors balloonpinnlib.py globals)
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


# ---------------------------------------------------------------------------
# Tensor ↔ NumPy converters
# ---------------------------------------------------------------------------
def tensor2np(tensor):
    """Turns a tensor into an numpy array
    
    Parameter
    ---------
    tensor : torch.Tensor
        Data to turn into a numpy.array
    
    Returns
    -------
    transformed : numpy.array
        Transformed array of the same shape as ``tensor``.
    """
    return tensor.detach().cpu().numpy()  

def np2tensor(vector):
    """Turns an array into an torch.Tensor 
    
    Parameter
    ---------
    tensor : numpy.array
        data to transform to torch.Tensor
    
    Returns
    -------
    transformed : torch.Tensor
        Data turned into a numpy.array
    """
    transformed = torch.tensor(
    vector, requires_grad=True, dtype=torch.float32
        ).view(-1, 1)
    return transformed


# ---------------------------------------------------------------------------
# Domain scaling
# ---------------------------------------------------------------------------

def scale_domains(
    data: torch.Tensor | np.ndarray,
    original_domains: list | tuple,
    new_domains: list | tuple,
    dim_to_transform: int | list | tuple | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a linear domain mapping to a tensor, optionally per-slice.

    When ``dim_to_transform`` is ``None``, maps the entire tensor from
    ``original_domains`` to ``new_domains``.  When a dimension is given,
    applies a *different* mapping to each slice along that dimension.

    Parameters
    ----------
    data : torch.Tensor or numpy.ndarray
        Input data of any shape.
    original_domains : tuple or list of tuples
        ``(min, max)`` for a global transform, or a list of ``N`` such
        tuples for per-slice transforms (``N`` = size of ``dim_to_transform``).
    new_domains : tuple or list of tuples
        Target domain(s), same format as ``original_domains``.
    dim_to_transform : int or None
        Axis along which per-slice transforms are applied.
        ``None`` applies a single transform to the whole tensor.
    dtype : torch.dtype
        Output tensor dtype.

    Returns
    -------
    transformed : torch.Tensor
        Transformed tensor of the same shape as ``data``.
    scale_factors : torch.Tensor
        Scale factor(s) used (shape ``(1,)`` global, ``(N,)`` per-slice).

    Raises
    ------
    ValueError
        On domain format errors, zero-width domains, or out-of-range dimensions.

    Examples
    --------
    Global transform ``[0, 30] -> [0, 1]``::

        >>> data = torch.tensor([[0., 10., 20.], [5., 15., 30.]])
        >>> t, s = scale_domains(data, (0, 30), (0, 1))

    Per-row transform (``dim_to_transform=0``)::

        >>> data = torch.tensor([[0., 5., 10.], [0., 50., 100.]])
        >>> t, s = scale_domains(data, [(0, 10), (0, 100)], [(0, 1), (0, 1)], dim_to_transform=0)
    """
    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data)
    else:
        data_tensor = data

    # Case 1: Global transform (single domain pair for the whole tensor)
    if dim_to_transform is None:
        if not (all(isinstance(x, (int, float)) for x in original_domains) and len(original_domains) == 2):
            raise ValueError(
                f"original_domains must be an ordered (min, max) pair. Got {original_domains}"
            )

        min_o, max_o = original_domains
        min_n, max_n = new_domains

        if max_o == min_o:
            raise ValueError(f"Original domain range is zero: [{min_o}, {max_o}]")

        scale = (max_n - min_n) / (max_o - min_o)
        transformed = (min_n * (max_o - data_tensor) + max_n * (data_tensor - min_o)) / (max_o - min_o)
        return transformed, torch.tensor([scale])

    # Case 2: Per-slice transform along one dimension
    if isinstance(dim_to_transform, (list, tuple)):
        if len(dim_to_transform) != 1:
            raise ValueError(
                f"Only single-dimension transformation supported. "
                f"Got dim_to_transform={dim_to_transform}."
            )
        dim_to_transform = dim_to_transform[0]
    elif not isinstance(dim_to_transform, int):
        raise ValueError(f"dim_to_transform must be int, got {type(dim_to_transform)}")

    if dim_to_transform < 0 or dim_to_transform >= data_tensor.ndim:
        raise ValueError(
            f"dim_to_transform={dim_to_transform} out of range for "
            f"tensor with {data_tensor.ndim} dimensions"
        )

    if not isinstance(original_domains, (list, tuple)):
        raise ValueError(
            f"original_domains must be a list or tuple of (min, max) pairs. "
            f"Got {type(original_domains)}"
        )
    if not isinstance(new_domains, (list, tuple)):
        raise ValueError(
            f"new_domains must be a list or tuple of (min, max) pairs. "
            f"Got {type(new_domains)}"
        )

    num_slices = data_tensor.shape[dim_to_transform]

    transformed = torch.zeros_like(data_tensor, dtype=dtype)
    scales = []

    for slice_idx in range(num_slices):
        orig = original_domains[slice_idx]
        new = new_domains[slice_idx]

        if not isinstance(orig, (tuple, list)) or len(orig) != 2:
            raise ValueError(
                f"original_domains[{slice_idx}] must be a (min, max) pair, got {orig}"
            )
        if not isinstance(new, (tuple, list)) or len(new) != 2:
            raise ValueError(
                f"new_domains[{slice_idx}] must be a (min, max) pair, got {new}"
            )

        min_o, max_o = orig
        min_n, max_n = new

        if max_o == min_o:
            raise ValueError(
                f"Original domain range is zero for slice {slice_idx}: [{min_o}, {max_o}]"
            )

        scale = (max_n - min_n) / (max_o - min_o)
        scales.append(scale)

        idx = [slice(None)] * data_tensor.ndim
        idx[dim_to_transform] = slice_idx
        transformed[tuple(idx)] = (data_tensor[tuple(idx)] - min_o) * scale + min_n

    return transformed, torch.tensor(scales)


# ---------------------------------------------------------------------------
# HRF factory
# ---------------------------------------------------------------------------

def DoubleGamma(
    A1: float, alpha1: float, beta1: float,
    A2: float, alpha2: float, beta2: float,
):
    """Return a Double-Gamma hemodynamic response function (HRF).

    The HRF is defined as the difference of two scaled Gamma PDFs:
    ``HRF(t) = A1 * Gamma(t; alpha1, 1/beta1) - A2 * Gamma(t; alpha2, 1/beta2)``.

    Parameters
    ----------
    A1 : float
        Amplitude of the first (peak) gamma component.
    alpha1 : float
        Shape parameter of the first gamma.
    beta1 : float
        Rate parameter of the first gamma.
    A2 : float
        Amplitude of the second (undershoot) gamma component.
    alpha2 : float
        Shape parameter of the second gamma.
    beta2 : float
        Rate parameter of the second gamma.

    Returns
    -------
    callable
        A function ``f(t) -> float`` evaluating the HRF at time *t*.
    """
    return lambda t: (A1 * gamma.pdf(t, a=alpha1, scale=(1 / beta1))) - (
        A2 * gamma.pdf(t, a=alpha2, scale=(1 / beta2))
    )


# ---------------------------------------------------------------------------
# Temporal matching
# ---------------------------------------------------------------------------

@torch.compile()
def timeBall(time_tensor1, time_tensor2, delta=0.005):
    """Find nearest-neighbour indices between two time tensors.

    For each value in ``time_tensor1``, locates the closest value in
    ``time_tensor2`` that falls within ``+/- delta``.

    Parameters
    ----------
    time_tensor1 : torch.Tensor
        Reference times, shape ``(N,)`` or ``(N, 1)``.
    time_tensor2 : torch.Tensor
        Search times, shape ``(M,)``.
    delta : float
        Matching tolerance.

    Returns
    -------
    indices : torch.Tensor
        Sorted indices into ``time_tensor2``, shape ``(N,)``.
    ball_index : None
        Reserved; always ``None``.
    """
    t1 = time_tensor1.squeeze()
    t2 = time_tensor2.squeeze()

    # Binary search to find insertion points
    indices = torch.bucketize(t1, t2)

    # Check left and right neighbors
    idx_left = (indices - 1).clamp(0, len(t2) - 1)
    idx_right = indices.clamp(0, len(t2) - 1)

    dist_left = torch.abs(t1 - t2[idx_left])
    dist_right = torch.abs(t1 - t2[idx_right])

    # Choose the nearest neighbor
    min_dist = torch.min(dist_left, dist_right)
    nearest_idx = torch.where(dist_left <= dist_right, idx_left, idx_right)

    # Filter to only include matches within delta
    valid_mask = min_dist <= delta
    result_indices = torch.where(valid_mask, nearest_idx, torch.zeros_like(nearest_idx))

    return result_indices.sort()[0], None


# ---------------------------------------------------------------------------
# 1-D Convolution
# ---------------------------------------------------------------------------

@torch.compile()
def pytorch_convolve(signal, kernel, mode="full", flip=False):
    """1-D convolution using :func:`torch.nn.functional.conv1d`.

    PyTorch equivalent of :func:`scipy.signal.convolve` for tensors of
    shape ``(n, 1)`` and ``(m, 1)`` with ``n > m``.

    Parameters
    ----------
    signal : torch.Tensor
        Input signal (stimulus), 1-D or ``(n, 1)``.
    kernel : torch.Tensor
        Convolution kernel (HRF), 1-D or ``(m, 1)``.
    mode : {``'full'``, ``'valid'``, ``'same'``}
        Output size convention (see :func:`scipy.signal.convolve`).
    flip : bool
        If ``True``, flip the kernel before convolving (cross-correlation).

    Returns
    -------
    torch.Tensor
        1-D convolution result.
    """
    signal = signal.squeeze() if signal.ndim >= 2 else signal
    kernel = kernel.squeeze() if kernel.ndim >= 2 else kernel

    if mode == "full":
        padding = len(kernel) - 1
        signal = F.pad(signal, (padding, padding))  # shape (n + 2(m - 1,))
    elif mode == "same":
        padding = len(kernel) - 1
        signal = F.pad(signal, (padding, 0))  # shape (n + m - 1,)

    signal = signal[None, None, :]
    kernel = kernel[None, None, :] if not flip else kernel.flip(-1)[None, None, :]

    return F.conv1d(signal, kernel, padding=0).squeeze()


# ---------------------------------------------------------------------------
# Stimulus–HRF convolution
# ---------------------------------------------------------------------------

def tofit(stim, hrf, time_max, dt=0.01):
    """Convolve stimulus with HRF to produce a predicted BOLD signal.

    Parameters
    ----------
    stim : torch.Tensor
        Stimulus time series.
    hrf : torch.Tensor
        Hemodynamic Response Function kernel.
    time_max : float
        Maximum time duration (seconds).
    dt : float
        Time step (seconds).

    Returns
    -------
    test : torch.Tensor
        Convolved (predicted BOLD) signal.
    test_time : torch.Tensor
        Corresponding time vector.

    """
    if isinstance(time_max, torch.Tensor):                                                                    
        time_max = float(time_max.detach())
    test_time = torch.arange(0, time_max, dt)
    test = pytorch_convolve(stim, hrf, mode="full", flip=True)[:test_time.size(0)]
    return test, test_time
