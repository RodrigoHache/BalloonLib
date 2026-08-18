"""
Centralised runtime configuration for BalloonLib.

Owns the mutable ``device`` and ``dtype`` globals consumed by every
submodule.  Update them through :func:`set_device` / :func:`set_dtype` so
that every consumer (which imports this module, not the names) sees the
new value on its next attribute access.

Example
-------
>>> import balloonlib
>>> balloonlib.set_device("cpu")
>>> balloonlib.set_dtype(torch.float32)
"""

import torch

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype: torch.dtype = torch.float32


def set_device(d) -> None:
    """Set the library-wide computation device.

    Parameters
    ----------
    d : str or torch.device
        Either a device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``) or
        a :class:`torch.device` instance.
    """
    global device
    device = d if isinstance(d, torch.device) else torch.device(d)


def set_dtype(d: torch.dtype) -> None:
    """Set the library-wide floating-point dtype."""
    global dtype
    dtype = d
