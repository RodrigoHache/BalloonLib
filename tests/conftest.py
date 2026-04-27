"""
Shared pytest fixtures for BalloonLib tests.
"""

import pytest
import torch


@pytest.fixture(scope="session")
def dtype():
    return torch.float32


@pytest.fixture(scope="session")
def device():
    return "cpu"


@pytest.fixture(scope="session")
def small_time(dtype):
    """Normalised time vector of 50 points in [-1, 1]."""
    t = torch.linspace(-1, 1, 50, dtype=dtype).view(-1, 1).requires_grad_(True)
    return t


@pytest.fixture(scope="session")
def small_stimulus(dtype):
    """Simple block stimulus at 100 Hz for 2 s."""
    t = torch.arange(0, 200, dtype=dtype)  # 200 samples @ 100 Hz
    stim = torch.zeros_like(t)
    stim[50:100] = 1.0
    return stim.view(-1, 1)
