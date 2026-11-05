"""Tests for balloonlib.ar1_whitening."""
import math
import numpy as np
import torch
import torch.nn as nn
from balloonlib.ar1_whitening import AR1Whitening

TR, N = 1.75, 165


def _ar1(phi, n, rng):
    e = np.empty(n); e[0] = rng.standard_normal()
    for k in range(1, n):
        e[k] = phi * e[k - 1] + math.sqrt(1 - phi ** 2) * rng.standard_normal()
    return torch.tensor(e)


def test_rho0_is_current_loss():
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(N, generator=g, dtype=torch.float64, requires_grad=True)
    y = torch.randn(N, generator=g, dtype=torch.float64)
    L1 = AR1Whitening(TR).loss(pred, y, None, nn.MSELoss())
    off = -torch.mean(pred - y).detach()
    L0 = nn.MSELoss()(off + pred, y)
    g1, = torch.autograd.grad(L1, pred); g0, = torch.autograd.grad(L0, pred)
    assert torch.allclose(L1, L0, atol=1e-12) and torch.allclose(g1, g0, atol=1e-12)


def test_whitening_inverts_ar1_correlation():
    w = AR1Whitening(TR); w.rho = 0.6
    W = torch.stack([w.whiten(c) for c in torch.eye(40, dtype=torch.float64)], 1)
    k = torch.arange(40, dtype=torch.float64)
    R = 0.6 ** (k[:, None] - k[None, :]).abs()
    assert torch.allclose(W.T @ W @ R, torch.eye(40, dtype=torch.float64), atol=1e-10)


def _estimate(phi, rng):
    w = AR1Whitening(TR, warmup=0, every=1)
    w.loss(_ar1(phi, N, rng) + 0.7, torch.zeros(N, dtype=torch.float64), None, nn.MSELoss())
    return w.update(0)


def test_recovers_rho():
    rng = np.random.default_rng(3)
    assert abs(np.mean([_estimate(0.4, rng)["phi"] for _ in range(100)]) - 0.4) < 0.04


def test_white_noise_is_reported_white():
    rng = np.random.default_rng(4)
    assert np.mean([_estimate(0.0, rng)["white"] for _ in range(300)]) > 0.9


def test_whitened_residual_is_white():
    rng = np.random.default_rng(5)
    assert abs(np.mean([_estimate(0.6, rng)["lag1_whitened"] for _ in range(100)])) < 0.05


def test_schedule():
    w = AR1Whitening(TR, warmup=100, every=50, freeze_after=250)
    assert [i for i in range(400) if w.due(i)] == [100, 150, 200, 250]
