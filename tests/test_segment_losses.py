"""Vectorised per-segment losses agree with the per-segment loop."""
import pytest, torch, torch.nn as nn
from balloonlib.physics import segment_losses, segment_temporal_residuals, weighted_temporal_ode_loss

LOSSES = [nn.MSELoss(), nn.MSELoss(reduction="sum"), nn.L1Loss(), nn.SmoothL1Loss(beta=0.5), nn.HuberLoss(delta=0.3)]


def _loop(residual, meFn, n_segments):
    return torch.stack([meFn(s, torch.zeros_like(s)) for s in segment_temporal_residuals(residual, n_segments)])


@pytest.mark.parametrize("meFn", LOSSES, ids=lambda f: f"{type(f).__name__}-{f.reduction}")
@pytest.mark.parametrize("N,S", [(3000, 30), (3001, 30), (2999, 7), (10, 10)])
def test_segment_losses_match_loop(meFn, N, S):
    torch.manual_seed(0)
    r = (torch.randn(N, 4) * torch.linspace(2, 0.1, N)[:, None]).requires_grad_(True)
    a, b = _loop(r, meFn, S), segment_losses(r, meFn, S)
    torch.testing.assert_close(b, a, rtol=1e-6, atol=0)
    ga, = torch.autograd.grad(a.sum(), r); gb, = torch.autograd.grad(b.sum(), r)
    torch.testing.assert_close(gb, ga, rtol=1e-6, atol=1e-12)


def test_fallback_for_arbitrary_callable():
    torch.manual_seed(1)
    r = torch.randn(300, 4)
    fn = lambda a, b: ((a - b) ** 4).mean()
    torch.testing.assert_close(segment_losses(r, fn, 30), _loop(r, fn, 30), rtol=0, atol=0)


@pytest.mark.parametrize("normalise", [False, True])
def test_weighted_loss_value(normalise):
    torch.manual_seed(2)
    r = torch.randn(3000, 4) * torch.linspace(2, 0.1, 3000)[:, None]
    seg = _loop(r, nn.MSELoss(), 30)
    w = torch.exp(-10.0 * torch.cat([torch.zeros(1), torch.cumsum(seg, 0)[:-1]]))
    if normalise:
        w = w / (w > 1e-6).sum()
    expected = torch.sum(w * seg)
    got = weighted_temporal_ode_loss(r, nn.MSELoss(), n_segments=30, epsilon=10.0, normalise_weights=normalise)
    torch.testing.assert_close(got, expected, rtol=1e-6, atol=0)
