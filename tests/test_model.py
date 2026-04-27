"""
Smoke tests for the Multihead PINN model.
"""

import pytest
import torch

from balloonlib.balloonpinnlib import Multihead


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_no_impulse():
    return Multihead(
        impulse=False,
        dtype=torch.float32,
    )


@pytest.fixture(scope="module")
def model_impulse():
    return Multihead(
        impulse=True,
        dtype=torch.float32,
    )


@pytest.fixture(scope="module")
def small_t():
    return torch.linspace(-1, 1, 30, dtype=torch.float32).view(-1, 1).requires_grad_(True)


@pytest.fixture(scope="module")
def small_t_stim(small_t):
    stim = torch.zeros(30, 1, dtype=torch.float32)
    stim[10:20] = 1.0
    return torch.cat([small_t, stim], dim=1)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestMultiheadInstantiation:
    def test_no_impulse_creates(self):
        m = Multihead(impulse=False)
        assert isinstance(m, Multihead)

    def test_impulse_creates(self):
        m = Multihead(impulse=True)
        assert isinstance(m, Multihead)

    def test_parameter_count_positive(self, model_no_impulse):
        n_params = sum(p.numel() for p in model_no_impulse.parameters())
        assert n_params > 0


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


class TestMultiheadForward:
    def test_forward_no_impulse_shape(self, model_no_impulse, small_t):
        model_no_impulse.eval()
        with torch.no_grad():
            out, _ = model_no_impulse(small_t)
        assert out is not None

    def test_forward_impulse_shape(self, model_impulse, small_t_stim):
        model_impulse.eval()
        with torch.no_grad():
            out, _ = model_impulse(small_t_stim)
        assert out is not None

    def test_state_variables_attached(self, model_no_impulse, small_t):
        """After forward, model should expose f, m, v, q as tensors."""
        model_no_impulse.eval()
        with torch.no_grad():
            model_no_impulse(small_t)
        for attr in ("f", "m", "v", "q"):
            val = getattr(model_no_impulse, attr, None)
            assert val is not None, f"model.{attr} is None after forward"
            assert isinstance(val, torch.Tensor)

    def test_predictor_shape(self, model_no_impulse, small_t):
        """predictor() should return a tensor matching temporal dim."""
        model_no_impulse.eval()
        with torch.no_grad():
            model_no_impulse(small_t)
            pred = model_no_impulse.predictor()
        assert pred.shape[0] == small_t.shape[0]


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_loss_has_grad(self, model_no_impulse, small_t):
        """Total loss should yield gradients through the model."""
        model_no_impulse.train()
        out, _ = model_no_impulse(small_t)
        # use a trivial MSE as proxy loss
        proxy = (model_no_impulse.predictor() ** 2).mean()
        proxy.backward()
        grads = [p.grad for p in model_no_impulse.parameters() if p.grad is not None]
        assert len(grads) > 0, "No parameter received a gradient"
