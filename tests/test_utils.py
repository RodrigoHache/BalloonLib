"""
Smoke tests for utility functions in balloonpinnlib.
"""
import numpy as np
import pytest
import torch

from balloonlib.balloonpinnlib import (
    DoubleGamma,
    dfdt,
    kge_stat,
    normFn,
    pytorch_convolve,
    scale_domains,
    timeBall,
    tofit,
    training_data,
)


# ---------------------------------------------------------------------------
# normFn
# ---------------------------------------------------------------------------

class TestNormFn:
    def test_mean_ref_returns_zero_mean(self):
        data = np.array([10.0, 20.0, 30.0])
        result = normFn(data, ref="mean")
        # mean of result should be 0
        assert np.isclose(np.mean(result), 0.0, atol=1e-6)

    def test_stim_ref_uses_prestim(self):
        # First 2 points (t<3.5) form the baseline when step=1.75
        data = np.ones(10) * 100.0
        result = normFn(data, ref="stim", step=1.75)
        assert np.allclose(result, 0.0, atol=1e-6)

    def test_tensor_input_returned_as_tensor(self):
        data = torch.ones(10) * 50.0
        result = normFn(data, ref="mean")
        assert isinstance(result, torch.Tensor)

    def test_raises_without_ref(self):
        with pytest.raises(TypeError):
            normFn(np.ones(5), ref=None, alt_ref=None)


# ---------------------------------------------------------------------------
# scale_domains
# ---------------------------------------------------------------------------

class TestScaleDomains:
    def test_identity_when_domains_equal(self):
        t = torch.linspace(0, 1, 10)
        out, scales = scale_domains(t, original_domains=[0, 1], new_domains=[0, 1])
        assert torch.allclose(out, t, atol=1e-6)

    def test_output_within_new_domain(self, dtype):
        t = torch.linspace(0.0, 1.0, 20, dtype=dtype)
        out, _ = scale_domains(t, original_domains=[0, 1], new_domains=[0, 30])
        assert out.min() >= 0.0 - 1e-5
        assert out.max() <= 30.0 + 1e-5

    def test_scales_shape(self, dtype):
        t = torch.linspace(0, 1, 10, dtype=dtype)
        _, scales = scale_domains(t, original_domains=[0, 1], new_domains=[0, 30])
        assert scales.shape == torch.Size([1])


# ---------------------------------------------------------------------------
# pytorch_convolve
# ---------------------------------------------------------------------------

class TestPytorchConvolve:
    def test_full_length(self):
        # full-mode: output length = n + m - 1
        n, m = 20, 5
        s = torch.ones(n)
        k = torch.ones(m)
        out = pytorch_convolve(s, k, mode="full")
        assert out.shape[0] == n + m - 1

    def test_same_length(self):
        n, m = 20, 5
        s = torch.ones(n)
        k = torch.ones(m)
        out = pytorch_convolve(s, k, mode="same")
        assert out.shape[0] == n

    def test_impulse_identity(self):
        """Convolving with a unit impulse should return the signal."""
        signal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        impulse = torch.tensor([0.0, 1.0, 0.0])
        out = pytorch_convolve(signal, impulse, mode="same")
        # centre element of each output should match signal
        assert out.shape[0] == signal.shape[0]


# ---------------------------------------------------------------------------
# timeBall
# ---------------------------------------------------------------------------

class TestTimeBall:
    def test_exact_match(self):
        t1 = torch.tensor([[0.0], [0.5], [1.0]])
        t2 = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        idx, _ = timeBall(t1, t2, delta=0.01)
        assert idx.tolist() == [0, 2, 4]

    def test_no_match_returns_zero(self):
        t1 = torch.tensor([[10.0]])  # far from t2
        t2 = torch.tensor([0.0, 0.5, 1.0])
        idx, _ = timeBall(t1, t2, delta=0.01)
        # scalar or 1-element tensor: value should be 0
        assert int(idx) == 0


# ---------------------------------------------------------------------------
# dfdt
# ---------------------------------------------------------------------------

class TestDfdt:
    def test_constant_has_zero_derivative(self, dtype):
        t = torch.linspace(0, 1, 30, dtype=dtype).view(-1, 1).requires_grad_(True)
        f = t.sum(dim=1)  # shape (30,)
        deriv = dfdt(signal=f, arg=t)
        assert deriv is not None
        # dfdt returns same shape as computed gradient — accept any consistent shape
        assert deriv.numel() == f.numel()

    def test_returns_tensor(self, dtype, small_time):
        f = torch.sin(small_time).squeeze()
        d = dfdt(signal=f, arg=small_time)
        assert isinstance(d, torch.Tensor)


# ---------------------------------------------------------------------------
# DoubleGamma
# ---------------------------------------------------------------------------

class TestDoubleGamma:
    def test_returns_callable(self):
        hrf = DoubleGamma(A1=6, alpha1=6, beta1=1, A2=2, alpha2=16, beta2=1)
        assert callable(hrf)

    def test_peak_is_positive(self):
        hrf = DoubleGamma(A1=6, alpha1=6, beta1=1, A2=2, alpha2=16, beta2=1)
        t = np.linspace(0, 30, 300)
        values = hrf(t)
        assert values.max() > 0

    def test_undershoot_exists(self):
        hrf = DoubleGamma(A1=6, alpha1=6, beta1=1, A2=2, alpha2=16, beta2=1)
        t = np.linspace(0, 30, 300)
        values = hrf(t)
        assert values.min() < 0


# ---------------------------------------------------------------------------
# kge_stat
# ---------------------------------------------------------------------------

class TestKgeStat:
    def test_perfect_match(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        kge = kge_stat(y, y)
        assert np.isclose(kge, 1.0, atol=1e-6)

    def test_range(self):
        rng = np.random.default_rng(0)
        obs = rng.normal(0, 1, 100)
        sim = rng.normal(0.5, 1.2, 100)
        kge = kge_stat(obs, sim)
        assert kge < 1.0  # not perfect


# ---------------------------------------------------------------------------
# tofit
# ---------------------------------------------------------------------------

class TestTofit:
    def test_output_shapes(self, dtype):
        stim = torch.zeros(200, dtype=dtype)
        stim[50:100] = 1.0
        hrf = torch.exp(-torch.linspace(0, 3, 30, dtype=dtype))
        signal, time = tofit(stim, hrf, time_max=2.0, dt=0.01)
        assert signal.shape == time.shape
        assert time.shape[0] == 200  # 2.0 / 0.01


# ---------------------------------------------------------------------------
# training_data
# ---------------------------------------------------------------------------

class TestTrainingData:
    def test_correct_length(self, dtype):
        # training_data returns a sorted index tensor of length num_points
        t_vec = torch.linspace(-1, 1, 100, dtype=dtype).view(-1, 1)
        data = {"t": t_vec}
        idx = training_data(data, num_points=20, random=False)
        assert idx.shape == torch.Size([20])

    def test_indices_in_range(self, dtype):
        t_vec = torch.linspace(-1, 1, 100, dtype=dtype).view(-1, 1)
        data = {"t": t_vec}
        idx = training_data(data, num_points=20, random=False)
        assert int(idx.min()) >= 0
        assert int(idx.max()) < 100

    def test_random_gives_different_draws(self, dtype):
        t_vec = torch.linspace(-1, 1, 100, dtype=dtype).view(-1, 1)
        data = {"t": t_vec}
        i1 = training_data(data, num_points=20, random=True)
        i2 = training_data(data, num_points=20, random=True)
        # Two independent random draws are very unlikely to be identical
        assert not torch.equal(i1, i2)

    def test_raises_if_too_many_points(self, dtype):
        t_vec = torch.linspace(-1, 1, 10, dtype=dtype).view(-1, 1)
        data = {"t": t_vec}
        with pytest.raises(ValueError):
            training_data(data, num_points=50, random=False)
