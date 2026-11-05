"""Tests for balloonlib.MOO_Loss against the published update rules."""
import math
import numpy as np
import pytest
import torch
import torch.nn as nn
from balloonlib.MOO_Loss import (InverseGradNorm, LRAnnealing, ReLoBRaLo, SoftAdapt,
                                 backward_fd_coefficients, softadapt_weights,
                                 per_loss_flat_gradients)


def _ld(**v):
    d = {k: torch.tensor(float(x)) for k, x in v.items()}
    d["total"] = torch.tensor(123.0)
    return d


# ---------------------------------------------------------------- SoftAdapt
def test_fd_two_point_and_polynomial_exactness():
    assert np.allclose(backward_fd_coefficients(2), [-1, 1])
    p = np.polynomial.Polynomial([0.3, -1.0, 2.0, 0.5, -0.2])           # degree 4
    x = np.arange(5.0)
    assert abs(backward_fd_coefficients(5) @ p(x) - p.deriv()(4.0)) < 1e-8


def test_softadapt_eq3_eq4_and_normalised():
    h = {"a": [1.0, 1.2], "b": [2.0, 1.5]}
    s = np.array([0.2, -0.5]); f = np.array([1.1, 1.75]); b = 1.5
    e3 = np.exp(b * s) / np.exp(b * s).sum()
    assert np.allclose(list(softadapt_weights(h, b).values()), e3, atol=1e-7)
    e4 = f * e3 / (f * e3).sum()
    assert np.allclose(list(softadapt_weights(h, b, ("loss_weighted",)).values()), e4, atol=1e-7)
    ns = s / np.abs(s).sum(); en = np.exp(b * ns) / np.exp(b * ns).sum()
    w, d = softadapt_weights(h, b, ("normalised",), return_diagnostics=True)
    assert np.allclose(list(w.values()), en, atol=1e-7) and np.allclose(list(d["ns"].values()), ns, atol=1e-7)


def test_softadapt_class_schedule_and_total_ignored():
    sa = SoftAdapt(n=3, beta=0.0, every=2, warmup=4)
    lw = {k: [1.0] for k in ("ode", "bold", "ic", "other")}
    ups = [i for i in range(10) if sa(_ld(ode=1 / (i + 1), bold=.5, ic=.1 * i, other=.2), lw, i) is not None]
    assert ups == [4, 6, 8] and all(len(v) == 4 for v in lw.values())
    assert all(abs(v[-1] - 1.0) < 1e-6 for v in lw.values())               # beta=0 -> all ones


# ---------------------------------------------------------------- ReLoBRaLo
def _bal(L, Lref, T):
    z = np.exp(L / (T * Lref)); return len(L) * z / z.sum()


def test_relobralo_matches_eq11_without_lookback():
    rb = ReLoBRaLo(alpha=0.7, temperature=0.5, expected_rho=1.0)       # rho = 1 always
    L0, L1, L2 = np.array([1.0, 2.0, 4.0]), np.array([0.5, 1.9, 4.2]), np.array([0.4, 1.5, 4.1])
    rb(_ld(a=L0[0], b=L0[1], c=L0[2]), None, 0)                            # records L(0)
    w1 = rb(_ld(a=L1[0], b=L1[1], c=L1[2]), None, 1)
    lam1 = 0.7 * np.ones(3) + 0.3 * _bal(L1, L0, 0.5)
    assert np.allclose(list(w1.values()), lam1)
    w2 = rb(_ld(a=L2[0], b=L2[1], c=L2[2]), None, 2)
    lam2 = 0.7 * lam1 + 0.3 * _bal(L2, L1, 0.5)
    assert np.allclose(list(w2.values()), lam2)


def test_relobralo_full_lookback_uses_initial_losses():
    rb = ReLoBRaLo(alpha=1.0, temperature=0.5, expected_rho=0.0)       # rho = 0 always
    rb(_ld(a=1.0, b=2.0), None, 0); rb(_ld(a=0.9, b=1.0), None, 1)
    w = rb(_ld(a=0.8, b=0.5), None, 2)
    assert np.allclose(list(w.values()), _bal(np.array([0.8, 0.5]), np.array([1.0, 2.0]), 0.5))


def test_relobralo_properties():
    rb = ReLoBRaLo(alpha=0.0, temperature=1e-4, expected_rho=1.0)      # no overflow at tiny T
    rb(_ld(a=1.0, b=1.0), None, 0)
    w = rb(_ld(a=1.0, b=0.1), None, 1)                                     # 'a' made no progress
    assert all(math.isfinite(v) for v in w.values()) and w["a"] > 1.99 and abs(sum(w.values()) - 2) < 1e-9
    big_T = ReLoBRaLo(alpha=0.0, temperature=1e6, expected_rho=1.0)
    big_T(_ld(a=1.0, b=1.0), None, 0)
    assert np.allclose(list(big_T(_ld(a=0.3, b=5.0), None, 1).values()), 1.0, atol=1e-5)


def test_relobralo_seed_reproducible():
    seq = [_ld(a=1 / (i + 1), b=1 / (i + 2) ** 0.5) for i in range(30)]
    runs = []
    for _ in range(2):
        rb = ReLoBRaLo(alpha=0.9, expected_rho=0.5, seed=7)
        runs.append([rb(d, None, i) for i, d in enumerate(seq)][-1])
    assert runs[0] == runs[1]


# ---------------------------------------------------------------- gradient-based
class _Lin(nn.Module):
    def __init__(self):
        super().__init__(); self.w = nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))


def _two_losses(m):
    a = torch.tensor([1.0, 0.0, 0.0]); b = torch.tensor([0.0, 3.0, 4.0])
    return {"ode": 0.5 * (m.w @ a) ** 2, "bold": (m.w @ b) ** 2, "total": torch.tensor(0.0)}


def test_inverse_gradnorm_equalises_weighted_gradient_norms():
    m = _Lin(); ld = _two_losses(m)
    w = InverseGradNorm(every=1)(ld, None, 0, model=m)
    g = per_loss_flat_gradients(m, ld, ["ode", "bold"])
    n = {k: float(g[k].norm()) for k in g}
    assert np.isclose(w["ode"], (n["ode"] + n["bold"]) / n["ode"])
    assert np.isclose(w["ode"] * n["ode"], w["bold"] * n["bold"])


@pytest.mark.parametrize("ref", ["ode", "bold"])
def test_lr_annealing_eq40_41_and_reference(ref):
    m = _Lin(); ld = _two_losses(m)
    g = per_loss_flat_gradients(m, ld, ["ode", "bold"])
    other = "bold" if ref == "ode" else "ode"
    lam_hat = float(g[ref].abs().max() / g[other].abs().mean())
    lr = LRAnnealing(ref=ref, every=1, alpha=0.9)
    w = lr(ld, None, 0, model=m)
    assert w[ref] == 1.0 and np.isclose(w[other], 0.1 * 1.0 + 0.9 * lam_hat)      # starts from lam = 1
    w2 = lr(ld, None, 1, model=m)
    assert np.isclose(w2[other], 0.1 * w[other] + 0.9 * lam_hat)
    w_mean = LRAnnealing(ref=ref, every=1, alpha=1.0, stat="mean")(ld, None, 0, model=m)
    assert np.isclose(w_mean[other], float(g[ref].abs().mean() / g[other].abs().mean()))


def test_gradient_balancers_leave_graph_usable_and_need_model():
    m = _Lin(); ld = _two_losses(m); total = ld["ode"] + ld["bold"]; ld["total"] = total
    LRAnnealing(every=1)(ld, None, 0, model=m)
    total.backward()                                                       # graph still alive
    assert m.w.grad is not None
    with pytest.raises(ValueError):
        InverseGradNorm(every=1)(_two_losses(_Lin()), None, 0)
    with pytest.raises(KeyError):
        LRAnnealing(ref="xyz", every=1)(_two_losses(m), None, 0, model=m)


# ---------------------------------------------------------------- compatibility
def test_legacy_functions_still_importable_from_training():
    from balloonlib.training import loss_reweight_paranoid, compute_per_loss_gradients  # noqa: F401
    m = _Lin(); ld = _two_losses(m)
    _, norms = compute_per_loss_gradients(m, ld)
    assert set(norms) == {"ode", "bold"}                                   # "total" skipped
