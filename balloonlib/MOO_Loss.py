"""Multi-objective loss balancing for the BalloonLib PINN.

Four methods, all acting on the component losses of ``training.loss``
("ode", "bold", "ic", "other"):

    SoftAdapt        Heydari, Thompson & Mehmood 2019, arXiv:1912.12355
    ReLoBRaLo        Bischof & Kraus 2021, arXiv:2110.09813 (eq. 11)
    InverseGradNorm  w_k = sum_j ||grad L_j|| / ||grad L_k||  (compute_per_loss_gradients)
    LRAnnealing      Wang, Teng & Perdikaris 2021, SIAM J. Sci. Comput. 43(5):A3055,
                     doi:10.1137/20M1318043; preprint arXiv:2001.04536v1, Algorithm 1,
                     eqs. 40-41.  Selectable reference term (default "ode", e.g. "bold")

Use one of the classes through ``training.train(..., balancer=...)``; ``train``
then fixes every ``amp`` at 1 and skips its dynamic amp rule.

The legacy stateless helpers ``loss_reweight_paranoid`` and
``compute_per_loss_gradients`` (formerly in training.py) live here as well and
are re-exported by ``training`` for backward compatibility.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List

import numpy as np
import torch

from balloonlib import config

__all__ = [
    "SoftAdapt", "ReLoBRaLo", "InverseGradNorm", "LRAnnealing",
    "softadapt_weights", "backward_fd_coefficients", "per_loss_flat_gradients",
    "loss_reweight_paranoid", "compute_per_loss_gradients",
]


# ---------------------------------------------------------------------------
# Adaptive loss reweighting
# ---------------------------------------------------------------------------


@torch.no_grad()
def loss_reweight_paranoid(
    loss_dict: Dict[str, torch.Tensor],
    loss_trace: Dict[str, List[float]],
    weights_history: Dict[str, List[float]],
    keys_to_skip: List[str] = [],
    temperature: float = 0.1,
    alpha: float = 0.999,
    rho: float = 0.95,
    device=None,
    eps: float = 1e-12,
    validate: bool = True,
) -> Dict[str, List[float]]:
    """Compute adaptive loss weights with Bernoulli history retention.

    Legacy, stateless form kept for backward compatibility.  It omits the factor
    m of Bischof & Kraus eq. 11 and its softmax is not max-shifted; use
    :class:`ReLoBRaLo` for the published rule.

    Implements the adaptive balancing algorithm from Bischof & Kraus
    (2021, eq. 11).  Combines softmax-based weights scaled by both
    *initial* and *recent* loss ratios, blended via exponential moving
    average and stochastic history selection.

    Parameters
    ----------
    loss_dict : dict of {str: torch.Tensor}
        Current loss values per component.
    loss_trace : dict of {str: list of float}
        Full loss history per component.
    weights_history : dict of {str: list of float}
        Weight history per component; new weights are appended in-place.
    keys_to_skip : list of str
        Components excluded from reweighting (always get weight 1.0).
    temperature : float
        Softmax temperature *T*.  ``T → inf`` yields uniform weights;
        ``T → 0`` concentrates weight on the slowest-progressing term.
    alpha : float
        EMA blending coefficient (higher = more past memory).
    rho : float
        Bernoulli probability for choosing the previous weight over the
        softmax-initial weight.
    device : str, torch.device, or None
        Computation device.  ``None`` uses :data:`balloonlib.config.device`.
    eps : float
        Clamping floor for numerical stability.
    validate : bool
        If ``True``, run paranoid post-hoc assertions.

    Returns
    -------
    dict of {str: list of float}
        The *same* ``weights_history`` object with new values appended.
    """
    if device is None:
        device = config.device

    if validate:
        original_keys = set(weights_history.keys()) - set(keys_to_skip)
        original_lengths = {k: len(weights_history[k]) for k in original_keys}

    keys = sorted(
        [
            k
            for k in weights_history.keys()
            if k in loss_dict and k in loss_trace and k not in keys_to_skip
        ]
    )

    if not keys:
        return weights_history

    current = torch.stack([loss_dict[k] for k in keys]).requires_grad_(False)
    initial = torch.tensor([loss_trace[k][0] for k in keys], device=device)

    prev_loss = torch.tensor(
        [loss_trace[k][-2] if len(loss_trace[k]) > 1 else loss_trace[k][0] for k in keys],
        device=device,
    )

    prev_weights = torch.tensor([weights_history[k][-1] for k in keys], device=device)

    logits_hist = current / (temperature * torch.clamp(initial, min=eps))
    logits_recent = current / (temperature * torch.clamp(prev_loss, min=eps))

    weights_hist = torch.softmax(logits_hist, dim=0)
    weights_recent = torch.softmax(logits_recent, dim=0)

    rho_sample = torch.bernoulli(torch.tensor(rho, device=device))
    hist_weights = rho_sample * prev_weights + (1 - rho_sample) * weights_hist
    new_weights = alpha * hist_weights + (1 - alpha) * weights_recent

    for k, w in zip(keys, new_weights.tolist()):
        weights_history[k].append(w)

    for k in keys_to_skip:
        weights_history[k].append(1.0)

    if validate:
        assert set(weights_history.keys()) == original_keys.union(keys_to_skip), "Keys changed!"
        for k in original_keys:
            if k != "total" and k in keys:
                assert len(weights_history[k]) == original_lengths[k] + 1, (
                    f"History length mismatch for {k}"
                )
            elif k != "total":
                assert len(weights_history[k]) == original_lengths[k], (
                    f"Non-updated key {k} changed length!"
                )

    return weights_history


# ---------------------------------------------------------------------------
# Per-loss gradient analysis
# ---------------------------------------------------------------------------


def compute_per_loss_gradients(
    model,
    loss_dict,
    every_n: int = 1,
    current_iter: int = 0,
    return_norms: bool = True,
    skip=("total",),
):
    """Compute per-component gradient norms w.r.t. model parameters.

    Keys in ``skip`` (default: "total") are ignored - including the total would
    count its gradient in the normalising sum.  Stateless legacy helper; for use
    inside ``train`` see :class:`InverseGradNorm`.

    For each entry in ``loss_dict``, back-propagates through the graph
    (with ``retain_graph=True``) to obtain the flattened gradient vector
    and its L2 norm.

    Parameters
    ----------
    model : torch.nn.Module
        The PINN model whose parameters are differentiated.
    loss_dict : dict of {str: torch.Tensor}
        Named scalar loss tensors.
    every_n : int
        Only compute when ``current_iter % every_n == 0`` to amortise
        the cost of ``retain_graph``.
    current_iter : int
        Current training iteration index.
    return_norms : bool
        If ``True``, also return normalised inverse-norm weights.

    Returns
    -------
    grads : dict of {str: torch.Tensor} or None
        Flattened gradient vectors per loss component.
    norms : dict of {str: float} or None
        Inverse-normalised L2 norms (``total / norm_i``).
        ``None`` on skipped iterations.
    """
    if current_iter % every_n != 0:
        return None, None

    params = list(model.parameters())
    grads = {}
    norms = {}

    for name, w_loss in loss_dict.items():
        if name in skip:
            continue
        gradients = torch.autograd.grad(
            outputs=w_loss,
            inputs=params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        flat = torch.cat(
            [
                g.detach().reshape(-1)
                if g is not None
                else torch.zeros(p.numel(), device=p.device, dtype=p.dtype)
                for g, p in zip(gradients, params)
            ]
        )
        grads[name] = flat

    if return_norms:
        norms = {key: value.norm(2).item() for key, value in grads.items()}
        total = sum(norms.values())
        norms = {key: (total / value) for key, value in norms.items()}

    return grads, norms


# ===========================================================================
# Stateful balancers for ``training.train(..., balancer=...)``
# ===========================================================================
#
# Common protocol
# ---------------
#   balancer(loss_dict, loss_weights, i, model=None) -> dict | None
#
#   * ``loss_dict`` is the dict returned by ``training.loss`` at iteration i
#     (keys "ode", "bold", "ic", "other", "total"; "total" is always ignored).
#     The component losses are UNWEIGHTED - the papers' L_i.
#   * When an update is due, the new weights are APPENDED to
#     ``loss_weights[k]``; ``training.loss`` reads ``loss_weights[k][-1]``, so
#     they act from the next iteration.  Returns the new {key: weight} or None.
#   * Gradient-based balancers (InverseGradNorm, LRAnnealing) need ``model``
#     and must be called BEFORE ``total.backward()``: they differentiate each
#     term with ``retain_graph=True``.  ``train`` calls every balancer there.
#   * ``state()`` returns a plain, picklable dict including the weight history.
#   * ``keys=`` selects which terms are re-weighted (default: all but "total");
#     ``every``, ``warmup``, ``prior`` are common too - see ``_COMMON_ARGS_DOC``.
#
# Units: every method below rescales whole loss terms.  Your terms are on very
# different scales (ODE residual sum over 30 causal segments, %^2 BOLD error,
# IC, "other"), which matters for the loss-statistics methods (SoftAdapt
# without normalisation, ReLoBRaLo uses ratios and is scale-free per term).


_COMMON_ARGS_DOC = """
    Common arguments (all balancers)
    --------------------------------
    keys : list of str or None
        Which loss terms the balancer re-weights.  ``None`` (default) means every
        term in ``loss_dict`` except ``"total"``; with ``training.loss`` that is
        ``["ode", "bold", "ic", "other"]``.  Pass a subset, e.g.
        ``keys=["ode", "bold"]``, to leave the other terms alone: no new weight is
        appended for them, so they keep their current weight
        ``loss_weights[k][-1]`` for the whole run.  Note that the balanced weights
        are on the method's own scale (e.g. ~1 for SoftAdapt/ReLoBRaLo, ratios of
        gradient norms for InverseGradNorm/LRAnnealing), so the fixed weights are
        not rescaled to match.  ``"total"`` raises ValueError; a key missing from
        ``loss_dict`` raises KeyError.  For SoftAdapt the weights sum to ``sum_to``
        over the balanced keys only; for LRAnnealing ``ref`` must be in ``keys``.
    every : int
        Update interval in iterations (>= 1).
    warmup : int
        First iteration at which an update may happen; updates then occur at
        ``i = warmup, warmup + every, ...``.  Before that the weights in
        ``loss_weights`` are not touched.
    prior : dict or None
        Fixed factors ``{key: factor}`` multiplied into the balanced weights at
        every update (missing keys -> 1.0), e.g. to keep a preferred ratio.
"""


class _Balancer:
    """Shared bookkeeping: key selection, schedule, optional prior, history.
""" + _COMMON_ARGS_DOC

    def __init__(self, keys=None, every: int = 1, warmup: int = 0,
                 prior: dict | None = None):
        if every < 1:
            raise ValueError("every must be >= 1")
        self.keys = list(keys) if keys is not None else None
        self.every, self.warmup, self.prior = int(every), int(warmup), prior
        self.history: list[dict] = []
        self._calls = 0

    def _keys(self, loss_dict) -> list:
        keys = self.keys or [k for k in loss_dict if k != "total"]
        if "total" in keys:
            raise ValueError("'total' cannot be balanced")
        missing = [k for k in keys if k not in loss_dict]
        if missing:
            raise KeyError(f"keys not in loss_dict: {missing}")
        return keys

    def _due(self, i: int) -> bool:
        return i >= self.warmup and (i - self.warmup) % self.every == 0

    def _commit(self, w: dict, loss_weights, i: int, **diag) -> dict:
        if self.prior:
            w = {k: w[k] * float(self.prior.get(k, 1.0)) for k in w}
        if loss_weights is not None:
            for k, v in w.items():
                loss_weights[k].append(float(v))
        self.history.append({"iter": int(i), "weights": {k: float(v) for k, v in w.items()}, **diag})
        return w

    def _base_state(self) -> dict:
        return {"keys": self.keys, "every": self.every, "warmup": self.warmup,
                "prior": self.prior, "history": list(self.history)}

    @staticmethod
    def _value(v) -> float:
        return float(v.detach().item()) if hasattr(v, "detach") else float(v)


def _stable_softmax(x: np.ndarray, eps: float = 0.0) -> np.ndarray:
    z = np.exp(x - x.max())
    return z / (z.sum() + eps)


# ---------------------------------------------------------------------------
# 1. SoftAdapt  (Heydari, Thompson & Mehmood 2019, arXiv:1912.12355)
# ---------------------------------------------------------------------------

def backward_fd_coefficients(n: int, order: int | None = None) -> np.ndarray:
    """Backward finite-difference stencil for df/di at the newest of n points
    (oldest -> newest).  Default ``order = n - 1`` = SoftAdapt Algorithm 1, line 3
    ("rate of change, up to (n-1)th order accurate").  Exact for polynomials of
    degree <= order.  n = 2 gives [-1, 1]."""
    order = n - 1 if order is None else order
    if not 1 <= order <= n - 1:
        raise ValueError(f"order must be in [1, n-1]; got order={order}, n={n}")
    m = order + 1
    x = np.arange(-(m - 1), 1, dtype=float)
    A = np.vander(x, m, increasing=True).T
    rhs = np.zeros(m); rhs[1] = 1.0
    return np.concatenate([np.zeros(n - m), np.linalg.solve(A, rhs)])


def softadapt_weights(history: dict, beta: float = 0.1, variants=(),
                      order: int | None = None, eps: float = 1e-8,
                      return_diagnostics: bool = False):
    """SoftAdapt weights (sum ~1) from per-term histories of equal length n >= 2.

    Algorithm 1:  s = rate of change (line 3); f = mean of the last n (line 4);
    "normalised": ns = s / (sum|s| + eps) (line 6); alpha = softmax(beta * ns or s)
    with the max shifted out (lines 7/9); "loss_weighted": alpha <- f alpha /
    (sum f alpha + eps) (line 12).

    Note: the authors' reference package normalises by sum(s) (signed, no eps);
    this follows the paper, sum(|s|) + eps.
    """
    keys = list(history)
    H = np.array([history[k] for k in keys], dtype=float)
    if H.shape[1] < 2:
        raise ValueError("SoftAdapt needs at least 2 stored values per loss")
    s = H @ backward_fd_coefficients(H.shape[1], order)
    f = H.mean(axis=1)
    variants = {v.lower().replace(" ", "_") for v in variants}
    unknown = variants - {"normalised", "loss_weighted"}
    if unknown:
        raise ValueError(f"unknown SoftAdapt variant(s): {sorted(unknown)}")
    ns = s / (np.abs(s).sum() + eps) if "normalised" in variants else s
    a = _stable_softmax(beta * ns, eps)
    if "loss_weighted" in variants:
        a = f * a / ((f * a).sum() + eps)
    w = dict(zip(keys, a.tolist()))
    if return_diagnostics:
        return w, {"s": dict(zip(keys, s.tolist())), "ns": dict(zip(keys, ns.tolist())),
                   "f": dict(zip(keys, f.tolist()))}
    return w


class SoftAdapt(_Balancer):
    """SoftAdapt (loss statistics only; no extra backward passes).

    n : stored values per term.  beta : temperature (paper default 0.1; with
    "normalised", ns lies in [-1, 1], so beta must be O(1-10) for the weights to
    move).  variants : subset of {"normalised", "loss_weighted"}.  order : finite
    difference order for s (default n-1; 1 = two-point difference, less noisy).
    sum_to : weights rescaled to sum to this (default: number of terms, so beta=0
    reproduces all-ones weights; paper: 1.0).  prior : fixed factors applied after.
    """

    def __init__(self, keys=None, n: int = 5, beta: float = 0.1, variants=("normalised",),
                 order: int | None = None, every: int = 1, warmup: int = 0,
                 sum_to: float | None = None, prior: dict | None = None, eps: float = 1e-8):
        super().__init__(keys, every, warmup, prior)
        if n < 2:
            raise ValueError("n must be >= 2")
        backward_fd_coefficients(n, order)
        self.n, self.beta, self.variants, self.order = n, beta, tuple(variants), order
        self.sum_to, self.eps = sum_to, eps
        self._hist: dict = {}

    def __call__(self, loss_dict, loss_weights=None, i=None, model=None):
        i = self._calls if i is None else i
        self._calls += 1
        keys = self._keys(loss_dict)
        for k in keys:
            self._hist.setdefault(k, deque(maxlen=self.n)).append(self._value(loss_dict[k]))
        if not self._due(i) or any(len(self._hist[k]) < self.n for k in keys):
            return None
        a, diag = softadapt_weights({k: list(self._hist[k]) for k in keys}, self.beta,
                                    self.variants, self.order, self.eps, return_diagnostics=True)
        total = self.sum_to if self.sum_to is not None else float(len(keys))
        norm = sum(a.values())
        return self._commit({k: total * a[k] / norm for k in keys}, loss_weights, i, **diag)

    def state(self) -> dict:
        return {"method": "SoftAdapt (Heydari et al. 2019)", "n": self.n, "beta": self.beta,
                "variants": list(self.variants), "order": self.order, "sum_to": self.sum_to,
                **self._base_state()}


# ---------------------------------------------------------------------------
# 2. ReLoBRaLo  (Bischof & Kraus 2021, arXiv:2110.09813, eq. 11)
# ---------------------------------------------------------------------------

class ReLoBRaLo(_Balancer):
    """Relative Loss Balancing with Random Lookback (loss statistics only).

        lam_bal(t, t') = m * softmax_i( L_i(t) / (T L_i(t')) )
        lam_hist(t)    = rho * lam(t-1) + (1 - rho) * lam_bal(t, 0)
        lam(t)         = alpha * lam_hist(t) + (1 - alpha) * lam_bal(t, t-1)
        rho ~ Bernoulli(expected_rho)   (one draw per update, shared by all terms)

    Defaults alpha = 0.999, T = 0.1, expected_rho = 0.999 (paper Sec. IX uses
    alpha in [0.9, 0.999], T in [1e-4, 1], E[rho] in [0.99, 1]).  lam(0) = 1.
    "t-1" is the previous UPDATE when every > 1.  The softmax is max-shifted, so
    small T cannot overflow.  ``seed`` makes the lookback draws reproducible.

    Differences from the legacy ``loss_reweight_paranoid``: the factor m is
    included (weights sum to m, so lam = 1 is the neutral value), and the softmax
    is numerically stable.
    """

    def __init__(self, keys=None, alpha: float = 0.999, temperature: float = 0.1,
                 expected_rho: float = 0.999, every: int = 1, warmup: int = 0,
                 prior: dict | None = None, seed: int | None = None, eps: float = 1e-12):
        super().__init__(keys, every, warmup, prior)
        if not 0.0 <= alpha <= 1.0 or not 0.0 <= expected_rho <= 1.0 or temperature <= 0:
            raise ValueError("need 0<=alpha<=1, 0<=expected_rho<=1, temperature>0")
        self.alpha, self.T, self.expected_rho, self.eps = alpha, temperature, expected_rho, eps
        self._rng = np.random.default_rng(seed)
        self._L0 = self._Lprev = self._lam = None

    def _bal(self, L, Lref):
        m = L.size
        return m * _stable_softmax(L / (self.T * np.maximum(Lref, self.eps)))

    def __call__(self, loss_dict, loss_weights=None, i=None, model=None):
        i = self._calls if i is None else i
        self._calls += 1
        keys = self._keys(loss_dict)
        L = np.array([self._value(loss_dict[k]) for k in keys])
        if self._L0 is None:                                    # first call: record L(0)
            self._L0, self._Lprev, self._lam = L.copy(), L.copy(), np.ones_like(L)
            return None
        if not self._due(i):
            return None
        rho = float(self._rng.random() < self.expected_rho)
        lam_hist = rho * self._lam + (1.0 - rho) * self._bal(L, self._L0)
        self._lam = self.alpha * lam_hist + (1.0 - self.alpha) * self._bal(L, self._Lprev)
        self._Lprev = L.copy()
        return self._commit(dict(zip(keys, self._lam.tolist())), loss_weights, i, rho=rho)

    def state(self) -> dict:
        return {"method": "ReLoBRaLo (Bischof & Kraus 2021)", "alpha": self.alpha,
                "temperature": self.T, "expected_rho": self.expected_rho, **self._base_state()}


# ---------------------------------------------------------------------------
# 3./4. Gradient-statistics balancers
# ---------------------------------------------------------------------------

def per_loss_flat_gradients(model, loss_dict, keys, params=None) -> dict:
    """Flattened gradient of each loss_dict[k] w.r.t. the trainable parameters,
    with ``retain_graph=True`` so ``total.backward()`` can follow.  Unused
    parameters contribute zeros."""
    params = [p for p in (params if params is not None else model.parameters()) if p.requires_grad]
    out = {}
    for k in keys:
        g = torch.autograd.grad(loss_dict[k], params, retain_graph=True, allow_unused=True)
        out[k] = torch.cat([(gi if gi is not None else torch.zeros_like(p)).detach().reshape(-1)
                            for gi, p in zip(g, params)])
    return out


class InverseGradNorm(_Balancer):
    """Inverse gradient-norm balancing (the rule of ``compute_per_loss_gradients``):

        w_k = sum_j ||grad L_j||_2 / ||grad L_k||_2

    so every weighted term pulls with the same gradient norm.  ``alpha`` is the
    weight of the NEW estimate in an exponential moving average (1.0 = none, the
    legacy behaviour).  Costs one extra backward pass per term on update steps,
    so use ``every`` >= 10.

    params : iterable of torch.nn.Parameter or None
        Parameters the per-term gradients are taken with respect to (default:
        all trainable ``model.parameters()``).  A subset, e.g. the last layer,
        makes each update much cheaper and changes the gradient statistics the
        weights are computed from.  Stored as a list, so ``model.parameters()``
        (a generator) is safe.  eps : floor on the gradient norm.
    """

    def __init__(self, keys=None, alpha: float = 1.0, every: int = 10, warmup: int = 0,
                 prior: dict | None = None, params=None, eps: float = 1e-12):
        super().__init__(keys, every, warmup, prior)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha, self.eps = alpha, eps
        self.params = list(params) if params is not None else None   # a generator would be used up after one update
        self._w = None

    def __call__(self, loss_dict, loss_weights=None, i=None, model=None):
        i = self._calls if i is None else i
        self._calls += 1
        if not self._due(i):
            return None
        if model is None:
            raise ValueError("InverseGradNorm needs model=")
        keys = self._keys(loss_dict)
        g = per_loss_flat_gradients(model, loss_dict, keys, self.params)
        norms = {k: float(g[k].norm()) for k in keys}
        S = sum(norms.values())
        w_hat = {k: S / max(norms[k], self.eps) for k in keys}
        self._w = w_hat if self._w is None else {k: (1 - self.alpha) * self._w[k] + self.alpha * w_hat[k] for k in keys}
        return self._commit(dict(self._w), loss_weights, i, grad_norms=norms)

    def state(self) -> dict:
        return {"method": "inverse gradient norm", "alpha": self.alpha, **self._base_state()}


class LRAnnealing(_Balancer):
    """Learning-rate annealing, with a selectable reference term.

    Wang, S., Teng, Y. & Perdikaris, P. (2021) Understanding and mitigating gradient
    flow pathologies in physics-informed neural networks.  SIAM J. Sci. Comput.
    43(5):A3055-A3081, doi:10.1137/20M1318043.  Equation numbers below refer to the
    preprint, arXiv:2001.04536v1 (Jan. 2020), Algorithm 1; the journal version
    numbers its equations differently.

        step (a), eq. 40:  lam_hat_k = stat_ref( grad L_ref ) / mean_theta | grad L_k |   (k != ref)
        step (b), eq. 41:  lam_k    <- (1 - alpha) lam_k + alpha lam_hat_k,  lam_k = 1 initially

    ref : the term whose weight stays fixed (paper: the PDE residual -> "ode";
        e.g. "bold" to scale everything relative to the data term).
    stat : "max" (paper: max_theta |grad L_ref|) or "mean" (mean |grad L_ref|).
        With ~1.7e5 parameters max/mean of the same gradient is ~1e4 here, so
        "max" makes every non-reference weight huge; "mean" balances the mean
        gradient magnitudes instead.  "max" is the published rule.
    alpha : weight of the NEW estimate, as printed in eq. 41 (paper: 0.9).
        Bischof & Kraus (arXiv:2110.09813, eq. 6) restate it with the opposite
        convention; alpha = 0.1 here equals their alpha = 0.9.
    every : the paper updates every 10 iterations.
    ref_weight : weight assigned to the reference term (paper: 1).
    params : parameters the gradients are taken with respect to (default: all
        trainable ``model.parameters()``; the paper uses all).  A subset, e.g. the
        last layer, is much cheaper but changes max/mean |grad|.  Stored as a
        list, so passing ``model.parameters()`` (a generator) is safe.
    eps : floor on mean |grad L_k|.
    """

    def __init__(self, ref: str = "ode", keys=None, stat: str = "max", alpha: float = 0.9,
                 every: int = 10, warmup: int = 0, ref_weight: float = 1.0,
                 prior: dict | None = None, params=None, eps: float = 1e-12):
        super().__init__(keys, every, warmup, prior)
        if stat not in ("max", "mean"):
            raise ValueError("stat must be 'max' or 'mean'")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.ref, self.stat, self.alpha, self.ref_weight = ref, stat, alpha, float(ref_weight)
        self.eps = eps
        self.params = list(params) if params is not None else None   # a generator would be used up after one update
        self._lam = None

    def __call__(self, loss_dict, loss_weights=None, i=None, model=None):
        i = self._calls if i is None else i
        self._calls += 1
        if not self._due(i):
            return None
        if model is None:
            raise ValueError("LRAnnealing needs model=")
        keys = self._keys(loss_dict)
        if self.ref not in keys:
            raise KeyError(f"reference term '{self.ref}' not among balanced keys {keys}")
        g = per_loss_flat_gradients(model, loss_dict, keys, self.params)
        gref = g[self.ref].abs()
        num = float(gref.max() if self.stat == "max" else gref.mean())
        lam_hat = {k: num / max(float(g[k].abs().mean()), self.eps) for k in keys if k != self.ref}
        if self._lam is None:
            self._lam = {k: 1.0 for k in lam_hat}     # Algorithm 1 starts from lam_i = 1
        self._lam = {k: (1 - self.alpha) * self._lam[k] + self.alpha * lam_hat[k] for k in lam_hat}
        w = {self.ref: self.ref_weight, **{k: self.ref_weight * v for k, v in self._lam.items()}}
        return self._commit({k: w[k] for k in keys}, loss_weights, i, lam_hat=lam_hat)

    def state(self) -> dict:
        return {"method": "learning-rate annealing (Wang et al. 2021)", "ref": self.ref,
                "stat": self.stat, "alpha": self.alpha, "ref_weight": self.ref_weight,
                **self._base_state()}


# Make help(SoftAdapt) etc. show the common arguments too.
for _cls in (SoftAdapt, ReLoBRaLo, InverseGradNorm, LRAnnealing):
    _cls.__doc__ = (_cls.__doc__ or "") + _COMMON_ARGS_DOC
del _cls
