"""
balloon_latent_prior.py
=======================
Learnable latent Gaussian distribution over Balloon model parameters,
implemented with Pyro (pyro-ppl) and PyTorch.

Mathematical framework
----------------------
K parameters theta_i each live on a constrained support (a_i, b_i) or (0, inf).
We define a per-parameter bijection h_i : support_i -> R:

    log-transform   (for positive-reals):   h_i(theta_i) = log(theta_i)
    logit-transform (for bounded interval): h_i(theta_i) = log((theta_i - a_i) / (b_i - theta_i))

In the transformed (unconstrained) space the joint distribution is:

    eta = h(theta) ~ N(mu, Sigma),    Sigma = L L^T

where L is lower-triangular with positive diagonal (Cholesky factor).
Sampling proceeds via the non-centred reparameterisation:

    z   ~ N(0, I_K)
    eta = mu + L @ z          (reparameterised sample, differentiable w.r.t. mu, L)
    theta = h^{-1}(eta)       (push forward to constrained space)

Both mu and L are learnable nn.Parameters.
Pyro's PyroParam enforces the lower_cholesky constraint on L automatically.

Dependencies
------------
    pip install pyro-ppl torch

Author:  [your name]
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributions as td
import torch.distributions.constraints as constraints
import torch.distributions.transforms as T

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam


# ---------------------------------------------------------------------------
# Parameter specification type
# ---------------------------------------------------------------------------

ParamSpec = Dict[
    str,
    Union[
        Tuple[float, float],   # bounded interval  (a, b)
        Tuple[float, None],    # positive reals     (0, None)  -- log transform
        Literal["positive"],   # alias for (0, None)
    ],
]

# Canonical Balloon / BOLD model parameter specification
# Sources: Buxton et al. 1998, Friston et al. 2000, Khalidov et al. 2011,
#          Stephan et al. 2007, Heinzle et al. 2016
BALLOON_PARAM_SPEC: ParamSpec = {
    # --- Neural / haemodynamic coupling ---
    "epsilon_n":  (0.0, None),   # neural efficacy          s^{-1}   log-transform
    "kappa":      (0.0, None),   # signal decay rate        s^{-1}   log-transform
    "gamma":      (0.0, None),   # autoregulatory feedback  s^{-1}   log-transform
    "tau_0":      (0.0, None),   # haemodynamic transit time s       log-transform
    # --- Balloon / Windkessel parameters ---
    "alpha":      (0.1, 0.9),    # Grubb exponent           dimensionless
    "E_0":        (0.1, 0.9),    # baseline OEF             dimensionless
    "V_0":        (0.01, 0.10),  # resting CBV fraction     dimensionless
}

# Literature-informed prior means in *physical* space
# (used to initialise mu via the forward bijection)
BALLOON_PRIOR_MEAN: Dict[str, float] = {
    "epsilon_n": 0.54,   # Friston 2003 estimate
    "kappa":     0.65,
    "gamma":     0.41,
    "tau_0":     0.98,
    "alpha":     0.32,   # Grubb 1974
    "E_0":       0.40,   # Fox & Raichle 1986
    "V_0":       0.04,   # Ogawa / Buxton
}


# ---------------------------------------------------------------------------
# Per-parameter bijection helpers
# ---------------------------------------------------------------------------

def _make_transform(spec_entry: Union[Tuple, str]) -> T.Transform:
    """
    Returns a Transform mapping eta (unconstrained, real) -> theta (physical).
    Called as _transforms[i](eta) in rsample(); .inv direction used in log_prob().

    For a positive real (a=0, b=None):
        forward (eta -> theta):  theta = exp(eta)
        inverse (theta -> eta):  eta   = log(theta)

    For a bounded interval (a, b):
        forward (eta -> theta):  theta = a + (b-a) * sigmoid(eta)
        inverse (theta -> eta):  eta   = logit((theta - a) / (b - a))
    """
    if spec_entry == "positive" or (
        isinstance(spec_entry, tuple) and spec_entry[1] is None
    ):
        return T.ExpTransform()   # eta -> exp(eta) = theta

    a, b = spec_entry
    # Compose: shift+scale to (0,1), then logit
    # SigmoidTransform.inv  = logit  (maps (0,1) -> R)
    scale = b - a
    a_t = torch.tensor(float(a-0.1*scale))
    b_t = torch.tensor(float(b+0.1*scale))
    affine = T.AffineTransform(loc=a_t, scale=torch.tensor(scale))   # (0,1) -> (a^-,b^+)
    logistic = T.SigmoidTransform()                                   # R -> (0,1)
    # Full chain R -> (a,b):   theta = a + (b-a)*sigmoid(eta)
    # We want the *forward* direction theta -> R for computing mu_init,
    # and the *inverse* direction R -> theta for sampling.
    # ComposeTransform([logistic, affine]):  R -> (0,1) -> (a^-,b^+)
    return T.ComposeTransform([logistic, affine])   # inverse of the logit+rescale


def _forward_bijection(theta: float, spec_entry) -> float:
    """Scalar helper: physical parameter -> unconstrained real (for mu initialisation)."""
    if spec_entry == "positive" or (
        isinstance(spec_entry, tuple) and spec_entry[1] is None
    ):
        return math.log(theta)
    a, b = spec_entry
    scale = b - a
    xi = (theta - (a-0.1*scale)) / (b - a + 0.2*scale)
    xi = max(1e-6, min(1 - 1e-6, xi))   # numerical safety
    return math.log(xi / (1.0 - xi))


# ---------------------------------------------------------------------------
# Core module
# ---------------------------------------------------------------------------

class BalloonLatentPrior(PyroModule):
    """
    Learnable latent Gaussian distribution over Balloon model parameters.

    The module exposes:
        mu  : (K,)   learnable mean in unconstrained (transformed) space
        L   : (K, K) learnable lower-triangular Cholesky factor (positive diagonal)
                     enforced via PyroParam with constraints.lower_cholesky

    Sampling API:
        theta_dict = prior.sample()             # single draw, returns dict
        theta_dict = prior.rsample()            # differentiable reparameterised sample
        log_p      = prior.log_prob(theta_dict) # log prior probability (with Jacobian)

    Both mu and L are optimisable via any standard torch.optim optimiser.

    Parameters
    ----------
    param_spec : ParamSpec
        Ordered dict mapping parameter name -> support specification.
        Each value is either:
            (a, b)   : bounded interval, b > a, both finite -> logit transform
            (a, None): positive reals, a >= 0               -> log transform
            'positive': alias for (0, None)

    prior_mean : dict, optional
        Initial prior mean in *physical* space. Keys must match param_spec.
        If not provided, midpoints of intervals (or log of 1.0 for positive reals)
        are used.

    init_log_std : float
        Initial diagonal log-standard-deviation in unconstrained space.
        Default -1.0  (std ~ 0.37, i.e. moderate uncertainty).

    Notes
    -----
    The Cholesky factor L is stored as a PyroParam constrained to
    `torch.distributions.constraints.lower_cholesky`.  Pyro internally
    keeps an unconstrained surrogate and maps it through
    `transform_to(lower_cholesky)`, which:
        - zeros the strict upper triangle
        - applies softplus to the diagonal to enforce positivity
    This means gradient-based optimisers see an unconstrained space
    and the constraint is always satisfied without projection or clamping.
    """

    def __init__(
        self,
        param_spec: ParamSpec = BALLOON_PARAM_SPEC,
        prior_mean: Optional[Dict[str, float]] = None,
        init_log_std: float = -1.0,
        fixed_params: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        # Parameters held at a constant value throughout training (not part of the Gaussian)
        self.fixed_params: Dict[str, float] = dict(fixed_params or {})

        # A parameter cannot be both latent (in param_spec) and fixed — catch this early
        overlap = set(self.fixed_params) & set(param_spec)
        if overlap:
            raise ValueError(
                f"fixed_params keys {sorted(overlap)} also appear in param_spec. "
                "Remove them from param_spec or from fixed_params."
            )

        self.param_names: List[str] = list(param_spec.keys())
        self.param_spec:  ParamSpec = param_spec
        self.K: int = len(self.param_names)

        # Build per-parameter transforms  (R -> support_i)
        # transforms[i] maps eta_i -> theta_i
        self._transforms: List[T.Transform] = [
            _make_transform(param_spec[name]) for name in self.param_names
        ]

        # ---- Initialise mu in transformed (unconstrained) space ----
        if prior_mean is None:
            prior_mean = {}
        mu_init = torch.zeros(self.K)
        for i, name in enumerate(self.param_names):
            spec = param_spec[name]
            if name in prior_mean:
                mu_init[i] = _forward_bijection(prior_mean[name], spec)
            elif spec == "positive" or (isinstance(spec, tuple) and spec[1] is None):
                mu_init[i] = 0.0   # log(1.0) = 0
            else:
                a, b = spec
                midpoint = 0.5 * (a + b)
                mu_init[i] = _forward_bijection(midpoint, spec)

        # ---- Learnable mean (unconstrained) ----
        # PyroParam with no constraint = plain nn.Parameter
        self.mu: torch.Tensor = PyroParam(mu_init)

        # ---- Learnable Cholesky factor ----
        # Initialise as scaled identity: L = exp(init_log_std) * I
        init_std = math.exp(init_log_std)
        L_init = torch.eye(self.K) * init_std
        # PyroParam enforces lower_cholesky:
        #   - strict upper triangle forced to zero
        #   - diagonal forced positive via softplus reparameterisation
        self.L: torch.Tensor = PyroParam(
            L_init,
            constraint=constraints.lower_cholesky,
        )

    # ------------------------------------------------------------------
    # Covariance helpers
    # ------------------------------------------------------------------

    @property
    def covariance(self) -> torch.Tensor:
        """Full covariance matrix Sigma = L L^T.  Shape: (K, K)."""
        return self.L @ self.L.T

    @property
    def correlation(self) -> torch.Tensor:
        """Pearson correlation matrix derived from Sigma."""
        sigma = self.covariance
        std = torch.sqrt(torch.diag(sigma))
        return sigma / (std.unsqueeze(0) * std.unsqueeze(1))

    @property
    def std_in_transformed_space(self) -> torch.Tensor:
        """Marginal standard deviations in unconstrained space.  Shape: (K,)."""
        return torch.sqrt(torch.diag(self.covariance))

    # ------------------------------------------------------------------
    # Distribution object
    # ------------------------------------------------------------------

    def _base_distribution(self) -> td.MultivariateNormal:
        """MVN in unconstrained space with current mu and L."""
        return td.MultivariateNormal(loc=self.mu, scale_tril=self.L)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def rsample(
        self, n: int = 1
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Differentiable reparameterised sample from the latent Gaussian,
        pushed forward to the constrained physical parameter space.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        theta_dict : Dict[str, Tensor]
            Parameter samples in physical space.  Each value has shape (n,).
        eta : Tensor
            Samples in unconstrained space.  Shape: (n, K).
            Retained for diagnostics / log-prob computation.
        """
        mvn = self._base_distribution()
        eta = mvn.rsample((n,))   # (n, K),  differentiable

        theta_dict: Dict[str, torch.Tensor] = {}
        for i, name in enumerate(self.param_names):
            # Apply inverse bijection: eta_i -> theta_i
            theta_dict[name] = self._transforms[i](eta[:, i])

        return theta_dict, eta

    def sample(self, n: int = 1) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Non-differentiable sample (wraps rsample with no_grad)."""
        with torch.no_grad():
            return self.rsample(n)

    # ------------------------------------------------------------------
    # Log probability
    # ------------------------------------------------------------------

    def log_prob(
        self, theta_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Log prior probability of a parameter vector in physical space.

        Applies the change-of-variables formula:
            log p(theta) = log p_eta(h(theta)) + sum_i log|dh_i/dtheta_i|

        where h_i is the forward bijection (physical -> unconstrained).

        Parameters
        ----------
        theta_dict : dict
            Physical parameter values.  Each value is a scalar Tensor or (n,).

        Returns
        -------
        log_p : Tensor
            Log prior.  Shape () for scalar input, (n,) for batched.
        """
        eta_list = []
        log_jac  = torch.tensor(0.0)

        for i, name in enumerate(self.param_names):
            theta_i = theta_dict[name]
            # Forward transform: theta -> eta  (inverse of the R->support map)
            # _transforms[i] is R -> support, so its .inv is support -> R
            eta_i = self._transforms[i].inv(theta_i)
            eta_list.append(eta_i)
            # Jacobian: log|d eta_i / d theta_i| = log|d/d theta transform.inv|
            log_jac = log_jac + self._transforms[i].inv.log_abs_det_jacobian(
                theta_i, eta_i
            )

        eta = torch.stack(eta_list, dim=-1)   # (..., K)
        mvn = self._base_distribution()
        return mvn.log_prob(eta) + log_jac

    # ------------------------------------------------------------------
    # KL divergence (closed-form, relative to standard normal in eta-space)
    # ------------------------------------------------------------------

    def kl_to_standard_normal(self) -> torch.Tensor:
        """
        Analytical KL divergence KL[N(mu, L L^T) || N(0, I)] in unconstrained space.

        Useful as a regularisation term in VAE-style objectives:
            L_total = -E[log p(y | theta)] + beta * KL[q(eta) || p(eta)]

        Returns
        -------
        kl : Tensor, scalar
        """
        mvn = self._base_distribution()
        standard = td.MultivariateNormal(
            loc=torch.zeros_like(self.mu),
            covariance_matrix=torch.eye(self.K, device=self.mu.device),
        )
        return td.kl_divergence(mvn, standard)

    # ------------------------------------------------------------------
    # Diagnostics / introspection
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """
        Return a summary dict of current prior geometry:
            - mean in physical space (mode of the marginal under log/logit)
            - marginal std in transformed space
            - full correlation matrix
        """
        with torch.no_grad():
            # Push mu through inverse bijections to get 'central' physical value
            physical_mean = {}
            for i, name in enumerate(self.param_names):
                physical_mean[name] = self._transforms[i](self.mu[i]).item()

            return {
                "param_names":            self.param_names,
                "mu_transformed":         self.mu.detach().cpu(),
                "physical_mode":          physical_mean,
                "std_transformed":        self.std_in_transformed_space.detach().cpu(),
                "correlation_matrix":     self.correlation.detach().cpu(),
                "L_cholesky":             self.L.detach().cpu(),
                "fixed_params":           dict(self.fixed_params),
            }

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n{'='*60}")
        print(f"  BalloonLatentPrior  —  K = {self.K} parameters")
        print(f"{'='*60}")
        print(f"{'Parameter':<14} {'Mode (physical)':>18} {'Std (η-space)':>14}")
        print(f"{'-'*48}")
        for name in s["param_names"]:
            i = s["param_names"].index(name)
            print(
                f"{name:<14} {s['physical_mode'][name]:>18.4f} "
                f"{s['std_transformed'][i]:>14.4f}"
            )
        print(f"\nCorrelation matrix (unconstrained space):")
        C = s["correlation_matrix"]
        header = " ".join(f"{n[:6]:>8}" for n in s["param_names"])
        print(f"{'':>14} {header}")
        for i, name in enumerate(s["param_names"]):
            row = " ".join(f"{C[i, j].item():>8.3f}" for j in range(self.K))
            print(f"  {name:<12} {row}")
        if s["fixed_params"]:
            print(f"\nFixed parameters (not sampled):")
            for name, val in s["fixed_params"].items():
                print(f"  {name:<14} {val}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Convenience factory: build from a plain Python dict
# ---------------------------------------------------------------------------

def make_balloon_prior(
    param_spec: Optional[ParamSpec] = None,
    prior_mean: Optional[Dict[str, float]] = None,
    init_log_std: float = -1.0,
) -> BalloonLatentPrior:
    """
    Convenience constructor.  Uses BALLOON_PARAM_SPEC and BALLOON_PRIOR_MEAN
    as defaults if not overridden.

    Example
    -------
    >>> prior = make_balloon_prior()
    >>> prior.print_summary()
    >>> theta_dict, eta = prior.rsample(n=32)
    >>> # theta_dict["alpha"].shape  ->  torch.Size([32])
    """
    spec       = param_spec if param_spec is not None else BALLOON_PARAM_SPEC
    mean_dict  = prior_mean if prior_mean is not None else BALLOON_PRIOR_MEAN
    return BalloonLatentPrior(
        param_spec=spec,
        prior_mean=mean_dict,
        init_log_std=init_log_std,
    )


# ---------------------------------------------------------------------------
# Integration with a gradient-based optimiser
# ---------------------------------------------------------------------------

class BalloonPosterior(PyroModule):
    """
    Variational posterior q(eta | y) = N(mu_q, L_q L_q^T) in unconstrained space.

    This is the *amortised* or *per-dataset* complement to BalloonLatentPrior.
    Use when fitting to a single BOLD time series:

        L_ELBO = E_q[log p(y | theta)] - KL[q(eta) || p(eta)]

    BalloonLatentPrior  ->  plays the role of the *prior*  p(eta)
    BalloonPosterior    ->  plays the role of the *posterior* q(eta | y)

    Both are learnable; optimise jointly or in stages.
    """

    def __init__(self, prior: BalloonLatentPrior):
        super().__init__()
        self.prior = prior
        self.K     = prior.K

        # Posterior mean: initialise at prior mean
        self.mu_q: torch.Tensor = PyroParam(prior.mu.detach().clone())

        # Posterior Cholesky: initialise tighter than prior
        L_init = torch.eye(self.K) * 0.1
        self.L_q: torch.Tensor = PyroParam(
            L_init,
            constraint=constraints.lower_cholesky,
        )

    def rsample(
        self, n: int = 1
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Draw reparameterised samples from q(eta | y)."""
        mvn = td.MultivariateNormal(loc=self.mu_q, scale_tril=self.L_q)
        eta = mvn.rsample((n,))   # (n, K)
        theta_dict = {}
        for i, name in enumerate(self.prior.param_names):
            theta_dict[name] = self.prior._transforms[i](eta[:, i])
        return theta_dict, eta

    def kl_to_prior(self) -> torch.Tensor:
        """
        Analytical KL[q(eta) || p(eta)] where both are Gaussian in eta-space.
        This is the exact ELBO regulariser.
        """
        q = td.MultivariateNormal(loc=self.mu_q, scale_tril=self.L_q)
        p = td.MultivariateNormal(loc=self.prior.mu, scale_tril=self.prior.L)
        return td.kl_divergence(q, p)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # --- 1. Build prior from spec ---
    prior = make_balloon_prior(init_log_std=-1.5)
    prior.print_summary()

    # --- 2. Reparameterised sampling ---
    theta_dict, eta = prior.rsample(n=256)
    print("Sample shapes:")
    for k, v in theta_dict.items():
        print(f"  {k:<14}: {tuple(v.shape)}   "
              f"min={v.min().item():.4f}  max={v.max().item():.4f}  "
              f"mean={v.mean().item():.4f}")

    # --- 3. Log probability of a single point ---
    single = {k: v[0] for k, v in theta_dict.items()}
    lp = prior.log_prob(single)
    print(f"\nlog p(theta[0]) = {lp.item():.4f}")

    # --- 4. KL to standard normal (regularisation) ---
    kl = prior.kl_to_standard_normal()
    print(f"KL[q || N(0,I)] = {kl.item():.4f}")

    # --- 5. Verify gradients flow through rsample ---
    theta_dict_grad, eta_grad = prior.rsample(n=8)
    loss = sum(v.mean() for v in theta_dict_grad.values())
    loss.backward()
    print(f"\nGradient check:")
    print(f"  grad(mu)    is not None: {prior.mu.grad is not None}")
    print(f"  grad(L)     is not None: {prior.L.grad is not None}")

    # --- 6. Custom minimal spec example ---
    custom_spec: ParamSpec = {
        "tau_0":   (0.5, 2.5),
        "alpha":   (0.1, 0.5),
        "E_0":     (0.2, 0.6),
    }
    custom_prior = make_balloon_prior(
        param_spec=custom_spec,
        prior_mean={"tau_0": 0.98, "alpha": 0.32, "E_0": 0.40},
    )
    custom_prior.print_summary()

    # --- 7. Posterior / prior pair for ELBO training ---
    posterior = BalloonPosterior(prior)
    kl_pq = posterior.kl_to_prior()
    print(f"KL[q || p] (prior-regularised) = {kl_pq.item():.4f}")

    # Simulate one ELBO gradient step
    opt = torch.optim.Adam(
        list(prior.parameters()) + list(posterior.parameters()),
        lr=1e-3,
    )
    opt.zero_grad()
    th, _ = posterior.rsample(n=16)
    # Dummy log-likelihood placeholder (replace with actual Balloon forward model)
    dummy_log_lik = -sum((v - 0.5).pow(2).mean() for v in th.values())
    elbo = dummy_log_lik - kl_pq
    (-elbo).backward()
    opt.step()
    print("ELBO gradient step completed successfully.")