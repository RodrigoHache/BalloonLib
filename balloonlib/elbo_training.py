"""
elbo_training.py
================
ELBO-based training loop integrating the BalloonLatentPrior / BalloonPosterior
with the existing PINN loss function.

Objective
---------
    L_ELBO = L_PINN(θ_sample) + β(i) · KL[q(η) ‖ p(η)]

where:
    θ_sample  ~ posterior.rsample(n=1)   reparameterised, differentiable
    L_PINN    = ODE residual + IC + BC + BOLD data-fit  (existing loss())
    β(i)      = beta_max · min(1, i / warmup_iters)     linear KL warmup

Parameters in the variational posterior (6 ODE coupling params)
---------------------------------------------------------------
    epsilon_n  →  lambdar_list  CBF amplitude; CMRO2 = cmro2_ratio × ε_n
    kappa      →  kappa_list    signal decay rate (shared for f and m)
    gamma      →  gamma_list    autoregulatory feedback (shared)
    tau_0      →  tau_MTT_list  mean transit time (s)
    tau_m      →  tau_m_list    viscoelastic time constant (s)
    alpha      →  alpha         Grubb's exponent (no longer a PINN parameter)

Fixed (not sampled)
-------------------
    I           high-resolution impulse tensor — pass in base_balloon_params
    E_0, V_0    BOLD signal coefficients — live in model.DEFAULT_PARAMS
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from balloonlib import config
from balloonlib.training import loss as pinn_loss
from balloonlib.data import segmentData, experimental_stims
from balloonlib.plotting import plot_balloon_fitting
from balloonlib.balloon_latent_prior import (
    BalloonLatentPrior,
    BalloonPosterior,
    ParamSpec,
)


# ---------------------------------------------------------------------------
# Parameter specification — 5-param ODE subset
# ---------------------------------------------------------------------------

ELBO_PARAM_SPEC: ParamSpec = {
    "epsilon_n": (0.0, None),  # CBF amplitude             log-transform
    "kappa":     (0.0, None),  # signal decay rate         log-transform
    "gamma":     (0.0, None),  # autoregulatory feedback   log-transform
    "tau_0":     (0.0, None),  # mean transit time (s)     log-transform
    "tau_m":     (0.0, None),  # viscoelastic time const   log-transform
    "alpha":     (0.1, 0.9),   # Grubb's exponent          logit-transform
}

ELBO_PRIOR_MEAN: Dict[str, float] = {
    "epsilon_n": 0.20,
    "kappa":     1.0 / 1.54,   # ≈ 0.649 s⁻¹
    "gamma":     1.0 / 2.46,   # ≈ 0.407 s⁻¹
    "tau_0":     3.0,           # s
    "tau_m":     20.0,          # s  (Friston 2000)
    "alpha":     0.40,
}


def make_ode_prior(init_log_std: float = -1.0) -> BalloonLatentPrior:
    """
    Build a BalloonLatentPrior over the 5 ODE coupling parameters, initialised
    at literature-informed means (ELBO_PRIOR_MEAN) with diagonal covariance
    exp(init_log_std) · I in unconstrained space.
    """
    return BalloonLatentPrior(
        param_spec=ELBO_PARAM_SPEC,
        prior_mean=ELBO_PRIOR_MEAN,
        init_log_std=init_log_std,
    )


# ---------------------------------------------------------------------------
# Parameter mapping: posterior sample → Balloon_params dict
# ---------------------------------------------------------------------------

def theta_to_balloon_params(
    theta: Dict[str, torch.Tensor],
    base_params: dict,
    cmro2_ratio: float = 0.25,
) -> dict:
    """
    Map a posterior sample θ (physical space, scalar tensors) into a
    Balloon_params dict compatible with balloonlib.training.loss().

    Parameters
    ----------
    theta : dict
        Output of BalloonPosterior.rsample(n=1), squeezed to scalar tensors.
        Required keys: 'epsilon_n', 'kappa', 'gamma', 'tau_0', 'tau_m', 'alpha'.
    base_params : dict
        Fixed Balloon_params entries: 'I', 't', 't_scale',
        'time_border_mask', 'first_non_zero_t'.
    cmro2_ratio : float
        CMRO2-to-CBF amplitude ratio (fixed at literature value 0.25).

    Returns
    -------
    dict
        Complete Balloon_params dict; all sampled tensors remain in the
        computation graph so gradients flow back to posterior parameters.
    """
    p = dict(base_params)  # shallow copy — fixed tensors shared, sampled ones new

    eps_n = theta["epsilon_n"].squeeze()
    kappa = theta["kappa"].squeeze()
    gamma = theta["gamma"].squeeze()

    p["lambdar_list"] = torch.stack([eps_n, cmro2_ratio * eps_n])
    p["kappa_list"]   = torch.stack([kappa, kappa])
    p["gamma_list"]   = torch.stack([gamma, gamma])
    p["tau_MTT_list"] = theta["tau_0"].squeeze()
    p["tau_m_list"]   = theta["tau_m"].squeeze()
    p["alpha"]        = theta["alpha"].squeeze()

    return p


# ---------------------------------------------------------------------------
# KL beta schedule
# ---------------------------------------------------------------------------

def _beta_schedule(step: int, warmup_iters: int, beta_max: float) -> float:
    """Linear warmup: 0 → beta_max over warmup_iters steps, then constant."""
    return beta_max * min(1.0, step / max(1, warmup_iters))


# ---------------------------------------------------------------------------
# Posterior diagnostic helper
# ---------------------------------------------------------------------------

def _posterior_mode_str(posterior: BalloonPosterior) -> str:
    """Format the posterior mode (mu_q pushed to physical space) as a string."""
    prior = posterior.prior
    with torch.no_grad():
        parts = [
            f"{name}={prior._transforms[j](posterior.mu_q[j]).item():.4f}"
            for j, name in enumerate(prior.param_names)
        ]
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# ELBO training loop
# ---------------------------------------------------------------------------

def elbo_train(
    model,
    posterior: BalloonPosterior,
    base_balloon_params: dict,
    data_params: dict,
    num_iter: int = 3000,
    beta_max: float = 1.0,
    warmup_iters: int = 500,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    loss_weights: Optional[Dict[str, List[float]]] = None,
    domain: Tuple[float, float] = (0, 30),
    random: bool = False,
    cmro2_ratio: float = 0.25,
    every: int = 500,
    dtype: Optional[torch.dtype] = None,
    device=None,
) -> Dict[str, List[float]]:
    """
    Train the Balloon PINN under an ELBO objective.

    Each step draws one reparameterised sample θ from the variational
    posterior, evaluates the PINN loss at those parameters, and adds a
    KL penalty (with linear warmup) relative to the prior.  Gradients
    flow through both the PINN weights and the posterior parameters
    (mu_q, L_q) via the reparameterisation trick.

    Parameters
    ----------
    model : Multihead
        The PINN model.  Its alpha parameter must NOT be an nn.Parameter
        (it is now owned by the posterior).
    posterior : BalloonPosterior
        Variational posterior q(θ|y); its .prior attribute is p(θ).
    base_balloon_params : dict
        Fixed Balloon_params entries: at minimum 'I' (impulse tensor).
        The ODE coupling parameters (including tau_m) are sampled each step.
        Note: this dict is mutated in-place with time tensors (same
        convention as balloonlib.training.train).
    data_params : dict
        Experimental data parameters.  Matches the format expected by
        balloonlib.training.loss().  Mutated in-place with stimulus info.
    num_iter : int
        Number of training steps.
    beta_max : float
        Maximum KL weight applied after warmup.
    warmup_iters : int
        Steps over which β increases linearly from 0 to beta_max.
    optimizer : torch.optim.Optimizer or None
        Joint optimiser over model + posterior parameters.
        Defaults to Adam(lr=4e-3) if None.
    scheduler : LRScheduler or None
        Optional learning-rate scheduler; stepped once per iteration.
    loss_weights : dict or None
        Per-component weight histories (same format as train()).
        Keys: 'ode', 'bold', 'ic', 'border', 'other'.
        Defaults to p=0.6 ODE / 0.4 BOLD split.
    domain : tuple
        Physical time domain (t_start, t_end) in seconds.
    random : bool
        Add stochastic jitter to collocation points each step.
    cmro2_ratio : float
        CMRO2-to-CBF amplitude ratio (fixed).
    every : int
        Progress print + plot interval (0 = silent).
    dtype : torch.dtype or None
    device : torch.device or None

    Returns
    -------
    dict
        Loss trace with keys:
            'ode', 'bold', 'ic', 'border', 'other'  — PINN components
            'pinn_total'                             — weighted PINN sum
            'kl'                                    — raw KL value
            'elbo_total'                             — pinn_total + β·KL
    """
    # ---- Device / dtype defaults -----------------------------------------
    if device is None:
        device = config.device
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    if dtype is None:
        dtype = config.dtype
    use_amp = device.type == "cuda"

    # ---- Loss weights default --------------------------------------------
    if loss_weights is None:
        loss_weights = {
            "ode":    [0.6],
            "bold":   [0.4],
            "ic":     [1.0],
            "border": [1.0],
            "other":  [1.0],
        }

    # Ensure 'other' key is always present — loss() always accesses it
    if "other" not in loss_weights:
        loss_weights["other"] = [0.0]

    # ---- Loss amplitude factors (mirrors train()) -------------------------
    amp = {"ode": 1e1, "bold": 1e0, "ic": 1e0, "border": 1e0, "other": 1e0}
    amp_p = torch.distributions.beta.Beta(6, 2)

    # ---- Optimiser -------------------------------------------------------
    if optimizer is None:
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(posterior.parameters()),
            lr=4e-3,
        )

    # ---- Time tensors (mirrors train() setup) ----------------------------
    max_elements = base_balloon_params["I"].size(0)
    first_non_zero_index = torch.argmax(base_balloon_params["I"]) - 1

    pinn_time = (torch.arange(0, max_elements) / max_elements).to(dtype)
    pinn_time = ((pinn_time - pinn_time.mean()) / pinn_time.std()).view(-1, 1)

    base_balloon_params.update({
        "first_non_zero_t": pinn_time[first_non_zero_index],
        "t_scale":          (pinn_time[-1] - pinn_time[0]) / (domain[1] - domain[0]),
        "time_border_mask": (
            (pinn_time.squeeze() <= pinn_time[first_non_zero_index])
            | (pinn_time.squeeze() >= pinn_time[-1])
        ),
    })

    # ---- Data preprocessing (mirrors train()) ----------------------------
    data_params["Bold_Signal"] = (
        torch.as_tensor(data_params["Bold_Signal"]).to(dtype).view(-1, 1)
    )

    if "Bold_Signal" in data_params:
        time_bf_stim = data_params["TR"]
        Bold_segments, time_corrected = segmentData(
            data_params["Bold_Signal"],
            Sti_Onsets=data_params["Sti_Onsets"],
            time_bf_stim=time_bf_stim,
            t0s=data_params["t0"],
            TR=data_params["TR"],
        )

        time_max = torch.ceil(
            torch.stack([seg.max() for seg in time_corrected]).max()
        ).to(dtype=torch.int32)

        stimulus, stimulus_time = experimental_stims(
            time_max.detach().item() / data_params["TR"],
            Sti_Onsets=[time_bf_stim],
            TR=data_params["TR"],
            block_len=data_params["stim_length [seg]"],
            stmxblck=data_params["stim_x_block"],
        )

        n_elements = data_params["Bold_Signal"].shape[0] * data_params["TR"]
        Bold_data_time = (
            torch.arange(0, n_elements, data_params["TR"]) + data_params["t0"]
        )

        Overall_stimuli, Overall_stim_time = experimental_stims(
            data_params["Bold_Signal"].shape[0]
            + (data_params["t0"] // data_params["TR"]),
            Sti_Onsets=data_params["Sti_Onsets"],
            TR=data_params["TR"],
            block_len=data_params["stim_length [seg]"],
            stmxblck=data_params["stim_x_block"],
        )

        data_params.update({
            "Bold_data_time":    Bold_data_time,
            "stimulus":          stimulus.view(-1, 1),
            "stimulus_time":     stimulus_time,
            "Overallstim":       Overall_stimuli,
            "Overall_stim_time": Overall_stim_time,
        })

    # ---- Stochastic collocation jitter -----------------------------------
    if not random:
        epsilon = torch.zeros(num_iter)
    else:
        distr = torch.distributions.beta.Beta(5, 5)
        epsilon = (distr.sample([num_iter]) - distr.mean) / max_elements

    # ---- Loss trace initialisation ---------------------------------------
    component_keys = ["ode", "bold", "ic", "border", "other"]
    loss_trace = {k: [] for k in component_keys}
    loss_trace.update({"pinn_total": [], "kl": [], "elbo_total": []})

    # ---- Training loop ---------------------------------------------------
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    amp_i  = 1e3  # dynamic amplitude initialised at warm-up value

    for i in tqdm(range(num_iter)):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # 1. Reparameterised sample → Balloon_params for this step
        theta, _ = posterior.rsample(n=1)
        theta     = {k: v.squeeze(0) for k, v in theta.items()}

        t_step    = torch.clamp(pinn_time + epsilon[i], min=pinn_time[0].item())
        balloon_i = theta_to_balloon_params(theta, base_balloon_params, cmro2_ratio)
        balloon_i["t"] = t_step

        # 2. PINN composite loss + KL inside autocast
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss_dict = pinn_loss(
                model,
                Balloon_params=balloon_i,
                data_params=data_params,
                loss_weights=loss_weights,
                amp=amp,
                domain=domain,
                random=random,
                dtype=dtype,
                meFn=data_params["errorFn"],
            )
            beta_i     = _beta_schedule(i, warmup_iters, beta_max)
            kl         = posterior.kl_to_prior()
            elbo_total = loss_dict["total"] + beta_i * kl

        # 3. Dynamic amplitude adjustment (mirrors train())
        if i <= 100:
            amp_i = 1e3
        elif len(loss_trace["bold"]) >= 11:
            amp_pi = amp_p.sample()
            denom  = np.mean(
                loss_trace["ode"][-11:-1]
                + loss_trace["ic"][-11:-1]
                + loss_trace["border"][-11:-1]
            )
            if denom > 0:
                tmp    = np.mean(loss_trace["bold"][-11:-1]) / denom
                amp_i  = (amp_pi.item() * amp_i + (1 - amp_pi.item()) * tmp)

        for k in loss_weights:
            if k != "bold" and loss_weights["bold"][0] > 0.0:
                amp[k] = max(1.0, round(amp_i, 1))

        # 4. Backward through PINN weights + posterior (mu_q, L_q)
        scaler.scale(elbo_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(posterior.parameters()),
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        # 5. Record
        for k in component_keys:
            loss_trace[k].append(loss_dict[k].detach().item())
        loss_trace["pinn_total"].append(loss_dict["total"].detach().item())
        loss_trace["kl"].append(kl.detach().item())
        loss_trace["elbo_total"].append(elbo_total.detach().item())

        # 6. Progress
        if every != 0 and (i + 1) % every == 0:
            print(
                f"\n[{i+1}/{num_iter}]  "
                f"elbo={elbo_total.item():.4e}  "
                f"pinn={loss_dict['total'].item():.4e}  "
                f"kl={kl.item():.4e}  β={beta_i:.3f}"
            )
            print(
                f"  ode={loss_dict['ode'].item():.3e}  "
                f"bold={loss_dict['bold'].item():.3e}  "
                f"ic={loss_dict['ic'].item():.3e}  "
                f"border={loss_dict['border'].item():.3e}"
            )
            print(f"  posterior mode →  {_posterior_mode_str(posterior)}")
            plot_balloon_fitting(
                model=model,
                t_normalized=pinn_time.requires_grad_(False),
                domain=domain,
                stimulus=data_params.get("stimulus"),
                title=f"ELBO  iter {i+1}   β={beta_i:.3f}",
                data_params=data_params if "Bold_Signal" in data_params else None,
                first_non_zero_index=first_non_zero_index,
                iteration=i,
                show_bold_signal="Bold_Signal" in data_params,
                dtype=dtype,
            )

    return loss_trace
