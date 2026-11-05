"""
Training utilities for the BalloonLib PINN framework.

Provides the composite loss function, the main training loop,
adaptive loss reweighting, and per-loss gradient analysis.
"""

import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List

from balloonlib import config
from balloonlib.physics import dfdt, weighted_temporal_ode_loss
from balloonlib.data import segmentData, experimental_stims
from balloonlib.utils import np2tensor, timeBall, tofit, SigmoidClamp, drive_moments
from balloonlib.plotting import plot_balloon_fitting
from balloonlib.MOO_Loss import loss_reweight_paranoid, compute_per_loss_gradients# noqa: F401  (moved; re-exported)

# ---------------------------------------------------------------------------
# Composite PINN loss
# ---------------------------------------------------------------------------
def loss(
    model,
    Balloon_params: dict,
    data_params: dict,
    loss_weights: dict,
    amp: dict,
    domain=(0, 30),
    random=False,
    dtype=torch.float32,
    meFn=nn.MSELoss(),
    causal_n_segments: int = 30,        
    causal_epsilon: float = 0.1,        
    causal_normalise: bool = False,     
):
    """Compute the composite PINN loss (ODE + IC + BC + data).

    Parameters
    ----------
    model : torch.nn.Module
        The Multihead PINN model.
    Balloon_params : dict
        Physiological parameters and time tensors for the Balloon model.
    data_params : dict
        Experimental data parameters (BOLD signal, stimulus info, etc.).
    data : dict
        Training data dictionary.
    loss_weights : dict of {str: list of float}
        Per-component weight histories; the last element is used.
    amp : dict of {str: float}
        Amplitude scaling factors per loss component.
    random : bool
        Whether to sample time points randomly.
    dtype : torch.dtype
        Tensor data type.
    meFn : torch.nn.Module
        Error metric, e.g. ``nn.MSELoss()``.

    Returns
    -------
    dict of {str: torch.Tensor}
        Loss dictionary with keys ``'total'``, ``'ode'``, ``'ic'``, ``'bold'``, ``'other'``.
    """
    # Evaluate model at sample points
    t = Balloon_params["t"].requires_grad_(True)
    if model.impulse:
        inputs = torch.cat([t, Balloon_params["I"]], dim=1).view(
            -1, 2
        )
    else:
        inputs = t
    
    output, _ = model(inputs)
    hrf_pinn = model.predictor()

    # Compute time derivatives
    dfindt, dmdt, dvdt, dqdt= [#, dpredt_num = [
        Balloon_params["t_scale"] * dfdt(signal=i.squeeze(), arg=Balloon_params["t"])
        for i in [model.f, model.m, model.v, model.q]#, hrf_pinn.squeeze()]
    ]
    d2findt2, d2mtdt2 = [
        Balloon_params["t_scale"] * dfdt(signal=d_dt, arg=Balloon_params["t"])
        for d_dt in [dfindt, dmdt]
    ]

    if any(x is None for x in [dfindt, dmdt, dvdt, dqdt, d2findt2, d2mtdt2]):
        raise ValueError("One or more time derivatives are None")
    if any(torch.isnan(x).any() for x in [dfindt, dmdt, dvdt, dqdt, d2findt2, d2mtdt2]):
        raise ValueError("One or more time derivatives contain NaN")

    # Graph-integrity checks
    for _n, _t in zip(
        ["model.f", "model.m", "model.v", "model.q"],
        [model.f, model.m, model.v, model.q],
    ):
        if not _t.requires_grad:
            warnings.warn(
                f"[GRAPH WARNING] {_n}.requires_grad is False — detached from graph!",
                RuntimeWarning,
                stacklevel=2,
            )
    for _n, _t in zip(
        ["dfindt", "dmdt", "dvdt", "dqdt", "d2findt2", "d2mtdt2"],#, "dpredt_num"],
        [dfindt, dmdt, dvdt, dqdt, d2findt2, d2mtdt2]#, dpredt_num],
    ):
        if not _t.requires_grad:
            warnings.warn(
                f"[GRAPH WARNING] {_n}.requires_grad is False — gradient will not flow!",
                RuntimeWarning,
                stacklevel=2,
            )
    
    # ODE residual
    Impulse = Balloon_params["I"]
    lambdar_list = Balloon_params["lambdar_list"]
    kappa_list = Balloon_params["kappa_list"]
    gamma_list = Balloon_params["gamma_list"]
    tau_m = Balloon_params["tau_m_list"]
    tau_MTT = Balloon_params["tau_MTT_list"]
    
    f_out = model.fout(model.v, tau_m=tau_m, dvdt=dvdt, t_scale = Balloon_params["t_scale"])

    residual = torch.cat(
        [
            d2findt2
            - (
                lambdar_list[0, 0] * Impulse
                - kappa_list[0, 0] * dfindt
                - gamma_list[0, 0] * (model.f - 1)
            ),
            d2mtdt2
            - (
                lambdar_list[0, 1] * Impulse
                - kappa_list[0, 1] * dmdt
                - gamma_list[0, 1] * (model.m - 1)
            ),
            tau_MTT * dvdt - (model.f - f_out),
            tau_MTT * dqdt - (model.m - (model.q / model.v.clamp(min=0.01)) * f_out),
        ],
        dim=1,
    )

    if torch.isnan(residual).any():
        print(f"NaN indices in residual: {torch.isnan(residual).nonzero()}")
        raise ValueError("residual is NaN! Terminating training.")

    ode_loss = weighted_temporal_ode_loss(
        residual, meFn, n_segments=causal_n_segments,
        epsilon=causal_epsilon, normalise_weights=causal_normalise
        )        
    if "Bold_Signal" in data_params:
        Bold_data = data_params["Bold_Signal"]
        bold_pinn, Bold_pinn_time = tofit(
            data_params["stimuli"],
            hrf_pinn,
            data_params["stimuli_t_max"],
        )
        bold_pinn = 100*(0.01/data_params['drive_area'])*bold_pinn
        
        samples_index, valid = timeBall(data_params["Bold_data_time"], Bold_pinn_time)
        assert bool(valid.all()), f"{int((~valid).sum())} acquisition times fall outside the stimulus grid"
        
        tmatch = Bold_pinn_time[samples_index]
        assert float((tmatch - data_params["Bold_data_time"].squeeze()).abs().max()) <= 0.005 + 1e-9

        noise = data_params.get("noise_model")
        if noise is None:                     # white-noise term (unchanged behaviour)
            bold_res = torch.add(bold_pinn[samples_index], - Bold_data)
            assert bold_res.shape == Bold_data.shape, f"broadcast: {tuple(bold_res.shape)}"
            offset = -torch.mean(bold_res).detach()
        
            bold_loss = meFn(torch.add(bold_res, offset), torch.zeros_like(bold_res))
        else:                                 # AR(1)-whitened term (balloonlib.ar1_whitening)
            bold_loss = noise.loss(bold_pinn[samples_index], Bold_data,
                                   data_params["Bold_data_time"], meFn)
                
    else:
        bold_loss = ode_loss.new_zeros(())

    # Physics violation penalties
    other_loss = meFn(tau_MTT*(model.v*dqdt-model.q*dvdt), model.m*model.v-model.q*model.f) + meFn(nn.functional.relu(1.0-model.v), torch.zeros_like(model.v))
                

    # Initial condition and boundary losses
    if Balloon_params["time_border_I"] == 0:
        # tmp_cauchy = torch.cat([Balloon_params["W"]*(output-1), hrf_pinn, d2findt2, d2mtdt2, dfindt, dmdt, dvdt, dqdt], dim=1)#, dpredt_num], dim=1)
        tmp_cauchy = torch.cat([output-1, hrf_pinn, d2findt2, d2mtdt2, dfindt, dmdt, dvdt, dqdt], dim=1)#, dpredt_num], dim=1)
        cauchy_ic = torch.index_select(tmp_cauchy, dim=0, index=Balloon_params["tmp_index"])
        ic_loss = meFn(cauchy_ic, torch.zeros_like(cauchy_ic))
        
    else:
        ic_loss = ode_loss.new_zeros(())

    loss_dict = {
        "ode": ode_loss,
        "bold": bold_loss,
        "ic": ic_loss,
        "other":  other_loss,
    }

    loss_dict["total"] = (
        amp["ode"] * loss_weights["ode"][-1] * loss_dict["ode"]
        + amp["bold"] * loss_weights["bold"][-1] * loss_dict["bold"]
        + amp["ic"] * loss_weights["ic"][-1] * loss_dict["ic"]
        + amp["other"]  * loss_weights["other"][-1]  * loss_dict["other"]
    )

    return loss_dict


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(
    model,
    optimizer,
    lossfn,
    num_iter,
    Balloon_params=None,
    data_params=None,
    random=False,
    every=3,
    loss_weights={"ode": [1.0], "ic": [1.0], "bold": [1.0], "other": [1.]},
    scheduler=None,
    dtype=None,
    device=None,
    causal_n_segments: int = 30,        
    causal_epsilon: float = 0.1,        
    causal_normalise: bool = False, 
    balancer=None,
):
    """Train the Multihead Balloon-PINN model.

    Parameters
    ----------
    model : torch.nn.Module
        The PINN model to train.
    optimizer : torch.optim.Optimizer
        Optimiser instance (e.g. Adam).
    lossfn : callable
        Loss function matching the :func:`loss` signature.
    num_iter : int
        Number of training iterations.
    Balloon_params : dict or None
        Balloon physiological parameters.
    data_params : dict or None
        Experimental data parameters.
    random : bool
        If ``True``, add stochastic jitter to time points each iteration.
    every : int
        Print progress and plot every *every* iterations (0 = silent).
    loss_weights : dict of {str: list of float}
        Initial per-component weight lists.
    scheduler : torch.optim.lr_scheduler or None
        Optional learning-rate scheduler.
    dtype : torch.dtype or None
        Tensor data type.  ``None`` uses :data:`balloonlib.config.dtype`.
    device : str, torch.device, or None
        Computation device.  ``None`` uses :data:`balloonlib.config.device`.

    balancer : balloonlib.MOO_Loss balancer or None
        Adaptive multi-objective loss weighting: ``SoftAdapt``, ``ReLoBRaLo``,
        ``InverseGradNorm`` or ``LRAnnealing``.  When given, every ``amp`` is fixed
        at 1, the dynamic amp rule is skipped, and the balancer appends new weights
        to a COPY of ``loss_weights`` (the caller's dict is not modified).  Its
        history is stored in ``data_params["balancer_state"]``.  ``None``
        (default) keeps the dynamic amp rule.
    Returns
    -------
    dict of {str: list of float}
        Merged loss trace (component traces + ``'total'`` key).
    """
    if device is None:
        device = config.device
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    if dtype is None:
        dtype = config.dtype
    use_amp = False  # fp32 training only — GradScaler/autocast intentionally disabled.

    amp = {
        "ode": 1e1,
        "bold": 1e0,
        "ic": 1e0,
        "other": 1e0,
    }
    soft_amp = SigmoidClamp(from_val = 0, to_val=1)
    amp_0 = 1e1
    amp_i = 1e1
    if balancer is not None:
        # The balancer owns the weighting: no amp rescaling, and never mutate
        # the caller's (or the default argument's) weight lists.
        amp = {k: 1.0 for k in amp}
        loss_weights = {k: list(v) for k, v in loss_weights.items()}
    loss_trace = {key: [] for key in loss_weights.keys()}
    total_trace = {"total": []}
    # data = {}
    
    if ("Bold_ode" in data_params) & ("Bold_Signal" in data_params):
        raise TypeError("Bold and Bold_segments cannot both be included during training")

    max_elements = Balloon_params["I"].size()[0]
    onset = int((Balloon_params["I"].flatten() > 0).nonzero()[0])
    first_non_zero_index = 0 if onset==0 else onset - 1
    
    Balloon_params.update({
        "I":            Balloon_params["I"].reshape(-1, 1),
        "lambdar_list": torch.as_tensor(Balloon_params["lambdar_list"], dtype=dtype).reshape(1, -1),
        "kappa_list":   torch.as_tensor(Balloon_params["kappa_list"],   dtype=dtype).reshape(1, -1),
        "gamma_list":   torch.as_tensor(Balloon_params["gamma_list"],   dtype=dtype).reshape(1, -1),
        "tau_m_list":   torch.as_tensor(Balloon_params["tau_m_list"],   dtype=dtype),
        "tau_MTT_list": torch.as_tensor(Balloon_params["tau_MTT_list"], dtype=dtype),
        })

    A, t_c, _ = drive_moments(Balloon_params["I"])
    data_params["drive_area"], data_params["drive_centroid"] = A, t_c
    
    domain = (0, (max_elements-1)*0.01)
    pinn_time = (torch.arange(0, max_elements) / max_elements).requires_grad_(False).to(dtype)
    pinn_t_mean, pinn_t_std = pinn_time.mean(), pinn_time.std()
    pinn_time = ((pinn_time - pinn_t_mean) / pinn_t_std).view(-1, 1)
    t_min = pinn_time[0]
    causal_epsilon_list = [1,5e-1,1e-1,5e-2,1e-2,1e-3]
    range_iter = np.arange(num_iter)
    causal_epsilon_loop = [int(i//(num_iter//(len(causal_epsilon_list)))) for i in range_iter]
    gate = soft_amp(0.05 * (range_iter - 100))
    Balloon_params.update(
        {
            "first_non_zero_t": pinn_time[first_non_zero_index],
            "t_scale": (pinn_time[-1] - t_min) / (domain[1] - domain[0]),
            "time_border_mask": (
                (pinn_time.squeeze() <= pinn_time[first_non_zero_index])
                | (pinn_time.squeeze() >= pinn_time[-50])
            ),
        }
    )
    Balloon_params.update(
        {
            "time_border_I": (Balloon_params["I"][Balloon_params["time_border_mask"]]).sum(),
            "tmp_index": torch.arange(max_elements)[Balloon_params["time_border_mask"]],
            "W":int(Balloon_params["time_border_mask"].sum())**(0.5)
        }
    )
    if not random:
        epsilon = pinn_time.new_zeros(num_iter)
    else:
        distr = torch.distributions.beta.Beta(5, 5)
        epsilon = (distr.sample([num_iter]) - distr.mean) / max_elements

    if "Bold_Signal" in data_params:
        data_params["Bold_Signal"] = torch.as_tensor(data_params["Bold_Signal"], dtype=dtype).squeeze()
        time_bf_stim = data_params["TR"]
        Bold_segments, time_corrected = segmentData(
            data_params["Bold_Signal"],
            Sti_Onsets=data_params["Sti_Onsets"],
            time_bf_stim=time_bf_stim,
            t0s=data_params["t0"],
            TR=data_params["TR"],
        )
        # optimal_combinatory = round(len(Bold_segments) / 2)
        # data_params.update({"index_size": optimal_combinatory})

        time_max = torch.ceil(torch.stack([i.max() for i in time_corrected]).max()).to(
            dtype=torch.int32
        )

        stimulus, stimulus_time = experimental_stims(
            time_max.detach().item() / data_params["TR"],
            Sti_Onsets=[time_bf_stim-t_c],
            TR=data_params["TR"],
            block_len=data_params["stim_length [seg]"],
            stmxblck=data_params["stim_x_block"],
        )

        stimuli, stimuli_time = experimental_stims(
            data_params["Bold_Signal"].shape[0] + (data_params["t0"] // data_params["TR"]),
            Sti_Onsets=data_params["Sti_Onsets"]-t_c,
            TR=data_params["TR"],
            block_len=data_params["stim_length [seg]"],
            stmxblck=data_params["stim_x_block"],
        )
        onset = int((stimuli.flatten()> 0).nonzero()[0])
        data_params.update(
            {
                "Bold_data_time": data_params["t0"] + torch.arange(data_params["Bold_Signal"].shape[0]) * data_params["TR"],
                "stimulus": stimulus.view(-1, 1),
                "stimulus_time": stimulus_time,
                "stimuli": stimuli,
                "stimuli_time": stimuli_time,
                "stimuli_Onset": onset,
                "stimuli_t_max":stimuli_time[-1], 
            }
        )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        
    for i in tqdm(range_iter):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        Balloon_params.update({"t": torch.clamp(pinn_time + epsilon[i], min=t_min)})
        # enabled=use_amp is False, so dtype is ignored and this runs in float32.
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss_dict = lossfn(
                model,
                Balloon_params=Balloon_params,
                data_params=data_params,
                loss_weights=loss_weights,
                amp=amp,
                random=random,
                dtype=dtype,
                meFn=data_params["errorFn"],
                causal_n_segments=causal_n_segments, 
                causal_epsilon=causal_epsilon_loop[i],
                causal_normalise=causal_normalise,
            )

        # Dynamic amplitude adjustment (activated after warm-up); skipped with a balancer
        if balancer is None:
            if i > 1:
                t = -(i+2) if i < 10 else -11
                tmp = np.mean(loss_trace["bold"][t:-1]) / (
                        np.mean(
                            np.asarray(loss_trace["ode"][t:-1])
                            + np.asarray(loss_trace["ic"][t:-1])
                            )
                        )
            
                amp_i = ((1-gate[i])*amp_0 + gate[i]*tmp).item()
        
            for k in loss_weights.keys():
                if k != "bold" and loss_weights["bold"][0] > 0.0:
                    amp[k] = np.round(amp_i, 1) if np.round(amp_i, 1) > 1 else 1.0

        # Multi-objective balancing: BEFORE backward, because the gradient-based
        # balancers differentiate each term (retain_graph=True).  New weights are
        # appended to loss_weights and act from the next iteration.
        if balancer is not None:
            balancer(loss_dict, loss_weights, i, model=model)

        scaler.scale(loss_dict["total"]).backward()

        # Re-estimate the residual AR(1) coefficient on schedule (no gradient);
        # the new rho is used from the next iteration's loss.
        noise = data_params.get("noise_model") if data_params is not None else None
        if noise is not None:
            est = noise.update(i)
            if est is not None and every != 0:
                print(f"[AR(1)] iter {i}: lag-1 {est['phi_raw']:.3f} -> rho {est['phi']:.3f}"
                      f" ({'white' if est['white'] else 'whitened'}), sigma {est['sigma']:.3g},"
                      f" lag-1 after whitening {est['lag1_whitened']:+.3f}"
                      + ("  AT BOUND: misfit read as noise?" if est["at_bound"] else ""))
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)  # wraps optimizer.step()
        scaler.update()  # adjusts scale factor

        if scheduler is not None and scaler.get_scale() >= scale_before_step:
            scheduler.step()
        total_trace["total"].append(loss_dict["total"].detach().item())
        
        for name_t, name_d in zip(loss_trace.keys(), loss_dict.keys()):
            if name_t == name_d:
                loss_trace[name_t].append(loss_dict[name_d].detach().item())
            else:
                raise ValueError(f"Loss name mismatch: {name_t} != {name_d}")

        if every != 0 and (i + 1) % every == 0:
            print(f"{i + 1}th Iter:")
            print(
                "total {:.4e}, ode:{:.3e}, bold:{:.3e}, CIC:{:.3e}, other:{:.3e}".format(
                    loss_dict["total"].detach().item(),
                    loss_dict["ode"].detach().item(),
                    loss_dict["bold"].detach().item(),
                    loss_dict["ic"].detach().item(),
                    loss_dict["other"].detach().item(),
                )
            )
            print(
                "total weights, ode:{:.3e}, bold:{:.3e}, CIC:{:.3e}, other:{:.3e}".format(
                    loss_weights["ode"][-1],
                    loss_weights["bold"][-1],
                    loss_weights["ic"][-1],
                    loss_weights["other"][-1],
                )
            )
            print("Physics amp", amp)
            plot_balloon_fitting(
                model=model,
                t_normalised=pinn_time.requires_grad_(False),
                domain= domain,
                title="Training Progress",
                data_params=data_params if "Bold_Signal" in data_params else None,
                first_non_zero_index=first_non_zero_index,
                iteration=i,
                show_bold_signal="Bold_Signal" in data_params,
                dtype=dtype,
            )

    loss_trace.update(total_trace)
    if balancer is not None and data_params is not None:
        data_params["balancer_state"] = balancer.state()                 # picklable summary
    if data_params is not None and data_params.get("noise_model") is not None:
        data_params["noise_state"] = data_params["noise_model"].state()   # picklable summary

    # Guarantee self.f/m/v/q reflect a full-precision forward pass over the
    # complete, ordered time domain on exit — independent of `every` and of
    # whatever precision the last autocast training step happened to use.
    model.eval()
    t_exit = pinn_time.detach().clone().requires_grad_(True)
    inputs = (
        torch.cat([t_exit, Balloon_params["I"].view(-1, 1)], dim=1)
        if model.impulse else t_exit
    )
    model(inputs)   # gradients ON: cached f/m/v/q stay differentiable w.r.t. model._x
    model.train()
    return loss_trace
