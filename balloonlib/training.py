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

from balloonlib import balloonmodellib as bml
from balloonlib.physics import dfdt, weighted_temporal_ode_loss
from balloonlib.data import training_data, segmentData, experimental_stims
from balloonlib.utils import timeBall, tofit
from balloonlib.plotting import plot_balloon_fitting

# Module-level device / dtype (mirrors balloonpinnlib globals)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda" 
dtype = torch.float32


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
    device: str = "cuda",
    eps: float = 1e-12,
    validate: bool = True,
) -> Dict[str, List[float]]:
    """Compute adaptive loss weights with Bernoulli history retention.

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
    device : str
        Computation device.
    eps : float
        Clamping floor for numerical stability.
    validate : bool
        If ``True``, run paranoid post-hoc assertions.

    Returns
    -------
    dict of {str: list of float}
        The *same* ``weights_history`` object with new values appended.
    """
    if validate:
        original_keys = set(weights_history.keys()) - set(keys_to_skip)
        original_lengths = {k: len(weights_history[k]) for k in original_keys}

    keys = sorted([
        k for k in weights_history.keys()
        if k in loss_dict and k in loss_trace and k not in keys_to_skip
    ])

    if not keys:
        return weights_history

    current = torch.stack([loss_dict[k] for k in keys]).requires_grad_(False)
    initial = torch.tensor([loss_trace[k][0] for k in keys], device=device)

    prev_loss = torch.tensor([
        loss_trace[k][-2] if len(loss_trace[k]) > 1 else loss_trace[k][0]
        for k in keys
    ], device=device)

    prev_weights = torch.tensor([weights_history[k][-1] for k in keys], device=device)

    logits_hist   = current / (temperature * torch.clamp(initial,   min=eps))
    logits_recent = current / (temperature * torch.clamp(prev_loss, min=eps))

    weights_hist   = torch.softmax(logits_hist,   dim=0)
    weights_recent = torch.softmax(logits_recent, dim=0)

    rho_sample  = torch.bernoulli(torch.tensor(rho, device=device))
    hist_weights = rho_sample * prev_weights + (1 - rho_sample) * weights_hist
    new_weights  = alpha * hist_weights + (1 - alpha) * weights_recent

    for k, w in zip(keys, new_weights.tolist()):
        weights_history[k].append(w)

    for k in keys_to_skip:
        weights_history[k].append(1.0)

    if validate:
        assert set(weights_history.keys()) == original_keys.union(keys_to_skip), "Keys changed!"
        for k in original_keys:
            if k != "total" and k in keys:
                assert len(weights_history[k]) == original_lengths[k] + 1, \
                    f"History length mismatch for {k}"
            elif k != "total":
                assert len(weights_history[k]) == original_lengths[k], \
                    f"Non-updated key {k} changed length!"

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
):
    """Compute per-component gradient norms w.r.t. model parameters.

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
        gradients = torch.autograd.grad(
            outputs=w_loss,
            inputs=params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        flat = torch.cat([
            g.detach().reshape(-1) if g is not None
            else torch.zeros(p.numel(), device=p.device, dtype=p.dtype)
            for g, p in zip(gradients, params)
        ])
        grads[name] = flat

    if return_norms:
        norms = {key: value.norm(2).item() for key, value in grads.items()}
        total = sum(norms.values())
        norms = {key: (total / value) for key, value in norms.items()}

    return grads, norms


# ---------------------------------------------------------------------------
# Composite PINN loss
# ---------------------------------------------------------------------------

def loss(
    model,
    Balloon_params: dict,
    data_params: dict,
    # data: dict,
    loss_weights: dict,
    amp: dict,
    domain=(0, 30),
    random=False,
    dtype=torch.float32,
    meFn=nn.MSELoss(),
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
    domain : tuple of float
        Time domain ``(t_start, t_end)``.
    random : bool
        Whether to sample time points randomly.
    dtype : torch.dtype
        Tensor data type.
    meFn : torch.nn.Module
        Error metric, e.g. ``nn.MSELoss()``.

    Returns
    -------
    dict of {str: torch.Tensor}
        Loss dictionary with keys ``'total'``, ``'ode'``, ``'ic'``,
        ``'border'``, ``'bold'``, ``'other'``.
    """
    # Evaluate model at sample points
    if model.impulse:
        inputs = torch.cat(
            [Balloon_params["t"], Balloon_params["I"].view(-1, 1)], dim=1
        ).view(-1, 2)
    else:
        inputs = Balloon_params["t"].requires_grad_(True)
    
    output, _ = model(inputs)
    hrf_pinn = model.predictor()

    # Compute time derivatives
    dfindt, dmdt, dvdt, dqdt, dpredt_num = [
        Balloon_params["t_scale"] * dfdt(signal=i.squeeze(), arg=Balloon_params["t"])
        for i in [model.f, model.m, model.v, model.q, hrf_pinn.squeeze()]
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
                RuntimeWarning, stacklevel=2,
            )
    for _n, _t in zip(
        ["dfindt", "dmdt", "dvdt", "dqdt", "d2findt2", "d2mtdt2", "dpredt_num"],
        [dfindt, dmdt, dvdt, dqdt, d2findt2, d2mtdt2, dpredt_num],
    ):
        if not _t.requires_grad:
            warnings.warn(
                f"[GRAPH WARNING] {_n}.requires_grad is False — gradient will not flow!",
                RuntimeWarning, stacklevel=2,
            )

    if dpredt_num is None:
        raise ValueError("dpredt_num is None")
    if torch.isnan(dpredt_num).any():
        raise ValueError("dpredt_num is NaN")
    
    # ODE residual
    Impulse = Balloon_params["I"].reshape(-1, 1)
    lambdar_list = torch.tensor(Balloon_params["lambdar_list"], dtype=dtype).unsqueeze(0)
    kappa_list   = torch.tensor(Balloon_params["kappa_list"],   dtype=dtype).unsqueeze(0)
    gamma_list   = torch.tensor(Balloon_params["gamma_list"],   dtype=dtype).unsqueeze(0)
    tau_m        = torch.tensor(Balloon_params["tau_m_list"],   dtype=dtype)
    tau_MTT      = torch.tensor(Balloon_params["tau_MTT_list"], dtype=dtype)
    
    f_out = model.fout(model.v, tau_m = tau_m, dvdt = dvdt)

    residual = torch.cat(
        [
            d2findt2 - (lambdar_list[0, 0] * Impulse - kappa_list[0, 0] * dfindt - gamma_list[0, 0] * (model.f - 1)),
            d2mtdt2  - (lambdar_list[0, 1] * Impulse - kappa_list[0, 1] * dmdt   - gamma_list[0, 1] * (model.m - 1)),
            tau_MTT * dvdt - (model.f - f_out),
            tau_MTT * dqdt - (model.m - (model.q / model.v.clamp(min=0.01)) * f_out),
        ],
        dim=1,
    )

    if torch.isnan(residual).any():
        print(f"NaN indices in residual: {torch.isnan(residual).nonzero()}")
        raise ValueError("residual is NaN! Terminating training.")

    ode_loss = weighted_temporal_ode_loss(
        residual, meFn, n_segments=30, epsilon=0.1, normalize_weights=False
    )

    if "Bold_Signal" in data_params:
        Bold_data = data_params["Bold_Signal"].squeeze()
        bold_pinn, Bold_pinn_time = tofit(
            data_params["Overallstim"],
            hrf_pinn,
            data_params["Overall_stim_time"][-1] + 0.01,
        )
        samples_index, _ = timeBall(data_params["Bold_data_time"], Bold_pinn_time)
        bold_pinn_sampled = bold_pinn[samples_index]
        offset = -torch.mean(bold_pinn_sampled - Bold_data)
        bold_loss = meFn(offset + bold_pinn_sampled, Bold_data)
    else : 
        bold_loss = torch.zeros_like(ode_loss, requires_grad=True)

    n6 = len(Balloon_params["I"]) // 6

    # Physics violation penalties
    viol_df = -dfindt[:n6].squeeze()[dfindt[:n6].squeeze() <= 0]
    viol_dm = -dmdt[:n6].squeeze()[dmdt[:n6].squeeze() <= 0]
    viol_dv = -dvdt[:n6].squeeze()[dvdt[:n6].squeeze() <= 0]

    mask_f1 = (model.f >= 1).squeeze()
    viol_m1 = (1 - model.m.squeeze()[mask_f1])
    viol_m1 = viol_m1[viol_m1 > 0]

    viol_v = -model.v.squeeze()[model.v.squeeze() < 0]

    violations = [v for v in [viol_df, viol_dm, viol_dv, viol_m1, viol_v] if v.numel() > 0]

    if violations:
        all_violations = torch.cat(violations)
        other_loss = meFn(all_violations, torch.zeros_like(all_violations))
    else:
        other_loss = ode_loss * 0.0  # stays on graph, zero contribution

    max_elements = Balloon_params["I"].size()[0]

    # Initial condition and boundary losses
    if (Balloon_params["I"][Balloon_params["time_border_mask"]]).sum() == 0:
        tmp_index = torch.arange(max_elements)[Balloon_params["time_border_mask"]]
        output_border = torch.index_select(
            torch.cat([model.f, model.m, model.v, model.q, hrf_pinn + 1], dim=1),
            dim=0, index=tmp_index,
        )
        ic_loss = Balloon_params["time_border_mask"].sum() * meFn(
            output_border, torch.ones_like(output_border)
        )

        tmp_border = torch.cat(
            [d2findt2, d2mtdt2, dfindt, dmdt, dvdt, dqdt, dpredt_num], dim=1
        )
        doutputdt_border = torch.index_select(tmp_border, dim=0, index=tmp_index)
        border_loss = meFn(
            doutputdt_border, torch.zeros_like(doutputdt_border)
        )
    else:
        ic_loss     = torch.zeros_like(ode_loss, requires_grad=True)
        border_loss = torch.zeros_like(ode_loss, requires_grad=True)

    loss_dict = {
        "ode":    ode_loss,
        "bold":   bold_loss,
        "ic":     ic_loss,
        "border": border_loss,
        "other":  other_loss,
    }

    loss_dict["total"] = (
          amp["ode"]    * loss_weights["ode"][-1]    * loss_dict["ode"]
        + amp["bold"]   * loss_weights["bold"][-1]   * loss_dict["bold"]
        + amp["ic"]     * loss_weights["ic"][-1]     * loss_dict["ic"]
        + amp["border"] * loss_weights["border"][-1] * loss_dict["border"]
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
    domain=(0, 30),
    random=False,
    every=3,
    loss_weights={"ode": [1.], "ic": [1.], "border": [1.], "bold": [1.], "other": [1.]},
    scheduler=None,
    dtype=torch.float32,
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
    domain : tuple of float
        Physical time domain ``(t_start, t_end)``.
    random : bool
        If ``True``, add stochastic jitter to time points each iteration.
    every : int
        Print progress and plot every *every* iterations (0 = silent).
    loss_weights : dict of {str: list of float}
        Initial per-component weight lists.
    scheduler : torch.optim.lr_scheduler or None
        Optional learning-rate scheduler.
    dtype : torch.dtype
        Tensor data type.

    Returns
    -------
    dict of {str: list of float}
        Merged loss trace (component traces + ``'total'`` key).
    """
    amp = {
        "ode":    1e1,
        "bold":   1e0,
        "ic":     1e0,
        "border": 1e0,
        "other":  0e1,
    }
    amp_p = torch.distributions.beta.Beta(6, 2)
    loss_trace  = {key: [] for key in loss_weights.keys()}
    total_trace = {"total": []}
    # data = {}

    if ("Bold_ode" in data_params) & ("Bold_Signal" in data_params):
        raise TypeError(
            "Bold and Bold_segments cannot both be included during training"
        )

    # data.update({"I": Balloon_params["I"]})
    # if "f_in" in data_params:
    #     data["f_in"] = data_params["ft"]
    # if "m" in data_params:
    #     data["m"] = data_params["m"]
    # if ("v" in data_params) and ("q" in data_params):
    #     data["v"] = data_params["v"]
    #     data["q"] = data_params["q"]

    max_elements = Balloon_params["I"].size()[0]
    first_non_zero_index = torch.argmax(Balloon_params["I"]) - 1

    pinn_time = (torch.arange(0, max_elements) / max_elements).requires_grad_(False).to(dtype)
    pinn_time = ((pinn_time - pinn_time.mean()) / pinn_time.std()).view(-1, 1)

    Balloon_params.update({
        "first_non_zero_t": pinn_time[first_non_zero_index],
        "t_scale": (pinn_time[-1] - pinn_time[0]) / (domain[1] - domain[0]),
        "time_border_mask": (
            (pinn_time.squeeze() <= pinn_time[first_non_zero_index])
            | (pinn_time.squeeze() >= pinn_time[-1])
        ),
    })

    data_params["Bold_Signal"] = torch.as_tensor(data_params["Bold_Signal"]).to(dtype).view(-1, 1)

    if not random:
        epsilon = torch.zeros(num_iter)
    else:
        distr = torch.distributions.beta.Beta(5, 5)
        epsilon = (distr.sample([num_iter]) - distr.mean) / max_elements

    if "Bold_Signal" in data_params:
        time_bf_stim = data_params["TR"]
        Bold_segments, time_corrected = segmentData(
            data_params["Bold_Signal"],
            Sti_Onsets=data_params["Sti_Onsets"],
            time_bf_stim=time_bf_stim,
            t0s=data_params["t0"],
            TR=data_params["TR"],
        )
        optimal_combinatory = round(len(Bold_segments) / 2)
        data_params.update({"index_size": optimal_combinatory})
        
        time_max = torch.ceil(
            torch.stack([i.max() for i in time_corrected]).max()
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
            data_params["Bold_Signal"].shape[0] + (data_params["t0"] // data_params["TR"]),
            Sti_Onsets=data_params["Sti_Onsets"],
            TR=data_params["TR"],
            block_len=data_params["stim_length [seg]"],
            stmxblck=data_params["stim_x_block"],
        )

        data_params.update({
            "Bold_data_time":   Bold_data_time,
            "stimulus":         stimulus.view(-1, 1),
            "stimulus_time":    stimulus_time,
            "Overallstim":      Overall_stimuli,
            "Overall_stim_time": Overall_stim_time,
        })
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    for i in tqdm(range(num_iter)):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # if "Bold_ode" in data_params:
        #     index = training_data(data, num_points=30)
        #     data_params.update({"index": index})
        
        Balloon_params.update({
            "t": torch.clamp(pinn_time + epsilon[i], min=pinn_time[0].item())
        })
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss_dict = lossfn(
                model,
                Balloon_params=Balloon_params,
                data_params=data_params,
                # data=data,
                loss_weights=loss_weights,
                amp=amp,
                domain=domain,
                random=random,
                dtype=dtype,
                meFn=data_params["errorFn"],
            )

        # Dynamic amplitude adjustment (activated after warm-up)
        if i > 100:
            amp_pi = amp_p.sample()
            tmp = np.mean(loss_trace["bold"][-11:-1]) / (
                np.mean(
                    loss_trace["ode"][-11:-1]
                    + loss_trace["ic"][-11:-1]
                    + loss_trace["border"][-11:-1]
                )
            )
            amp_i = (amp_pi * amp_i + (1 - amp_pi) * tmp).item()
        else:
            amp_i = 1e3

        for k in loss_weights.keys():
            if k != "bold" and loss_weights["bold"][0] > 0.0:
                amp[k] = np.round(amp_i, 1) if np.round(amp_i, 1) > 1 else 1.0
        
        scaler.scale(loss_dict["total"]).backward()                                                                  
        scaler.unscale_(optimizer)                                                                                   
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)                                             
        scaler.step(optimizer)    # wraps optimizer.step()                                                           
        scaler.update()           # adjusts scale factor
                
        if scheduler is not None:
            scheduler.step()
        total_trace['total'].append(loss_dict['total'].detach().item())
        for name_t, name_d in zip(loss_trace.keys(), loss_dict.keys()):
            if name_t == name_d:
                loss_trace[name_t].append(loss_dict[name_d].detach().item())
            else:
                raise ValueError(f"Loss name mismatch: {name_t} != {name_d}")

        if every != 0 and (i + 1) % every == 0:
            print(f"{i + 1}th Iter:")
            print(
                "total {:.4e}, ode:{:.3e}, bold:{:.3e}, Dic:{:.3e}, Cic:{:.3e}, other:{:.3e}".format(
                    loss_dict["total"].detach().item(),
                    loss_dict["ode"].detach().item(),
                    loss_dict["bold"].detach().item(),
                    loss_dict["ic"].detach().item(),
                    loss_dict["border"].detach().item(),
                    loss_dict["other"].detach().item(),
                )
            )
            print(
                "total weights, ode:{:.3e}, bold:{:.3e}, Dic:{:.3e}, Cic:{:.3e}, other:{:.3e}".format(
                    loss_weights["ode"][-1],
                    loss_weights["bold"][-1],
                    loss_weights["ic"][-1],
                    loss_weights["border"][-1],
                    loss_weights["other"][-1],
                )
            )
            print("Physics amp", amp)
            plot_balloon_fitting(
                model=model,
                t_normalized=pinn_time.requires_grad_(False),
                domain=domain,
                stimulus=stimulus,
                title="Training Progress",
                data_params=data_params if "Bold_Signal" in data_params else None,
                first_non_zero_index=first_non_zero_index,
                iteration=i,
                show_bold_signal="Bold_Signal" in data_params,
                dtype=dtype,
            )

    loss_trace.update(total_trace)
    return loss_trace
