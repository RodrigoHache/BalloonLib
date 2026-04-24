"""
Normalized Time Training Module for Balloon PINN

This module implements time normalization strategies for training Physics-Informed
Neural Networks (PINNs) on the balloon hemodynamic model. 

KEY CONCEPT:
-----------
Neural networks train more stably with inputs in the [0, 1] range. However, physical
equations (ODEs) and visualization require the original time scale (e.g., 0-30 seconds).

STRATEGY:
--------
1. NORMALIZE time before passing to the network: t_norm = (t - t_min) / (t_max - t_min)
2. Network trains on t_norm ∈ [0, 1]
3. DENORMALIZE time for physics computations: t_real = t_norm * (t_max - t_min) + t_min
4. Use t_real for ODE residuals and visualization
5. Gradients computed using t_norm (network input)

BENEFITS:
--------
- Improved numerical stability
- Better gradient flow
- Faster convergence
- Network architecture agnostic to physical time scales

Author: Created for BalloonPINN project
Date: 2025-12-16
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
from balloonlib.balloonpinnlib import dfdt


def normalize_time(t: torch.Tensor, domain: Tuple[float, float]) -> torch.Tensor:
    """
    Normalize time from physical domain to [0, 1].
    
    Args:
        t: Time tensor in physical units (e.g., seconds)
        domain: Tuple (t_min, t_max) defining the physical time range
        
    Returns:
        t_norm: Normalized time in [0, 1]
        
    Example:
        >>> t = torch.tensor([0., 10., 20., 30.])
        >>> t_norm = normalize_time(t, domain=(0, 30))
        >>> print(t_norm)  # [0., 0.333, 0.667, 1.0]
    """
    t_min, t_max = domain
    t_norm = (t - t_min) / (t_max - t_min)
    return t_norm


def denormalize_time(t_norm: torch.Tensor, domain: Tuple[float, float]) -> torch.Tensor:
    """
    Denormalize time from [0, 1] back to physical domain.
    
    Args:
        t_norm: Normalized time in [0, 1]
        domain: Tuple (t_min, t_max) defining the physical time range
        
    Returns:
        t: Time in physical units (e.g., seconds)
        
    Example:
        >>> t_norm = torch.tensor([0., 0.333, 0.667, 1.0])
        >>> t = denormalize_time(t_norm, domain=(0, 30))
        >>> print(t)  # [0., 10., 20., 30.]
    """
    t_min, t_max = domain
    t_real = t_norm * (t_max - t_min) + t_min
    return t_real


def loss_ballon_random_NORM_TIME(
    model,
    Balloon_params,
    data_params,
    data,
    domain=(0, 30),
    random=False,
    sample_size=None,
    dtype=torch.float32,
    loss_weights=torch.tensor([0.80, 0.20], dtype=torch.float32),
    bamp=90,
    meFn=nn.MSELoss(),
):
    """
    Computes the total loss for the PINN with NORMALIZED TIME INPUTS.
    
    This is a modified version of `loss_ballon_random()` that:
    1. Maps time from [domain[0], domain[1]] to [0, 1] for network input
    2. Uses normalized time for network forward pass and gradient computation
    3. Uses denormalized (physical) time for ODE residuals and visualization
    
    TIME FLOW:
    ---------
    Physical Time (0-30s) → [Normalize] → t_norm (0-1) → Network(t_norm) 
    → Output (v, q, f, m) → [Denormalize for physics] → ODE Residuals
    → [Keep denormalized for plots] → Visualization
    
    CRITICAL: All automatic differentiation (dfdt) uses t_norm, but the values
    used in physics equations (e.g., Impulse I, time for plotting) use t_real.
    
    Args:
        model (nn.Module): The PINN model.
        Balloon_params (dict): Parameters for the Balloon model.
        data_params (dict): Parameters for experimental data.
        data (dict): Dictionary containing training data.
        domain (tuple, optional): Physical time domain (start, end). Defaults to (0, 30).
        random (bool, optional): Whether to sample time points randomly. Defaults to False.
        sample_size (int, optional): Number of collocation points. Defaults to None.
        dtype (torch.dtype, optional): Data type. Defaults to torch.float32.
        loss_weights (torch.Tensor, optional): Weights for different loss components.
        bamp (float, optional): BOLD amplitude scaling factor. Defaults to 90.
        meFn (nn.Module, optional): Loss function (e.g., MSELoss). Defaults to nn.MSELoss().
        
    Returns:
        dict: A dictionary containing 'total_loss', 'ode_loss', 'ic_loss', 
              'border_loss', and 'Bold_loss'.
    """
    max_samples = (
        data["I"].size()[0] if sample_size == None else sample_size
    )  # length of the impulse tensor (in samples)

    # ============================================
    # STEP 1: Generate time samples in [0, 1]
    # ============================================
    if not random:
        t_norm = torch.arange(0, max_samples) / max_samples
    else:
        # With jittering (your recent implementation)
        distr = torch.distributions.beta.Beta(5, 5)
        t_norm = torch.arange(0, max_samples) 
        epsilon = (distr.sample([1]) - torch.tensor(0.5))
        t_norm = (t_norm + epsilon) / max_samples

    # Require gradients on NORMALIZED time (this is what the network sees)
    t_norm = t_norm.requires_grad_(True).to(dtype).view(-1, 1)  # shape (sample_size, 1)

    # ============================================
    # STEP 2: Network forward pass with NORMALIZED time
    # ============================================
    if model.impulse:  # does the input include the Impulse?
        inputs = torch.cat([t_norm, data["I"].view(-1, 1)], dim=1).view(-1, 2)
    else:
        inputs = t_norm

    output, _ = model(inputs)  # Network sees t_norm ∈ [0, 1]
    r = output[0, :, :].view(-1, 2)  # shape (sample_size, 2) - (f, m)

    # ============================================
    # STEP 3: Compute derivatives w.r.t. NORMALIZED time
    # ============================================
    # CRITICAL: dfdt must use t_norm for autograd to work
    drdt = torch.cat([dfdt(signal=r[:, i], arg=t_norm) for i in range(2)], dim=1)
    dsdt = torch.cat([dfdt(signal=drdt[:, i], arg=t_norm) for i in range(2)], dim=1)
    if any(x is None for x in [drdt, dsdt]):
        raise ValueError("drdt or dsdt is None")

    dvdt, dqdt = [dfdt(signal=output[1, :, i], arg=t_norm) for i in range(2)]
    if any(x is None for x in [dvdt, dqdt]):
        raise ValueError("Core derivatives are None")

    dpredt_num = dfdt(signal=model.predictor(), arg=t_norm)
    if dpredt_num is None:
        raise ValueError("dpredicted/dt_num is None")

    # ============================================
    # STEP 4: DENORMALIZE time for physics and visualization
    # ============================================
    # Now we need PHYSICAL time for:
    # - ODE residual computation (using impulse I)
    # - Boundary conditions
    # - Visualization
    t_real = denormalize_time(t_norm, domain)  # shape (sample_size, 1)

    # ============================================
    # STEP 5: Compute ODE residual (using PHYSICAL time values)
    # ============================================
    I = data["I"].view(-1, 1).expand(-1, 2)  # shape (sample_size, 2)
    
    lambdar_list = torch.tensor(Balloon_params["lambdar_list"], dtype=dtype).unsqueeze(0)
    kappa_list = torch.tensor(Balloon_params["kappa_list"], dtype=dtype).unsqueeze(0)
    gamma_list = torch.tensor(Balloon_params["gamma_list"], dtype=dtype).unsqueeze(0)
    tau_m = torch.tensor(Balloon_params["tau_m_list"], dtype=dtype)
    tau_MTT = torch.tensor(Balloon_params["tau_MTT_list"], dtype=dtype)

    # NOTE: Derivatives (dvdt, dqdt) are computed w.r.t. normalized time
    # To convert to physical time derivatives: d/dt_real = d/dt_norm * (1 / time_scale)
    # where time_scale = (domain[1] - domain[0])
    time_scale = domain[1] - domain[0]
    
    # Adjust derivatives for physical time scale
    # dvdt_real = dvdt / time_scale
    # dqdt_real = dqdt / time_scale
    # For the ODE, we need: τ_MTT * dv/dt_real = f_in - f_out
    # This becomes: τ_MTT * (dv/dt_norm) / time_scale = f_in - f_out
    # Or equivalently: (τ_MTT / time_scale) * dv/dt_norm = f_in - f_out
    
    tau_MTT_scaled = tau_MTT / time_scale  # Effective tau in normalized time
    
    f_out = model.fout(output[1, :, 0], Balloon_params["alpha"], tau_m, dvdt / time_scale)
    
    core_residual = torch.cat(
        [
            ((dvdt * tau_MTT_scaled) - (r[:, 0].view(-1, 1) - f_out)).view(-1, 1),
            ((dqdt * tau_MTT_scaled) - 
             (r[:, 1].view(-1, 1) - (output[1, :, 1] / output[1, :, 0]).view(-1, 1) * f_out)).view(-1, 1),
        ],
        dim=1,
    )  # shape (sample_size, 2)

    # Find the index of the first non-zero value (using PHYSICAL time for interpretation)
    first_non_zero_index = torch.argmax(data["I"]) - 1
    print()(first_non_zero_index * time_scale / max_samples).item()
    first_non_zero_t = (first_non_zero_index * time_scale / max_samples).item()
    combined_condition = (t_real.squeeze(1) < domain[0] + first_non_zero_t) | (
        t_real.squeeze(1) >= domain[1] - 2
    )

    residual = torch.cat(
        [
            9 * (dsdt - ((lambdar_list * I) - (kappa_list * drdt) - (gamma_list * (r - 1)))),
            6 * core_residual,
            (dpredt_num - model.dpredt(dvdt=dvdt / time_scale, dqdt=dqdt / time_scale, t=t_real)),
        ],
        dim=1,
    )  # shape (sample_size, 5)

    # Penalty terms for v, m >= 1
    tmp = torch.relu(-(r[:, 1][r[:, 0] >= 1] - 1)).requires_grad_(True)
    m_lowerThan_1 = 10.0 * tmp if tmp.numel() > 0 else torch.zeros(1, requires_grad=True)
    v_lowerThan_1 = 10.0 * torch.relu(-(output[1, :, 0] - 1))[
        torch.logical_not(combined_condition)
    ].requires_grad_(True)

    ode_loss = [
        meFn(residual, torch.zeros_like(residual))
        + meFn(v_lowerThan_1, torch.zeros_like(v_lowerThan_1))
        + meFn(m_lowerThan_1, torch.zeros_like(m_lowerThan_1))
    ][0]

    if torch.isnan(ode_loss):
        raise ValueError("ode_loss is NaN! Terminating training.")

    hrf_pinn = model.predictor()

    # ============================================
    # STEP 6: Data loss (if applicable)
    # ============================================
    bold_loss = torch.zeros_like(ode_loss)
    
    # Note: Data comparison uses the same indexing, but time interpretation 
    # should use t_real for plotting/debugging purposes
    if (
        ("f" in data)
        | ("m" in data)
        | ("Bold_ode" in data)
        | (("v" in data) and ("q" in data))
    ):
        samples_index = data_params["index"]
        t_samples_real = t_real[samples_index]
        
        epsilon_val = 1e-3
        t_samples_real[t_samples_real < domain[0]] = domain[0] + epsilon_val
        t_samples_real[t_samples_real > domain[1]] = domain[1] - epsilon_val
        t_samples_real = t_samples_real.to(dtype).view(-1, 1)

        if ("f" in data) | ("m" in data):
            r_samples = r[samples_index, :].to(dtype)
            f_loss = meFn(data["f"][samples_index], r_samples[:, 0]) if "f" in data else torch.zeros_like(ode_loss)
            m_loss = meFn(data["m"][samples_index], r_samples[:, 1]) if "m" in data else torch.zeros_like(ode_loss)
            nv_ode_loss = f_loss + m_loss
        else:
            nv_ode_loss = torch.zeros_like(ode_loss)

        if ("v" in data) and ("q" in data):
            v_ode = data["v"][samples_index].to(dtype)
            q_ode = data["q"][samples_index].to(dtype)
            core_ode_loss = meFn(v_ode, model.v[samples_index]) + meFn(
                q_ode, model.q[samples_index]
            )
        else:
            core_ode_loss = torch.zeros_like(ode_loss)

        if "Bold_ode" in data:
            Bold_ode_samps = data["Bold_ode"][samples_index].view(-1, 1).to(dtype)
            hrf_pinn_samps = hrf_pinn[samples_index].view(-1, 1)
            bold_loss = meFn(bamp * hrf_pinn_samps, bamp * Bold_ode_samps)

    # ============================================
    # STEP 7: Initial/Boundary conditions
    # ============================================
    if (I[combined_condition, 0] == 0).all():
        tmp_index = torch.arange(max_samples)[combined_condition]
        output_border = torch.index_select(output, dim=1, index=tmp_index).view(-1, 4)
        hrf_pinn_border = hrf_pinn[combined_condition].requires_grad_(True).view(-1, 1)

        uber_ic = (
            torch.cat([output_border, 1 + bamp * hrf_pinn_border], dim=1)
            .requires_grad_(True)
            .view(-1, 5)
        )
        ic_loss = combined_condition.sum() * meFn(uber_ic, torch.ones_like(uber_ic))

        border_zerodS = torch.index_select(dsdt, dim=0, index=tmp_index)
        border_zerodR = torch.index_select(drdt, dim=0, index=tmp_index)
        border_zerodV = torch.index_select(dvdt, dim=0, index=tmp_index)
        border_zerodQ = torch.index_select(dqdt, dim=0, index=tmp_index)
        border_zerodP = torch.index_select(dpredt_num, dim=0, index=tmp_index)

        border_zeros = (
            torch.cat(
                [
                    border_zerodS.view(-1, 2),
                    border_zerodR.view(-1, 2),
                    border_zerodV.view(-1, 1),
                    border_zerodQ.view(-1, 1),
                    10 * border_zerodP.view(-1, 1),
                ],
                dim=1,
            )
            .requires_grad_(True)
            .view(-1, 7)
        )

        border_loss = meFn(border_zeros, torch.zeros_like(border_zeros))
    else:
        border_loss = torch.zeros_like(ode_loss, requires_grad=True)
        ic_loss = torch.zeros_like(ode_loss)

    # ============================================
    # STEP 8: Total loss
    # ============================================
    w_max = 100
    total_loss = (
        loss_weights[0] * ode_loss
        + loss_weights[1] * bold_loss
        + w_max * (ic_loss + border_loss)
    )

    return {
        "total_loss": total_loss,
        "ode_loss": ode_loss,
        "ic_loss": ic_loss,
        "border_loss": border_loss,
        "Bold_loss": bold_loss,
        "t_normalized": t_norm,  # For debugging
        "t_physical": t_real,    # For debugging/plotting
    }
