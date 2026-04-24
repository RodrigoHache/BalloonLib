from __future__ import annotations

import colorsys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator

from balloonlib.utils import scale_domains, tensor2np, timeBall, tofit
from balloonlib.data import segmentData, experimental_stims

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

# Base hues for each dataset (HSL hue in [0, 1])
_DATASET_HUES = [0.60, 0.08]          # blue-ish, orange-ish
_SATURATION   = 0.85
_LIGHTNESS_LEVELS = [0.30, 0.50, 0.68]  # dark → light for triplets 0, 1, 2


def _palette(n_datasets: int, n_triplets_per_dataset: int) -> list[list[tuple]]:
    """
    Build a 2-D list of RGB colours  palette[dataset_idx][triplet_idx].

    Parameters
    ----------
    n_datasets : int
        Number of datasets (≤ len(_DATASET_HUES)).
    n_triplets_per_dataset : int
        Number of triplet curves per dataset (≤ len(_LIGHTNESS_LEVELS)).

    Returns
    -------
    list[list[tuple]]
        RGB tuples, each in [0, 1]³.
    """
    palette: list[list[tuple]] = []
    for ds_idx in range(n_datasets):
        hue = _DATASET_HUES[ds_idx % len(_DATASET_HUES)]
        triplet_colors: list[tuple] = []
        for tr_idx in range(n_triplets_per_dataset):
            lightness = _LIGHTNESS_LEVELS[tr_idx % len(_LIGHTNESS_LEVELS)]
            rgb = colorsys.hls_to_rgb(hue, lightness, _SATURATION)
            triplet_colors.append(rgb)
        palette.append(triplet_colors)
    return palette


# ---------------------------------------------------------------------------
# _to_numpy
# ---------------------------------------------------------------------------

def _to_numpy(data) -> np.ndarray:
    """Convert tensor or array to a 2-D numpy array (n_signals, signal_length)."""
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)

    arr = np.squeeze(arr)

    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr if arr.shape[0] <= arr.shape[1] else arr.T
    raise TypeError(
        f"Expected 1-D or 2-D data, got shape {arr.shape}."
    )


# ---------------------------------------------------------------------------
# plotSignals
# ---------------------------------------------------------------------------

_Y_LIMITS: dict[str, tuple[float, float]] = {
    "f":   (0.970, 1.165),
    "m":   (0.970, 1.165),
    "v":   (0.995, 1.025),
    "q":   (0.995, 1.025),
    "hrf": (-0.0045, 0.010),
}


def plotSignals(
    datasets: list[dict],
    key: str,
    Subject: str,
    dataset_labels: list[str] | None = None,
    grid_mode: str = "11x11",
    figsize: tuple[float, float] | None = None,
    linewidth: float = 1.2,
    alpha: float = 0.75,
    custom_ylim: dict | None = None,
) -> pd.DataFrame:
    """
    Overlay training results from multiple datasets and triplets on a grid of
    subplots, using HLS colour coding to distinguish sets and triplets.

    Parameters
    ----------
    datasets : list[dict]
        List of 1 or 2 dicts.  Each dict maps triplet labels (str) to
        sub-dicts containing the signal array/tensor under ``key``.
        Each sub-dict may optionally contain ``d_comparison`` (float) and
        ``d_reconstruction`` (float) scalar metrics.

        Example::

            datasets = [
                {"A1": {"f": arr, "d_comparison": 1e-3, "d_reconstruction": 2e-4},
                 "A2": {"f": arr, "d_comparison": 1e-3, "d_reconstruction": 2e-4}},
                {"B1": {"f": arr, "d_comparison": 5e-4, "d_reconstruction": 1e-4}},
            ]

    key : str
        Which signal to plot (e.g. 'f', 'm', 'v', 'q', 'hrf').
    Subject : str
        Subject label used in the figure title (e.g. "Patient 01 Stroke").
    dataset_labels : list[str], optional
        Human-readable names for each dataset.  Defaults to ["Dataset 0", ...].
    grid_mode : str
        ``"11x11"`` (default) or ``"2x5"``.  See module docstring.
    figsize : tuple[float, float], optional
        Figure size.  Defaults to (25, 25) for 11×11 and (18, 7) for 2×5.
    linewidth : float
        Curve line width.  Default 1.2.
    alpha : float
        Curve opacity.  Default 0.75.
    custom_ylim : dict, optional
        Per-key y-axis overrides merged with ``_Y_LIMITS``.

    Returns
    -------
    pd.DataFrame
        Tidy table with one row per (cell, dataset, replicate).
        Columns: ``cell_idx``, ``p``, ``1-p``, ``dataset``, ``replicate``,
        ``d_reconstruction``, ``d_comparison``.

    Notes
    -----
    * **Hue** encodes dataset identity.
    * **Lightness** encodes triplet rank within a dataset (darker = first).
    * Each subplot is annotated with the **mean** of d_reconstruction and
      d_comparison across all replicates of that dataset (4 annotations total
      per cell: 2 datasets × 2 metrics).
    * Curves missing from a dataset/triplet for a particular grid cell are
      silently skipped.
    """
    if not isinstance(datasets, list) or len(datasets) == 0:
        raise ValueError("`datasets` must be a non-empty list of dicts.")

    # ── Grid mode setup ───────────────────────────────────────────────────────
    if grid_mode == "11x11":
        x        = torch.arange(0.00, 1.1, 0.1)
        w_Raws   = torch.cartesian_prod(x, x.flip(dims=(0,)))  # (121, 2)
        n_rows, n_cols   = 11, 11
        default_figsize  = (25, 25)
        def title_fmt(coord):
            return f"({coord[0]:.1f}, {coord[1]:.1f})"

    elif grid_mode == "2x5":
        p        = torch.arange(0.00, 1.0, 0.1)               # 10 values: 0.0 … 0.9
        w_Raws   = torch.stack([p, 1 - p], dim=1)             # (10, 2)
        n_rows, n_cols   = 2, 5
        default_figsize  = (18, 7)
        def title_fmt(coord):
            return f"p={coord[0]:.1f}"

    else:
        raise ValueError(f"grid_mode must be '11x11' or '2x5', got '{grid_mode!r}'.")

    if figsize is None:
        figsize = default_figsize

    n_cells = n_rows * n_cols

    # ── Palette & y-limits ────────────────────────────────────────────────────
    n_datasets = len(datasets)
    n_triplets  = max(len(d) for d in datasets)
    palette = _palette(n_datasets, n_triplets)

    ylim_map = {**_Y_LIMITS, **(custom_ylim or {})}
    ylim = ylim_map.get(key, (None, None))

    if dataset_labels is None:
        dataset_labels = [f"Dataset {i}" for i in range(n_datasets)]

    def _get_at(val, cell_idx: int):
        """
        Extract the scalar at position cell_idx from a metric value.
        val can be a list, np.ndarray, torch.Tensor, or a plain float.
        Returns float, or None if val is None / index is out of range.
        """
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().numpy()
        arr = np.asarray(val)
        if arr.ndim == 0:               # plain scalar
            return float(arr)
        if cell_idx < len(arr):
            return float(arr[cell_idx])
        return None

    # ── Accumulate DataFrame rows ─────────────────────────────────────────────
    df_rows: list[dict] = []

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    plt.subplots_adjust(hspace=0.45, wspace=0.35)

    # Track max signal length for x-axis
    max_len: int = 0

    # ── Draw curves ───────────────────────────────────────────────────────────
    for ds_idx, dataset in enumerate(datasets):
        triplet_keys = list(dataset.keys())

        for tr_idx, triplet_key in enumerate(triplet_keys):
            color = palette[ds_idx][tr_idx]
            label = f"{dataset_labels[ds_idx]} \u2013 {triplet_key}"
            sub_dict = dataset[triplet_key]

            if key not in sub_dict:
                continue

            data_arr = _to_numpy(sub_dict[key])
            n_signals, signal_length = data_arr.shape
            max_len = max(max_len, signal_length)

            _raw_rec = sub_dict.get("d_reconstruction")
            _raw_cmp = sub_dict.get("d_comparison")
            _has_metrics = (_raw_rec is not None) or (_raw_cmp is not None)

            # One DataFrame row per (cell_idx, replicate): value = val[cell_idx]
            if _has_metrics:

                for cell_idx in range(min(n_signals, n_cells)):
                    d_rec = _get_at(_raw_rec, cell_idx)
                    d_cmp = _get_at(_raw_cmp, cell_idx)
                    if d_rec is not None or d_cmp is not None:
                        coord = w_Raws[cell_idx].numpy() if cell_idx < len(w_Raws) else [None, None]
                        df_rows.append({
                            "cell_idx":         cell_idx,
                            "p":                float(coord[0]) if coord[0] is not None else None,
                            "1-p":              float(coord[1]) if coord[1] is not None else None,
                            "dataset":          dataset_labels[ds_idx],
                            "replicate":        triplet_key,
                            "d_reconstruction": d_rec,
                            "d_comparison":     d_cmp,
                        })

            # Plot
            for cell_idx, ax in enumerate(axes.ravel()):
                if cell_idx >= n_signals or cell_idx >= n_cells:
                    break
                ax.plot(
                    data_arr[cell_idx],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    label=label if cell_idx == 0 else None,  # label once
                )

    # ── Annotate each cell: mean of val[cell_idx] across replicates ───────────
    # Plot N annotation: mean = (db['0']['metric'][N] + db['1']['metric'][N] + ...) / n_replicates
    _annot_y = [0.04, 0.22]   # Dataset 0 near bottom, Dataset 1 just above it
    for cell_idx, ax in enumerate(axes.ravel()):
        if cell_idx >= n_cells:
            break
        for ds_idx, dataset in enumerate(datasets):
            d_cmps_at = [_get_at(v["d_comparison"],     cell_idx) for v in dataset.values()
                         if "d_comparison"     in v and _get_at(v["d_comparison"],     cell_idx) is not None]
            d_recs_at = [_get_at(v["d_reconstruction"], cell_idx) for v in dataset.values()
                         if "d_reconstruction" in v and _get_at(v["d_reconstruction"], cell_idx) is not None]
            if not d_cmps_at and not d_recs_at:
                continue
            mean_cmp = float(np.mean(d_cmps_at)) if d_cmps_at else None
            mean_rec = float(np.mean(d_recs_at)) if d_recs_at else None
            ann_color = palette[ds_idx][0]
            msg = (
                f"{dataset_labels[ds_idx]}\n"
                + (f"  HRF: {mean_cmp:.3e}"      if mean_cmp is not None else "")
                + (f"\n  Bold: {mean_rec:.3e}" if mean_rec is not None else "")
            )
            ax.annotate(
                msg,
                xy=(0.02, _annot_y[ds_idx % len(_annot_y)]),
                xycoords="axes fraction",
                fontsize=5,
                color=ann_color,
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.55, lw=0),
            )

    # ── Cosmetics for every subplot ────────────────────────────────────────────
    for cell_idx, ax in enumerate(axes.ravel()):
        if cell_idx >= n_cells:
            break

        # Title using grid coordinate
        if cell_idx < len(w_Raws):
            coord = w_Raws[cell_idx].numpy()
            ax.set_title(title_fmt(coord), fontsize=7, pad=2)

        ax.tick_params(axis="both", labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ylim[0] is not None:
            ax.set_ylim(*ylim)
        if max_len > 0:
            ax.set_xlim(-1, max_len)

    # ── Legend (first axis) ───────────────────────────────────────────────────
    axes.ravel()[0].legend(
        fontsize=6,
        loc="upper right",
        framealpha=0.6,
        handlelength=1.5,
    )

    # ── Remove unused subplots ────────────────────────────────────────────────
    n_used = max(
        (
            _to_numpy(datasets[ds][tk][key]).shape[0]
            for ds in range(n_datasets)
            for tk in datasets[ds]
            if key in datasets[ds][tk]
        ),
        default=0,
    )
    for j in range(n_used, n_cells):
        fig.delaxes(axes.ravel()[j])

    # ── Global title ──────────────────────────────────────────────────────────
    n_ds_str = " vs ".join(dataset_labels)
    fig.suptitle(
        f"Signal '{key}' — {Subject}\n"
        f"{n_ds_str}  |  {n_triplets} triplets  |  {n_used} cells  [{grid_mode}]",
        y=1.01,
        fontsize=13,
    )

    plt.tight_layout()
    plt.show()

    # ── Build and return the summary DataFrame ────────────────────────────────
    df = pd.DataFrame(df_rows, columns=[
        "cell_idx", "p", "1-p", "dataset", "replicate",
        "d_reconstruction", "d_comparison",
    ])
    return df


# ============================================================================
# Migrated from balloonpinnlib.py
# ============================================================================

def plot_trace(loss_trace: dict, title: str, step_size: int = 0):
    """Plot training loss traces on a two-panel figure.

    Parameters
    ----------
    loss_trace : dict
        Dictionary with keys ``'total'``, ``'ode'``, ``'ic'``, ``'border'``,
        ``'bold'``, and ``'other'``.  Each value is a list of scalar loss
        values recorded per iteration.
    title : str
        Figure super-title.
    step_size : int, optional
        Unused; reserved for future vertical-line annotations.

    Notes
    -----
    Left panel: total loss (log scale).
    Right panel: individual loss components (log scale).
    """
    for key, value in loss_trace.items():
        if key == "total":
            total_trace = value
        elif key == "ode":
            ode_trace = value
        elif key == "ic":
            ic_trace = value
        elif key == "border":
            border_loss_trace = value
        elif key == "bold":
            bold_loss_trace = value
        elif key == "other":
            other_loss_trace = value
        else:
            raise ValueError("Unknown loss key: " + key)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.85)

    num_iter = len(total_trace)
    xaxis = range(1, num_iter + 1)

    ax[0].set_yscale("log")
    ax[0].plot(xaxis, total_trace, label="Total Loss")
    ax[0].set_xlabel("Number of iterations", fontsize=14)
    ax[0].set_ylabel("Loss", fontsize=12)
    ax[0].set_title("Total Loss vs. Iteration")
    ax[0].legend()

    ax[1].set_yscale("log")
    ax[1].plot(xaxis, ode_trace, label="ODE Loss", alpha=0.9)
    ax[1].plot(xaxis, ic_trace, label="Dirichlet IC Loss", alpha=0.6)
    ax[1].plot(xaxis, border_loss_trace, label="Cauchy IC Loss", alpha=0.3)
    if bold_loss_trace is not None:
        ax[1].plot(xaxis, bold_loss_trace, label="Data Loss", alpha=0.4)
    ax[1].set_xlabel("Number of iterations", fontsize=14)
    ax[1].set_ylabel("Loss", fontsize=12)
    ax[1].set_ylim(1e-18, 1e2)
    ax[1].set_title("ODE, DIC, CIC and Data Loss vs. Iteration")
    ax[1].legend(fontsize=12)

    plt.show()


def plot_balloon_fitting(
    model,
    t_normalized,
    domain,
    stimulus=None,
    title="BOLD Fitting Results",
    numerical_solutions=None,
    data_params=None,
    first_non_zero_index=None,
    iteration=None,
    show_bold_signal=False,
    dtype=torch.float32,
):
    """Visualise Balloon-PINN model fitting results.

    Creates a multi-panel figure showing model state variables (f, m, v, q),
    HRF predictions, and (optionally) BOLD fitting against experimental data.
    Supports both training-progress and evaluation modes.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PINN model exposing ``predictor()``, ``f``, ``m``, ``v``, ``q``.
    t_normalized : torch.Tensor
        Normalised time points, shape ``(N, 1)`` or ``(N,)``.
    domain : tuple of float
        Physical time domain ``(t_min, t_max)`` used to rescale
        ``t_normalized`` for plotting.
    stimulus : torch.Tensor or None
        Stimulus tensor; required when ``model.impulse`` is ``True``.
    title : str
        Figure super-title.
    numerical_solutions : dict or None
        Ground-truth solutions keyed by ``'f'``, ``'m'``, ``'v'``, ``'q'``,
        ``'bold'`` for overlay comparison.
    data_params : dict or None
        Experimental parameters needed for BOLD subplots.  Required keys
        when ``show_bold_signal=True``:
        ``'Sti_Onsets'``, ``'TR'``, ``'stim_length [seg]'``,
        ``'Bold_Signal'``, ``'Bold_data_time'``, ``'Overallstim'``,
        ``'Overall_stim_time'``, ``'stimulus'``, ``'stimulus_time'``.
    first_non_zero_index : int or None
        Index of first stimulus onset (computed automatically if ``None``).
    iteration : int or None
        Current training iteration (shown in subplot titles).
    show_bold_signal : bool
        If ``True``, add BOLD fitting subplots (requires ``data_params``).
    dtype : torch.dtype
        Tensor data type.

    Notes
    -----
    When ``show_bold_signal=True`` the layout is a 3 x 4 GridSpec with
    five panels (f/m, v/q, HRF, single-trial BOLD, overall BOLD).
    Otherwise a simple 1 x 3 layout is used.
    """
    # Compute time points in original domain
    
    t_plot, plot_scale = scale_domains(
        t_normalized.squeeze(),
        original_domains=t_normalized[[0, -1]].squeeze().tolist(),
        new_domains=domain,
    )
    t_plot = tensor2np(t_plot)
    plot_scale = plot_scale.item()
    
    # Prepare model inputs
    if hasattr(model, "impulse") and model.impulse:
        if stimulus is None:
            raise ValueError("stimulus must be provided when model.impulse is True")
        if t_normalized.ndim == 1:
            t_normalized = t_normalized.view(-1, 1)
        inputs = torch.cat([t_normalized, stimulus.view(-1, 1)], dim=1)
    else:
        inputs = t_normalized.view(-1, 1)

    if first_non_zero_index is None:
        first_non_zero_index = 0
    
    model.eval()
    with torch.no_grad():
        pred, _ = model(inputs)
        r_pred = tensor2np(torch.cat([model.f, model.m], dim=1))
        vq_pred = tensor2np(torch.cat([model.v, model.q], dim=1))
        hrf_predict = model.predictor()
        hrf_predict_linear = model.predictor(linear=True)
        hrf_pred_np = tensor2np(hrf_predict)
        hrf_pred_linear_np = tensor2np(hrf_predict_linear)
    
    if show_bold_signal:
        fig = plt.figure(figsize=(14, 6), layout="constrained")
        gs = GridSpec(3, 4, figure=fig)
        ax0 = fig.add_subplot(gs[0:2, 0])
        ax1 = fig.add_subplot(gs[0:2, 1])
        ax2 = fig.add_subplot(gs[0:2, 2])
        ax3 = fig.add_subplot(gs[0:2, 3])
        ax4 = fig.add_subplot(gs[2, :])
        ax0t = ax0.twiny()
        ax1t = ax1.twiny()
        ax2t = ax2.twiny()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(14, 6))
        ax0, ax1, ax2 = axes.flatten()
        ax0t = ax0.twiny()
        ax1t = ax1.twiny()
        ax2t = ax2.twiny()

    if iteration is not None:
        plt.suptitle(f"{title} - Iteration {iteration}")
    else:
        plt.suptitle(title)
    
    # Subplot 0: f and m
    ax0.plot(t_plot, r_pred[:, 0], lw=1.5, alpha=0.7, label="PINN f_in")
    ax0.plot(t_plot, r_pred[:, 1], lw=1.5, alpha=0.7, label="PINN m")
    if numerical_solutions is not None and "f" in numerical_solutions and "m" in numerical_solutions:
        ax0.plot(t_plot, tensor2np(numerical_solutions["f"]), "--", lw=1,
                 c="midnightblue", label="Numerical f_in")
        ax0.plot(t_plot, tensor2np(numerical_solutions["m"]), "--", lw=1,
                 c="midnightblue", label="Numerical m")
    ax0t.axvline(x=t_normalized[first_non_zero_index].item(), color="orange", ls="-.")
    ax0.axvline(x=t_plot[first_non_zero_index], color="r", ls="--")
    ax0.axhline(y=1 + 1e-4, color="r", ls="--")
    ax0.axhline(y=1 - 1e-4, color="r", ls="--")
    ax0.xaxis.set_minor_locator(MultipleLocator(2.5))
    ax0.grid(visible=True, which="both")
    ax0.legend(fontsize=10)
    ax0.set_xlabel("PI time")
    ax0.set_title("f_in and m")
    new_tlims = t_normalized[0].item() + (np.array(ax0.get_xlim()) - t_plot[0]) / plot_scale
    new_tticks = np.round(t_normalized[0].item() + (ax0.get_xticks() - t_plot[0]) / plot_scale, 2)[1:-1]
    ax0t.set_xlabel("NN time")
    ax0t.set_xlim(new_tlims)
    ax0t.set_xticks(new_tticks)
    
    # Subplot 1: v and q
    ax1.plot(t_plot, vq_pred[:, 0], lw=1.5, alpha=0.7, label="PINN v")
    ax1.plot(t_plot, vq_pred[:, 1], lw=1.5, alpha=0.7, label="PINN q")
    if numerical_solutions is not None and "v" in numerical_solutions and "q" in numerical_solutions:
        ax1.plot(t_plot, tensor2np(numerical_solutions["v"]), "--", lw=1,
                 c="midnightblue", label="Numerical v")
        ax1.plot(t_plot, tensor2np(numerical_solutions["q"]), "--", lw=1,
                 c="midnightblue", label="Numerical q")
    ax1.axvline(x=t_plot[first_non_zero_index], color="r", ls="--")
    ax1.axhline(y=1 + 1e-4, color="r", ls="--")
    ax1.axhline(y=1 - 1e-4, color="r", ls="--")
    ax1.xaxis.set_minor_locator(MultipleLocator(2.5))
    ax1.grid(visible=True, which="both")
    ax1.legend(fontsize=10)
    ax1.set_xlabel("PI time")
    ax1.set_title("v and q")
    ax1t.set_xlabel("NN time")
    ax1t.set_xlim(new_tlims)
    ax1t.set_xticks(new_tticks)
    
    # Subplot 2: HRF
    ax2.plot(t_plot, hrf_pred_np, lw=1.5, alpha=0.7, label="PINN HRF")
    ax2.plot(t_plot, hrf_pred_linear_np, lw=1.5, alpha=0.7, label="PINN HRF (linear)")
    if numerical_solutions is not None and "bold" in numerical_solutions:
        ax2.plot(t_plot, tensor2np(numerical_solutions["bold"]), "--", lw=1,
                 c="midnightblue", label="Numerical HRF")
    ax2.axvline(x=t_plot[first_non_zero_index], color="r", ls="--")
    ax2.axhline(y=1e-4, color="r", ls="--")
    ax2.axhline(y=-1e-4, color="r", ls="--")
    ax2.xaxis.set_minor_locator(MultipleLocator(2.5))
    ax2.grid(visible=True, which="both")
    ax2.legend(fontsize=10)
    ax2.set_xlabel("PI time")
    ax2.set_title("HRF")
    ax2t.set_xlabel("NN time")
    ax2t.set_xlim(new_tlims)
    ax2t.set_xticks(new_tticks)
    
    # Subplots 3 and 4: BOLD signal fitting (if requested)
    if show_bold_signal:
        if data_params is None:
            raise ValueError("data_params must be provided when show_bold_signal=True")

        Bold_Signal = data_params.get("Bold_Signal")
        Sti_Onsets = data_params.get("Sti_Onsets")
        TR = data_params.get("TR")
        stim_length = data_params.get("stim_length [seg]")
        
        if any(x is None for x in [Bold_Signal, Sti_Onsets, TR, stim_length]):
            raise ValueError(
                "data_params must contain 'Bold_Signal', 'Sti_Onsets', "
                "'TR', and 'stim_length [seg]' keys"
            )
        
        if "stimulus" in data_params and "stimulus_time" in data_params:
            stimulus_single = data_params["stimulus"]
            stimulus_time = data_params["stimulus_time"]
        else:
            time_bf_stim = data_params.get("TR", TR)
            time_max = torch.ceil(torch.tensor([domain[1]])).to(dtype=torch.int32)
            stimulus_single, stimulus_time = experimental_stims(
                time_max.item() / TR,
                Sti_Onsets=[time_bf_stim],
                TR=TR,
                block_len=stim_length,
                stmxblck=data_params.get("stim_x_block", 1),
            )

        single_trial_bold, trial_time = tofit(
            stimulus_single, hrf_predict, stimulus_time[-1].item() + 0.01
        )
        trial_time_np = tensor2np(trial_time)
        
        if "Overallstim" in data_params and "Overall_stim_time" in data_params:
            Overall_stimuli = data_params["Overallstim"]
            Overall_stim_time = data_params["Overall_stim_time"]
        else:
            Overall_stimuli, Overall_stim_time = experimental_stims(
                Bold_Signal.shape[0] + (data_params.get("t0", 0) // TR),
                Sti_Onsets=Sti_Onsets,
                TR=TR,
                block_len=stim_length,
                stmxblck=data_params.get("stim_x_block", 1),
            )

        Overall_bold_pinn, Bold_pinn_time = tofit(
            Overall_stimuli,
            hrf_predict,
            Overall_stim_time[-1].item() + 0.01,
        )
        
        Bold_data_time = data_params.get("Bold_data_time")
        if Bold_data_time is not None:
            samples_index, _ = timeBall(Bold_data_time, Bold_pinn_time)
            Bold_arr = Bold_Signal if isinstance(Bold_Signal, np.ndarray) else tensor2np(Bold_Signal)
            offset = -np.mean(tensor2np(Overall_bold_pinn[samples_index]) - Bold_arr)
        else:
            offset = 0.0

        ax3.plot(stimulus_time.cpu(), tensor2np(stimulus_single), alpha=0.7,
                 color="green", label="Stimulus")
        ax3.plot(trial_time_np, offset + tensor2np(single_trial_bold),
                 alpha=0.7, label="Estimated BOLD")
        
        if hasattr(Bold_Signal, "__len__"):
            time_bf_stim = data_params.get("TR", TR)
            Bold_segments, time_corrected = segmentData(
                Bold_Signal,
                Sti_Onsets=Sti_Onsets,
                time_bf_stim=time_bf_stim,
                t0s=data_params.get("t0", 0),
                TR=TR,
            )
            for k, j in zip(time_corrected, Bold_segments):
                ax3.scatter(tensor2np(k), tensor2np(j), color="orange")

        ax3.axvline(x=trial_time_np[torch.argmax(stimulus_single).item()], color="r", ls="--")
        ax3.xaxis.set_minor_locator(MultipleLocator(2.5))
        ax3.grid(visible=True, which="both")
        ax3.legend(fontsize=10)
        ax3.set_xlabel("time")
        ax3.set_title("Estimated BOLD, single stimulus")

        if Bold_data_time is not None:
            Bold_arr = Bold_Signal if isinstance(Bold_Signal, np.ndarray) else tensor2np(Bold_Signal)
            ax4.scatter(tensor2np(Bold_data_time), Bold_arr, label="data")
        ax4.plot(tensor2np(Bold_pinn_time), offset + tensor2np(Overall_bold_pinn),
                 label="Estimated BOLD")
        ax4.plot(tensor2np(Overall_stim_time), tensor2np(Overall_stimuli),
                 label="Stimuli", color="green")
        ax4.xaxis.set_minor_locator(MultipleLocator(2.5))
        ax4.legend(fontsize=10, loc="lower center", ncol=3)
        ax4.grid(visible=True, which="both")
        ax4.set_title(f"BOLD fitting, estimation and data, after iteration n°:{iteration}")

    plt.show()


def plot_weights(
    weights_history, title, keys_to_skip, step_size: int = 1400,
):
    """Plot adaptive loss weight traces over training iterations.

    Parameters
    ----------
    weights_history : dict of {str: list of float}
        Weight history per component (keys: ``'ode'``, ``'ic'``,
        ``'border'``, ``'bold'``, ``'other'``).
    title : str
        Figure super-title.
    keys_to_skip : list of str
        Components to omit from the plot.
    step_size : int
        Interval for vertical dashed guide-lines.
    """
    keys = list(set(weights_history.keys()) - set(keys_to_skip))

    for key, value in weights_history.items():
        if key == "ode":
            ode_trace = value
        elif key == "ic":
            ic_trace = value
        elif key == "border":
            border_loss_trace = value
        elif key == "bold":
            bold_loss_trace = value
        elif key == "other":
            other_loss_trace = value
        else:
            raise ValueError("Unknown key: " + key)

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.85)

    num_iter = len(weights_history[keys[0]])
    xaxis = range(1, num_iter + 1)

    for i in range(num_iter // step_size):
        ax.axvline(x=step_size * (i + 1), color="gray", ls="--")

    ax.plot(xaxis, ode_trace, label="ODE weights", alpha=0.9)
    ax.plot(xaxis, ic_trace, label="Dirichlet IC weights", alpha=0.6)
    ax.plot(xaxis, border_loss_trace, label="Cauchy IC weights", alpha=0.3)
    if bold_loss_trace is not None:
        ax.plot(xaxis, bold_loss_trace, label="Data weights", alpha=0.4)

    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Loss Weights")
    ax.set_title("Weights Value vs. Iteration")
    ax.grid()
    ax.legend()
    ax.set_ylim(-0.1, 1.1)

    plt.show()


def plotHRFs(data):
    """Plot a 10 x 10 grid of HRF time series.

    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        HRF waveforms, shape ``(n_signals, length)`` or
        ``(length, n_signals)``.  At most 100 signals are shown.
    """
    tmp = data.squeeze() if isinstance(data, torch.Tensor) else np.squeeze(data)
    data = np.copy(tensor2np(tmp)) if isinstance(tmp, torch.Tensor) else np.copy(tmp)

    if data.ndim == 2:
        data = data if data.shape[0] < data.shape[1] else data.T
    elif data.ndim == 1:
        data = data.reshape(1, -1)
    else:
        raise TypeError(
            "Several signals should be delivered using a order-2 tensor (matrix)"
        )

    n_signals, signal_length = data.shape

    fig, axes = plt.subplots(10, 10, figsize=(25, 25))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    for idx, ax in enumerate(axes.ravel()):
        if idx < n_signals:
            ax.plot(data[idx], linewidth=1.5, color="steelblue")
            ax.set_title(f"TS-{idx}", fontsize=8, pad=2)
            ax.tick_params(axis="both", labelsize=6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_ylim(np.percentile(data, 1), np.percentile(data, 99))
            ax.set_xlim(-1, signal_length)

    fig.suptitle(
        "Time Series Array Visualization\n(100 series × 3000 points each)",
        y=1.02,
        fontsize=14,
    )

    for j in range(n_signals, 100):
        fig.delaxes(axes.ravel()[j])

    plt.tight_layout()
    plt.show()
