"""
Optuna Hyperparameter Optimization for BalloonPINN Multihead Model

This script uses Optuna to find optimal hyperparameters for the
Physics-Informed Neural Network (PINN) used in the Balloon hemodynamic model.

Hyperparameters searched:
    - Fourier feature parameters (scale, mapping size, learnable)
    - Activation functions (hidden, NV heads, core heads)
    - Random Weight Factorization (on/off)
    - Learning rate and optimizer (Adam / AdamW / RAdam)
    - LR scheduler type and parameters
    - Loss weights (ODE vs. IC)

Usage:
    python optimising_w_optuna.py
    python optimising_w_optuna.py --trials 100 --timeout 7200

Author: BalloonPINN Project
Date: January 2026 (updated April 2026)
"""

import os
import pickle
import argparse
from typing import Dict, Any, Optional

import optuna
from optuna.trial import TrialState
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ── BalloonLib imports (direct submodules — no shim)
from balloonlib import balloonmodellib as bml
from balloonlib.model    import Multihead
from balloonlib.training import loss as balloon_loss, train
from balloonlib.data     import experimental_stims
from balloonlib.utils    import np2tensor, tensor2np

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float32)

# Study configuration
N_TRIALS        = 50        # Number of optimization trials
TIMEOUT         = 3600 * 2  # 2-hour timeout
N_STARTUP_TRIALS = 10       # Random trials before TPE kicks in
N_WARMUP_STEPS  = 5         # Pruner warmup steps

# Training configuration for each trial
EPOCHS_PER_TRIAL = 25000    # Iterations per trial
EVAL_INTERVAL    = 100      # Report to pruner every N iterations


# ============================================================================
# DATA LOADING
# ============================================================================

def load_balloon_data() -> Dict[str, Any]:
    """
    Load or generate training data for the Balloon model.

    Replace the body of this function with your actual data loading.
    The returned dict must contain at least ``'I'`` (stimulus tensor).

    Returns
    -------
    dict
        Keys: ``'I'`` (stimulus, shape ``(N,)``), ``'max_elements'`` (int).
    """
    max_elements = 3000

    # Synthetic stimulus: on from t ≈ 0.067 to t ≈ 0.10  (physical: 2–3 s)
    I = torch.zeros(max_elements)
    stim_start = int(0.067 * max_elements)
    stim_end   = int(0.10  * max_elements)
    I[stim_start:stim_end] = 1.0

    return {"I": I, "max_elements": max_elements}


def default_balloon_params(impulse: torch.Tensor) -> Dict[str, Any]:
    """
    Return default physiological Balloon-model parameters.

    Parameters
    ----------
    impulse : torch.Tensor
        High-resolution stimulus tensor, shape ``(N,)``.

    Returns
    -------
    dict
        ``Balloon_params`` dict ready for :func:`~balloonlib.training.loss`.
    """
    return {
        "lambdar_list": [0.2, 0.05],
        "kappa_list":   [1 / 1.54, 1 / 1.54],
        "gamma_list":   [1 / 2.46, 1 / 2.46],
        "tau_m_list":   20,
        "tau_MTT_list": 3.0,
        "alpha":        0.4,
        "I":            impulse,
    }


def default_data_params() -> Dict[str, Any]:
    """
    Return a minimal ``data_params`` dict for ODE-only training (no BOLD data).

    Returns
    -------
    dict
    """
    return {
        "errorFn": nn.MSELoss(),
        # No Bold_Signal key → BOLD data loss is disabled
    }


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_multihead_model(trial: optuna.Trial) -> nn.Module:
    """
    Build a :class:`~balloonlib.model.Multihead` with trial-suggested HPs.

    Parameters
    ----------
    trial : optuna.Trial

    Returns
    -------
    Multihead
    """
    # ── Hidden activation
    act_name = trial.suggest_categorical(
        "activation",
        ["Tanh", "GELU", "SiLU", "Mish", "Softplus", "ELU", "Sigmoid", "Hardtanh"],
    )
    act_map = {
        "Tanh": nn.Tanh(), "GELU": nn.GELU(), "SiLU": nn.SiLU(),
        "Mish": nn.Mish(), "Softplus": nn.Softplus(), "ELU": nn.ELU(),
        "Sigmoid": nn.Sigmoid(), "Hardtanh": nn.Hardtanh(),
    }
    act = act_map[act_name]

    # ── Fourier features
    use_fourier = trial.suggest_categorical("use_fourier", [True, False])
    if use_fourier:
        fourier_mapping_size = trial.suggest_int("fourier_mapping_size", 1, 32)
        fourier_scale        = trial.suggest_float("fourier_scale", 0.1, 5.0, log=True)
        fourier_learnable    = trial.suggest_categorical("fourier_learnable", [True, False])
        multi_scale_fourier  = trial.suggest_categorical("multi_scale_fourier", [True, False])
    else:
        fourier_mapping_size = 1
        fourier_scale        = 1.0
        fourier_learnable    = False
        multi_scale_fourier  = False

    # ── NV-head activations (f and m outputs)
    nv_fn_name  = trial.suggest_categorical("nv_fn", ["ELU", "LeakyReLU", "Softplus", "GELU", "SiLU"])
    nv_alpha_f  = trial.suggest_float("nv_alpha_f", 0.01, 1.0, log=True)
    nv_alpha_m  = trial.suggest_float("nv_alpha_m", 0.01, 1.0, log=True)

    nv_fn_map = {
        "ELU":       lambda a: nn.ELU(alpha=a),
        "LeakyReLU": lambda a: nn.LeakyReLU(negative_slope=a),
        "Softplus":  lambda a: nn.Softplus(beta=max(1.0 / a, 0.01)),
        "GELU":      lambda a: nn.GELU(),
        "SiLU":      lambda a: nn.SiLU(),
    }
    nv_fn = (nv_fn_map[nv_fn_name](nv_alpha_f), nv_fn_map[nv_fn_name](nv_alpha_m))

    # ── Core-head activations (v and q outputs — must stay positive)
    core_fn_name = trial.suggest_categorical("core_fn", ["Softplus", "ReLU", "ELU", "Sigmoid"])
    core_fn_map  = {
        "Softplus": nn.Softplus(), "ReLU": nn.ReLU(),
        "ELU": nn.ELU(alpha=1.0), "Sigmoid": nn.Sigmoid(),
    }
    core_fn_inst = core_fn_map[core_fn_name]

    # ── Random Weight Factorization
    use_rwf = trial.suggest_categorical("use_rwf", [True, False])

    # ── Seed
    seed = trial.suggest_int("seed", 1, 1000)

    model = Multihead(
        impulse              = False,
        act                  = act,
        nv_fn                = nv_fn,
        core_fn              = (core_fn_inst, core_fn_inst),
        dtype                = torch.float32,
        use_fourier          = use_fourier,
        fourier_mapping_size = fourier_mapping_size,
        fourier_scale        = fourier_scale,
        fourier_learnable    = fourier_learnable,
        multi_scale_fourier  = multi_scale_fourier,
        random_weightsMatrix = use_rwf,
        seed                 = seed,
    )
    model.init_nn_params()
    return model


# ============================================================================
# OBJECTIVE
# ============================================================================

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function for BalloonPINN HPO.

    Uses :func:`~balloonlib.training.train` (the canonical training loop)
    so that every trial runs through the exact same physics as production.

    Parameters
    ----------
    trial : optuna.Trial

    Returns
    -------
    float
        Best total loss achieved during the trial (lower = better).
    """
    # ── Data
    data     = load_balloon_data()
    I        = data["I"]
    domain   = (0.0, len(I) * 0.01)   # physical domain in seconds

    # ── Model
    try:
        model = create_multihead_model(trial)
    except Exception as e:
        print(f"Model creation failed: {e}")
        raise optuna.exceptions.TrialPruned()

    # ── Optimizer
    opt_name     = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RAdam"])
    lr           = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    use_wd       = trial.suggest_categorical("use_weight_decay", [True, False])
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True) if use_wd else 0.0

    optimizer = getattr(optim, opt_name)(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    # ── Scheduler
    use_scheduler = trial.suggest_categorical("use_scheduler", [True, False])
    scheduler     = None
    sched_name    = None

    if use_scheduler:
        sched_name = trial.suggest_categorical(
            "scheduler",
            ["StepLR", "ExponentialLR", "CosineAnnealingLR", "ReduceLROnPlateau", "OneCycleLR"],
        )
        if sched_name == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=trial.suggest_int("step_size", 500, 5000),
                gamma=trial.suggest_float("gamma", 0.1, 0.9),
            )
        elif sched_name == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=trial.suggest_float("exp_gamma", 0.99, 0.9999, log=True),
            )
        elif sched_name == "CosineAnnealingLR":
            eta_min_ratio = trial.suggest_float("eta_min_ratio", 1e-4, 1e-1, log=True)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=trial.suggest_int("T_max", 1000, EPOCHS_PER_TRIAL),
                eta_min=lr * eta_min_ratio,
            )
        elif sched_name == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min",
                factor=trial.suggest_float("factor", 0.1, 0.5),
                patience=trial.suggest_int("patience", 100, 1000),
            )
        elif sched_name == "OneCycleLR":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr * trial.suggest_float("max_lr_mult", 2.0, 10.0),
                total_steps=EPOCHS_PER_TRIAL,
                pct_start=trial.suggest_float("pct_start", 0.1, 0.4),
                anneal_strategy="cos",
            )

    # ── Loss weights
    ode_w = trial.suggest_float("ode_weight", 0.1, 10.0, log=True)
    ic_w  = trial.suggest_float("ic_weight",  0.1, 10.0, log=True)

    loss_weights = {
        "ode":    [ode_w],
        "ic":     [ic_w],
        "border": [1.0],
        "bold":   [0.0],   # no BOLD data in this objective
        "other":  [0.0],
    }

    # ── Parameters dicts
    balloon_params = default_balloon_params(I)
    data_params    = default_data_params()

    # ── Train via the canonical train() loop
    # train() returns a merged dict: {'ode': [...], 'ic': [...], ..., 'total': [...]}
    try:
        loss_trace = train(
            model          = model,
            optimizer      = optimizer,
            lossfn         = balloon_loss,
            num_iter       = EPOCHS_PER_TRIAL,
            Balloon_params = balloon_params,
            data_params    = data_params,
            domain         = domain,
            random         = False,
            every          = 0,              # silent during HPO
            loss_weights   = loss_weights,
            scheduler      = scheduler,
            dtype          = torch.float32,
        )
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Training failed: {e}")
        raise optuna.exceptions.TrialPruned()

    # ── Extract best total loss from trace
    total_trace = loss_trace.get("total", [float("inf")])
    best_loss   = min(total_trace) if total_trace else float("inf")

    if not np.isfinite(best_loss):
        raise optuna.exceptions.TrialPruned()

    return best_loss


# ============================================================================
# STUDY RUNNER
# ============================================================================

def run_optimization(
    n_trials:   int           = N_TRIALS,
    timeout:    Optional[int] = TIMEOUT,
    study_name: str           = "balloonpinn_optuna",
    storage:    Optional[str] = None,
) -> optuna.Study:
    """
    Run the Optuna hyperparameter optimisation study.

    Parameters
    ----------
    n_trials : int
        Number of trials.
    timeout : int or None
        Wall-clock timeout in seconds.
    study_name : str
        Study name for persistence.
    storage : str or None
        SQLite / database URL, e.g. ``"sqlite:///study.db"``.

    Returns
    -------
    optuna.Study
    """
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS, seed=42)
    pruner  = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_WARMUP_STEPS)

    study = optuna.create_study(
        study_name    = study_name,
        storage       = storage,
        load_if_exists= True,
        direction     = "minimize",
        sampler       = sampler,
        pruner        = pruner,
    )

    study.optimize(
        objective,
        n_trials          = n_trials,
        timeout           = timeout,
        show_progress_bar = True,
        gc_after_trial    = True,
    )

    return study


def print_study_results(study: optuna.Study) -> None:
    """Print a summary of optimisation results and save best params."""
    pruned   = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\n" + "=" * 70)
    print("OPTUNA STUDY RESULTS — BalloonPINN Optimization")
    print("=" * 70)
    print(f"\n  Total trials:  {len(study.trials)}")
    print(f"  Completed:     {len(complete)}")
    print(f"  Pruned:        {len(pruned)}")

    if complete:
        trial = study.best_trial
        print(f"\nBest trial (#{trial.number}):")
        print(f"  Loss: {trial.value:.6f}")
        print("\nBest hyperparameters:")
        for k, v in trial.params.items():
            print(f"  {k}: {v}")

        save_path = os.path.join(os.path.dirname(__file__), "best_params.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(trial.params, f)
        print(f"\nBest params saved to: {save_path}")
    else:
        print("\nNo trials completed successfully.")


def create_model_from_best_params(params: Dict[str, Any]) -> nn.Module:
    """
    Instantiate a :class:`~balloonlib.model.Multihead` from saved best params.

    Parameters
    ----------
    params : dict
        Dictionary returned by ``study.best_trial.params``.

    Returns
    -------
    Multihead
    """
    act_map = {
        "Tanh": nn.Tanh(), "GELU": nn.GELU(), "SiLU": nn.SiLU(),
        "Mish": nn.Mish(), "Softplus": nn.Softplus(), "ELU": nn.ELU(),
        "Sigmoid": nn.Sigmoid(), "Hardtanh": nn.Hardtanh(),
    }
    core_fn_map = {
        "Softplus": nn.Softplus(), "ReLU": nn.ReLU(),
        "ELU": nn.ELU(alpha=1.0), "Sigmoid": nn.Sigmoid(),
    }
    nv_fn_name = params.get("nv_fn", "Softplus")
    nv_fn_map  = {
        "ELU":       lambda a: nn.ELU(alpha=a),
        "LeakyReLU": lambda a: nn.LeakyReLU(negative_slope=a),
        "Softplus":  lambda a: nn.Softplus(beta=max(1.0 / a, 0.01)),
        "GELU":      lambda a: nn.GELU(),
        "SiLU":      lambda a: nn.SiLU(),
    }

    core_inst = core_fn_map.get(params.get("core_fn", "Softplus"), nn.Softplus())
    nv_f = nv_fn_map[nv_fn_name](params.get("nv_alpha_f", 0.1))
    nv_m = nv_fn_map[nv_fn_name](params.get("nv_alpha_m", 0.1))

    model = Multihead(
        impulse              = False,
        act                  = act_map.get(params.get("activation", "Tanh"), nn.Tanh()),
        nv_fn                = (nv_f, nv_m),
        core_fn              = (core_inst, core_inst),
        dtype                = torch.float32,
        use_fourier          = params.get("use_fourier", False),
        fourier_mapping_size = params.get("fourier_mapping_size", 9),
        fourier_scale        = params.get("fourier_scale", 0.85),
        fourier_learnable    = params.get("fourier_learnable", True),
        multi_scale_fourier  = params.get("multi_scale_fourier", False),
        random_weightsMatrix = params.get("use_rwf", True),
        seed                 = params.get("seed", 42),
    )
    model.init_nn_params()
    return model


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BalloonPINN Optuna HPO")
    parser.add_argument("--trials",  type=int, default=N_TRIALS,  help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=TIMEOUT,   help="Wall-clock timeout (seconds)")
    parser.add_argument("--storage", type=str, default=None,      help="SQLite URL, e.g. sqlite:///study.db")
    args = parser.parse_args()

    print("Starting BalloonPINN HPO with Optuna ...")
    print(f"  Device:           {DEVICE}")
    print(f"  Trials:           {args.trials}")
    print(f"  Epochs per trial: {EPOCHS_PER_TRIAL}")
    print(f"  Timeout:          {args.timeout} s")

    study = run_optimization(
        n_trials   = args.trials,
        timeout    = args.timeout,
        storage    = args.storage,
    )
    print_study_results(study)

    try:
        import optuna.visualization as vis
        fig = vis.plot_param_importances(study)
        fig.write_html("param_importance.html")
        fig = vis.plot_optimization_history(study)
        fig.write_html("optimization_history.html")
        print("Plots saved: param_importance.html, optimization_history.html")
    except ImportError:
        print("Install plotly for visualisations: pip install plotly")
    except Exception as e:
        print(f"Visualisation error: {e}")