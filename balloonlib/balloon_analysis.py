"""
balloon_analysis — downstream analysis pipeline for BalloonLib HRF loops.

Turns the ``.pkl`` dumps produced by the 100x PINN training loop
(``BalloonLoop*.ipynb``) into the paper's descriptor tables, goodness-of-fit
tables, group-comparison statistics and figures — the steps currently
redefined by hand inside every dated ``HundredSignalAnalysis<date>.ipynb``.

Stages
------
1. ``load_loop``        — read a loop ``.pkl`` and normalise to numpy arrays
                          (replaces the notebook ``detach_structure``).
2. ``describe_hrfs``    — HRF shape descriptors as a tidy DataFrame
                          (wraps ``balloonlib.metrics.hrf_description``).
3. ``filter_implausible`` — drop runs with NaN descriptors (Shan-2014-style
                          plausibility filter; replaces ``DS_noNan``).
4. ``goodness_of_fit``  — R2 / KGE / MI / RMSE / L2RE / Pearson / Spearman
                          of the run-mean state vs. ground truth
                          (replaces the notebook ``efficiency``, which was
                          never in the library).
5. ``compare_groups``   — Shapiro-Wilk -> Welch-t / Mann-Whitney-U ->
                          Bonferroni -> effect size, one row per descriptor
                          (the paper's group-comparison table).
6. ``run_analysis``     — one call that chains 1-5 for a single experiment.

The descriptor maths lives in ``balloonlib``; this module is orchestration
only, so the library stays the single source of truth.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib as mpl
from matplotlib.lines import Line2D

# Print-readability floor for every figure this module draws. IEEE/JBHI
# requires >=300 dpi for colour/grayscale figures and >=600 dpi for line art
# and tables; 600 dpi as a module-wide default satisfies both. Axis/tick/
# legend/title sizes below raise anything that would otherwise fall back to
# matplotlib's 10 pt default -- explicit fontsize= calls elsewhere in this
# module (which override these) have been bumped independently (+2 pt each)
# so the intended text is never smaller than what a reader gets here.
# ``figure.dpi`` stays at the matplotlib default: 600 is a print
# requirement, not a screen one, and raising it only bloats inline previews.
_RC: dict[str, object] = {
    "savefig.dpi": 600,
    # screen/inline dpi stays at the matplotlib default; 600 applies to
    # savefig only, where it is actually required by the journal.
    "figure.dpi": 100,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
}
mpl.rcParams.update(_RC)

# Canonical descriptor column order (matches balloonlib.metrics.hrf_description).
DESCRIPTORS: tuple[str, ...] = (
    "HP", "TTP[s]", "FWHM[s]", "TO[s]", "AUC", "MU", "TTU[s]", "TT0[s]",
)
STATE_KEYS: tuple[str, ...] = ("f", "m", "v", "q", "hrf")

# Group colours, shared by every comparison figure so a given group is the
# same colour everywhere. Group order matches the ``labels`` you pass:
# first group -> red, second -> blue (e.g. right=red, left=blue). Extra
# entries cover >2-group comparisons.
GROUP_PALETTE: tuple[str, ...] = (
    "#c0392b",  # red
    "#2c7fb8",  # blue
    "#27ae60",  # green
    "#8e44ad",  # purple
    "#e67e22",  # orange
    "#16a085",  # teal
)

# IEEE Transactions/Journals physical figure limits (color template, June
# 2023): a full-page-width figure (spanning both columns, `figure*`) is
# 7.16 in / 181 mm wide; the maximum printable depth is 8.5 in / 216 mm.
# Reference values only -- the figures below are NOT drawn to these limits.
# Forcing a multi-panel figure into 7.16 in makes the (fixed, literal-point)
# text swamp the panels, so each figure keeps its native canvas and is
# scaled by \includegraphics at inclusion time. Pass ``figsize=`` if you
# want to draw at page width and re-check legibility yourself.
IEEE_PAGE_WIDTH_IN: float = 7.16
IEEE_MAX_DEPTH_IN: float = 8.5


def _get_hrf_description():
    """Return balloonlib's descriptor function (single source of truth)."""
    from balloonlib.metrics import hrf_description
    return hrf_description


def _get_bpl():
    """Return the ``balloonpinnlib`` module, whichever way it is installed.

    In the repository it lives *inside* the ``balloonlib`` package, so a bare
    ``import balloonpinnlib`` fails when this module is imported as
    ``balloonlib.balloon_analysis``. Try the package-qualified path first and
    fall back to the top-level name for a flat sys.path layout.
    """
    try:
        from balloonlib import balloonpinnlib as bpl
    except ImportError:
        import importlib
        bpl = importlib.import_module("balloonpinnlib")
    return bpl


def _to_cpu(x):
    """Detach a tensor and move it to the CPU; pass anything else through.

    The BOLD reconstruction below runs its convolutions on the CPU. Training
    notebooks typically call ``torch.set_default_device('cuda')``, so any
    tensor built here without an explicit ``device=`` silently lands on the
    GPU and the convolution raises "Input type (torch.FloatTensor) and weight
    type (torch.cuda.FloatTensor) should be the same". Every tensor entering
    the BOLD path is normalised through this function, so callers may hand in
    ``data_params`` entries on either device (or a mix of both).
    """
    return x.detach().cpu() if hasattr(x, "detach") else x


# --------------------------------------------------------------------------- #
# Stage 1 — load
# --------------------------------------------------------------------------- #
def load_loop(source: str | dict) -> dict[str, np.ndarray]:
    """Load a loop ``.pkl`` (or accept an in-memory dict) and normalise.

    Every state key (``f, m, v, q, hrf``) is returned as a float array of
    shape ``(n_runs, T)``; loss keys (``total, ode, ic, other, bold``) as
    1-D arrays of length ``n_runs``. Handles both numpy dumps and lists of
    torch tensors (``.detach().cpu().numpy()``), matching the two dump
    conventions seen across the notebook lineage.
    """
    if isinstance(source, str):
        with open(source, "rb") as fh:
            raw = pickle.load(fh)
    else:
        raw = source

    out: dict[str, np.ndarray] = {}
    for key, values in raw.items():
        if isinstance(values, list) and len(values) and hasattr(values[0], "detach"):
            arr = np.asarray([v.detach().cpu().numpy() for v in values])
        else:
            arr = np.asarray(values)
        if key in STATE_KEYS:
            out[key] = np.squeeze(arr)               # (n_runs, T)
        else:
            out[key] = np.asarray(arr, dtype=float).ravel()
    return out


# --------------------------------------------------------------------------- #
# Stage 2 — descriptors
# --------------------------------------------------------------------------- #
def describe_hrfs(hrf: np.ndarray, max_time: float = 30.0,
                  integration_rule: str = "rectangle") -> pd.DataFrame:
    """HRF shape descriptors as a DataFrame (one row per run).

    ``hrf`` has shape ``(n_runs, T)``. Delegates to
    ``balloonlib.metrics.hrf_description`` so the descriptor definitions never
    drift from the library.
    """
    hrf_description = _get_hrf_description()
    desc = hrf_description(np.asarray(hrf), max_time=max_time,
                           integration_rule=integration_rule)
    return pd.DataFrame(desc)[list(DESCRIPTORS)]


# --------------------------------------------------------------------------- #
# Stage 3 — plausibility filter
# --------------------------------------------------------------------------- #
@dataclass
class FilterResult:
    states: dict[str, np.ndarray]        # filtered state arrays (n_kept, T)
    descriptors: pd.DataFrame            # filtered descriptor table (n_kept rows)
    dropped_idx: np.ndarray              # indices removed
    n_in: int
    n_kept: int

    @property
    def n_dropped(self) -> int:
        return self.n_in - self.n_kept


def filter_implausible(loop: dict[str, np.ndarray],
                       descriptors: pd.DataFrame | None = None,
                       flag_column: str | None = None,
                       max_time: float = 30.0) -> FilterResult:
    """Drop implausible runs (Shan-2014-style), keyed on NaN descriptors.

    A run is discarded when ``hrf_description`` could not resolve a plausible
    HRF shape and returned NaN for *any* descriptor — matching the notebook
    diagnostic ``set(np.where(Description.isna())[0])``, which flags a run if
    any column is NaN, not only ``TO[s]``. This is the default
    (``flag_column=None``): a run with a valid ``TO[s]`` but a NaN elsewhere
    (e.g. an unresolved HP or FWHM on noisy data) is still dropped.

    Pass ``flag_column="TO[s]"`` (or any single column) to reproduce the older
    single-column behaviour of the ``DS_noNan`` cell. All state arrays and the
    descriptor table are filtered consistently.
    """
    if descriptors is None:
        descriptors = describe_hrfs(loop["hrf"], max_time=max_time)

    n_in = len(descriptors)
    if flag_column is None:
        na_mask = descriptors.isna().any(axis=1).to_numpy()   # any descriptor NaN
    else:
        na_mask = descriptors[flag_column].isna().to_numpy()
    dropped = np.where(na_mask)[0]

    states = {}
    for key in STATE_KEYS:
        if key in loop:
            states[key] = np.delete(np.asarray(loop[key]), dropped, axis=0)

    kept_desc = descriptors.drop(index=dropped).reset_index(drop=True)
    return FilterResult(states=states, descriptors=kept_desc,
                        dropped_idx=dropped, n_in=n_in, n_kept=n_in - len(dropped))


# --------------------------------------------------------------------------- #
# Stage 4 — goodness of fit
# --------------------------------------------------------------------------- #
def _kge(y_obs: np.ndarray, y_sim: np.ndarray) -> float:
    r = stats.pearsonr(y_sim, y_obs)[0]
    mu_rate = np.mean(y_sim) / np.mean(y_obs)
    std_rate = np.std(y_sim) / np.std(y_obs)
    return 1.0 - np.sqrt((r - 1) ** 2 + (std_rate - 1) ** 2 + (mu_rate - 1) ** 2)


def _efficiency_row(obs: np.ndarray, sim: np.ndarray,
                    with_mi: bool = True) -> dict[str, float]:
    y_o = np.asarray(obs, dtype=float).ravel()
    y_s = np.asarray(sim, dtype=float).ravel()
    resid = y_o - y_s
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_o - y_o.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(resid ** 2))
    l2re = np.sqrt(ss_res) / np.sqrt(np.sum(y_o ** 2))
    row = {
        "R2": r2,
        "KGE": _kge(y_o, y_s),
        "RMSE": rmse,
        "L2re": l2re,
        "Pr": stats.pearsonr(y_o, y_s)[0],
        "Sr": stats.spearmanr(y_o, y_s)[0],
    }
    if with_mi:
        try:
            from sklearn.feature_selection import mutual_info_regression
            row["MI"] = float(mutual_info_regression(
                y_o.reshape(-1, 1), y_s)[0])
        except Exception:
            row["MI"] = np.nan
    return row


def _summarize_runs(per_df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """Collapse a per-run efficiency table ``(n_runs, metrics)`` to across-run
    summary statistics. Returns a Series with a ``(metric, stat)`` MultiIndex,
    ``stat`` in ``{mean, sd, median, q25, q75}`` (sample SD, ddof=1). NaNs
    (e.g. a failed MI) are dropped per metric before summarising.
    """
    out = {}
    for c in cols:
        if c not in per_df.columns:
            continue
        a = per_df[c].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        out[(c, "mean")]   = np.mean(a) if a.size else np.nan
        out[(c, "sd")]     = np.std(a, ddof=1) if a.size > 1 else np.nan
        out[(c, "median")] = np.median(a) if a.size else np.nan
        out[(c, "q25")]    = np.percentile(a, 25) if a.size else np.nan
        out[(c, "q75")]    = np.percentile(a, 75) if a.size else np.nan
    s = pd.Series(out)
    s.index = pd.MultiIndex.from_tuples(s.index)
    return s


def goodness_of_fit(states: dict[str, np.ndarray],
                    ground_truth: dict[str, np.ndarray],
                    reducer: str = "mean",
                    with_mi: bool = True,
                    agg: str = "point") -> pd.DataFrame:
    """GOF table comparing PINN state ensembles to ground truth.

    ``states`` are filtered arrays ``(n_runs, T)``; ``ground_truth`` maps each
    key to the reference 1-D signal. Columns: R2, KGE, MI, RMSE, L2re, Pr, Sr.
    One row per state variable present in both inputs.

    ``agg`` chooses what the ensemble of runs collapses to:

    * ``"point"`` (default) — reduce the run ensemble (``mean`` or ``median``
      per ``reducer``) to a single consensus signal, then score it once.
      Matches the notebook, which averages the 100 runs before computing
      ``efficiency``. This scores the *ensemble-mean reconstruction*, which is
      smoother than any single run, so it reads better than a typical run.
    * ``"per_run"`` — score every run individually; returns a
      ``(n_runs, metrics)`` table per state, concatenated with a
      ``(state, run)`` MultiIndex.
    * ``"summary"`` — score every run, then report across-run mean, SD,
      median and IQR (q25/q75) per metric (columns become a
      ``(metric, stat)`` MultiIndex). This is the dispersion-aware report:
      the SD/IQR quantify run-to-run reliability, which the point estimate
      hides.
    """
    cols = ["R2", "KGE", "MI", "RMSE", "L2re", "Pr", "Sr"]
    reduce_fn = {"mean": np.mean, "median": np.median}[reducer]

    if agg == "point":
        rows, index = [], []
        for key in STATE_KEYS:
            if key in states and key in ground_truth:
                reduced = reduce_fn(np.asarray(states[key]), axis=0).ravel()
                rows.append(_efficiency_row(ground_truth[key], reduced, with_mi=with_mi))
                index.append(f"{key}_pinn")
        df = pd.DataFrame(rows, index=index)
        return df[[c for c in cols if c in df.columns]]

    if agg not in ("per_run", "summary"):
        raise ValueError("agg must be 'point', 'per_run' or 'summary'")

    per_state, summ_rows, summ_index = {}, [], []
    for key in STATE_KEYS:
        if key in states and key in ground_truth:
            arr = np.asarray(states[key])                 # (n_runs, T)
            truth = np.asarray(ground_truth[key]).ravel()
            per = pd.DataFrame(
                [_efficiency_row(truth, arr[i].ravel(), with_mi=with_mi)
                 for i in range(arr.shape[0])],
                index=[f"run_{i}" for i in range(arr.shape[0])])
            per = per[[c for c in cols if c in per.columns]]
            per_state[f"{key}_pinn"] = per
            summ_rows.append(_summarize_runs(per, cols))
            summ_index.append(f"{key}_pinn")

    if agg == "per_run":
        return pd.concat(per_state, names=["state", "run"])
    return pd.DataFrame(summ_rows, index=summ_index)


def bold_efficiency(experiment, data_params, bpl, *, reducer: str = "mean",
                    agg: str = "point", with_mi: bool = True,
                    per_run: bool = False) -> pd.DataFrame:
    """BOLD-reconstruction efficiency against a measured BOLD trace.

    Reconstructs the ensemble BOLD by convolving the HRF with the stimulus
    (``bpl.tofit``), samples the reconstruction at the measured-data time
    points (``bpl.timeBall``), then scores it against the measured
    ``Bold_Signal`` with the same seven-metric efficiency used by
    :func:`goodness_of_fit` (R2, KGE, MI, RMSE, L2re, Pr, Sr).

    Unlike :func:`goodness_of_fit` (state-vs-ground-truth), this needs only a
    stimulus and a measured BOLD trace, never a ground-truth HRF, so it
    applies to every experiment:

    * simulated (noiseless / noise-added): pass the simulated BOLD as
      ``data_params['Bold_Signal']`` — scores the reconstruction against the
      clean or noisy synthetic BOLD.
    * in-vivo (right / left hemisphere): pass the measured in-vivo BOLD —
      scores against the patient data. No ground truth needed or used.

    Matches the notebook's
    ``efficiency(bold_vals, Overall_bold_pinn[samples_index])``: the
    reconstruction is scored **raw**. The DC offset drawn in the BOLD panel is
    a plotting-only alignment and is NOT applied before scoring.

    Parameters
    ----------
    experiment : ExperimentResult | FilterResult | dict
        Anything :func:`_as_states` accepts; must expose an ``hrf`` ensemble
        of shape ``(n_runs, T)``.
    data_params : dict
        Requires ``Bold_Signal`` (measured trace), ``Overallstim`` (stimulus
        train, torch tensor), ``Overall_stim_time`` (torch tensor) and
        ``Bold_data_time`` (torch tensor — the measured-sample time axis,
        needed to align the reconstruction to the data). Same keys as the
        BOLD panel; here ``Bold_data_time`` is required (no time axis => no
        way to sample the reconstruction at the data points).
    bpl : module
        The imported ``balloonpinnlib`` (kept a parameter so this module never
        hard-imports torch). Provides ``tofit``, ``timeBall`` and
        ``tensor2np``.
    reducer : {"mean", "median"}
        How the HRF ensemble is reduced before the ``agg="point"`` score
        (ignored for the other modes). ``mean`` matches the notebook's
        ``hrf_predict``.
    agg : {"point", "per_run", "summary"}
        What the ensemble of runs collapses to (mirrors
        :func:`goodness_of_fit`):

        * ``"point"`` (default) — reconstruct BOLD from the reduced HRF and
          score it once (the notebook number). One row ``bold_<reducer>``.
          Scores the *ensemble-mean reconstruction*, which is smoother than
          any single run, so it reads better than a typical run.
        * ``"per_run"`` — reconstruct + score every run's own HRF; one row per
          run (``run_<i>``).
        * ``"summary"`` — score every run, then report across-run mean, SD,
          median and IQR (q25/q75) per metric (columns become a
          ``(metric, stat)`` MultiIndex). The dispersion-aware report: SD/IQR
          quantify how much a single restart's BOLD fit varies.
    with_mi : bool
        Include the mutual-information column (needs scikit-learn).
    per_run : bool
        Deprecated alias for ``agg="per_run"`` (kept for older call sites).
        Overrides ``agg`` when True.

    Returns
    -------
    pd.DataFrame
        ``agg="point"`` -> one row ``bold_<reducer>``; ``agg="per_run"`` ->
        one row per run; ``agg="summary"`` -> one row with a
        ``(metric, stat)`` column MultiIndex. Metrics R2, KGE, MI, RMSE,
        L2re, Pr, Sr — same as :func:`goodness_of_fit`.
    """
    import torch

    states = _as_states(experiment)
    if "hrf" not in states:
        raise KeyError("experiment has no 'hrf' ensemble to reconstruct BOLD from")
    hrf_ens = np.asarray(states["hrf"])          # (n_runs, T)

    required = ["Bold_Signal", "Overallstim", "Overall_stim_time", "Bold_data_time"]
    missing = [k for k in required if data_params.get(k) is None]
    if missing:
        raise ValueError(f"bold_efficiency needs data_params keys: {missing}")

    stim   = data_params["Overallstim"]
    stim_t = data_params["Overall_stim_time"]
    bdt    = data_params["Bold_data_time"]
    bsig   = data_params["Bold_Signal"]
    bvals  = bsig.values if hasattr(bsig, "values") else np.asarray(bsig)
    bvals  = np.asarray(bvals, dtype=float).ravel()

    def _score_hrf(hrf_1d):
        hrf_t = torch.tensor(np.asarray(hrf_1d).ravel(), dtype=torch.float32)
        bold, tt = bpl.tofit(stim.detach(), hrf_t.to(stim.device), stim_t[-1].item() + 0.01)
        idx, _ = bpl.timeBall(bdt, tt)
        idx = idx.detach().cpu()
        sampled = bpl.tensor2np(bold[idx]).ravel()
        return _efficiency_row(bvals, sampled, with_mi=with_mi)

    cols = ["R2", "KGE", "MI", "RMSE", "L2re", "Pr", "Sr"]
    if per_run:                                   # deprecated alias
        agg = "per_run"
    if agg not in ("point", "per_run", "summary"):
        raise ValueError("agg must be 'point', 'per_run' or 'summary'")

    if agg == "point":
        reduce_fn = {"mean": np.mean, "median": np.median}[reducer]
        reduced = reduce_fn(hrf_ens, axis=0).ravel()
        df = pd.DataFrame([_score_hrf(reduced)], index=[f"bold_{reducer}"])
        return df[[c for c in cols if c in df.columns]]

    per = pd.DataFrame(
        [_score_hrf(hrf_ens[i]) for i in range(hrf_ens.shape[0])],
        index=[f"run_{i}" for i in range(hrf_ens.shape[0])])
    per = per[[c for c in cols if c in per.columns]]
    if agg == "per_run":
        return per
    return _summarize_runs(per, cols).to_frame("bold").T


def full_efficiency(experiment, data_params, bpl, *, ground_truth=None,
                    reducer: str = "mean", agg: str = "point",
                    with_mi: bool = True) -> pd.DataFrame:
    """Combined efficiency over all state variables **and** BOLD, in one table.

    Stacks :func:`goodness_of_fit` (state ensembles f, m, v, q, hrf vs
    ``ground_truth``) on top of :func:`bold_efficiency` (BOLD reconstruction vs
    the measured ``Bold_Signal``). Rows: ``f_pinn, m_pinn, v_pinn, q_pinn,
    hrf_pinn, bold_<reducer>``.

    The state rows are included only when ``ground_truth`` is given — so for a
    simulated experiment you get all six rows, and for an in-vivo hemisphere
    (no ground truth) you get the single BOLD row. This is the one call that
    answers "efficiency over f, m, v, q, hrf and BOLD" for whichever
    experiment applies.

    ``agg`` is threaded to both sub-functions (``"point"`` /  ``"per_run"`` /
    ``"summary"``), so the whole table is either point estimates, per-run
    distributions, or across-run mean/SD/median/IQR. ``ground_truth`` maps
    ``f/m/v/q/hrf`` to reference 1-D signals; ``data_params`` and ``bpl`` are
    as in :func:`bold_efficiency`.
    """
    parts = []
    if ground_truth is not None:
        gof = goodness_of_fit(_as_states(experiment), ground_truth,
                              reducer=reducer, with_mi=with_mi, agg=agg)
        if not gof.empty:
            parts.append(gof)
    bold = bold_efficiency(experiment, data_params, bpl,
                           reducer=reducer, agg=agg, with_mi=with_mi)
    parts.append(bold)
    if agg == "per_run":
        # goodness_of_fit per_run has a (state, run) MultiIndex; bold has a
        # flat run index. Tag bold under a 'bold' state level for a uniform
        # (state, run) index across the stack.
        tagged = []
        for p in parts:
            if isinstance(p.index, pd.MultiIndex):
                tagged.append(p)
            else:
                p2 = p.copy()
                p2.index = pd.MultiIndex.from_product(
                    [["bold"], p.index], names=["state", "run"])
                tagged.append(p2)
        return pd.concat(tagged)
    return pd.concat(parts)


# --------------------------------------------------------------------------- #
# Stage 5 — group comparison statistics
# --------------------------------------------------------------------------- #
def compare_groups(desc_a: pd.DataFrame, desc_b: pd.DataFrame,
                   labels: tuple[str, str] = ("A", "B"),
                   alpha: float = 0.05,
                   normality_alpha: float = 0.05,
                   features: Sequence[str] = DESCRIPTORS) -> pd.DataFrame:
    """Per-descriptor two-group comparison with multiple-testing correction.

    For each descriptor:
      1. Shapiro-Wilk normality test on each group.
      2. If both normal -> Welch's t-test (Cohen's d effect size);
         else -> Mann-Whitney U (rank-biserial correlation effect size).
      3. Bonferroni correction across the ``n`` descriptors tested.

    Reproduces the paper's group-comparison table. ``alpha`` is the
    family-wise level; the per-test Bonferroni threshold is ``alpha / n``.
    """
    features = list(features)
    n_tests = len(features)
    rows = []
    for feat in features:
        x = desc_a[feat].dropna().to_numpy()
        y = desc_b[feat].dropna().to_numpy()
        normal_x = stats.shapiro(x).pvalue > normality_alpha if len(x) >= 3 else False
        normal_y = stats.shapiro(y).pvalue > normality_alpha if len(y) >= 3 else False

        if normal_x and normal_y:
            test_name = "Welch t-test"
            test_stat, p = stats.ttest_ind(x, y, equal_var=False)
            pooled_sd = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
            effect = (np.mean(x) - np.mean(y)) / pooled_sd
            effect_name = "Cohen d"
        else:
            test_name = "Mann-Whitney U"
            test_stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            effect = 1.0 - (2.0 * test_stat) / (len(x) * len(y))  # rank-biserial
            effect_name = "rank-biserial"

        p_adj = min(p * n_tests, 1.0)
        rows.append({
            "Feature": feat,
            "Test": test_name,
            f"n_{labels[0]}": len(x),
            f"n_{labels[1]}": len(y),
            f"Normal_{labels[0]}": normal_x,
            f"Normal_{labels[1]}": normal_y,
            "statistic": test_stat,
            "p_value": p,
            "p_adj_bonferroni": p_adj,
            f"significant(alpha={alpha:g})": p_adj < alpha,
            "effect_size": effect,
            "effect_type": effect_name,
        })
    return pd.DataFrame(rows)


def compare_to_ground_truth(desc: pd.DataFrame,
                            ground_truth: "pd.Series | dict | pd.DataFrame",
                            label: str = "ensemble",
                            alpha: float = 0.05,
                            normality_alpha: float = 0.05,
                            features: Sequence[str] = DESCRIPTORS) -> pd.DataFrame:
    """Per-descriptor **one-sample** test: does the ensemble differ from truth?

    The ground truth is a single deterministic HRF, so each descriptor has one
    fixed reference value ``mu0`` — this is a one-sample problem, not the
    two-group comparison of :func:`compare_groups`. For each descriptor:

      1. Shapiro-Wilk normality test on the ensemble values.
      2. If normal -> one-sample t-test against ``mu0`` (one-sample Cohen's d);
         else -> Wilcoxon signed-rank on ``(values - mu0)`` (matched
         rank-biserial effect size).
      3. Bonferroni correction across the ``n`` descriptors tested.

    ``ground_truth`` maps each descriptor to its scalar truth value (a Series,
    a dict, or a one-row DataFrame). Column layout parallels
    :func:`compare_groups` so the same plotting/significance code applies.
    """
    features = list(features)
    if isinstance(ground_truth, pd.DataFrame):
        gt = ground_truth.iloc[0]
    else:
        gt = pd.Series(dict(ground_truth))
    n_tests = len(features)
    rows = []
    for feat in features:
        x = desc[feat].dropna().to_numpy()
        mu0 = float(gt[feat])
        normal_x = stats.shapiro(x).pvalue > normality_alpha if len(x) >= 3 else False

        d = x - mu0
        if normal_x:
            test_name = "one-sample t-test"
            test_stat, p = stats.ttest_1samp(x, mu0)
            sd = np.std(x, ddof=1)
            effect = (np.mean(x) - mu0) / sd if sd > 0 else np.nan
            effect_name = "Cohen d (1-sample)"
        else:
            test_name = "Wilcoxon signed-rank"
            nz = d[d != 0]
            if len(nz) == 0:                       # ensemble identical to truth
                test_stat, p, effect = np.nan, 1.0, 0.0
            else:
                test_stat, p = stats.wilcoxon(nz)
                # matched-pairs rank-biserial: (W+ - W-) / sum(ranks)
                ranks = stats.rankdata(np.abs(nz))
                w_pos = ranks[nz > 0].sum()
                w_neg = ranks[nz < 0].sum()
                effect = (w_pos - w_neg) / ranks.sum()
            effect_name = "rank-biserial (matched)"

        p_adj = min(p * n_tests, 1.0)
        rows.append({
            "Feature": feat,
            "Test": test_name,
            f"n_{label}": len(x),
            f"Normal_{label}": normal_x,
            "ground_truth": mu0,
            f"mean_{label}": float(np.mean(x)),
            "statistic": test_stat,
            "p_value": p,
            "p_adj_bonferroni": p_adj,
            f"significant(alpha={alpha:g})": p_adj < alpha,
            "effect_size": effect,
            "effect_type": effect_name,
        })
    return pd.DataFrame(rows)


def ground_truth_table(experiment, ground_truth, label: str = "ensemble",
                       alpha: float = 0.05, normality_alpha: float = 0.05,
                       features: Sequence[str] = DESCRIPTORS) -> pd.DataFrame:
    """Compact one-sample ensemble-vs-truth table for a simulated experiment.

    A thin, presentation-oriented wrapper around :func:`compare_to_ground_truth`
    that returns exactly the reporting columns

        Feature | Test | n | Normal | statistic | p_value | p_adj |
        significant(alpha=…) | effect_size | effect_type

    (label-suffixed ``n_``/``Normal_`` columns renamed to plain ``n``/``Normal``,
    and ``p_adj_bonferroni`` → ``p_adj``). Use this for the per-experiment
    significance table of the noiseless and noise-added simulations, where every
    descriptor has a single deterministic true value.

    Parameters
    ----------
    experiment : ExperimentResult | pandas.DataFrame
        Either a :class:`ExperimentResult` (its ``.descriptors`` is used) or a
        descriptor DataFrame directly.
    ground_truth : pandas.Series | dict | pandas.DataFrame
        The reference descriptor row (e.g. ``description_real`` built once from
        the simulated HRF). The noiseless and noise-added runs share the *same*
        ground truth — noise is added to the BOLD signal, not to the HRF.
    label : str
        Name used only to key the ``n``/``Normal`` columns internally.
    alpha, normality_alpha, features
        Passed through to :func:`compare_to_ground_truth`.

    Notes
    -----
    The test and effect measure are chosen per descriptor: Shapiro-normal →
    one-sample t-test with one-sample Cohen's d; otherwise Wilcoxon signed-rank
    with matched-pairs rank-biserial. ``effect_type`` names which, so the two
    effect scales are never confused (they are comparable only within a test
    family). Bonferroni correction is applied across ``features``.
    """
    desc = getattr(experiment, "descriptors", experiment)
    s = compare_to_ground_truth(desc, ground_truth, label=label, alpha=alpha,
                                normality_alpha=normality_alpha,
                                features=features)
    rename = {f"n_{label}": "n", f"Normal_{label}": "Normal",
              "p_adj_bonferroni": "p_adj"}
    cols = ["Feature", "Test", f"n_{label}", f"Normal_{label}", "statistic",
            "p_value", "p_adj_bonferroni", f"significant(alpha={alpha:g})",
            "effect_size", "effect_type"]
    return s[cols].rename(columns=rename)


def _resolve_margin(spec, mu0: float) -> float:
    """Turn one ROPE-margin spec into an absolute half-width in the
    descriptor's own units.

    ``spec`` is either a number (absolute margin, e.g. ``0.1`` → ±0.1 s) or a
    percentage string like ``"2%"`` (fraction of ``|mu0|`` → ±0.02·|mu0|).
    """
    if isinstance(spec, str):
        s = spec.strip()
        if not s.endswith("%"):
            raise ValueError(f"margin string {spec!r} must end with '%' "
                             "(e.g. '2%'); use a number for an absolute margin")
        return float(s[:-1]) / 100.0 * abs(mu0)
    return float(spec)


def ground_truth_bayes(experiment, ground_truth, margins,
                       label: str = "ensemble", cred_mass: float = 0.95,
                       features: Sequence[str] = DESCRIPTORS) -> pd.DataFrame:
    """Bayesian ensemble-vs-truth equivalence table (analytic conjugate model).

    The frequentist :func:`ground_truth_table` tests the point null "bias = 0",
    which any large ensemble (here n≈100 random-init runs) rejects for
    scientifically trivial offsets. This companion instead **estimates** the
    bias and asks whether it is *practically* zero, via a Region Of Practical
    Equivalence (ROPE). Large n now sharpens the estimate instead of
    manufacturing significance.

    Model (per descriptor): a Normal likelihood with unknown mean and variance
    under the non-informative Jeffreys prior gives a **closed-form** posterior
    for the bias δ = μ − μ₀:

        δ  ~  Student-t( loc = x̄ − μ₀,  scale = s/√n,  df = n − 1 )

    No sampling, no optimisation — exact, pure SciPy. From this posterior:

      * the ``cred_mass`` **HDI** (equal-tailed = HDI for the symmetric t),
      * ``pct_in_rope`` — posterior mass inside ±Δ,
      * ``P_direction`` — posterior probability the bias has its observed sign,
      * a three-way **verdict** (Kruschke's HDI-vs-ROPE rule):
          - HDI entirely inside  ±Δ → ``"equivalent"`` (recovered to tolerance),
          - HDI entirely outside ±Δ → ``"biased"``,
          - HDI straddling the ROPE → ``"undecided"`` (need more runs).

    Parameters
    ----------
    experiment : ExperimentResult | pandas.DataFrame
        Ensemble descriptors (``.descriptors`` used if an ExperimentResult).
    ground_truth : pandas.Series | dict | pandas.DataFrame
        Reference descriptor row (the same ``description_real`` for the
        noiseless and noise-added runs — noise is on the BOLD, not the HRF).
    margins : dict[str, float | str]
        ROPE half-width per descriptor — the one *scientific* input, so it is
        **required** and every tested feature must appear. Each value is either
        a number (absolute margin in the descriptor's units, e.g. ``0.1`` for
        ``±0.1 s``) or a percentage string (fraction of ``|truth|``, e.g.
        ``"2%"``). The phrasing "0.1 s for all times, 2 % for AUC/HP/MU" is::

            margins = {"TTP[s]": 0.1, "FWHM[s]": 0.1, "TO[s]": 0.1,
                       "TTU[s]": 0.1, "TT0[s]": 0.1,
                       "AUC": "2%", "HP": "2%", "MU": "2%"}
    label : str
        Names the mean column (``mean_{label}``).
    cred_mass : float
        Credible-interval mass for the HDI (default 0.95).
    features : sequence of str
        Descriptors to test (default all eight).

    Returns
    -------
    pandas.DataFrame with columns
        Feature | n | ground_truth | mean_{label} | bias | bias_pct |
        ROPE | HDI_low | HDI_high | pct_in_rope | P_direction | verdict

    Notes
    -----
    ``bias`` is in the descriptor's physical units; ``bias_pct`` expresses it as
    a percentage of the true value for cross-descriptor comparison. Because each
    descriptor is estimated independently there is no multiplicity correction to
    apply — this reports estimates, not a family of accept/reject decisions. The
    n here is optimiser restarts, so the posterior describes bias of the
    *ensemble procedure*; its width shrinks with more runs.
    """
    from scipy import stats

    desc = getattr(experiment, "descriptors", experiment)
    features = list(features)
    gt = ground_truth.iloc[0] if isinstance(ground_truth, pd.DataFrame) \
        else pd.Series(dict(ground_truth))

    missing = [f for f in features if f not in margins]
    if missing:
        raise ValueError(
            f"margins is missing an entry for {missing}. The ROPE margin is a "
            "required scientific choice (there is no safe default) — give each "
            "descriptor an absolute value (e.g. 0.1) or a percentage (e.g. '2%').")

    q = (1.0 + cred_mass) / 2.0
    rows = []
    for feat in features:
        x = desc[feat].dropna().to_numpy()
        n = len(x)
        mu0 = float(gt[feat])
        mean = float(np.mean(x))
        s = float(np.std(x, ddof=1))
        se = s / np.sqrt(n)                    # posterior scale of the bias
        b = mean - mu0                          # posterior location of the bias
        df = n - 1
        rope = _resolve_margin(margins[feat], mu0)

        post = stats.t(df, loc=b, scale=se)     # exact bias posterior
        tcrit = stats.t.ppf(q, df)
        lo, hi = b - tcrit * se, b + tcrit * se
        pct_in = float(post.cdf(rope) - post.cdf(-rope))
        p_dir = float(max(post.cdf(0.0), 1.0 - post.cdf(0.0)))

        if lo > -rope and hi < rope:
            v = "equivalent"
        elif hi < -rope or lo > rope:
            v = "biased"
        else:
            v = "undecided"

        rows.append({
            "Feature": feat,
            "n": n,
            "ground_truth": mu0,
            f"mean_{label}": mean,
            "bias": b,
            "bias_pct": 100.0 * b / mu0 if mu0 != 0 else np.nan,
            "ROPE": rope,
            "HDI_low": lo,
            "HDI_high": hi,
            "pct_in_rope": 100.0 * pct_in,
            "P_direction": p_dir,
            "verdict": v,
        })
    return pd.DataFrame(rows)


def compare_groups_bayes(desc_a, desc_b, margins,
                         labels: tuple[str, str] = ("A", "B"),
                         cred_mass: float = 0.95, method: str = "welch",
                         n_draws: int = 200_000, seed: int = 0,
                         features: Sequence[str] = DESCRIPTORS) -> pd.DataFrame:
    """Bayesian two-group practical-equivalence table (unequal variances).

    The Bayesian counterpart of :func:`compare_groups`. Where that function
    tests the point null "the two hemispheres' descriptor means are equal"
    (which n≈100 restarts reject for scientifically trivial gaps), this
    **estimates** the between-group difference

        Delta = mu_a - mu_b

    and asks whether it is *practically* zero, via the same Region Of
    Practical Equivalence (ROPE) logic as :func:`ground_truth_bayes`.

    Model
    -----
    Each group gets an independent Normal likelihood with unknown mean and
    variance under the Jeffreys prior, so each mean posterior is a Student-t::

        mu_a ~ t(df=n_a-1, loc=xbar_a, scale=s_a/sqrt(n_a))
        mu_b ~ t(df=n_b-1, loc=xbar_b, scale=s_b/sqrt(n_b))

    The variances are **never pooled** — the noisier ensemble (e.g. the
    lower-tSNR hemisphere) keeps its own wider spread, and that uncertainty
    flows honestly into the Delta posterior. Two ways to obtain Delta:

      * ``method="welch"`` (default) — analytic Bayesian Behrens-Fisher: Delta
        is approximated as Student-t with ``loc = xbar_a - xbar_b``,
        ``scale = sqrt(s_a^2/n_a + s_b^2/n_b)`` and Welch-Satterthwaite df.
        Closed form, no sampling — the Bayesian mirror of Welch's t-test.
      * ``method="mc"`` — draw from each group's t-posterior and subtract; read
        the HDI and ROPE mass off the difference samples. Exact, no
        distributional approximation to Delta.

    This is the analytic special case of Kruschke's BEST (2013) with a Normal
    (rather than Student-t) likelihood — appropriate when the descriptor
    ensembles are free of heavy-tailed outlier restarts.

    Parameters
    ----------
    desc_a, desc_b : ExperimentResult | pandas.DataFrame
        Per-group descriptor ensembles (``.descriptors`` used if an
        ExperimentResult). ``desc_a`` is the minuend (Delta = a - b).
    margins : dict[str, float | str]
        ROPE half-width per descriptor — required, same convention as
        :func:`ground_truth_bayes`: a number is an absolute margin in the
        descriptor's units; a percentage string (e.g. ``"2%"``) is a fraction
        of the pooled-mean magnitude ``|(xbar_a + xbar_b) / 2|``.
    labels : tuple[str, str]
        Group names; ``Delta`` is ``labels[0] - labels[1]``.
    cred_mass : float
        Credible mass for the HDI (default 0.95).
    method : {"welch", "mc"}
        Analytic Welch-t (default) or Monte-Carlo difference of posteriors.
    n_draws, seed : int
        Monte-Carlo settings (ignored for ``method="welch"``).
    features : sequence of str
        Descriptors to test (default all eight).

    Returns
    -------
    pandas.DataFrame with columns
        Feature | n_{a} | n_{b} | mean_{a} | mean_{b} | delta | delta_pct |
        ROPE | HDI_low | HDI_high | pct_in_rope | P_direction | verdict

    where verdict is ``"equivalent"`` (HDI inside +/-Delta), ``"different"``
    (HDI outside), or ``"undecided"`` (HDI straddles a ROPE limit).

    Notes
    -----
    The groups are treated as independent (separate sets of restarts), not
    paired. ``delta_pct`` is expressed relative to the pooled-mean magnitude.
    Cohen's d is deliberately not reported: it needs a pooled SD, which the
    unequal-variance model avoids — report ``delta`` in physical units and as
    a percentage instead. As in the one-sample case, ``n`` indexes optimiser
    restarts, so Delta describes the difference between the two ensemble
    *procedures* on this patient, and no multiplicity correction is applied
    because intervals (not a family of tests) are reported.
    """
    from scipy import stats

    la, lb = labels
    da = getattr(desc_a, "descriptors", desc_a)
    db = getattr(desc_b, "descriptors", desc_b)
    features = list(features)

    missing = [f for f in features if f not in margins]
    if missing:
        raise ValueError(
            f"margins is missing an entry for {missing}. The ROPE margin is a "
            "required scientific choice (there is no safe default) — give each "
            "descriptor an absolute value (e.g. 0.1) or a percentage (e.g. '2%').")

    q = (1.0 + cred_mass) / 2.0
    rng = np.random.default_rng(seed)
    rows = []
    for feat in features:
        x = da[feat].dropna().to_numpy()
        y = db[feat].dropna().to_numpy()
        na, nb = len(x), len(y)
        mean_a, mean_b = float(np.mean(x)), float(np.mean(y))
        sa = float(np.std(x, ddof=1))
        sb = float(np.std(y, ddof=1))
        sea2, seb2 = sa**2 / na, sb**2 / nb
        delta = mean_a - mean_b
        pooled_mag = abs((mean_a + mean_b) / 2.0)
        rope = _resolve_margin(margins[feat], pooled_mag)

        if method == "mc":
            draws_a = stats.t(na - 1, loc=mean_a, scale=sa / np.sqrt(na)).rvs(
                n_draws, random_state=rng)
            draws_b = stats.t(nb - 1, loc=mean_b, scale=sb / np.sqrt(nb)).rvs(
                n_draws, random_state=rng)
            d = draws_a - draws_b
            lo, hi = np.quantile(d, [1 - q, q])
            pct_in = float(np.mean((d > -rope) & (d < rope)))
            p_dir = float(max(np.mean(d > 0), np.mean(d < 0)))
        else:                                   # analytic Behrens-Fisher / Welch
            se = np.sqrt(sea2 + seb2)           # posterior scale of Delta
            df = (sea2 + seb2) ** 2 / (
                sea2**2 / (na - 1) + seb2**2 / (nb - 1))   # Welch-Satterthwaite
            post = stats.t(df, loc=delta, scale=se)
            tcrit = stats.t.ppf(q, df)
            lo, hi = delta - tcrit * se, delta + tcrit * se
            pct_in = float(post.cdf(rope) - post.cdf(-rope))
            p_dir = float(max(post.cdf(0.0), 1.0 - post.cdf(0.0)))

        if lo > -rope and hi < rope:
            v = "equivalent"
        elif hi < -rope or lo > rope:
            v = "different"
        else:
            v = "undecided"

        rows.append({
            "Feature": feat,
            f"n_{la}": na,
            f"n_{lb}": nb,
            f"mean_{la}": mean_a,
            f"mean_{lb}": mean_b,
            "delta": delta,
            "delta_pct": 100.0 * delta / pooled_mag if pooled_mag != 0 else np.nan,
            "ROPE": rope,
            "HDI_low": lo,
            "HDI_high": hi,
            "pct_in_rope": 100.0 * pct_in,
            "P_direction": p_dir,
            "verdict": v,
        })
    return pd.DataFrame(rows)


def bayes_comparison_table(desc_a, desc_b, margins,
                           labels: tuple[str, str] = ("A", "B"),
                           cred_mass: float = 0.95, method: str = "welch",
                           features: Sequence[str] = DESCRIPTORS,
                           **kwargs) -> pd.DataFrame:
    """Compact two-group Bayesian equivalence table (paper Fig. descriptor).

    A thin, presentation-oriented wrapper around :func:`compare_groups_bayes`,
    the two-group mirror of :func:`ground_truth_table`. Returns exactly the
    reporting columns

        Feature | n_a | n_b | mean_a | mean_b | delta | delta_pct |
        ROPE | HDI_low | HDI_high | pct_in_rope | P_direction | verdict

    with the ``mean_``/``n_`` columns keyed by ``labels``. This is the companion
    table to :func:`plot_descriptor_comparison` in its Bayesian mode: the figure
    shows the raw group distributions plus a per-panel verdict glyph, and this
    table carries the numbers the figure deliberately does *not* draw — the
    between-group difference ``delta = mean_a - mean_b`` with its ``cred_mass``
    HDI, the ROPE half-width, and the practical-equivalence ``verdict``
    (``equivalent`` / ``different`` / ``undecided``).

    Because the ROPE and HDI describe ``delta`` (a single derived quantity in
    *difference* space), not either group individually, they are reported here
    rather than as geometry on the group boxes.

    Parameters
    ----------
    desc_a, desc_b : ExperimentResult | pandas.DataFrame
        The two ensembles' descriptors (``.descriptors`` is used if present).
    margins : dict
        Per-descriptor ROPE half-widths, same convention as
        :func:`ground_truth_bayes`: a number is absolute units, a ``"2%"`` string
        is a fraction of the pooled-mean magnitude.
    labels : tuple[str, str]
        Group names, used to key the ``mean_``/``n_`` columns.
    cred_mass, method, features, **kwargs
        Passed through to :func:`compare_groups_bayes` (``method`` is ``"welch"``
        analytic or ``"mc"`` Monte-Carlo).

    Notes
    -----
    Variances are never pooled: each group keeps its own spread, so the noisier
    ensemble widens the HDI of ``delta`` honestly. No Cohen's d is reported (it
    needs a pooled SD); the effect is the difference itself, in physical units
    and as ``delta_pct`` of the pooled-mean magnitude.
    """
    la, lb = labels
    s = compare_groups_bayes(desc_a, desc_b, margins, labels=labels,
                             cred_mass=cred_mass, method=method,
                             features=features, **kwargs)
    cols = ["Feature", f"n_{la}", f"n_{lb}", f"mean_{la}", f"mean_{lb}",
            "delta", "delta_pct", "ROPE", "HDI_low", "HDI_high",
            "pct_in_rope", "P_direction", "verdict"]
    return s[cols]


# --------------------------------------------------------------------------- #
# One-shot orchestration
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentResult:
    name: str
    loop: dict[str, np.ndarray]
    descriptors_raw: pd.DataFrame
    filtered: FilterResult
    gof: pd.DataFrame | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def descriptors(self) -> pd.DataFrame:
        return self.filtered.descriptors


def run_analysis(pkl: str | dict, name: str = "experiment",
                 ground_truth: dict[str, np.ndarray] | None = None,
                 max_time: float = 30.0,
                 with_mi: bool = True,
                 reducer: str = "mean",
                 agg: str = "point") -> ExperimentResult:
    """Load -> describe -> filter -> (optional GOF) for one experiment.

    ``reducer`` and ``agg`` are forwarded to :func:`goodness_of_fit` when a
    ``ground_truth`` is given, so the stored ``.gof`` can be a point estimate
    (``agg="point"``, default), a per-run table (``"per_run"``) or an
    across-run mean/SD/median/IQR summary (``"summary"``).
    """
    loop = load_loop(pkl)
    desc_raw = describe_hrfs(loop["hrf"], max_time=max_time)
    filt = filter_implausible(loop, descriptors=desc_raw, max_time=max_time)
    gof = None
    if ground_truth is not None:
        gof = goodness_of_fit(filt.states, ground_truth,
                              reducer=reducer, with_mi=with_mi, agg=agg)
    return ExperimentResult(
        name=name, loop=loop, descriptors_raw=desc_raw,
        filtered=filt, gof=gof,
        meta={"n_in": filt.n_in, "n_kept": filt.n_kept,
              "n_dropped": filt.n_dropped, "max_time": max_time},
    )


# --------------------------------------------------------------------------- #
# Stage 6 — reporting / figures
# --------------------------------------------------------------------------- #
def _descriptor_box(ax, datasets: Sequence[np.ndarray], colors: Sequence,
                    labels: Sequence[str]) -> None:
    """Draw the shared descriptor aesthetic: filled boxplots + jittered points.

    Used by both :func:`plot_hrf_descriptors` (one dataset) and
    :func:`plot_descriptor_comparison` (two+), so the visual feel stays locked
    across every descriptor figure.
    """
    bp = ax.boxplot(list(datasets), patch_artist=True, widths=0.6,
                    showfliers=False, medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    for i, (data, c) in enumerate(zip(datasets, colors), start=1):
        jit = np.random.default_rng(0).normal(0, 0.05, len(data))
        ax.plot(i + jit, data, ".", color=c, ms=3, alpha=0.5, zorder=3)
    ax.set_xticks(range(1, len(datasets) + 1))
    ax.set_xticklabels(labels)


def _descriptor_raincloud(ax, datasets: Sequence[np.ndarray], colors: Sequence,
                          labels: Sequence[str]) -> None:
    """Raincloud counterpart of :func:`_descriptor_box` — same signature.

    For each group draws, in the group's own colour: a half-violin KDE *cloud*
    (left of the position), a narrow *box* (median/IQR/whiskers), and jittered
    *rain* of raw points (right). Shows each group's distribution shape (skew,
    tails), so no mean marker is needed. Needs :func:`scipy.stats.gaussian_kde`
    (already a project dependency).
    """
    rng = np.random.default_rng(0)
    for i, (data, c) in enumerate(zip(datasets, colors), start=1):
        y = np.asarray(data, float)
        y = y[np.isfinite(y)]
        if len(y) == 0:
            continue
        if len(y) > 1 and np.ptp(y) > 0:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(y)
            ys = np.linspace(y.min(), y.max(), 200)
            d = kde(ys)
            d = d / d.max() * 0.32
            ax.fill_betweenx(ys, i - d, i, color=c, alpha=0.40, lw=0, zorder=2)
            ax.plot(i - d, ys, color=c, lw=1.0, zorder=2)
        ax.boxplot(y, positions=[i], widths=0.10, showfliers=False,
                   medianprops=dict(color="black", linewidth=1.6),
                   boxprops=dict(linewidth=1.2, color=c),
                   whiskerprops=dict(linewidth=1.2, color=c),
                   capprops=dict(linewidth=1.2, color=c))
        ax.plot(i + 0.20 + rng.normal(0, 0.04, len(y)), y, ".",
                color=c, ms=3, alpha=0.5, zorder=3)
    n = len(datasets)
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.4, n + 0.7)


def _ground_truth_descriptors(hrf_true=None, ground_truth=None,
                              max_time: float = 30.0) -> "pd.Series | None":
    """Resolve a ground-truth descriptor Series from either input form.

    ``hrf_true`` is a 1-D ground-truth HRF curve (run through
    :func:`describe_hrfs`); ``ground_truth`` is precomputed descriptor values
    (Series / dict / one-row DataFrame). Returns ``None`` if neither given.
    """
    if ground_truth is not None:
        if isinstance(ground_truth, pd.DataFrame):
            return ground_truth.iloc[0]
        return pd.Series(dict(ground_truth))
    if hrf_true is not None:
        arr = np.asarray(hrf_true).ravel()[None, :]     # (1, T)
        return describe_hrfs(arr, max_time=max_time).iloc[0]
    return None


# Locked scientific exponents per descriptor (keeps y-ticks clean & aligned)
_DESCRIPTOR_EXP = {"HP": -3, "TTP[s]": 0, "FWHM[s]": 0, "TO[s]": 0,
                   "AUC": -2, "MU": -4, "TTU[s]": 1, "TT0[s]": 1}
# Marker colours (threaded across the whole figure)
_MEAN_COLOR = "#e41a1c"       # red  — mean
_MEDIAN_COLOR = "#ff7f0e"     # orange — median
_TRUTH_COLOR = "#1f4ed8"      # blue — numerical/theoretic ground truth
_SIG_COLOR = "black"          # significance asterisk (kept off the colour triad)
_ROPE_COLOR = "#1f4ed8"       # ROPE tolerance band — shares the truth hue (one entity)
_UNDECIDED_COLOR = "0.45"     # grey "?" for a straddling HDI (undecided verdict)


def _lock_exponent(ax, exponent):
    import matplotlib.ticker as mticker
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((exponent, exponent))
    ax.yaxis.set_major_formatter(fmt)
    ax.ticklabel_format(axis="y", style="sci")


_CLOUD_COLOR = "#5b9bd5"      # KDE cloud / HRF estimation (light blue)


def _raincloud_panel(ax, y, rng, pos=1.0, cloud_color=_CLOUD_COLOR):
    """Vertical raincloud in one panel: half-violin KDE *cloud* (left) + a
    narrow *box* (median/IQR/whiskers) + jittered *rain* of raw points (right).

    Shows the distribution's shape (skew, tails, bimodality) directly, so no
    mean marker is needed. Needs :func:`scipy.stats.gaussian_kde` (already a
    project dependency)."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return
    if len(y) > 1 and np.ptp(y) > 0:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(y)
        ys = np.linspace(y.min(), y.max(), 200)
        d = kde(ys)
        d = d / d.max() * 0.33
        ax.fill_betweenx(ys, pos - d, pos, color=cloud_color, alpha=0.40, lw=0)
        ax.plot(pos - d, ys, color=cloud_color, lw=1.0)
    ax.boxplot(y, positions=[pos], widths=0.10, showfliers=False,
               medianprops=dict(color=_MEDIAN_COLOR, linewidth=2),
               boxprops=dict(linewidth=1.3),
               whiskerprops=dict(linewidth=1.3),
               capprops=dict(linewidth=1.3))
    ax.plot(pos + 0.22 + rng.normal(0, 0.045, len(y)), y, ".",
            color="0.35", ms=3, alpha=0.45, zorder=1)
    ax.set_xlim(pos - 0.5, pos + 0.55)


def plot_hrf_descriptors(desc: pd.DataFrame, label: str = "experiment",
                         hrf_true=None, ground_truth=None, hrf_ensemble=None,
                         stats_df: pd.DataFrame | None = None,
                         margins: dict | None = None,
                         alpha: float = 0.05, show_significance: bool = True,
                         sig_loc: str = "top", show_jitter: bool = True,
                         style: str = "box",
                         gt_marker: str = "*", dt: float = 0.01,
                         max_time: float = 30.0,
                         features: Sequence[str] = DESCRIPTORS,
                         figsize=None, title: str = "HRF descriptors"):
    """Single-group descriptor figure — journal layout.

    2 x 4 grid of unfilled boxplots (one descriptor each) beside a tall HRF
    ensemble panel. Per box: orange median, red mean dot, light jittered raw
    points (the spread *and* its skew), and — when a ground truth is given — a
    blue star at the true value. The mean-vs-median offset is the minimalist
    skewness cue; no separate SD bar is drawn (jitter already carries spread).

    Ground truth (optional): pass ``hrf_true`` (a 1-D ground-truth HRF curve,
    run through :func:`describe_hrfs`) or ``ground_truth`` (precomputed
    descriptor values as Series/dict/one-row DataFrame). The blue star threads
    to the blue dashed "theoretic HRF" in the side panel — same entity, same
    colour everywhere.

    Verdict layer (on by default; two flavours, auto-detected from ``stats_df``):

    * **Bayesian ROPE** — pass a ``ground_truth_bayes`` table as ``stats_df``,
      or a ``margins`` dict to have it computed internally. Each panel then gets
      a shaded **ROPE band** ``[mu0-D, mu0+D]`` around the blue truth star (the
      tolerance you can *see* the box sit inside), and a three-way mark: no mark
      for ``equivalent`` (the band tells the story), a black ``*`` for
      ``biased`` (HDI outside the ROPE), a grey ``?`` for ``undecided`` (HDI
      straddling it). This is the recommended annotation for a recovery study.
    * **Frequentist** — pass a ``ground_truth_table`` /
      ``compare_to_ground_truth`` result, or neither ``stats_df`` nor
      ``margins`` (the one-sample test is then computed internally): panels
      where the ensemble differs significantly from truth (Bonferroni,
      family-wise ``alpha``) get a black ``*``. Note this flags *any* detectable
      offset, so with n~100 nearly every panel stars even when the bias is
      practically negligible — prefer the Bayesian layer for the headline.

    ``sig_loc="top"`` (default) centres the mark in the panel's upper part;
    ``sig_loc="marker"`` places it beside the truth star. Set
    ``show_significance=False`` to suppress.

    ``style``: ``"box"`` (default) draws unfilled boxplots with orange median,
    red mean dot, and light jitter (mean-vs-median offset = skewness cue).
    ``"raincloud"`` draws a half-violin KDE cloud + narrow box + rain of raw
    points, which shows distribution shape directly — so the mean dot is
    dropped (redundant); the blue truth star is kept. Needs
    ``scipy.stats.gaussian_kde``.

    Side panel (optional, legacy): pass ``hrf_ensemble`` (the ``(n_runs, T)``
    HRF array, e.g. ``res.filtered.states['hrf']``) to draw the HRF estimation
    (mean +/- SD band) with the theoretic HRF overlaid beside the descriptor
    grid. This mode draws at the notebook's original geometry (2.53 in per
    grid unit, i.e. ~3.8 in per descriptor box beside the wide HRF column),
    which is screen/slide sized rather than journal-column sized — a wider
    default canvas instead (use for slides / exploratory figures, not the
    journal submission path). The default (no ``hrf_ensemble``) path is the
    one used for manuscript figures: a plain 2x4 descriptor grid at the full
    IEEE page width (7.16 in), matching :func:`plot_descriptor_comparison`'s
    proven per-panel geometry. Pass ``figsize`` explicitly to override either
    default. Returns ``(fig, axes)`` where ``axes`` is a dict:
    ``{'boxes': [...8 axes...], 'hrf': ax_or_None}``.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    features = list(features)
    gt = _ground_truth_descriptors(hrf_true, ground_truth, max_time=max_time)

    # Verdict layer. Two flavours, auto-detected from the columns of stats_df:
    #   * Bayesian ROPE (ground_truth_bayes): 'verdict' + 'ROPE' columns → draw
    #     the tolerance band and a per-panel three-way mark.
    #   * frequentist (ground_truth_table / compare_to_ground_truth):
    #     'significant' column → the classic "* differs from truth" star.
    # If no stats_df is passed, `margins` decides which is computed internally.
    mode = None                      # "bayes" | "freq" | None
    sig_map, verdict_map, rope_map = {}, {}, {}
    if show_significance and gt is not None:
        if stats_df is None:
            if margins is not None:
                stats_df = ground_truth_bayes(desc, gt, margins, label=label,
                                              features=features)
            else:
                stats_df = compare_to_ground_truth(desc, gt, label=label,
                                                   alpha=alpha, features=features)
        if "verdict" in stats_df.columns:
            mode = "bayes"
            verdict_map = {r["Feature"]: str(r["verdict"]) for _, r in stats_df.iterrows()}
            if "ROPE" in stats_df.columns:
                rope_map = {r["Feature"]: float(r["ROPE"]) for _, r in stats_df.iterrows()}
        else:
            mode = "freq"
            sig_col = [c for c in stats_df.columns if c.startswith("significant")][0]
            sig_map = {r["Feature"]: bool(r[sig_col]) for _, r in stats_df.iterrows()}

    show_hrf = hrf_ensemble is not None
    ncol = 4
    nrow = int(np.ceil(len(features) / ncol))

    if show_hrf:
        # Legacy side-by-side layout: notebook geometry (see
        # docstring). Width picked so each descriptor box keeps the same
        # ~3.8 in per descriptor box, with the HRF column taking its
        # 3.5:1.5 share of the mosaic width ratios
        # of the extra width rather than crushing the boxes.
        # Notebook geometry: 2.526 in per grid unit -> 3.79 in per descriptor
        # box alongside the wide HRF column. Screen/slide sized by design.
        total_units = ncol * 1.5 + 3.5
        default_figsize = (total_units * 2.526, 9)
        fs = figsize or default_figsize
        mosaic = [[f"b{r*ncol+c+1}" for c in range(ncol)] + ["hrf"]
                  for r in range(nrow)]
        wr = [1.5] * ncol + [3.5]
        fig, axd = plt.subplot_mosaic(mosaic, figsize=fs,
                                      constrained_layout=True,
                                      gridspec_kw={"width_ratios": wr})
        boxes = [axd[f"b{i+1}"] for i in range(len(features))]
        hrf_ax = axd["hrf"]
    else:
        # Box-only path: plain nrow x ncol grid at the same per-panel
        # geometry as the legacy mode's descriptor boxes, matching
        # plot_descriptor_comparison.
        fs = figsize or (ncol * 3.79 * 0.72, 3.4 * nrow)
        fig, axarr = plt.subplots(nrow, ncol, figsize=fs,
                                  constrained_layout=True)
        boxes = list(np.atleast_1d(axarr).ravel())
        hrf_ax = None

    rng = np.random.default_rng(0)
    raincloud = style == "raincloud"
    for k, feat in enumerate(features):
        ax = boxes[k]
        y = desc[feat].dropna().to_numpy()
        if raincloud:
            _raincloud_panel(ax, y, rng, pos=1.0)   # no mean dot in raincloud
            ax.set_xticks([1.0]); ax.set_xticklabels([feat])
        else:
            ax.boxplot(y, tick_labels=[feat], widths=0.35,
                       medianprops=dict(color=_MEDIAN_COLOR, linewidth=2),
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       flierprops=dict(marker="o", markersize=3,
                                       markerfacecolor="none",
                                       markeredgecolor="0.5"))
            if show_jitter and len(y):
                ax.plot(1 + rng.normal(0, 0.045, len(y)), y, ".",
                        color="0.35", ms=3, alpha=0.35, zorder=1)
            if len(y):
                ax.plot(1, y.mean(), "o", color=_MEAN_COLOR, ms=9, zorder=5)

        if gt is not None and feat in gt.index:
            mu0 = float(gt[feat])
            # ROPE tolerance (Bayesian mode) — a short vertical bar to the right
            # of the box spanning [μ₀-Δ, μ₀+Δ]. Drawn as a range indicator
            # rather than a full-width band/lines: the ROPE is often much wider
            # than the ensemble spread, so a band would swamp the panel and
            # full-width lines dominate it; a compact bar beside the box keeps
            # the tolerance legible without forcing the y-range open.
            rope = rope_map.get(feat)
            x_star = 1.0                         # truth star x-position
            if mode == "bayes" and rope is not None:
                x_rope = 1.55
                x_star = x_rope                  # star rides on the ROPE bar
                ax.plot([x_rope, x_rope], [mu0 - rope, mu0 + rope],
                        color=_ROPE_COLOR, lw=1.4, alpha=0.85,
                        solid_capstyle="butt", zorder=4)
                # small caps so the interval endpoints read clearly
                for edge in (mu0 - rope, mu0 + rope):
                    ax.plot([x_rope - 0.05, x_rope + 0.05], [edge, edge],
                            color=_ROPE_COLOR, lw=1.4, alpha=0.85, zorder=4)
                ax.set_xlim(0.55, 1.8)
            # truth star — centred on μ₀; on the ROPE bar in Bayesian mode
            ax.plot(x_star, mu0, marker=gt_marker, ms=14, color=_TRUTH_COLOR,
                    ls="none", zorder=6)

            # verdict mark
            mark = None                                  # (glyph, color, fontsize)
            if mode == "freq" and sig_map.get(feat):
                mark = ("*", _SIG_COLOR, 22)
            elif mode == "bayes":
                v = verdict_map.get(feat)
                if v == "biased":
                    mark = ("*", _SIG_COLOR, 22)
                elif v == "undecided":
                    mark = ("?", _UNDECIDED_COLOR, 18)
                # "equivalent" → no mark; the box inside the band tells the story
            if mark is not None:
                glyph, mcol, mfs = mark
                if sig_loc == "top":
                    ax.annotate(glyph, xy=(0.5, 0.97), xycoords="axes fraction",
                                ha="center", va="top", color=mcol,
                                fontsize=mfs, fontweight="bold", zorder=7)
                else:                                    # beside the star
                    ax.annotate(glyph, xy=(x_star, mu0), xytext=(14, 4),
                                textcoords="offset points", ha="left",
                                va="bottom", color=mcol, fontsize=mfs - 3,
                                fontweight="bold", zorder=7)

        if feat in _DESCRIPTOR_EXP:
            _lock_exponent(ax, _DESCRIPTOR_EXP[feat])
        ax.margins(y=0.08)
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=12)

    for k in range(len(features), len(boxes)):
        boxes[k].set_visible(False)

    # --- HRF ensemble side panel -------------------------------------------
    if hrf_ax is not None:
        import seaborn as sns
        arr = np.squeeze(np.asarray(hrf_ensemble))
        if arr.ndim == 1:
            arr = arr[None, :]
        t = np.arange(arr.shape[1]) * dt
        df_hrf = pd.DataFrame({"time[s]": np.tile(t, arr.shape[0]),
                               "au": arr.ravel()})
        sns.lineplot(data=df_hrf, x="time[s]", y="au", errorbar="sd",
                     ax=hrf_ax, alpha=0.8, color="#5b9bd5",
                     label="HRF estimation")
        if hrf_true is not None:
            ht = np.asarray(hrf_true).ravel()
            hrf_ax.plot(t[:len(ht)], ht, color=_TRUTH_COLOR, lw=1.8, ls="--",
                        label="Theoretic HRF")
        hrf_ax.set_xlabel("time [s]", fontsize=15)
        hrf_ax.set_ylabel("a.u.", fontsize=14)
        hrf_ax.legend(fontsize=14, frameon=False, loc="upper right")

    # --- one shared figure-level legend ------------------------------------
    handles = [mpatches.Patch(color=_TRUTH_COLOR, label="ground truth (numerical/theoretic)"),
               mpatches.Patch(color=_MEDIAN_COLOR, label="median")]
    if raincloud:
        handles.append(mpatches.Patch(color=_CLOUD_COLOR, label="density (raw points)"))
    else:
        handles.append(mpatches.Patch(color=_MEAN_COLOR, label="mean"))
    if show_significance and gt is not None:
        if mode == "bayes":
            handles.append(mlines.Line2D([], [], color=_ROPE_COLOR, lw=1.4,
                                         label="ROPE (±Δ tolerance bar)"))
            handles.append(mpatches.Patch(color=_SIG_COLOR,
                                          label="* biased (HDI outside ROPE)"))
            handles.append(mpatches.Patch(color=_UNDECIDED_COLOR,
                                          label="? undecided (HDI straddles ROPE)"))
        elif mode == "freq":
            handles.append(mpatches.Patch(color=_SIG_COLOR,
                                          label=f"* differs from truth (p<{alpha:g}, Bonferroni)"))
    # "outside lower center" (constrained-layout-aware placement) reserves
    # real space for the legend row instead of floating it via bbox_to_anchor,
    # which the layout engine doesn't budget for -- that mismatch is what let
    # the legend collide with the bottom row of panels once fonts grew. The
    # long entries (e.g. the Bonferroni verdict label) can be wider than the
    # narrow IEEE-width box-only canvas at a naive ncol, overflowing the
    # figure edges -- fewer legend columns (and a slightly smaller legend
    # font) keeps every row within the canvas at 7.16 in; the wider legacy
    # canvas can afford more columns.
    legend_ncol = min(len(handles), 3) if show_hrf else min(len(handles), 2)
    legend_fs = 13 if show_hrf else 11
    fig.legend(handles=handles, loc="outside lower center",
               ncol=legend_ncol, fontsize=legend_fs, frameon=False)
    if title:
        fig.suptitle(title, fontsize=20)
    return fig, {"boxes": boxes, "hrf": hrf_ax}


def plot_descriptor_comparison(desc_a: pd.DataFrame, desc_b: pd.DataFrame,
                               labels: tuple[str, str] = ("A", "B"),
                               colors: tuple[str, str] = GROUP_PALETTE[:2],
                               stats_df: pd.DataFrame | None = None,
                               style: str = "box",
                               features: Sequence[str] = DESCRIPTORS,
                               axes=None):
    """Per-descriptor distributions for two groups (paper Fig. descriptor).

    ``style``: ``"box"`` (default) draws filled boxplots + jittered points;
    ``"raincloud"`` draws a half-violin KDE cloud + narrow box + rain of raw
    points per group. Both use the shared helpers so group colours (from
    ``GROUP_PALETTE``, group 0 = red, group 1 = blue) thread identically to
    :func:`plot_hrf_descriptors`.

    If ``stats_df`` is supplied, a verdict marker is drawn in the upper part of
    each panel. The layer is auto-detected from the columns:

    * **Bayesian** (:func:`compare_groups_bayes`, has a ``verdict`` column) —
      ``different`` → black ``*``, ``undecided`` → grey ``?``, ``equivalent`` →
      no mark. No ROPE band is drawn on the raw boxes: the ROPE and 95% HDI are
      properties of the between-group difference ``delta = mu_a - mu_b`` and live
      in difference space, not on the descriptor axis, so they belong in an
      accompanying table rather than as geometry on the group distributions.
    * **Frequentist** (:func:`compare_groups`, has a ``significant`` column) —
      a bracket + ``*`` spans the two boxes when the Bonferroni-adjusted p-value
      clears the family-wise ``alpha``.

    Returns ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt
    features = list(features)
    if axes is None:
        ncol = 4
        nrow = int(np.ceil(len(features) / ncol))
        # Per-panel width kept at the notebook value (3.3 in,
        # i.e. a `figure*` spanning both columns) so the bumped font sizes
        # print at their literal point size instead of being shrunk by a
        # later \includegraphics rescale.
        width_per_col = 3.3
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(width_per_col * ncol, 2.7 * nrow))
        axes = np.atleast_1d(axes).ravel()
    else:
        fig = axes[0].figure

    sig_map = {}
    if stats_df is not None:
        for _, r in stats_df.iterrows():
            sig_map[r["Feature"]] = r

    for k, feat in enumerate(features):
        ax = axes[k]
        a = desc_a[feat].dropna().to_numpy()
        b = desc_b[feat].dropna().to_numpy()
        draw = _descriptor_raincloud if style == "raincloud" else _descriptor_box
        draw(ax, [a, b], colors, labels)
        ax.tick_params(axis="x", labelsize=10)
        ax.set_title(feat)
        if feat in sig_map:
            r = sig_map[feat]
            if "verdict" in stats_df.columns:
                # Bayesian two-group verdict (compare_groups_bayes). The ROPE and
                # HDI live in *difference* space (delta = mu_a - mu_b), not on the
                # descriptor axis, so we do NOT draw a band on the raw boxes — only
                # a verdict glyph centred in the upper part of the panel. The full
                # delta / HDI / ROPE numbers belong in an accompanying table.
                glyph, gcol, gfs = {
                    "different": ("*", _SIG_COLOR, 22),
                    "undecided": ("?", _UNDECIDED_COLOR, 18),
                }.get(str(r["verdict"]), (None, None, None))
                if glyph is not None:
                    lo, hi = ax.get_ylim()
                    ax.set_ylim(lo, hi + 0.12 * (hi - lo))   # headroom for the glyph
                    ax.annotate(glyph, xy=(0.5, 0.97), xycoords="axes fraction",
                                ha="center", va="top", color=gcol,
                                fontsize=gfs, fontweight="bold")
            else:
                # Frequentist significance bracket (compare_groups).
                sig_col = [c for c in stats_df.columns if c.startswith("significant")][0]
                if bool(r[sig_col]):
                    ymax = max(a.max() if len(a) else 0, b.max() if len(b) else 0)
                    ymin = min(a.min() if len(a) else 0, b.min() if len(b) else 0)
                    yr = (ymax - ymin) or 1.0
                    y = ymax + 0.08 * yr
                    ax.plot([1, 1, 2, 2], [y, y + 0.03*yr, y + 0.03*yr, y],
                            lw=0.8, color="black")
                    ax.text(1.5, y + 0.04*yr, "*", ha="center", va="bottom", fontsize=13)
                    ax.set_ylim(top=y + 0.18*yr)
    for k in range(len(features), len(axes)):
        axes[k].set_visible(False)
    fig.tight_layout()
    return fig, axes


# --------------------------------------------------------------------------- #
# Stage 6b — ensemble reporting (ported from HundredSignalAnalysis notebook)
# --------------------------------------------------------------------------- #
# These reproduce the notebook's ``plot_balloon_fitting`` (single experiment)
# and ``plot_balloon_comparison`` (n experiments overlaid), renamed to
# ``plot_ensemble_states`` / ``plot_ensemble_comparison``. The seaborn ``sd``
# band across runs *is* the 100-run ensemble spread — mean line + shaded SD.
#
# Fixes applied vs. the notebook original:
#   * ``arr2DF`` (array->tidy DataFrame) is an internal helper taking a plain
#     ``(n_runs, T)`` array, so it plugs straight into ``FilterResult.states``;
#     it still accepts the notebook's ``{key: {'noNan': arr}}`` wrapper.
#   * the per-iter BOLD reconstruction (notebook ``process_iter``/``BOLD_iter``)
#     no longer closes over module globals — the stimulus is passed in.
#   * the ``'Overall_sim_time'`` typo (silently ignored in the notebook) is
#     corrected to ``'Overall_stim_time'``.

def _as_states(obj) -> dict[str, np.ndarray]:
    """Normalise any accepted container to ``{state: (n_runs, T)}``.

    Accepts an :class:`ExperimentResult` (from :func:`run_analysis`), a
    :class:`FilterResult`, a plain ``{key: (n_runs, T)}`` dict, or the
    notebook's ``{key: {'noNan': arr}}`` wrapper.
    """
    if hasattr(obj, "filtered"):            # ExperimentResult -> its FilterResult
        obj = obj.filtered
    if hasattr(obj, "states"):              # FilterResult (or anything exposing states)
        return obj.states
    out = {}
    for k, v in obj.items():
        if isinstance(v, dict) and "noNan" in v:
            v = v["noNan"]
        arr = np.asarray(v)
        if k in STATE_KEYS:
            out[k] = np.squeeze(arr)
    return out


def _arr2df(arr: np.ndarray, key: str, dt: float = 0.01) -> pd.DataFrame:
    """Melt a ``(n_runs, T)`` array to tidy long form (notebook ``arr2DF``).

    Columns: ``time[s]``, ``iter``, ``stateVar``, ``au``. Time base is
    ``arange(T) * dt`` (the notebook used ``/100`` == ``dt=0.01``).
    """
    a = np.asarray(arr)
    a = a.squeeze() if a.ndim > 2 else a
    n_cells, T = np.min(a.shape), np.max(a.shape)
    if a.shape != (n_cells, T):
        a = a.reshape(n_cells, T)
    time_vals = np.arange(T) * dt
    return pd.DataFrame({
        "time[s]": np.tile(time_vals, n_cells),
        "iter": np.repeat(np.arange(n_cells), T),
        "stateVar": np.repeat(key, n_cells * T),
        "au": a.ravel(),
    })


def _bold_per_iter(df_hrf: pd.DataFrame, stimulus, stim_time, bpl) -> pd.DataFrame:
    """Convolve each run's HRF with ``stimulus`` (notebook ``process_iter``).

    Returns tidy ``time`` / ``au`` rows for every iteration, so seaborn draws
    the ensemble BOLD mean +/- SD. ``bpl`` is the imported ``balloonpinnlib``
    (kept as a parameter so this module never hard-imports torch).
    """
    import torch

    def _one(group):
        hrf = torch.tensor(group["au"].values, device="cpu",
                           requires_grad=False, dtype=torch.float32)
        bold, tt = bpl.tofit(_to_cpu(stimulus), hrf,
                             stim_time[-1].item() + 0.01)
        bold = bold.cpu().view(-1, 1)
        tt = tt.cpu().view(-1, 1)
        res = torch.concat((tt, bold), dim=1).detach().numpy()
        return pd.DataFrame(res, columns=["time", "au"])

    return df_hrf.groupby("iter").apply(_one, include_groups=False).reset_index()


def _overall_bold_panel(ax, df_hrf: pd.DataFrame, data_params: dict, bpl,
                        color=None, est_label: str = "Estimated BOLD",
                        show_data: bool = True) -> None:
    """Draw one "overall" BOLD-fitting panel: measured data (scatter) + the
    ensemble-reconstructed BOLD (mean +/- SD band, offset-aligned to the data)
    + the stimulus train. Mirrors the ``ax4`` panel of
    :func:`plot_ensemble_states` so the two functions stay identical, and is
    reused per group by :func:`plot_ensemble_comparison`.

    ``bpl`` is the imported ``balloonpinnlib`` (kept as a parameter so this
    module never hard-imports torch). Requires the same ``data_params`` keys.
    """
    import seaborn as sns
    import torch

    Overall_stimuli = data_params["Overallstim"]
    Overall_stim_time = data_params["Overall_stim_time"]
    Bold_Signal = data_params["Bold_Signal"]
    Bold_data_time = data_params.get("Bold_data_time")

    overall = _bold_per_iter(df_hrf, Overall_stimuli, Overall_stim_time, bpl)
    offset = 0.0
    if Bold_data_time is not None:
        hrf_mean = df_hrf.pivot(values="au", index="iter",
                                columns="time[s]").mean(axis=0)
        hrf_mean = torch.tensor(hrf_mean.values, dtype=torch.float32,
                                    device="cpu")
        ov_mean, ov_time = bpl.tofit(_to_cpu(Overall_stimuli), hrf_mean,
                                     Overall_stim_time[-1].item() + 0.01)
        idx, _ = bpl.timeBall(_to_cpu(Bold_data_time), ov_time)
        idx = idx.detach().cpu()
        bvals = Bold_Signal.values if hasattr(Bold_Signal, "values") else np.asarray(Bold_Signal)
        offset = -np.mean(bpl.tensor2np(ov_mean[idx]) - bvals)

    if show_data and Bold_data_time is not None:
        bvals = Bold_Signal.values if hasattr(Bold_Signal, "values") else np.asarray(Bold_Signal)
        ax.scatter(bpl.tensor2np(Bold_data_time), bvals, label="data", s=12,
                   color=color, alpha=0.6)
    sns.lineplot(data=overall.assign(au=lambda d: offset + d["au"]),
                 x="time", y="au", err_style="band", errorbar="sd",
                 alpha=0.8, ax=ax, label=est_label, color=color)
    ax.plot(bpl.tensor2np(Overall_stim_time), bpl.tensor2np(Overall_stimuli),
            color="green", label="Stimulus")
    ax.legend(fontsize=12, loc="lower center", ncol=3)
    ax.set_xlabel("time[s]"); ax.set_ylabel("au")


def plot_ensemble_states(states, *, dt: float = 0.01,
                         numerical_solutions: dict | None = None,
                         first_non_zero_index: int = 0,
                         title: str = "Ensemble fit",
                         data_params: dict | None = None,
                         show_bold_signal: bool = False,
                         figsize: tuple[float, float] | None = None,
                         axes=None):
    """Ensemble state/HRF (and optionally BOLD) fit for one experiment.

    ``states`` is a :class:`FilterResult`, a ``{state: (n_runs, T)}`` dict, or
    the notebook ``{state: {'noNan': arr}}`` wrapper. The three state panels
    (f/m, v/q, HRF) show the across-run mean with a +/-1 SD band (seaborn
    ``errorbar='sd'``). ``numerical_solutions`` (keys ``f, m, v, q, hrf``)
    overlays dashed ground truth.

    ``show_bold_signal=True`` adds the two BOLD panels; it requires ``torch`` +
    ``balloonpinnlib`` and a fully-populated ``data_params`` (keys as in the
    notebook: ``TR``, ``Sti_Onsets``, ``stim_length [seg]``, ``Bold_Signal``,
    ``Bold_data_time``, ``Overallstim``, ``Overall_stim_time``, ``stimulus``,
    ``stimulus_time``). Pure state/HRF panels need only numpy + seaborn.
    Tensors in ``data_params`` may live on any device (CPU or CUDA), or a mix.

    ``figsize`` overrides the drawn canvas; the defaults are the notebook
    geometry, (14, 6) in both modes. These are screen/slide sized, NOT journal
    page sized -- fonts are literal points and do not scale with the canvas, so
    forcing this figure into a 7.16 in column makes the text swamp the panels.
    For a print figure, keep the wide canvas and let ``\\includegraphics``
    scale it, or pass a smaller ``figsize`` and re-check legibility.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MultipleLocator

    st = _as_states(states)
    dfs = {k: _arr2df(st[k], k, dt=dt) for k in STATE_KEYS if k in st}
    T = dfs["hrf"]["time[s]"].nunique()
    t_plot = np.arange(T) * dt

    fs = figsize if figsize is not None else (14, 6)
    if show_bold_signal:
        fig = plt.figure(figsize=fs, layout="constrained")
        gs = GridSpec(3, 4, figure=fig)
        ax0 = fig.add_subplot(gs[0:2, 0]); ax1 = fig.add_subplot(gs[0:2, 1])
        ax2 = fig.add_subplot(gs[0:2, 2]); ax3 = fig.add_subplot(gs[0:2, 3])
        ax4 = fig.add_subplot(gs[2, :])
    else:
        fig, axs = plt.subplots(1, 3, figsize=fs, layout="constrained")
        ax0, ax1, ax2 = axs.flatten()
    fig.suptitle(title, fontsize=18)

    def _panel(ax, frame, name, num_keys):
        sns.lineplot(data=frame, x="time[s]", y="au", err_style="band",
                     errorbar="sd", n_boot=100, ls="-", alpha=0.7,
                     hue="stateVar", ax=ax)
        if numerical_solutions is not None:
            for nk, col in num_keys:
                if nk in numerical_solutions:
                    ax.plot(t_plot, np.asarray(numerical_solutions[nk]).ravel(),
                            "--", lw=1, c=col, label=f"Numerical {nk}")
        ax.axvline(x=t_plot[first_non_zero_index], color="r", ls="--")
        ax.xaxis.set_minor_locator(MultipleLocator(2.5))
        ax.legend(fontsize=11, frameon=False)
        ax.set_xlabel("time[s]"); ax.set_ylabel("au"); ax.set_title(name)

    _panel(ax0, pd.concat([dfs["f"], dfs["m"]], ignore_index=True),
           "f_in and m", [("f", "midnightblue"), ("m", "orange")])
    _panel(ax1, pd.concat([dfs["v"], dfs["q"]], ignore_index=True),
           "v and q", [("v", "midnightblue"), ("q", "midnightblue")])
    _panel(ax2, dfs["hrf"], "HRF", [("hrf", "midnightblue")])

    if show_bold_signal:
        if data_params is None:
            raise ValueError("data_params must be provided when show_bold_signal=True")
        bpl = _get_bpl()
        required = ["Bold_Signal", "Sti_Onsets", "TR", "stim_length [seg]",
                    "stimulus", "stimulus_time", "Overallstim", "Overall_stim_time"]
        missing = [k for k in required if k not in data_params]
        if missing:
            raise ValueError(f"data_params missing keys: {missing}")

        stimulus = _to_cpu(data_params["stimulus"])
        stimulus_time = _to_cpu(data_params["stimulus_time"])
        Overall_stimuli = _to_cpu(data_params["Overallstim"])
        Overall_stim_time = _to_cpu(data_params["Overall_stim_time"])
        Bold_Signal = data_params["Bold_Signal"]
        Bold_data_time = _to_cpu(data_params.get("Bold_data_time"))
        TR = data_params["TR"]
        Sti_Onsets = data_params["Sti_Onsets"]

        # single-trial ensemble BOLD
        single = _bold_per_iter(dfs["hrf"], stimulus, stimulus_time, bpl)
        # overall ensemble BOLD (offset-aligned to data)
        overall = _bold_per_iter(dfs["hrf"], Overall_stimuli, Overall_stim_time, bpl)
        offset = 0.0
        if Bold_data_time is not None:
            import torch
            hrf_mean = dfs["hrf"].pivot(values="au", index="iter",
                                        columns="time[s]").mean(axis=0)
            hrf_mean = torch.tensor(hrf_mean.values, dtype=torch.float32,
                                    device="cpu")
            ov_mean, ov_time = bpl.tofit(Overall_stimuli, hrf_mean,
                                         Overall_stim_time[-1].item() + 0.01)
            idx, _ = bpl.timeBall(Bold_data_time, ov_time)
            idx = idx.detach().cpu()
            bvals = Bold_Signal.values if hasattr(Bold_Signal, "values") else np.asarray(Bold_Signal)
            offset = -np.mean(bpl.tensor2np(ov_mean[idx]) - bvals)

        sns.lineplot(data=single.assign(au=lambda d: offset + d["au"]),
                     x="time", y="au", err_style="band", errorbar="sd",
                     n_boot=100, alpha=0.7, ax=ax3, label="Estimated \nBOLD")
        ax3.plot(np.asarray(stimulus_time.cpu()) if hasattr(stimulus_time, "cpu")
                 else stimulus_time,
                 bpl.tensor2np(stimulus), color="green", alpha=0.7, label="Stimulus")

        # stimulus-locked epochs of the measured signal, as in the notebook's
        # plot_balloon_fitting: the point of this panel is the single-trial
        # prediction *against* the per-epoch measurements.
        bvals = Bold_Signal.values if hasattr(Bold_Signal, "values") else np.asarray(Bold_Signal)
        try:
            Bold_segments, time_corrected = bpl.segmentData(
                bvals, Sti_Onsets=Sti_Onsets,
                time_bf_stim=data_params.get("time_bf_stim", TR),
                t0s=data_params.get("t0", 0), TR=TR)
            for _tc, _seg in zip(time_corrected, Bold_segments):
                ax3.scatter(bpl.tensor2np(_tc), bpl.tensor2np(_seg),
                            color="orange", s=12, alpha=0.8)
        except Exception as exc:                      # pragma: no cover
            warnings.warn(f"segmentData failed, ax3 epochs omitted: {exc}")

        ax3.legend(fontsize=10, frameon=False, loc = 'upper right', ncol = 1)
        ax3.set_xlabel("time[s]"); ax3.set_ylabel("au")
        
        ax3.set_title("Estimated BOLD, single stimulus")

        if Bold_data_time is not None:
            ax4.scatter(bpl.tensor2np(Bold_data_time), bvals, color="orange", label="data", s=12)
        sns.lineplot(data= overall.assign(au=lambda d: offset + d["au"]),
                     x="time", y="au", err_style="band", errorbar="sd",
                     alpha=0.7, ax=ax4, label="Estimated BOLD")
        ax4.plot(bpl.tensor2np(Overall_stim_time), bpl.tensor2np(Overall_stimuli),
                 color="green", label="Stimuli")
        ax4.legend(fontsize=12, bbox_to_anchor=(0.75, -0.5), loc="lower center", ncol=3, frameon=False)
        ax4.set_xlabel("time[s]"); ax4.set_ylabel("au")
        ax4.set_title("BOLD fitting, estimation and data")
        ax4.set_ylim(overall["au"].min()*8, overall["au"].max()*1.5)
        return fig, (ax0, ax1, ax2, ax3, ax4)

    return fig, (ax0, ax1, ax2)



def plot_ensemble_comparison(states_list: Sequence, labels: Sequence[str],
                             *, dt: float = 0.01,
                             numerical_solutions: dict | None = None,
                             first_non_zero_index: int = 0,
                             title: str = "Ensemble comparison",
                             colors: Sequence | None = None,
                             show_bold_signal: bool = False,
                             data_params_list: Sequence[dict] | None = None,
                             figsize: tuple[float, float] | None = None):
    """Overlay n ensembles (e.g. right vs left) on shared state/HRF axes.

    ``states_list`` is a list of the containers accepted by
    :func:`plot_ensemble_states`; ``labels`` names each. Each ensemble is drawn
    as its own mean +/- SD band in a distinct hue.

    With ``show_bold_signal=True`` the figure gains one BOLD-fitting panel per
    group beneath the state/HRF row (measured data + offset-aligned
    reconstruction + stimulus, in the group's colour) — reproducing the
    notebook's ``plot_balloon_comparison`` layout. It then requires
    ``data_params_list``: one ``data_params`` dict per group (same keys as
    :func:`plot_ensemble_states`), in the same order as ``states_list``, plus
    ``torch`` and ``balloonpinnlib``. Without it the three state/HRF panels use
    only numpy + seaborn.
    
    ``figsize`` overrides the drawn canvas; the defaults are the notebook
    geometry, (16, 6) for the state row and (16, 6 + 2.2 per group) with
    ``show_bold_signal=True``. These are screen/slide sized, NOT journal page
    sized -- see the note in :func:`plot_ensemble_states`.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MultipleLocator

    import matplotlib.colors as mcolors

    n = len(states_list)
    if len(labels) != n:
        raise ValueError(f"len(labels)={len(labels)} != len(states_list)={n}")
    if colors is None:
        if n > len(GROUP_PALETTE):
            raise ValueError(
                f"{n} groups but GROUP_PALETTE defines only {len(GROUP_PALETTE)} "
                "colours; pass an explicit `colors=` sequence.")
        colors = GROUP_PALETTE[:n]

    sts = [_as_states(s) for s in states_list]
    T = np.max(np.asarray(sts[0]["hrf"]).shape)
    t_plot = np.arange(T) * dt

    def _long(key):
        frames = []
        for st, lbl in zip(sts, labels):
            df = _arr2df(st[key], key, dt=dt)
            df["curve"] = f"{key} ({lbl})"
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    # Each group keeps its GROUP_PALETTE colour across every figure. Where a
    # panel overlays two states per group (f/m, v/q), a darker/lighter shade
    # of the *same* group colour keeps the two states apart without changing
    # the group's identity.
    def _shade(hexc, factor):
        r, g, b = mcolors.to_rgb(hexc)
        if factor < 1:                      # darken
            return (r * factor, g * factor, b * factor)
        return tuple(min(1.0, c + (1 - c) * (factor - 1)) for c in (r, g, b))

    palette = {}
    for i, lbl in enumerate(labels):
        base = colors[i]
        palette[f"hrf ({lbl})"] = base                      # pure group colour
        palette[f"f ({lbl})"] = palette[f"v ({lbl})"] = _shade(base, 0.65)  # darker
        palette[f"m ({lbl})"] = palette[f"q ({lbl})"] = _shade(base, 1.5)   # lighter

    if show_bold_signal:
        if data_params_list is None or len(data_params_list) != n:
            raise ValueError(
                "show_bold_signal=True needs data_params_list: one data_params "
                f"dict per group (got {None if data_params_list is None else len(data_params_list)}, "
                f"need {n}).")
        from matplotlib.gridspec import GridSpec
        fs = figsize if figsize is not None else (16, 6 + 2.2 * n)
        fig = plt.figure(figsize=fs, layout="constrained")
        gs = GridSpec(2 + n, 3, figure=fig, height_ratios=[1, 1] + [0.9] * n)
        ax0 = fig.add_subplot(gs[0:2, 0])
        ax1 = fig.add_subplot(gs[0:2, 1])
        ax2 = fig.add_subplot(gs[0:2, 2])
        bold_axes = [fig.add_subplot(gs[2 + i, :]) for i in range(n)]
    else:
        fs = figsize if figsize is not None else (16, 6)
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=fs,
                                            layout="constrained")
        bold_axes = []
    fig.suptitle(title, fontsize=18)

    def _panel(ax, frame, name, num):
        sns.lineplot(data=frame, x="time[s]", y="au", hue="curve",
                     palette=palette, err_style="band", errorbar="sd",
                     n_boot=100, alpha=0.75, ax=ax)
        if numerical_solutions is not None:
            for nk, col in num:
                if nk in numerical_solutions:
                    ax.plot(t_plot, np.asarray(numerical_solutions[nk]).ravel(),
                            "--", lw=1, c=col, label=f"Num {nk}")
        ax.axvline(x=t_plot[first_non_zero_index], color="r", ls="--", lw=1)
        ax.xaxis.set_minor_locator(MultipleLocator(2.5))
        ax.legend(fontsize=10)
        ax.set_xlabel("time[s]"); ax.set_ylabel("au"); ax.set_title(name)

    _panel(ax0, pd.concat([_long("f"), _long("m")], ignore_index=True),
           "f_in and m", [("f", "midnightblue"), ("m", "orange")])
    _panel(ax1, pd.concat([_long("v"), _long("q")], ignore_index=True),
           "v and q", [("v", "midnightblue"), ("q", "midnightblue")])
    _panel(ax2, _long("hrf"), "HRF", [("hrf", "midnightblue")])

    if show_bold_signal:
        bpl = _get_bpl()
        required = ["Bold_Signal", "Overallstim", "Overall_stim_time"]
        for i, (st, lbl, dp, ax) in enumerate(
                zip(sts, labels, data_params_list, bold_axes)):
            missing = [k for k in required if k not in dp]
            if missing:
                raise ValueError(f"data_params_list[{i}] ({lbl}) missing keys: {missing}")
            df_hrf = _arr2df(st["hrf"], "hrf", dt=dt)
            _overall_bold_panel(ax, df_hrf, dp, bpl, color=colors[i],
                                est_label=f"Est. BOLD ({lbl})")
            ax.set_title(f"BOLD fitting — {lbl}")
        return fig, (ax0, ax1, ax2, *bold_axes)

    return fig, (ax0, ax1, ax2)
