"""Recursive out-of-sample VoC engine for KMZ (2024), Section V.C.

One repetition (seed) draws ``P_max`` random Fourier features once, then runs the
rolling ``T``-month recursion: at each decision point it trains dual-form ridge on
the ``T`` most recent pairs, forecasts the next return, and books the timing-
strategy return. The nested complexity grid reuses the single RFF draw (slice the
first ``P`` columns) and one Gram eigendecomposition per ``(window, P)`` across all
shrinkage levels ``z`` (see :mod:`voc.rff` and :mod:`voc.kernel_ridge`).

:func:`run_grid` runs many seeds — independent and parallelizable — and returns a
long-format table of per-``(seed, P, z)`` statistics plus the across-seed means:
the paper's aggregate-the-statistics-not-the-forecasts convention (2025 reply,
Section 4.2.1). The engine is pure (no file IO, no data-pipeline imports); a
driver loads the standardized dataset, calls :func:`run_grid`, and caches the
result.

Timing (get this exactly right)
-------------------------------
Predictors ``G[t]`` and the return ``R[t]`` arrive on the same row of the
standardized dataset; the engine applies the single uniform shift, pairing
``G[t]`` with the NEXT month's return ``R[t+1]``. At decision point ``t`` it trains
on the pairs ``(G[t-T], R[t-T+1]), ..., (G[t-1], R[t])`` — every training return
already realized — forecasts ``R[t+1]`` from ``G[t]``, and books
``forecast * R[t+1]``. Nothing dated after ``t`` enters the fit or the
standardization, so there is no lookahead.

The ridgeless (``z -> 0`` minimum-norm) column is recorded as ``z = 0.0``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from voc.kernel_ridge import ridge_dual, ridgeless
from voc.performance_metrics import compute_metrics
from voc.rff import compute_rff, draw_rff_weights, standardize_by_training_window

GAMMA_DEFAULT = 2.0
Z_GRID_DEFAULT = tuple(10.0 ** k for k in range(-3, 4))  # 1e-3 .. 1e3
# Complexity grid P = c * T; includes P=2, P=12 (c=1), and P=12000 (c=1000, the
# anchor). All even (sin/cos pairs).
P_GRID_DEFAULT = (2, 4, 8, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12000)
_STAT_COLUMNS = (
    "r2", "beta_norm", "mean_return", "volatility", "sharpe",
    "alpha", "information_ratio", "alpha_tstat",
)


def _oos_windows(n_rows, T):
    """Decision points and their training/forecast row indices.

    For decision point ``ts[i] = t`` the training feature rows are ``t-T..t-1``,
    the training-return rows are ``t-T+1..t`` (shifted one month forward), and the
    forecast uses row ``t`` against the realized ``R[t+1]``.
    """
    ts = np.arange(T, n_rows - 1)  # need R[t+1], so t <= n-2
    feat_rows = ts[:, None] + np.arange(-T, 0)[None, :]
    ret_rows = ts[:, None] + np.arange(-T + 1, 1)[None, :]
    return ts, feat_rows, ret_rows


def run_recursive_oos(
    G, R, seed, T=12, p_grid=P_GRID_DEFAULT, z_grid=Z_GRID_DEFAULT,
    gamma=GAMMA_DEFAULT, include_ridgeless=True, return_forecasts=False,
):
    """One repetition (one RFF seed) of the recursive OOS analysis.

    Parameters
    ----------
    G : array-like, shape (n, d)
        Volatility-standardized predictors; row t is month t.
    R : array-like, shape (n,)
        Volatility-standardized market excess returns; row t is month t.
    seed : int
        RFF repetition index.
    T, p_grid, z_grid, gamma : see module defaults.
    include_ridgeless : bool
        Also fit the ``z -> 0`` minimum-norm model, recorded as ``z = 0.0``.
    return_forecasts : bool
        Additionally return the per-``(P, z)`` forecast and realized-return series.

    Returns
    -------
    list of dict
        One row per ``(P, z)`` — the metrics of :func:`compute_metrics` plus
        ``seed``, ``P``, ``z``, and ``c = P / T``. If ``return_forecasts``, also a
        dict with ``"forecasts"`` (``(P, z) -> array``) and ``"realized"``.
    """
    G = np.asarray(G, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    p_grid = tuple(sorted(p_grid))
    if any(P % 2 for P in p_grid):
        raise ValueError("all P must be even (RFFs come in sin/cos pairs)")
    z_grid = tuple(float(z) for z in z_grid)
    if any(z == 0.0 for z in z_grid):
        raise ValueError("z_grid must be > 0; use include_ridgeless for the z=0 limit")

    omega = draw_rff_weights(p_grid[-1] // 2, G.shape[1], seed)
    S = compute_rff(G, omega, gamma=gamma)  # one max-size draw, sliced for nesting

    ts, feat_rows, ret_rows = _oos_windows(G.shape[0], T)
    n_oos = ts.size
    realized = R[ts + 1]

    z_labels = list(z_grid) + ([0.0] if include_ridgeless else [])
    forecasts = {(P, z): np.empty(n_oos) for P in p_grid for z in z_labels}
    beta_norms = {(P, z): np.empty(n_oos) for P in p_grid for z in z_labels}

    for i in range(n_oos):
        # Standardize every feature column by its training-window volatility once,
        # then slice the first P for each model (columns nest).
        train_std, oos_std = standardize_by_training_window(S[feat_rows[i]], S[ts[i]])
        R_train = R[ret_rows[i]]
        for P in p_grid:
            X = train_std[:, :P]
            s_oos = oos_std[:P]
            betas = ridge_dual(X, R_train, z_grid)  # one eigendecomposition, all z
            forecasts_z = betas @ s_oos
            norms_z = np.linalg.norm(betas, axis=1)
            for j, z in enumerate(z_grid):
                forecasts[(P, z)][i] = forecasts_z[j]
                beta_norms[(P, z)][i] = norms_z[j]
            if include_ridgeless:
                beta_rl = ridgeless(X, R_train)
                forecasts[(P, 0.0)][i] = s_oos @ beta_rl
                beta_norms[(P, 0.0)][i] = np.linalg.norm(beta_rl)

    results = []
    for P in p_grid:
        for z in z_labels:
            stats = compute_metrics(forecasts[(P, z)], realized, beta_norms[(P, z)])
            stats.update({"seed": int(seed), "P": int(P), "z": float(z), "c": P / T})
            results.append(stats)
    if return_forecasts:
        return results, {"forecasts": forecasts, "realized": realized}
    return results


def _aggregate(seed_stats, T):
    """Stack per-seed rows and average each statistic across seeds per (P, z)."""
    per_seed = pd.DataFrame([row for rows in seed_stats for row in rows])
    averaged = (
        per_seed.groupby(["P", "z"], sort=False)[list(_STAT_COLUMNS)]
        .mean()
        .reset_index()
    )
    averaged["c"] = averaged["P"] / T
    return per_seed, averaged


def _write_forecast_series(seed_bundles, dates, T, study_name, data_dir):
    """Persist per-seed forecast / realized / strategy series for each (P, z).

    ``seed_bundles`` is an iterable of ``(seed, bundle)`` from
    :func:`run_recursive_oos` with ``return_forecasts=True``. The realized return
    at out-of-sample window i is ``R[T+1+i]``, so its label is ``dates[T+1+i]`` when
    ``dates`` is given, else the integer position.
    """
    frames = []
    for seed, bundle in seed_bundles:
        realized = bundle["realized"]
        n_oos = realized.shape[0]
        obs = (
            np.asarray(dates)[T + 1 : T + 1 + n_oos]
            if dates is not None
            else np.arange(n_oos)
        )
        for (P, z), forecast in bundle["forecasts"].items():
            frames.append(
                pd.DataFrame(
                    {
                        "seed": seed, "P": P, "z": z, "obs": obs,
                        "forecast": forecast, "realized": realized,
                        "strategy": forecast * realized,
                    }
                )
            )
    path = Path(data_dir) / f"forecasts_{study_name}.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(path)
    return path


def run_voc_study(
    target, predictors, *, dates=None, T=12, p_grid=P_GRID_DEFAULT,
    z_grid=Z_GRID_DEFAULT, seeds=range(50), gamma=GAMMA_DEFAULT,
    include_ridgeless=True, n_jobs=1, save_forecasts=False,
    study_name="market", data_dir=None,
):
    """Run a virtue-of-complexity study on any (target, predictors) pair.

    The reusable entry point: the KMZ market replication is one call, and
    asset-class studies (bonds, international) or experiments (e.g. the Nagel
    counterfactual) are other calls with different inputs. Inputs are assumed
    ALREADY volatility-standardized (see :mod:`voc.preprocessing`) — the engine
    never standardizes for you, so no lookahead is smuggled in on your behalf.

    Parameters
    ----------
    target : array-like, shape (n,)
        Standardized target returns; row t is month t.
    predictors : array-like or DataFrame, shape (n, d)
        Standardized predictors; the RFF dimension follows the width d.
    dates : array-like, shape (n,), optional
        Labels used for the forecast export's ``obs`` column.
    save_forecasts : bool
        If True, write per-seed forecast / realized / strategy series for every
        ``(P, z)`` to ``<data_dir>/forecasts_<study_name>.parquet`` (requires
        ``data_dir``). Keep the grid small when enabling this (e.g. just the anchor
        configuration).
    T, p_grid, z_grid, seeds, gamma, include_ridgeless, n_jobs :
        As in :func:`run_recursive_oos`; ``n_jobs`` parallelizes across seeds.

    Returns
    -------
    (per_seed, averaged) : tuple of pandas.DataFrame
        One row per ``(seed, P, z)``, and the across-seed mean per ``(P, z)``.
    """
    R = np.asarray(target, dtype=np.float64).ravel()
    G = np.asarray(predictors, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError("predictors must be 2-D (n, d)")
    if G.shape[0] != R.shape[0]:
        raise ValueError("target and predictors must have the same number of rows")
    if save_forecasts and data_dir is None:
        raise ValueError("save_forecasts=True requires data_dir")
    seeds = list(seeds)

    def _one_seed(seed):
        return run_recursive_oos(
            G, R, seed, T=T, p_grid=p_grid, z_grid=z_grid, gamma=gamma,
            include_ridgeless=include_ridgeless, return_forecasts=save_forecasts,
        )

    if n_jobs == 1:
        seed_out = [_one_seed(s) for s in seeds]
    else:
        from joblib import Parallel, delayed

        seed_out = Parallel(n_jobs=n_jobs)(delayed(_one_seed)(s) for s in seeds)

    if save_forecasts:
        seed_stats = [out[0] for out in seed_out]
        _write_forecast_series(
            list(zip(seeds, [out[1] for out in seed_out])),
            dates, T, study_name, data_dir,
        )
    else:
        seed_stats = seed_out

    return _aggregate(seed_stats, T)


def run_grid(
    dataset, target_col="mkt_excess", predictor_cols=None, T=12,
    p_grid=P_GRID_DEFAULT, z_grid=Z_GRID_DEFAULT, seeds=range(50),
    gamma=GAMMA_DEFAULT, include_ridgeless=True, n_jobs=1,
):
    """DataFrame convenience wrapper over :func:`run_voc_study` (market schema).

    Splits a standardized dataset into its ``target_col`` and predictor columns
    (everything except ``"date"`` and the target, so the RFF dimension follows the
    frame's width) and runs the study.

    Returns
    -------
    (per_seed, averaged) : tuple of pandas.DataFrame
    """
    if predictor_cols is None:
        predictor_cols = [c for c in dataset.columns if c not in ("date", target_col)]
    dates = dataset["date"].to_numpy() if "date" in dataset.columns else None
    return run_voc_study(
        dataset[target_col], dataset[predictor_cols], dates=dates, T=T,
        p_grid=p_grid, z_grid=z_grid, seeds=seeds, gamma=gamma,
        include_ridgeless=include_ridgeless, n_jobs=n_jobs,
    )
