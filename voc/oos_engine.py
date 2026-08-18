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

The decision points are ``t = T, ..., n-2`` — ``n - T - 1`` OOS months, one
FEWER than the paper's literal ``t in {T, ..., 1091}`` (Section V.C step iii,
``1092 - T`` months): the paper's final decision books ``R[1092]``, which an
n = 1092 row sample (1930-01 to 2020-12) cannot supply. The paper attains the
extra month only because its raw data begin before this project's 1930-01
sample floor. Do not "restore" it by starting at ``t = T - 1``: feature row
``t - T`` would wrap to index -1 and train the first window on the LAST rows.

The ridgeless (``z -> 0`` minimum-norm) column is recorded as ``z = 0.0``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from voc.kernel_ridge import predict, ridge_dual, ridgeless
from voc.performance_metrics import compute_metrics
from voc.rff import (
    GAMMA_DEFAULT,
    compute_rff,
    draw_rff_weights,
    standardize_by_training_window,
)

Z_GRID_DEFAULT = tuple(10.0**k for k in range(-3, 4))  # 1e-3 .. 1e3
# Complexity grid P = c * T; includes P=2, P=12 (c=1), and P=12000 (c=1000, the
# anchor). All even (sin/cos pairs).
P_GRID_DEFAULT = (2, 4, 8, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12000)


def run_recursive_oos(
    G, R, seed, T=12, p_grid=P_GRID_DEFAULT, z_grid=Z_GRID_DEFAULT,
    gamma=GAMMA_DEFAULT, include_ridgeless=True, return_forecasts=False,
):
    """One repetition (one RFF seed) of the recursive OOS analysis.

    Parameters
    ----------
    G : array-like, shape (n, d)
        Volatility-standardized predictors; row t is month t. Must be finite.
    R : array-like, shape (n,)
        Volatility-standardized market excess returns; row t is month t.
    seed : int
        RFF repetition index.
    T, p_grid, z_grid, gamma : see module defaults.
        ``T`` is a positive integer leaving at least 3 OOS months
        (``n >= T + 4``); ``p_grid`` holds positive even integers and ``z_grid``
        finite values ``> 0``. Duplicates in either grid are dropped.
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
    if G.ndim != 2 or R.shape != (G.shape[0],):
        raise ValueError(f"G must be (n, d) and R (n,); got G {G.shape}, R {R.shape}")
    if not (np.isfinite(G).all() and np.isfinite(R).all()):
        raise ValueError("G and R must be finite; the input has NaN/inf cells")
    if T != int(T) or int(T) < 1:
        raise ValueError(f"T must be a positive integer; got {T!r}")
    T = int(T)
    n_oos = G.shape[0] - T - 1
    if n_oos < 3:
        raise ValueError(
            "need at least 3 OOS months (n_rows >= T + 4) for the performance "
            f"statistics; n_rows={G.shape[0]} with T={T} gives n_oos={n_oos}"
        )
    if any(P != int(P) for P in p_grid):
        raise ValueError(f"p_grid must hold integers; got {tuple(p_grid)!r}")
    p_grid = tuple(sorted({int(P) for P in p_grid}))
    if not p_grid or p_grid[0] < 2 or any(P % 2 for P in p_grid):
        raise ValueError(
            "every P must be a positive even integer (RFFs come in sin/cos "
            f"pairs); got {p_grid!r}"
        )
    z_grid = tuple(dict.fromkeys(float(z) for z in z_grid))  # dedupe, keep order
    if any(not np.isfinite(z) or z <= 0.0 for z in z_grid):
        raise ValueError(
            f"z_grid must hold finite z > 0, got {z_grid!r}; "
            "use include_ridgeless for the z=0 minimum-norm limit"
        )
    z_labels = list(z_grid) + ([0.0] if include_ridgeless else [])
    if not z_labels:
        raise ValueError(
            "empty model grid: z_grid is empty and include_ridgeless is False"
        )

    omega = draw_rff_weights(p_grid[-1] // 2, G.shape[1], seed)
    S = compute_rff(G, omega, gamma=gamma)  # one max-size draw, sliced for nesting

    ts = np.arange(T, G.shape[0] - 1)  # decision points; each books R[t + 1]
    realized = R[ts + 1]
    forecasts = {(P, z): np.empty(n_oos) for P in p_grid for z in z_labels}
    beta_norms = {(P, z): np.empty(n_oos) for P in p_grid for z in z_labels}

    for i, t in enumerate(ts):
        # Standardize every feature column by its training-window volatility once,
        # then slice the first P for each model (columns nest). Training features
        # are rows t-T..t-1, training returns rows t-T+1..t (the single shift);
        # plain slices are views, matching the brute-force test reference.
        train_std, oos_std = standardize_by_training_window(S[t - T : t], S[t])
        R_train = R[t - T + 1 : t + 1]
        for P in p_grid:
            X = train_std[:, :P]
            s_oos = oos_std[:P]
            if z_grid:
                betas = ridge_dual(X, R_train, z_grid)  # one eigendecomp, all z
                forecasts_z = betas @ s_oos
                norms_z = np.linalg.norm(betas, axis=1)
                for j, z in enumerate(z_grid):
                    forecasts[(P, z)][i] = forecasts_z[j]
                    beta_norms[(P, z)][i] = norms_z[j]
            if include_ridgeless:
                beta_rl = ridgeless(X, R_train)
                forecasts[(P, 0.0)][i] = predict(s_oos, beta_rl)
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


def run_grid(
    dataset, target_col="mkt_excess", predictor_cols=None, T=12,
    p_grid=P_GRID_DEFAULT, z_grid=Z_GRID_DEFAULT, seeds=range(50),
    gamma=GAMMA_DEFAULT, include_ridgeless=True, n_jobs=1,
):
    """Run the recursive OOS grid across seeds; return long-format statistics.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Volatility-standardized data with ``target_col`` and ``predictor_cols``.
        If ``predictor_cols`` is None, uses every column except ``"date"`` and the
        target (the RFF dimension follows the predictor frame's width). Must be
        free of NaN/inf.
    seeds : iterable of int
        RFF repetition seeds; must be non-empty. Duplicates are dropped.
    n_jobs : int
        joblib parallelism across seeds (``-1`` = all cores). Seeds are
        independent, so results are identical for any ``n_jobs``.

    Returns
    -------
    (per_seed, averaged) : tuple of pandas.DataFrame
        ``per_seed`` has one row per ``(seed, P, z)``; ``averaged`` holds the
        across-seed mean per ``(P, z)`` of every statistic column in
        ``per_seed``, so statistics added to ``compute_metrics`` flow through.
    """
    if predictor_cols is None:
        predictor_cols = [c for c in dataset.columns if c not in ("date", target_col)]
    G = dataset[predictor_cols].to_numpy(dtype=np.float64)
    R = dataset[target_col].to_numpy(dtype=np.float64)
    if not (np.isfinite(G).all() and np.isfinite(R).all()):
        raise ValueError(
            "dataset has NaN/inf in the predictors or target; clean it upstream"
        )
    seeds = list(dict.fromkeys(int(s) for s in seeds))  # dedupe, keep order
    if not seeds:
        raise ValueError("seeds must be a non-empty iterable of ints")

    def _one_seed(seed):
        return run_recursive_oos(
            G, R, seed, T=T, p_grid=p_grid, z_grid=z_grid, gamma=gamma,
            include_ridgeless=include_ridgeless,
        )

    # joblib runs n_jobs=1 sequentially in-process, so one dispatch path serves
    # every setting (and the tested path IS the production path).
    seed_results = Parallel(n_jobs=n_jobs)(delayed(_one_seed)(s) for s in seeds)

    per_seed = pd.DataFrame([row for rows in seed_results for row in rows])
    # c is constant within each (P, z) group, so it averages through unchanged;
    # no separate statistic-column list to keep in sync with compute_metrics.
    averaged = (
        per_seed.drop(columns="seed")
        .groupby(["P", "z"], sort=False)
        .mean()
        .reset_index()
    )
    return per_seed, averaged
