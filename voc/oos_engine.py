"""Recursive out-of-sample VoC engine for KMZ (2024), Section V.C.

One repetition (seed) draws ``P_max`` random Fourier features once, then runs the
rolling ``T``-month recursion: at each decision point it trains dual-form ridge on
the ``T`` most recent pairs, forecasts the next return, and books the timing-
strategy return. The nested complexity grid reuses the single RFF draw (slice the
first ``P`` columns) and one Gram eigendecomposition per ``(window, P)`` across all
shrinkage levels ``z`` (see :mod:`voc.rff` and :mod:`voc.kernel_ridge`).

:func:`run_voc_study` is the generic entry point (issue #14): it runs many
seeds — independent and parallelizable — on ANY already-standardized
``(target, predictors)`` pair and returns a long-format table of
per-``(seed, P, z)`` statistics plus the across-seed means: the paper's
aggregate-the-statistics-not-the-forecasts convention (2025 reply, Section
4.2.1). :func:`run_grid` is its DataFrame convenience wrapper in the market
schema. The engine is pure (no data-pipeline imports; the only file IO is the
optional ``save_forecasts`` export); a driver loads standardized data, calls
the entry point, and caches the result.

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

RFF standardization uses the CENTERED training-window convention by default
(``uncentered=False``), pinned by the issue #9 anchor investigation: centered
scaling plus the workbook market series reproduces the paper's Figure 8
ridgeless anchors within ~1% (see :func:`voc.rff.standardize_by_training_window`).
The kwarg on :func:`run_recursive_oos` / :func:`run_grid` is retained for A/B
checks against the uncentered variant.
"""

from __future__ import annotations

from pathlib import Path

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
    G,
    R,
    seed,
    T=12,
    p_grid=P_GRID_DEFAULT,
    z_grid=Z_GRID_DEFAULT,
    gamma=GAMMA_DEFAULT,
    include_ridgeless=True,
    uncentered=False,
    return_forecasts=False,
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
    uncentered : bool
        RFF training-window standardization convention (see
        :func:`voc.rff.standardize_by_training_window`). The pipeline pins the
        CENTERED default (uncentered=False), validated against the paper's
        Figure 8 anchors in issue #9; the kwarg is retained for A/B checks.
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
        train_std, oos_std = standardize_by_training_window(
            S[t - T : t], S[t], uncentered=uncentered
        )
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


def _write_forecast_series(seed_bundles, dates, T, study_name, data_dir):
    """Persist per-seed forecast / realized / strategy series for each (P, z).

    ``seed_bundles`` iterates ``(seed, bundle)`` pairs from
    :func:`run_recursive_oos` with ``return_forecasts=True``. The realized
    return at OOS step i is ``R[T + 1 + i]``, so its label is
    ``dates[T + 1 + i]`` when ``dates`` is given, else the integer position.
    """
    frames = []
    for seed, bundle in seed_bundles:
        realized = bundle["realized"]
        n_oos = realized.shape[0]
        if dates is not None:
            obs = np.asarray(dates)[T + 1 : T + 1 + n_oos]
        else:
            obs = np.arange(n_oos)
        for (P, z), forecast in bundle["forecasts"].items():
            frames.append(
                pd.DataFrame(
                    {
                        "seed": int(seed),
                        "P": int(P),
                        "z": float(z),
                        "obs": obs,
                        "forecast": forecast,
                        "realized": realized,
                        "strategy": forecast * realized,
                    }
                )
            )
    path = Path(data_dir) / f"forecasts_{study_name}.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(path)
    return path


def run_voc_study(
    target,
    predictors,
    *,
    dates=None,
    T=12,
    p_grid=P_GRID_DEFAULT,
    z_grid=Z_GRID_DEFAULT,
    seeds=range(50),
    gamma=GAMMA_DEFAULT,
    include_ridgeless=True,
    uncentered=False,
    n_jobs=1,
    save_forecasts=False,
    study_name="market",
    data_dir=None,
):
    """Run a virtue-of-complexity study on any (target, predictors) pair.

    The reusable entry point (issue #14): the KMZ market replication is one
    call, and asset-class studies (bonds, international) or experiments (the
    Nagel counterfactual) are other calls with different inputs. Inputs are
    assumed ALREADY volatility-standardized, which is the caller's job (see
    :mod:`voc.preprocessing`) — the engine never standardizes your data for
    you, so no lookahead can be smuggled in on your behalf. (The RFF
    training-window standardization is a separate, internal step; see
    ``uncentered``.)

    Parameters
    ----------
    target : array-like, shape (n,)
        Standardized target returns; row t is month t. Must be finite.
    predictors : array-like or DataFrame, shape (n, d)
        Standardized predictors; the RFF dimension follows the width d.
        Must be finite — drop the standardization burn-in rows first.
    dates : array-like, shape (n,), optional
        Row labels, used only for the forecast export's ``obs`` column.
    seeds : iterable of int
        RFF repetition seeds; must be non-empty. Duplicates are dropped.
    uncentered : bool
        RFF training-window standardization convention. The pipeline pins the
        CENTERED default (False), validated against the paper's Figure 8
        anchors in issue #9 (see :func:`run_recursive_oos`); the kwarg is
        retained for A/B checks.
    n_jobs : int
        joblib parallelism across seeds (``-1`` = all cores). Seeds are
        independent, so results are identical for any ``n_jobs``.
    save_forecasts : bool
        If True, also write the per-seed forecast / realized / strategy
        series of every ``(P, z)`` cell to
        ``<data_dir>/forecasts_<study_name>.parquet`` (requires ``data_dir``).
        The file grows as seeds x cells x months, so keep the grid small when
        enabling this — e.g. just the anchor configuration.
    T, p_grid, z_grid, gamma, include_ridgeless :
        As in :func:`run_recursive_oos`.

    Returns
    -------
    (per_seed, averaged) : tuple of pandas.DataFrame
        ``per_seed`` has one row per ``(seed, P, z)``; ``averaged`` holds the
        across-seed mean per ``(P, z)`` of every statistic column in
        ``per_seed``, so statistics added to ``compute_metrics`` flow through.
    """
    R = np.asarray(target, dtype=np.float64).ravel()
    G = np.asarray(predictors, dtype=np.float64)
    if G.ndim != 2 or G.shape[0] != R.shape[0]:
        raise ValueError(
            "predictors must be (n, d) with one row per target row; got "
            f"predictors {G.shape}, target {R.shape}"
        )
    if not (np.isfinite(G).all() and np.isfinite(R).all()):
        raise ValueError(
            "target or predictors have NaN/inf; standardize and drop the "
            "burn-in rows upstream (see voc.preprocessing.standardize_inputs)"
        )
    if save_forecasts and data_dir is None:
        raise ValueError("save_forecasts=True requires data_dir")
    if dates is not None and len(np.asarray(dates)) != R.shape[0]:
        raise ValueError(
            f"dates must have one label per data row; got "
            f"{len(np.asarray(dates))} labels for {R.shape[0]} rows"
        )
    seeds = list(dict.fromkeys(int(s) for s in seeds))  # dedupe, keep order
    if not seeds:
        raise ValueError("seeds must be a non-empty iterable of ints")

    def _one_seed(seed):
        return run_recursive_oos(
            G,
            R,
            seed,
            T=T,
            p_grid=p_grid,
            z_grid=z_grid,
            gamma=gamma,
            include_ridgeless=include_ridgeless,
            uncentered=uncentered,
            return_forecasts=save_forecasts,
        )

    # joblib runs n_jobs=1 sequentially in-process, so one dispatch path serves
    # every setting (and the tested path IS the production path).
    seed_out = Parallel(n_jobs=n_jobs)(delayed(_one_seed)(s) for s in seeds)

    if save_forecasts:
        seed_stats = [stats for stats, _ in seed_out]
        _write_forecast_series(
            zip(seeds, (bundle for _, bundle in seed_out)),
            dates,
            T,
            study_name,
            data_dir,
        )
    else:
        seed_stats = seed_out

    per_seed = pd.DataFrame([row for rows in seed_stats for row in rows])
    # c is constant within each (P, z) group, so it averages through unchanged;
    # no separate statistic-column list to keep in sync with compute_metrics.
    averaged = (
        per_seed.drop(columns="seed")
        .groupby(["P", "z"], sort=False)
        .mean()
        .reset_index()
    )
    return per_seed, averaged


def run_grid(
    dataset,
    target_col="mkt_excess",
    predictor_cols=None,
    T=12,
    p_grid=P_GRID_DEFAULT,
    z_grid=Z_GRID_DEFAULT,
    seeds=range(50),
    gamma=GAMMA_DEFAULT,
    include_ridgeless=True,
    uncentered=False,
    n_jobs=1,
):
    """DataFrame convenience wrapper over :func:`run_voc_study` (market schema).

    Splits a standardized dataset into ``target_col`` and predictor columns
    (default: every column except ``"date"`` and the target, so the RFF
    dimension follows the frame's width) and runs the study. Kept with this
    exact signature so the market and variable-importance drivers and the
    engine tests are wrapper-agnostic.
    """
    if predictor_cols is None:
        predictor_cols = [c for c in dataset.columns if c not in ("date", target_col)]
    dates = dataset["date"].to_numpy() if "date" in dataset.columns else None
    return run_voc_study(
        dataset[target_col],
        dataset[predictor_cols],
        dates=dates,
        T=T,
        p_grid=p_grid,
        z_grid=z_grid,
        seeds=seeds,
        gamma=gamma,
        include_ridgeless=include_ridgeless,
        uncentered=uncentered,
        n_jobs=n_jobs,
    )
