"""Nagel (2025) critique toolkit for the VoC engine - reusable across studies.

Nagel, "Seemingly Virtuous Complexity in Return Prediction" (SSRN 5335012),
argues that at P >> T a ridgeless RFF forecast is a similarity-weighted average of
the T training-window returns; because persistent predictors make similarity ~
recency and kernel distances scale with predictor volatility, the strategy is
mechanically recency-weighted, volatility-timed momentum. These functions run his
three tests on ANY study's standardized returns and anchor forecast export, so the
market study and the asset-class studies (#15/#16) share one implementation.

- :func:`momentum_benchmark` - his hand-built mimic: a linearly declining-weighted
  average of the trailing T standardized returns, and its timing-strategy return.
  (The returns are already volatility-standardized, so the vol-timing Nagel
  describes is baked into the inputs.)
- :func:`forecast_anatomy` - regress a study's VoC forecasts on the trailing T
  returns; the lag weights are the shape to compare with the benchmark's.
- :func:`spanning_regression` - regress the VoC strategy return on the benchmark
  return (plus the static market); the alpha Nagel expects to vanish.
- :func:`benchmark_metrics` - score any (forecast, realized) with the same metric
  definitions as the VoC strategy.

All three tests use only lagged returns, so nothing here peeks at the future.
"""

from __future__ import annotations

import numpy as np

from voc.performance_metrics import alpha_ir_tstat, oos_r2, sharpe_ratio


def declining_weights(T=12):
    """Linearly declining weights on the trailing T returns, summing to 1.

    ``w_k`` is proportional to ``T - k`` for k = 0 (most recent) .. T-1 (oldest),
    so the most recent training return gets the most weight - the recency Nagel's
    kernel argument implies.
    """
    w = np.arange(T, 0, -1, dtype=np.float64)
    return w / w.sum()


def momentum_benchmark(returns, T=12, weights=None):
    """Nagel's momentum mimic on standardized returns.

    Forecasts ``R[t+1]`` as a declining-weighted average of the trailing T returns
    ``R[t], R[t-1], ..., R[t-T+1]`` (the same T training-window returns the engine
    fits), over the SAME out-of-sample months as the engine (t = T .. n-2).

    Returns
    -------
    dict with ``oos_index`` (the decision points t), ``forecast`` (n_oos,),
    ``realized`` (``R[t+1]``), and ``strategy`` (``forecast * R[t+1]``).
    """
    R = np.asarray(returns, dtype=np.float64).ravel()
    n = R.shape[0]
    if n < T + 2:
        raise ValueError("need at least T + 2 observations")
    w = declining_weights(T) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (T,):
        raise ValueError(f"weights must have shape ({T},)")
    ts = np.arange(T, n - 1)
    # Columns are R[t], R[t-1], ..., R[t-T+1] for each decision point t.
    windows = R[ts[:, None] - np.arange(T)[None, :]]
    forecast = windows @ w
    realized = R[ts + 1]
    return {
        "oos_index": ts,
        "forecast": forecast,
        "realized": realized,
        "strategy": forecast * realized,
    }


def forecast_anatomy(voc_forecast, returns, oos_index, T=12):
    """Regress a study's VoC forecasts on the trailing T standardized returns.

    ``voc_forecast[i]`` is the forecast at decision point ``oos_index[i]`` (which
    predicts ``R[oos_index[i]+1]``); the regressors are ``R[t], ..., R[t-T+1]`` plus
    an intercept. Returns the estimated ``lag_weights`` (T,), the ``intercept``, and
    the regression ``r2`` - the shape and tightness Nagel predicts. Uses only
    lagged returns, so no lookahead.
    """
    R = np.asarray(returns, dtype=np.float64).ravel()
    ts = np.asarray(oos_index)
    y = np.asarray(voc_forecast, dtype=np.float64)
    X = R[ts[:, None] - np.arange(T)[None, :]]
    design = np.column_stack([np.ones(ts.shape[0]), X])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    r2 = 1.0 - resid.var() / y.var()
    return {"intercept": float(coef[0]), "lag_weights": coef[1:], "r2": float(r2)}


def spanning_regression(voc_strategy, benchmark_strategy, market):
    """Regress the VoC strategy return on the benchmark return + the static market.

    Returns ``alpha`` (intercept), the ``beta_benchmark`` and ``beta_market``
    loadings, the ``alpha_tstat`` from the OLS standard error, and the residual SD.
    Nagel finds the benchmark drives the VoC ``alpha`` to about zero.
    """
    y = np.asarray(voc_strategy, dtype=np.float64)
    design = np.column_stack(
        [
            np.ones(y.shape[0]),
            np.asarray(benchmark_strategy, dtype=np.float64),
            np.asarray(market, dtype=np.float64),
        ]
    )
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    n, k = design.shape
    sigma2 = resid @ resid / (n - k)
    cov = sigma2 * np.linalg.inv(design.T @ design)
    return {
        "alpha": float(coef[0]),
        "beta_benchmark": float(coef[1]),
        "beta_market": float(coef[2]),
        "alpha_tstat": float(coef[0] / np.sqrt(cov[0, 0])),
        "resid_std": float(resid.std()),
    }


def benchmark_metrics(forecast, realized, periods_per_year=12):
    """Score any (forecast, realized) pair with the VoC strategy's own metrics.

    Same definitions as :func:`voc.performance_metrics.compute_metrics`, minus the
    coefficient norm (a benchmark has no ridge coefficients), so the benchmark and
    the VoC strategy sit in one comparison table.
    """
    forecast = np.asarray(forecast, dtype=np.float64)
    realized = np.asarray(realized, dtype=np.float64)
    strategy = forecast * realized
    alpha, information_ratio, alpha_tstat = alpha_ir_tstat(
        strategy, realized, periods_per_year
    )
    return {
        "r2": float(oos_r2(forecast, realized)),
        "mean_return": float(strategy.mean()),
        "volatility": float(strategy.std()),
        "sharpe": float(sharpe_ratio(strategy, periods_per_year)),
        "alpha": float(alpha),
        "information_ratio": float(information_ratio),
        "alpha_tstat": float(alpha_tstat),
    }
