"""Reusable diagnostics and counterfactuals for Nagel's RFF critique.

This import-light module implements Nagel (2025), Sections II.F--II.G:
equation (14)'s volatility-timed momentum rule, equation (15)'s MA(2)
reversal, equations (16)--(17)'s predictor wild bootstrap, and the matched
``+/-`` twin-market extension used by this project.

Nagel's printed equation (14) has an apparent indexing typo: its newest term
is labelled ``r[t+1]`` although the derivation defines a weighted average of
realized training returns.  Here the newest term is ``r[t]``.  All functions
follow the project convention that row ``X[t]`` forecasts ``R[t+1]``.
"""

from __future__ import annotations

import numpy as np

from voc.performance_metrics import alpha_ir_tstat, oos_r2, sharpe_ratio


def declining_weights(T=12):
    """Linearly declining weights on the trailing ``T`` returns, summing to 1."""
    if T != int(T) or int(T) < 1:
        raise ValueError(f"T must be a positive integer; got {T!r}")
    weights = np.arange(int(T), 0, -1, dtype=np.float64)
    return weights / weights.sum()


def _aligned_oos_inputs(returns, T):
    """Return validated target, decision indices, and trailing-return windows."""
    values = np.asarray(returns, dtype=np.float64).ravel()
    if not np.isfinite(values).all():
        raise ValueError("returns must be finite")
    if T != int(T) or int(T) < 1:
        raise ValueError(f"T must be a positive integer; got {T!r}")
    T = int(T)
    if values.size < T + 2:
        raise ValueError("need at least T + 2 observations")
    decision_points = np.arange(T, values.size - 1)
    # Columns: R[t], R[t-1], ..., R[t-T+1].  R[t+1] never enters a forecast.
    windows = values[decision_points[:, None] - np.arange(T, dtype=int)[None, :]]
    return values, decision_points, windows


def momentum_benchmark(returns, T=12, weights=None):
    """Recency-only momentum diagnostic (not the full Nagel equation (14))."""
    values, decision_points, windows = _aligned_oos_inputs(returns, T)
    weights = (
        declining_weights(T)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if weights.shape != (int(T),):
        raise ValueError(f"weights must have shape ({int(T)},)")
    forecast = windows @ weights
    realized = values[decision_points + 1]
    return {
        "oos_index": decision_points,
        "forecast": forecast,
        "realized": realized,
        "strategy": forecast * realized,
    }


def volatility_timed_momentum(
    returns,
    predictors,
    T=12,
    weights=None,
    position_scale=0.05,
    variance_floor=1e-12,
):
    """Nagel equation (14): declining momentum / mean predictor variance.

    At decision ``t`` the variance is the mean, across predictor columns, of
    sample variances in ``X[t-T:t]`` (ending at ``t-1``).  The momentum window
    uses returns ``R[t]`` through ``R[t-T+1]`` and therefore remains feasible.
    """
    if T != int(T) or int(T) < 2:
        raise ValueError("T must be an integer of at least two for sample variance")
    values, decision_points, windows = _aligned_oos_inputs(returns, T)
    predictors = np.asarray(predictors, dtype=np.float64)
    if predictors.ndim != 2 or predictors.shape[0] != values.size:
        raise ValueError(
            f"predictors must be (n, K) with n={values.size}; got {predictors.shape}"
        )
    if not np.isfinite(predictors).all():
        raise ValueError("predictors must be finite")
    if not np.isfinite(position_scale):
        raise ValueError("position_scale must be finite")
    if not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be finite and positive")
    weights = (
        declining_weights(T)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if weights.shape != (int(T),):
        raise ValueError(f"weights must have shape ({int(T)},)")

    predictor_variance = np.array(
        [
            predictors[t - int(T) : t].var(axis=0, ddof=1).mean()
            for t in decision_points
        ],
        dtype=np.float64,
    )
    predictor_variance = np.maximum(predictor_variance, variance_floor)
    forecast = position_scale * (windows @ weights) / predictor_variance
    realized = values[decision_points + 1]
    return {
        "oos_index": decision_points,
        "forecast": forecast,
        "realized": realized,
        "strategy": forecast * realized,
        "predictor_variance": predictor_variance,
    }


def add_ma2_reversal(
    returns,
    *,
    seed,
    theta=(0.2, 0.2),
    innovation_variance=0.01,
    return_innovations=False,
):
    """Add Nagel equation (15)'s MA(2) reversal to raw returns.

    ``xi[t] ~ N(0, innovation_variance)`` and ``R_tilde[t]`` equals ``R[t]``
    plus ``xi[t] - theta[0]*xi[t-1] - theta[1]*xi[t-2]``.  Two pre-sample
    innovations make the first retained observation follow the same rule.
    """
    values = np.asarray(returns, dtype=np.float64).ravel()
    theta = np.asarray(theta, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("returns must be finite")
    if theta.shape != (2,) or not np.isfinite(theta).all():
        raise ValueError("theta must contain two finite coefficients")
    if not np.isfinite(innovation_variance) or innovation_variance < 0:
        raise ValueError("innovation_variance must be finite and nonnegative")
    innovations = np.random.default_rng(seed).normal(
        scale=np.sqrt(innovation_variance), size=values.size + 2
    )
    component = (
        innovations[2:] - theta[0] * innovations[1:-1] - theta[1] * innovations[:-2]
    )
    transformed = values + component
    if return_innovations:
        return transformed, {
            "innovations": innovations,
            "component": component,
            "theta": theta.copy(),
        }
    return transformed


def fit_ar1_kendall(series, *, bias_adjust=True, stationarity_bound=None):
    """Fit an intercept AR(1) with an explicit first-order Kendall correction.

    Nagel cites but does not spell out the finite-sample correction.  We use
    the common direct plug-in convention
    ``phi_bc = phi_ols + (1 + 3*phi_ols)/(n-1)`` and reset the intercept to
    ``(1-phi_bc)*mean(x)``. Nagel does not specify coefficient clipping, so
    the default leaves the corrected coefficient unchanged. An explicit
    ``stationarity_bound`` can be supplied for non-replication applications.
    """
    values = np.asarray(series, dtype=np.float64).ravel()
    if values.size < 5 or not np.isfinite(values).all():
        raise ValueError("series must contain at least five finite observations")
    design = np.column_stack([np.ones(values.size - 1), values[:-1]])
    intercept_ols, phi_ols = np.linalg.lstsq(design, values[1:], rcond=None)[0]
    phi = float(phi_ols)
    if bias_adjust:
        phi += (1.0 + 3.0 * phi) / (values.size - 1)
    if stationarity_bound is not None:
        if not 0 < stationarity_bound <= 1:
            raise ValueError("stationarity_bound must lie in (0, 1] or be None")
        phi = float(np.clip(phi, -stationarity_bound, stationarity_bound))
    intercept = float((1.0 - phi) * values.mean())
    residuals = values[1:] - intercept - phi * values[:-1]
    return {
        "intercept": intercept,
        "phi": phi,
        "intercept_ols": float(intercept_ols),
        "phi_ols": float(phi_ols),
        "residuals": residuals,
    }


def wild_bootstrap_predictors(
    predictors,
    *,
    seed,
    bias_adjust=True,
    stationarity_bound=None,
    return_diagnostics=False,
):
    """Generate Nagel equations (16)--(17)'s wild-bootstrap predictors.

    Each predictor gets a full-sample AR(1).  Independent Rademacher signs
    flip its fitted innovations, exactly retaining every innovation magnitude
    and hence its time-varying volatility.  The original first row initializes
    each recursive artificial series.
    """
    predictors = np.asarray(predictors, dtype=np.float64)
    if predictors.ndim != 2 or predictors.shape[0] < 5 or predictors.shape[1] < 1:
        raise ValueError("predictors must have shape (n >= 5, K >= 1)")
    if not np.isfinite(predictors).all():
        raise ValueError("predictors must be finite")
    rng = np.random.default_rng(seed)
    bootstrapped = np.empty_like(predictors)
    bootstrapped[0] = predictors[0]
    diagnostics = []
    for column in range(predictors.shape[1]):
        fit = fit_ar1_kendall(
            predictors[:, column],
            bias_adjust=bias_adjust,
            stationarity_bound=stationarity_bound,
        )
        signs = rng.choice(np.array([-1.0, 1.0]), size=predictors.shape[0] - 1)
        bootstrap_innovations = fit["residuals"] * signs
        for row in range(1, predictors.shape[0]):
            bootstrapped[row, column] = (
                fit["intercept"]
                + fit["phi"] * bootstrapped[row - 1, column]
                + bootstrap_innovations[row - 1]
            )
        diagnostics.append(
            {
                **fit,
                "signs": signs,
                "bootstrap_innovations": bootstrap_innovations,
            }
        )
    if return_diagnostics:
        return bootstrapped, diagnostics
    return bootstrapped


def nonlinear_signal(predictors, predictor_names, *, calibration_size=120):
    """Fixed nonlinear rule: curved valuation plus bond-credit interaction.

    The bounded rule ``.6*cos(dp) + .4*sin(ltr)*sin(dfr)`` is normalized on
    the first ``calibration_size`` rows only.  Extending the end of the sample
    therefore never revises the already-planted historical signal.
    """
    predictors = np.asarray(predictors, dtype=np.float64)
    names = list(predictor_names)
    if predictors.ndim != 2 or predictors.shape[1] != len(names):
        raise ValueError("predictor_names must name every predictor column")
    if not np.isfinite(predictors).all():
        raise ValueError("predictors must be finite")
    missing = {"dp", "ltr", "dfr"}.difference(names)
    if missing:
        raise ValueError(f"nonlinear signal is missing named inputs: {sorted(missing)}")
    dp, ltr, dfr = (predictors[:, names.index(name)] for name in ("dp", "ltr", "dfr"))
    raw = 0.6 * np.cos(dp) + 0.4 * np.sin(ltr) * np.sin(dfr)
    calibration_rows = min(int(calibration_size), raw.size)
    if calibration_rows < 2:
        raise ValueError("calibration_size must leave at least two rows")
    center = raw[:calibration_rows].mean()
    scale = raw[:calibration_rows].std(ddof=1)
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("nonlinear signal is degenerate on its calibration block")
    return (raw - center) / scale


def make_twin_markets(
    predictors,
    predictor_names,
    *,
    signal_r2,
    shock_seed,
    calibration_size=120,
):
    """Construct matched ``+/-`` markets sharing exactly one shock path.

    For ``t < n-1``, ``R_plus[t+1] = eps[t+1] + a*g(X[t])`` and the minus
    twin changes only the sign of the signal.  With unit-variance shocks and
    calibration-period unit-variance ``g``, ``a=sqrt(R2/(1-R2))`` gives the
    requested oracle signal-to-noise R-squared.
    """
    if not np.isfinite(signal_r2) or not 0 < signal_r2 < 1:
        raise ValueError("signal_r2 must lie strictly between zero and one")
    predictors = np.asarray(predictors, dtype=np.float64)
    signal = nonlinear_signal(
        predictors, predictor_names, calibration_size=calibration_size
    )
    # Keep the seeded Gaussian path unmodified. Full-sample normalization
    # would leak future shocks into earlier simulated returns and make the
    # historical prefix change whenever SAMPLE_END is extended.
    shocks = np.random.default_rng(shock_seed).standard_normal(predictors.shape[0])
    amplitude = float(np.sqrt(signal_r2 / (1.0 - signal_r2)))
    planted = np.zeros_like(shocks)
    planted[1:] = amplitude * signal[:-1]
    return {
        "plus": shocks + planted,
        "minus": shocks - planted,
        "epsilon": shocks,
        "g": signal,
        "truth": amplitude * signal,
        "amplitude": amplitude,
        "signal_r2": float(signal_r2),
    }


def recovery_metrics(forecast_plus, forecast_minus, truth):
    """Score ``(forecast_plus - forecast_minus)/2`` against planted truth."""
    forecast_plus = np.asarray(forecast_plus, dtype=np.float64).ravel()
    forecast_minus = np.asarray(forecast_minus, dtype=np.float64).ravel()
    truth = np.asarray(truth, dtype=np.float64).ravel()
    if not (forecast_plus.shape == forecast_minus.shape == truth.shape):
        raise ValueError("paired forecasts and truth must have the same shape")
    if (
        truth.size < 3
        or not np.isfinite(np.concatenate([forecast_plus, forecast_minus, truth])).all()
    ):
        raise ValueError("paired forecasts and truth need at least three finite rows")
    recovered = 0.5 * (forecast_plus - forecast_minus)
    intercept, slope = np.linalg.lstsq(
        np.column_stack([np.ones(truth.size), truth]), recovered, rcond=None
    )[0]
    error = recovered - truth
    rmse = float(np.sqrt(np.mean(error**2)))
    truth_sd = float(truth.std())
    if truth_sd <= 1e-12:
        raise ValueError("truth must have nonzero variance")
    return {
        "correlation": float(np.corrcoef(recovered, truth)[0, 1]),
        "intercept": float(intercept),
        "slope": float(slope),
        "rmse": rmse,
        "nrmse": rmse / truth_sd,
        "bias": float(error.mean()),
        "recovered": recovered,
    }


def forecast_anatomy(voc_forecast, returns, oos_index, T=12):
    """Project forecasts on trailing returns (a proxy, not interpolation weights)."""
    returns = np.asarray(returns, dtype=np.float64).ravel()
    decision_points = np.asarray(oos_index)
    forecast = np.asarray(voc_forecast, dtype=np.float64)
    trailing = returns[decision_points[:, None] - np.arange(T, dtype=int)[None, :]]
    design = np.column_stack([np.ones(decision_points.size), trailing])
    coefficients, *_ = np.linalg.lstsq(design, forecast, rcond=None)
    residual = forecast - design @ coefficients
    variance = forecast.var()
    r2 = np.nan if variance == 0 else 1.0 - residual.var() / variance
    return {
        "intercept": float(coefficients[0]),
        "lag_weights": coefficients[1:],
        "r2": float(r2),
    }


def spanning_regression(voc_strategy, benchmark_strategy, market):
    """Regress a payoff on intercept, Nagel payoff, and its own market."""
    strategy = np.asarray(voc_strategy, dtype=np.float64)
    benchmark = np.asarray(benchmark_strategy, dtype=np.float64)
    market = np.asarray(market, dtype=np.float64)
    if (
        strategy.ndim != 1
        or benchmark.shape != strategy.shape
        or market.shape != strategy.shape
    ):
        raise ValueError("all spanning series must be one-dimensional and aligned")
    design = np.column_stack([np.ones(strategy.size), benchmark, market])
    coefficients, *_ = np.linalg.lstsq(design, strategy, rcond=None)
    residual = strategy - design @ coefficients
    n_obs, n_coefficients = design.shape
    sigma2 = residual @ residual / (n_obs - n_coefficients)
    covariance = sigma2 * np.linalg.pinv(design.T @ design)
    alpha = float(coefficients[0])
    residual_std = float(residual.std())
    return {
        "alpha": alpha,
        "beta_benchmark": float(coefficients[1]),
        "beta_market": float(coefficients[2]),
        "alpha_tstat": float(alpha / np.sqrt(covariance[0, 0])),
        "information_ratio": float(np.sqrt(12.0) * alpha / residual_std),
        "resid_std": residual_std,
    }


def benchmark_metrics(forecast, realized, periods_per_year=12):
    """Score a forecast/payoff with the VoC strategy's metric definitions."""
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
