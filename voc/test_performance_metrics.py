"""Tests for the performance-metrics module (issue #7).

Hand-computed on tiny arrays so the formulas — OOS R^2 (with the paper's
zero-forecast benchmark), Sharpe, and the alpha/IR/t-stat OLS — are pinned
exactly. The OLS point estimates are cross-checked against ``numpy.polyfit``, an
independent least-squares route.
"""

import numpy as np

from voc.performance_metrics import (
    alpha_ir_tstat,
    compute_metrics,
    oos_r2,
    sharpe_ratio,
)


def test_zero_forecast_scores_r2_zero():
    """The zero forecast is the benchmark, so its R^2 is exactly 0 (footnote 40)."""
    realized = np.array([0.1, -0.2, 0.05, -0.03, 0.08])
    assert abs(oos_r2(np.zeros_like(realized), realized)) < 1e-15


def test_perfect_forecast_scores_r2_one():
    realized = np.array([0.1, -0.2, 0.05, -0.03, 0.08])
    assert abs(oos_r2(realized, realized) - 1.0) < 1e-15


def test_oos_r2_matches_hand_value():
    """errors of +/-0.5 on a series of variance 1 give R^2 = 1 - 0.25 = 0.75."""
    realized = np.array([1.0, -1.0, 1.0, -1.0])
    forecasts = np.array([0.5, -0.5, 0.5, -0.5])
    assert np.isclose(oos_r2(forecasts, realized), 0.75)


def test_sharpe_matches_hand_value():
    """[1, 3] has mean 2 and (population) SD 1, so annualized Sharpe = sqrt(12)*2."""
    assert np.isclose(sharpe_ratio(np.array([1.0, 3.0])), np.sqrt(12.0) * 2.0)


def test_alpha_ir_tstat_against_polyfit():
    """alpha matches an independent OLS (polyfit); IR and the t-stat match a
    recomputation from the polyfit residuals."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50)
    y = 0.3 + 1.5 * x + 0.2 * rng.standard_normal(50)
    alpha, information_ratio, alpha_tstat = alpha_ir_tstat(y, x)

    slope_pf, intercept_pf = np.polyfit(x, y, 1)
    resid = y - (intercept_pf + slope_pf * x)
    n, xbar, sxx = y.size, x.mean(), np.sum((x - x.mean()) ** 2)
    expected_ir = np.sqrt(12.0) * intercept_pf / resid.std()
    expected_se = np.sqrt(np.sum(resid ** 2) / (n - 2) * (1.0 / n + xbar ** 2 / sxx))

    assert np.isclose(alpha, intercept_pf)
    assert np.isclose(information_ratio, expected_ir)
    assert np.isclose(alpha_tstat, intercept_pf / expected_se)


def test_compute_metrics_keys_and_consistency():
    """compute_metrics returns every statistic, finite, and consistent with the
    standalone functions."""
    rng = np.random.default_rng(1)
    forecasts = 0.1 * rng.standard_normal(50)
    realized = 0.1 * rng.standard_normal(50)
    beta_norms = rng.random(50) + 0.5

    m = compute_metrics(forecasts, realized, beta_norms)
    for key in (
        "r2", "beta_norm", "mean_return", "volatility", "sharpe",
        "alpha", "information_ratio", "alpha_tstat",
    ):
        assert key in m and np.isfinite(m[key])
    assert np.isclose(m["r2"], oos_r2(forecasts, realized))
    assert np.isclose(m["sharpe"], sharpe_ratio(forecasts * realized))
    assert np.isclose(m["beta_norm"], beta_norms.mean())
