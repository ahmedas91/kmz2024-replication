"""Tests for the Nagel critique toolkit (issue #17).

Synthetic; no data on disk. They pin the benchmark on a hand-computed series, the
spanning regression's alpha behaviour (near zero for a scaled copy of the
benchmark, positive for an orthogonal drift), and that the anatomy uses only
lagged returns.
"""

import numpy as np
import pytest

from voc.nagel import (
    add_ma2_reversal,
    benchmark_metrics,
    declining_weights,
    fit_ar1_kendall,
    forecast_anatomy,
    make_twin_markets,
    momentum_benchmark,
    recovery_metrics,
    spanning_regression,
    volatility_timed_momentum,
    wild_bootstrap_predictors,
)
from voc.performance_metrics import sharpe_ratio


def test_declining_weights_sum_to_one_and_decrease():
    w = declining_weights(12)
    assert w.shape == (12,)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(np.diff(w) < 0)  # strictly declining, most recent largest


def test_momentum_benchmark_hand_value():
    """T=3, weights (3,2,1)/6 on [R[t], R[t-1], R[t-2]] over the engine's OOS months."""
    R = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = momentum_benchmark(R, T=3)
    # decision points t = 3, 4 (n=6 -> arange(3, 5))
    np.testing.assert_array_equal(out["oos_index"], [3, 4])
    # forecast at t=3: (3*R[3] + 2*R[2] + 1*R[1]) / 6 = (12 + 6 + 2)/6
    assert np.isclose(out["forecast"][0], 20.0 / 6.0)
    np.testing.assert_array_equal(out["realized"], [5.0, 6.0])  # R[t+1]
    np.testing.assert_allclose(out["strategy"], out["forecast"] * out["realized"])


def test_volatility_timed_momentum_matches_equation_14_by_hand():
    """The missing inverse predictor-variance term is part of Nagel's rule."""
    returns = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # Every three-row training block has column variances 1 and 4 (ddof=1),
    # hence mean predictor variance 2.5.
    predictors = np.column_stack([np.arange(6.0), 2.0 * np.arange(6.0)])
    out = volatility_timed_momentum(returns, predictors, T=3)
    expected_momentum = (3 * 4 + 2 * 3 + 1 * 2) / 6
    assert np.isclose(out["predictor_variance"][0], 2.5)
    assert np.isclose(out["forecast"][0], 0.05 * expected_momentum / 2.5)


def test_volatility_timed_momentum_requires_two_rows_for_sample_variance():
    with pytest.raises(ValueError, match="at least two"):
        volatility_timed_momentum(np.arange(5.0), np.arange(5.0)[:, None], T=1)


def test_volatility_timed_momentum_has_no_lookahead():
    rng = np.random.default_rng(41)
    returns = rng.standard_normal(30)
    predictors = rng.standard_normal((30, 3))
    base = volatility_timed_momentum(returns, predictors, T=5)
    changed_returns = returns.copy()
    changed_predictors = predictors.copy()
    # Forecast at decision t=5 cannot use R[6:] or X[5:].
    changed_returns[6:] += 100.0
    changed_predictors[5:] += 100.0
    changed = volatility_timed_momentum(changed_returns, changed_predictors, T=5)
    assert np.isclose(base["forecast"][0], changed["forecast"][0])


def test_anatomy_recovers_a_pure_momentum_shape():
    """If the 'VoC' forecast IS declining-weighted momentum, the anatomy recovers
    those weights with R^2 ~ 1."""
    rng = np.random.default_rng(0)
    R = rng.standard_normal(200)
    bench = momentum_benchmark(R, T=12)
    a = forecast_anatomy(bench["forecast"], R, bench["oos_index"], T=12)
    assert a["r2"] > 0.999
    np.testing.assert_allclose(a["lag_weights"], declining_weights(12), atol=1e-6)


def test_anatomy_uses_only_lagged_returns():
    """Perturbing returns after the last decision point leaves the anatomy
    unchanged (no lookahead)."""
    rng = np.random.default_rng(1)
    R = rng.standard_normal(120)
    bench = momentum_benchmark(R, T=12)
    ts = bench["oos_index"]
    base = forecast_anatomy(bench["forecast"], R, ts, T=12)
    R2 = R.copy()
    R2[ts.max() + 1 :] += 5.0
    perturbed = forecast_anatomy(bench["forecast"], R2, ts, T=12)
    np.testing.assert_allclose(base["lag_weights"], perturbed["lag_weights"])


def test_spanning_alpha_near_zero_for_scaled_benchmark():
    rng = np.random.default_rng(2)
    n = 300
    bench = rng.standard_normal(n)
    market = rng.standard_normal(n)
    voc = 2.0 * bench + 0.001 * rng.standard_normal(n)  # a scaled copy + noise
    res = spanning_regression(voc, bench, market)
    assert abs(res["alpha"]) < 0.02
    assert np.isclose(res["beta_benchmark"], 2.0, atol=0.02)


def test_spanning_positive_alpha_for_orthogonal_drift():
    rng = np.random.default_rng(3)
    n = 300
    bench = rng.standard_normal(n)
    market = rng.standard_normal(n)
    voc = bench + 0.5  # a constant drift the benchmark cannot span
    res = spanning_regression(voc, bench, market)
    assert res["alpha"] > 0.3


def test_benchmark_metrics_matches_performance_metrics():
    rng = np.random.default_rng(4)
    forecast = 0.1 * rng.standard_normal(80)
    realized = 0.1 * rng.standard_normal(80)
    m = benchmark_metrics(forecast, realized)
    for key in ("r2", "sharpe", "alpha", "information_ratio", "alpha_tstat"):
        assert key in m and np.isfinite(m[key])
    assert np.isclose(m["sharpe"], sharpe_ratio(forecast * realized))


def test_benchmark_metrics_matches_compute_metrics():
    """benchmark_metrics must equal compute_metrics on every shared key, so
    the Nagel comparison table scores both strategies with the SAME metric
    definitions (checkup finding: the claimed parity was never asserted)."""
    import numpy as np

    from voc.nagel import benchmark_metrics
    from voc.performance_metrics import compute_metrics

    rng = np.random.default_rng(11)
    forecast = rng.standard_normal(240)
    realized = rng.standard_normal(240)
    bench = benchmark_metrics(forecast, realized)
    full = compute_metrics(forecast, realized, np.ones(240))
    for key, value in bench.items():
        assert np.isclose(value, full[key]), key


def test_ma2_reversal_uses_variance_point_zero_one_and_exact_formula():
    returns = np.linspace(-0.02, 0.03, 8)
    transformed, details = add_ma2_reversal(returns, seed=7, return_innovations=True)
    xi = details["innovations"]
    expected = returns + xi[2:] - 0.2 * xi[1:-1] - 0.2 * xi[:-2]
    np.testing.assert_allclose(transformed, expected)
    # Pin the RNG scale: 0.01 is the variance, not the standard deviation.
    expected_xi = np.random.default_rng(7).normal(scale=0.1, size=10)
    np.testing.assert_allclose(xi, expected_xi)


def test_kendall_correction_pins_documented_convention():
    rng = np.random.default_rng(8)
    x = rng.standard_normal(50).cumsum()
    unadjusted = fit_ar1_kendall(x, bias_adjust=False, stationarity_bound=None)
    adjusted = fit_ar1_kendall(x, bias_adjust=True, stationarity_bound=None)
    expected = unadjusted["phi"] + (1 + 3 * unadjusted["phi"]) / (len(x) - 1)
    assert np.isclose(adjusted["phi"], expected)
    assert np.isclose(adjusted["intercept"], (1 - expected) * x.mean())


def test_wild_bootstrap_is_reproducible_and_preserves_innovation_magnitudes():
    rng = np.random.default_rng(9)
    predictors = rng.standard_normal((80, 3)).cumsum(axis=0)
    first, diagnostics = wild_bootstrap_predictors(
        predictors, seed=10, return_diagnostics=True
    )
    second = wild_bootstrap_predictors(predictors, seed=10)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first[0], predictors[0])
    for column, fit in enumerate(diagnostics):
        innovations = (
            first[1:, column] - fit["intercept"] - fit["phi"] * first[:-1, column]
        )
        np.testing.assert_allclose(np.abs(innovations), np.abs(fit["residuals"]))
        assert set(np.unique(fit["signs"])).issubset({-1.0, 1.0})


def _named_twin_predictors(n=180):
    rng = np.random.default_rng(11)
    names = ["dp", "ltr", "dfr", "other"]
    return rng.standard_normal((n, len(names))), names


def test_twin_markets_share_shocks_and_use_next_return_timing():
    predictors, names = _named_twin_predictors()
    twins = make_twin_markets(predictors, names, signal_r2=0.2, shock_seed=12)
    np.testing.assert_allclose(twins["plus"][0], twins["minus"][0])
    np.testing.assert_allclose(
        twins["plus"][1:] - twins["minus"][1:],
        2.0 * twins["truth"][:-1],
    )
    np.testing.assert_allclose(
        0.5 * (twins["plus"][1:] + twins["minus"][1:]),
        twins["epsilon"][1:],
    )


def test_twin_market_history_is_unchanged_when_sample_end_extends():
    predictors, names = _named_twin_predictors(n=180)
    short = make_twin_markets(
        predictors[:150], names, signal_r2=0.2, shock_seed=12
    )
    long = make_twin_markets(predictors, names, signal_r2=0.2, shock_seed=12)
    for key in ("plus", "minus", "epsilon", "g", "truth"):
        np.testing.assert_allclose(short[key], long[key][:150])


def test_recovery_metrics_reports_pattern_scale_and_error():
    truth = np.linspace(-2.0, 2.0, 101)
    shared = np.sin(np.arange(101))
    plus = shared + 1.5 * truth + 0.25
    minus = shared - 1.5 * truth - 0.25
    result = recovery_metrics(plus, minus, truth)
    assert np.isclose(result["correlation"], 1.0)
    assert np.isclose(result["slope"], 1.5)
    assert np.isclose(result["intercept"], 0.25)
    np.testing.assert_allclose(result["recovered"], 1.5 * truth + 0.25)
