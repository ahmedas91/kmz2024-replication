"""Tests for the Nagel critique toolkit (issue #17).

Synthetic; no data on disk. They pin the benchmark on a hand-computed series, the
spanning regression's alpha behaviour (near zero for a scaled copy of the
benchmark, positive for an orthogonal drift), and that the anatomy uses only
lagged returns.
"""

import numpy as np

from voc.nagel import (
    benchmark_metrics,
    declining_weights,
    forecast_anatomy,
    momentum_benchmark,
    spanning_regression,
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
