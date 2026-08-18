"""Tests for the recursive OOS engine (issue #7).

Synthetic; no data on disk. They pin the engine's correctness: the dual-form
recursion reproduces an explicit window-by-window primal ridge, forecasts use no
future information, a known linear signal is recovered as positive OOS R^2, the
grid output has the right schema, and a fixed seed list is deterministic.
"""

import numpy as np
import pandas as pd

from voc.oos_engine import run_grid, run_recursive_oos
from voc.rff import compute_rff, draw_rff_weights, standardize_by_training_window

RNG = np.random.default_rng(707)


def _ridge_primal(S, R, z):
    """Independent primal-space reference for the KMZ estimator."""
    n_obs, n_feat = S.shape
    A = z * np.eye(n_feat) + S.T @ S / n_obs
    return np.linalg.solve(A, S.T @ R / n_obs)


def _synthetic_dataset(n, d, seed=0):
    """A standardized-looking dataset: date, mkt_excess, and d predictor columns."""
    rng = np.random.default_rng(seed)
    data = {
        "date": pd.date_range("1930-01-31", periods=n, freq="ME"),
        "mkt_excess": rng.standard_normal(n),
    }
    for j in range(d):
        data[f"x{j}"] = rng.standard_normal(n)
    return pd.DataFrame(data)


def test_dual_recursion_matches_brute_force_primal():
    """The engine's forecasts equal explicit window-by-window primal ridge."""
    n, d, T = 60, 3, 12
    G = RNG.standard_normal((n, d))
    R = RNG.standard_normal(n)
    p_grid, z_grid = (2, 8), (1e-2, 1.0)

    _, out = run_recursive_oos(
        G, R, seed=11, T=T, p_grid=p_grid, z_grid=z_grid,
        include_ridgeless=False, return_forecasts=True,
    )
    S = compute_rff(G, draw_rff_weights(max(p_grid) // 2, d, 11))
    ts = np.arange(T, n - 1)
    for P in p_grid:
        for z in z_grid:
            expected = np.empty(ts.size)
            for i, t in enumerate(ts):
                X_std, s_oos_std = standardize_by_training_window(
                    S[t - T:t, :P], S[t, :P]
                )
                beta = _ridge_primal(X_std, R[t - T + 1:t + 1], z)
                expected[i] = s_oos_std @ beta
            np.testing.assert_allclose(
                out["forecasts"][(P, z)], expected, rtol=1e-7, atol=1e-9
            )


def test_no_lookahead_in_forecasts():
    """Perturbing returns from month k on leaves every earlier forecast unchanged."""
    n, d, T, k = 60, 4, 12, 45
    G = RNG.standard_normal((n, d))
    R = RNG.standard_normal(n)
    R_future = R.copy()
    R_future[k:] += 10.0

    _, base = run_recursive_oos(
        G, R, seed=5, T=T, p_grid=(8,), z_grid=(1.0,),
        include_ridgeless=False, return_forecasts=True,
    )
    _, perturbed = run_recursive_oos(
        G, R_future, seed=5, T=T, p_grid=(8,), z_grid=(1.0,),
        include_ridgeless=False, return_forecasts=True,
    )
    unaffected = k - T  # decision points t < k cannot use R[k:]
    np.testing.assert_allclose(
        base["forecasts"][(8, 1.0)][:unaffected],
        perturbed["forecasts"][(8, 1.0)][:unaffected],
        rtol=1e-12,
    )


def test_known_signal_recovers_positive_r2():
    """When R[t+1] is an exact linear function of the RFFs of G[t], a
    correctly-sized model recovers it as a strongly positive OOS R^2."""
    n, d, T, P, seed = 80, 3, 12, 8, 9
    G = RNG.standard_normal((n, d))
    S = compute_rff(G, draw_rff_weights(P // 2, d, seed))
    beta_true = RNG.standard_normal(P)
    R = np.empty(n)
    R[0] = 0.0
    R[1:] = S[:-1] @ beta_true  # R[t+1] = S[t] @ beta_true

    results = run_recursive_oos(
        G, R, seed=seed, T=T, p_grid=(P,), z_grid=(1e-6,), include_ridgeless=False
    )
    r2 = next(r for r in results if r["P"] == P)["r2"]
    assert r2 > 0.5


def test_grid_schema_and_finite_and_ridgeless_column():
    """run_grid returns the full schema with finite stats and one averaged row
    per (P, z) including the ridgeless (z=0) column."""
    dataset = _synthetic_dataset(60, 3)
    p_grid, z_grid = (2, 8, 12), (1e-2, 1.0, 1e2)
    per_seed, averaged = run_grid(
        dataset, seeds=range(2), p_grid=p_grid, z_grid=z_grid
    )
    expected_cols = {
        "seed", "P", "z", "c", "r2", "beta_norm", "mean_return",
        "volatility", "sharpe", "alpha", "information_ratio", "alpha_tstat",
    }
    assert expected_cols <= set(per_seed.columns)
    stats = per_seed[
        ["r2", "beta_norm", "mean_return", "volatility",
         "sharpe", "alpha", "information_ratio", "alpha_tstat"]
    ].to_numpy()
    assert np.isfinite(stats).all()
    # 3 P values x (3 z + ridgeless) models, averaged across seeds.
    assert len(averaged) == len(p_grid) * (len(z_grid) + 1)
    assert (averaged["z"] == 0.0).any()  # ridgeless present


def test_determinism_same_seed_list():
    """The same seed list yields identical grid output."""
    dataset = _synthetic_dataset(60, 3)
    kwargs = dict(seeds=range(3), p_grid=(2, 8), z_grid=(1.0,), include_ridgeless=False)
    first, _ = run_grid(dataset, **kwargs)
    second, _ = run_grid(dataset, **kwargs)
    pd.testing.assert_frame_equal(first, second)
