"""Tests for the recursive OOS engine (issue #7).

Synthetic; no data on disk. They pin the engine's correctness: the dual-form
recursion reproduces an explicit window-by-window primal ridge (the shared
reference from test_kernel_ridge, so both suites pin the SAME normalization),
forecasts use no future information while later forecasts do respond to it,
a known linear signal is recovered as positive OOS R^2, the grid output has
the right schema, degenerate inputs are rejected, and a fixed seed list is
deterministic. Each test seeds its own generator, so tests stay independent
of execution order and subsets.
"""

import numpy as np
import pandas as pd
import pytest

from voc.oos_engine import run_grid, run_recursive_oos
from voc.rff import compute_rff, draw_rff_weights, standardize_by_training_window
from voc.test_kernel_ridge import _ridge_primal


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
    rng = np.random.default_rng(707)
    n, d, T = 60, 3, 12
    G = rng.standard_normal((n, d))
    R = rng.standard_normal(n)
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
    """Perturbing returns from month k on leaves every earlier forecast unchanged
    AND moves the forecasts whose training windows include the perturbed months.
    The second leg keeps this test two-sided: an engine that ignored training
    returns altogether would pass the first assertion alone."""
    rng = np.random.default_rng(708)
    n, d, T, k = 60, 4, 12, 45
    G = rng.standard_normal((n, d))
    R = rng.standard_normal(n)
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
    affected = np.abs(
        base["forecasts"][(8, 1.0)][unaffected:]
        - perturbed["forecasts"][(8, 1.0)][unaffected:]
    )
    assert affected.min() > 0.0  # every later window trains on a changed return


def test_known_signal_recovers_positive_r2():
    """When R[t+1] is an exact linear function of the RFFs of G[t], a
    correctly-sized model recovers it as a strongly positive OOS R^2."""
    rng = np.random.default_rng(709)
    n, d, T, P, seed = 80, 3, 12, 8, 9
    G = rng.standard_normal((n, d))
    S = compute_rff(G, draw_rff_weights(P // 2, d, seed))
    beta_true = rng.standard_normal(P)
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
    # Every per-seed statistic (c included) must survive into the averaged
    # frame; only the seed key is dropped.
    assert set(averaged.columns) == set(per_seed.columns) - {"seed"}


def test_determinism_same_seed_list():
    """The same seed list yields identical grid output."""
    dataset = _synthetic_dataset(60, 3)
    kwargs = dict(seeds=range(3), p_grid=(2, 8), z_grid=(1.0,), include_ridgeless=False)
    first, _ = run_grid(dataset, **kwargs)
    second, _ = run_grid(dataset, **kwargs)
    pd.testing.assert_frame_equal(first, second)


def test_rejects_bad_grids():
    """Config-reachable grid typos fail fast with clear messages, BEFORE the
    expensive RFF build: odd/nonpositive P, nonpositive or non-finite z, and an
    empty model grid."""
    rng = np.random.default_rng(21)
    G, R = rng.standard_normal((30, 2)), rng.standard_normal(30)
    with pytest.raises(ValueError, match="positive even"):
        run_recursive_oos(G, R, seed=0, p_grid=(3,), z_grid=(1.0,))
    with pytest.raises(ValueError, match="positive even"):
        run_recursive_oos(G, R, seed=0, p_grid=(-4, 8), z_grid=(1.0,))
    with pytest.raises(ValueError, match="positive even"):
        run_recursive_oos(G, R, seed=0, p_grid=(0, 8), z_grid=(1.0,))
    with pytest.raises(ValueError, match="finite z > 0"):
        run_recursive_oos(G, R, seed=0, p_grid=(8,), z_grid=(-1.0,))
    with pytest.raises(ValueError, match="finite z > 0"):
        run_recursive_oos(G, R, seed=0, p_grid=(8,), z_grid=(float("nan"),))
    with pytest.raises(ValueError, match="empty model grid"):
        run_recursive_oos(
            G, R, seed=0, p_grid=(8,), z_grid=(), include_ridgeless=False
        )


def test_rejects_short_samples_bad_T_and_nonfinite_data():
    """Degenerate samples raise clearly instead of crashing deep in the metrics
    layer or silently caching NaN/garbage statistic rows."""
    rng = np.random.default_rng(22)
    G, R = rng.standard_normal((15, 2)), rng.standard_normal(15)
    with pytest.raises(ValueError, match="at least 3 OOS months"):
        run_recursive_oos(G, R, seed=0, T=12, p_grid=(4,), z_grid=(1.0,))
    with pytest.raises(ValueError, match="positive integer"):
        run_recursive_oos(G, R, seed=0, T=0, p_grid=(4,), z_grid=(1.0,))
    with pytest.raises(ValueError, match="positive integer"):
        run_recursive_oos(G, R, seed=0, T=-12, p_grid=(4,), z_grid=(1.0,))
    G_bad = G.copy()
    G_bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        run_recursive_oos(G_bad, R, seed=0, T=4, p_grid=(4,), z_grid=(1.0,))


def test_duplicate_grid_values_collapse_to_one_row():
    """z_grid=(1, 1.0) and p_grid=(8, 8) must not emit duplicated (P, z) rows;
    duplicates would break downstream pivots on the cached parquet."""
    rng = np.random.default_rng(23)
    G, R = rng.standard_normal((30, 2)), rng.standard_normal(30)
    results = run_recursive_oos(
        G, R, seed=0, p_grid=(8, 8), z_grid=(1, 1.0), include_ridgeless=False
    )
    assert len(results) == 1


def test_run_grid_rejects_empty_seeds_and_dirty_datasets():
    """N_SEEDS=0 or a NaN cell must fail with a clear error, not a KeyError deep
    in pandas or a LinAlgError deep in eigh."""
    dataset = _synthetic_dataset(40, 2)
    with pytest.raises(ValueError, match="non-empty"):
        run_grid(dataset, seeds=())
    dirty = dataset.copy()
    dirty.loc[5, "x0"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run_grid(dirty, seeds=range(1), p_grid=(4,), z_grid=(1.0,))
