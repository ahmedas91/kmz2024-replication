"""Tests for the reusable VoC study API (issue #14).

Synthetic; no data on disk. They pin the public surface: an end-to-end study
returns the expected schema and finite values; the market DataFrame wrapper
(`run_grid`) reproduces the array API (`run_voc_study`) exactly; a Sharpe
recomputed from an exported per-seed strategy series matches the cached statistic;
`save_forecasts` requires a `data_dir`; and the standardization steps are strictly
backward-looking.
"""

import numpy as np
import pandas as pd
import pytest

from voc import run_voc_study, standardize_inputs
from voc.oos_engine import run_grid
from voc.performance_metrics import sharpe_ratio


def _synthetic(n=60, d=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n), rng.standard_normal((n, d))


def test_end_to_end_schema_and_finite():
    target, predictors = _synthetic()
    per_seed, averaged = run_voc_study(
        target, predictors, p_grid=(2, 8, 12), z_grid=(1e-2, 1.0), seeds=range(2)
    )
    # 3 P x (2 z + ridgeless) models.
    assert len(averaged) == 3 * (2 + 1)
    assert {"P", "z", "c"}.issubset(averaged.columns)
    stats = per_seed[
        ["r2", "beta_norm", "sharpe", "alpha", "information_ratio", "alpha_tstat"]
    ].to_numpy()
    assert np.isfinite(stats).all()


def test_market_wrapper_matches_array_api():
    """run_grid (DataFrame) reproduces run_voc_study (arrays) for the same seeds."""
    target, predictors = _synthetic()
    ds = pd.DataFrame(
        {
            "date": pd.date_range("1930-01-31", periods=len(target), freq="ME"),
            "mkt_excess": target,
        }
    )
    for j in range(predictors.shape[1]):
        ds[f"x{j}"] = predictors[:, j]

    a1, b1 = run_grid(ds, seeds=range(2), p_grid=(2, 8), z_grid=(1.0,))
    a2, b2 = run_voc_study(
        target, predictors, seeds=range(2), p_grid=(2, 8), z_grid=(1.0,)
    )
    pd.testing.assert_frame_equal(a1, a2)
    pd.testing.assert_frame_equal(b1, b2)


def test_forecast_export_consistency(tmp_path):
    """Sharpe recomputed from an exported strategy series matches the cached stat."""
    target, predictors = _synthetic()
    per_seed, _ = run_voc_study(
        target,
        predictors,
        p_grid=(8,),
        z_grid=(1.0,),
        seeds=range(2),
        include_ridgeless=False,
        save_forecasts=True,
        study_name="unit",
        data_dir=tmp_path,
    )
    exported = pd.read_parquet(tmp_path / "forecasts_unit.parquet")
    cell = exported[
        (exported["seed"] == 0) & (exported["P"] == 8) & (exported["z"] == 1.0)
    ]
    recomputed = sharpe_ratio(cell["strategy"].to_numpy())
    cached = per_seed[
        (per_seed["seed"] == 0) & (per_seed["P"] == 8) & (per_seed["z"] == 1.0)
    ]["sharpe"].iloc[0]
    assert np.isclose(recomputed, cached)


def test_save_forecasts_requires_data_dir():
    target, predictors = _synthetic()
    with pytest.raises(ValueError):
        run_voc_study(
            target,
            predictors,
            save_forecasts=True,
            seeds=range(1),
            p_grid=(2,),
            z_grid=(1.0,),
        )


def test_mismatched_lengths_rejected():
    target, predictors = _synthetic(n=60)
    with pytest.raises(ValueError):
        run_voc_study(
            target[:-1], predictors, seeds=range(1), p_grid=(2,), z_grid=(1.0,)
        )


def test_standardize_inputs_is_backward_looking():
    """Burn-in rows are invalid; settled rows are finite (no lookahead)."""
    rng = np.random.default_rng(1)
    target = rng.standard_normal(120)
    predictors = rng.standard_normal((120, 3))
    target_std, predictors_std, valid = standardize_inputs(target, predictors)
    assert not valid[:36].any()  # 36-month predictor burn-in
    assert valid[36:].all()
    assert np.isfinite(target_std[valid]).all()
    assert np.isfinite(predictors_std[valid]).all()


def test_dates_length_mismatch_rejected():
    """A dates array not matching the sample length must fail loudly, not
    silently mislabel the forecast export (checkup finding)."""
    target, predictors = _synthetic()
    with pytest.raises(ValueError, match="one label per data row"):
        run_voc_study(
            target,
            predictors,
            dates=np.arange(len(target) + 24),
            seeds=range(1),
            p_grid=(2,),
            z_grid=(1.0,),
        )


def test_standardize_inputs_rejects_mid_sample_gaps():
    """A NaN month after the burn-in would make the valid mask non-contiguous
    and splice non-adjacent months into the engine's recursion; refuse."""
    rng = np.random.default_rng(3)
    target = rng.standard_normal(120)
    target[60] = np.nan
    predictors = rng.standard_normal((120, 3))
    with pytest.raises(ValueError, match="non-finite mid-sample"):
        standardize_inputs(target, predictors)


def test_centered_standardization_requires_two_rows():
    """T=1 centered-mode scales are undefined (ddof=1); the engine must error
    rather than silently disable the pinned standardization."""
    from voc.rff import standardize_by_training_window

    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="at least 2 training rows"):
        standardize_by_training_window(
            rng.standard_normal((1, 4)), rng.standard_normal(4), uncentered=False
        )
