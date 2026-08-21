"""Tests for the Nagel critique and matched-twin driver layer.

The ``voc/nagel.py`` toolkit has its own synthetic tests; this module covers
the driver glue that was merged without coverage: the anchor-forecast loader
(ridgeless filter, across-seed averaging, the loud failure on a ridge-only
export) and the structure of the cached result parquets (skip-gated until
``doit nagel_analysis`` has run).

``load_anchor_forecasts`` supplies the across-seed ensemble path used only
for the supplemental forecast-projection plot. Performance and spanning
statistics use ``load_anchor_forecasts_by_seed`` and are completed per draw
before averaging, matching the paper's convention.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from nagel_analysis import (
    ANATOMY_PATH,
    METRICS_PATH,
    SPANNING_PATH,
    load_anchor_forecasts,
    load_anchor_forecasts_by_seed,
)
from nagel_experiments import (
    COUNTERFACTUAL_SPANNING_PATH,
    COUNTERFACTUALS_PATH,
    TWIN_PATHS_PATH,
    TWIN_RECOVERY_PATH,
    TWIN_SPANNING_PATH,
    _twin_seed_rows,
)
from voc.nagel import make_twin_markets

requires_results = pytest.mark.skipif(
    not (METRICS_PATH.exists() and ANATOMY_PATH.exists() and SPANNING_PATH.exists()),
    reason="Nagel result parquets not found; run `doit nagel_analysis`",
)

requires_experiments = pytest.mark.skipif(
    not all(
        path.exists()
        for path in (
            COUNTERFACTUALS_PATH,
            COUNTERFACTUAL_SPANNING_PATH,
            TWIN_RECOVERY_PATH,
            TWIN_SPANNING_PATH,
            TWIN_PATHS_PATH,
        )
    ),
    reason="Nagel experiment parquets not found; run `doit nagel_analysis`",
)


def _fake_export(tmp_path, with_ridgeless=True):
    """A tiny forecasts parquet in the export schema: 2 seeds x 2 z x 3 months."""
    z_values = [1000.0] + ([0.0] if with_ridgeless else [])
    rows = [
        {
            "seed": seed,
            "P": 12_000,
            "z": z,
            "obs": pd.Timestamp("2000-01-31") + pd.offsets.MonthEnd(month),
            "forecast": float(seed + 1) * (month + 1),
            "realized": float(month + 1),
            "strategy": float(seed + 1) * (month + 1) ** 2,
        }
        for seed in (0, 1)
        for z in z_values
        for month in range(3)
    ]
    path = tmp_path / "forecasts_unit.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_loader_filters_ridgeless_and_averages_seeds(tmp_path):
    """Only z = 0 rows enter, averaged across seeds month by month."""
    voc = load_anchor_forecasts(_fake_export(tmp_path))
    assert len(voc) == 3  # one row per month
    # seeds 0 and 1 forecast (month+1) and 2*(month+1): mean = 1.5*(month+1)
    np.testing.assert_allclose(voc["forecast"].to_numpy(), [1.5, 3.0, 4.5])
    np.testing.assert_allclose(voc["realized"].to_numpy(), [1.0, 2.0, 3.0])


def test_per_seed_loader_preserves_draws(tmp_path):
    forecasts = load_anchor_forecasts_by_seed(_fake_export(tmp_path))
    assert list(forecasts["seed"].unique()) == [0, 1]
    assert len(forecasts) == 6
    np.testing.assert_allclose(
        forecasts.loc[forecasts["seed"] == 1, "forecast"], [2.0, 4.0, 6.0]
    )


def test_loader_rejects_export_without_ridgeless(tmp_path):
    """A ridge-only export must fail loudly, not silently score z = 1000."""
    path = _fake_export(tmp_path, with_ridgeless=False)
    with pytest.raises(ValueError, match="no ridgeless"):
        load_anchor_forecasts(path)


@lru_cache(maxsize=1)
def _results():
    return (
        pd.read_parquet(METRICS_PATH),
        pd.read_parquet(ANATOMY_PATH),
        pd.read_parquet(SPANNING_PATH),
    )


@requires_results
def test_result_parquets_structure():
    """Two strategies in the metrics table with finite values; one lag weight
    per training month; the spanning row carries the regression outputs."""
    metrics, anatomy, spanning = _results()
    assert list(metrics["strategy"]) == [
        "VoC (ridgeless, c=1000)",
        "Volatility-timed momentum",
    ]
    stat_cols = [c for c in metrics.select_dtypes(include=[np.number]).columns]
    assert np.isfinite(metrics[stat_cols].to_numpy()).all()
    assert list(anatomy["lag"]) == list(range(1, 13))
    assert np.isfinite(anatomy[["voc_lag_weight", "benchmark_weight"]].to_numpy()).all()
    assert len(spanning) == 1
    for column in ("alpha", "alpha_tstat", "beta_benchmark", "anatomy_r2"):
        assert np.isfinite(spanning[column].iloc[0])


def test_twin_driver_exercises_both_predictor_designs_with_paired_seeds():
    """A tiny engine run pins the 14/15-input orchestration and tidy keys."""
    rng = np.random.default_rng(31)
    names = ["dp", "ltr", "dfr"] + [f"x{i}" for i in range(11)]
    predictors = rng.standard_normal((40, 14))
    twins = {
        "easy": make_twin_markets(
            predictors,
            names,
            signal_r2=0.20,
            shock_seed=32,
            calibration_size=20,
        ),
        "realistic": make_twin_markets(
            predictors,
            names,
            signal_r2=0.02,
            shock_seed=32,
            calibration_size=20,
        ),
    }
    recovery, spanning, paths = _twin_seed_rows(0, predictors, twins, T=4, P=8)
    assert {(row["predictor_set"], row["strength"]) for row in recovery} == {
        (design, strength)
        for design in ("x14_shared", "x15_world_lag")
        for strength in ("easy", "realistic")
    }
    assert {
        (row["predictor_set"], row["strength"], row["market"]) for row in spanning
    } == {
        (design, strength, market)
        for design in ("x14_shared", "x15_world_lag")
        for strength in ("easy", "realistic")
        for market in ("plus", "minus")
    }
    assert len(paths) == 4
    assert all(len(row["truth"]) == 35 for row in paths)


@requires_experiments
def test_experiment_cache_schemas_and_keys():
    counterfactuals = pd.read_parquet(COUNTERFACTUALS_PATH)
    counter_spanning = pd.read_parquet(COUNTERFACTUAL_SPANNING_PATH)
    recovery = pd.read_parquet(TWIN_RECOVERY_PATH)
    spanning = pd.read_parquet(TWIN_SPANNING_PATH)
    paths = pd.read_parquet(TWIN_PATHS_PATH)

    assert set(counterfactuals["experiment"]) == {
        "historical",
        "ma2_reversal",
        "wild_bootstrap",
    }
    assert set(counterfactuals["strategy"]) == {
        "RFF",
        "Volatility-timed momentum",
    }
    assert set(counter_spanning["experiment"]) == set(counterfactuals["experiment"])
    assert set(counterfactuals["target_scale"]) == {"raw decimal return"}
    assert set(recovery["predictor_set"]) == {"x14_shared", "x15_world_lag"}
    assert set(recovery["strength"]) == {"easy", "realistic"}
    assert set(spanning["market"]) == {"plus", "minus"}
    assert set(paths["predictor_set"]) == set(recovery["predictor_set"])
    for frame in (counterfactuals, counter_spanning, recovery, spanning, paths):
        assert np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()
