"""Tests for the Nagel critique driver layer (checkup follow-up to #17).

The ``voc/nagel.py`` toolkit has its own synthetic tests; this module covers
the driver glue that was merged without coverage: the anchor-forecast loader
(ridgeless filter, across-seed averaging, the loud failure on a ridge-only
export) and the structure of the cached result parquets (skip-gated until
``doit export_forecasts nagel`` has run).

Note, documented rather than asserted: ``load_anchor_forecasts`` averages
FORECASTS across seeds and computes one statistic set on the averaged path,
which deviates from the engine's average-the-statistics convention (the
averaged forecast is denoised, so its Sharpe exceeds the per-seed mean in
the Figure 8 cache). Flagged in the checkup for a deliberate decision; the
tests pin what the code does today.
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
)

requires_results = pytest.mark.skipif(
    not (METRICS_PATH.exists() and ANATOMY_PATH.exists() and SPANNING_PATH.exists()),
    reason="Nagel result parquets not found; run `doit export_forecasts nagel`",
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
        "Momentum benchmark",
    ]
    stat_cols = [c for c in metrics.columns if c != "strategy"]
    assert np.isfinite(metrics[stat_cols].to_numpy()).all()
    assert list(anatomy["lag"]) == list(range(1, 13))
    assert np.isfinite(anatomy[["voc_lag_weight", "benchmark_weight"]].to_numpy()).all()
    assert len(spanning) == 1
    for column in ("alpha", "alpha_tstat", "beta_benchmark", "anatomy_r2"):
        assert np.isfinite(spanning[column].iloc[0])
