"""Sanity tests for a non-paper (updated-sample) artifact set (issue #11).

Test gating between the two periods works through PATHS, not skip logic in
the anchor tests: the paper-period artifacts keep the bare canonical
filenames, and the anchor tests (``test_figure7.py``, ``test_figure8.py``,
``test_figure11.py``) read those bare names, so they always validate the
paper numbers against the paper artifacts, whatever ``SAMPLE_END`` is
currently configured, and skip only when the paper artifacts are absent.

This module covers the other side. When ``SAMPLE_END`` is configured away
from the paper period (README: run ``doit`` once with the default and once
with ``SAMPLE_END=2024-12``), the suffixed artifact set must exist and be
structurally sound: the OOS grid holds the same (P, z) cells as the paper
grid with finite statistics, the variable-importance cache holds all 16
configurations, and the figure and summary files are on disk. No numeric
anchors here, deliberately: the paper's published values pertain to its
1930-2020 sample only, so an updated period has nothing published to pin to.

Everything skips when the configured period IS the paper period (the bare
set is then fully covered by the anchor tests) or when the suffixed
artifacts have not been built yet.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sample_period import PAPER_SAMPLE_END, SAMPLE_END, SAMPLE_SUFFIX
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)

GRID_PATH = DATA_DIR / f"oos_grid_T{TRAIN_WINDOW}{SAMPLE_SUFFIX}.parquet"
VI_PATH = DATA_DIR / f"variable_importance_T{TRAIN_WINDOW}{SAMPLE_SUFFIX}.parquet"

is_paper_period = SAMPLE_END == PAPER_SAMPLE_END
requires_updated_run = pytest.mark.skipif(
    is_paper_period or not (GRID_PATH.exists() and VI_PATH.exists()),
    reason=(
        "updated-sample sanity tests run only when SAMPLE_END is configured "
        f"away from the paper period ({PAPER_SAMPLE_END}) and the suffixed "
        "artifacts have been built (SAMPLE_END=... doit)"
    ),
)

METRIC_COLUMNS = [
    "r2",
    "beta_norm",
    "mean_return",
    "volatility",
    "sharpe",
    "alpha",
    "information_ratio",
    "alpha_tstat",
]


@lru_cache(maxsize=1)
def _grid():
    return pd.read_parquet(GRID_PATH)


@lru_cache(maxsize=1)
def _vi():
    return pd.read_parquet(VI_PATH)


@requires_updated_run
def test_updated_grid_has_full_cell_structure():
    """Same model grid as the paper run: 14 complexity levels, 7 ridge z's
    plus the ridgeless z = 0 rows, one row per (P, z), c = P / T."""
    grid = _grid()
    assert grid["P"].nunique() == 14
    assert set(np.log10(grid.loc[grid["z"] > 0, "z"].unique()).round()) == set(
        range(-3, 4)
    )
    assert (grid["z"] == 0.0).sum() == grid["P"].nunique()
    assert len(grid) == grid["P"].nunique() * grid["z"].nunique()
    assert np.allclose(grid["c"], grid["P"] / TRAIN_WINDOW)


@requires_updated_run
def test_updated_grid_statistics_are_finite_and_sane():
    """All statistics finite; volatility strictly positive everywhere."""
    grid = _grid()
    assert np.isfinite(grid[METRIC_COLUMNS]).all().all()
    assert (grid["volatility"] > 0).all()


@requires_updated_run
def test_updated_vi_cache_structure():
    """All 16 variable-importance configurations, same seed set in each,
    finite metrics throughout."""
    vi = _vi()
    assert vi["excluded"].nunique() == 16
    assert "none" in set(vi["excluded"])
    seed_sets = vi.groupby("excluded")["seed"].agg(frozenset)
    assert seed_sets.nunique() == 1
    assert np.isfinite(vi[["r2", "sharpe"]]).all().all()


@requires_updated_run
def test_updated_output_files_exist():
    """The full suffixed output set is on disk next to the paper set."""
    expected = [
        f"figure7{SAMPLE_SUFFIX}.png",
        f"figure7_data{SAMPLE_SUFFIX}.parquet",
        f"figure8{SAMPLE_SUFFIX}.png",
        f"figure8_data{SAMPLE_SUFFIX}.parquet",
        f"figure11{SAMPLE_SUFFIX}.png",
        f"figure11_data{SAMPLE_SUFFIX}.parquet",
        f"predictor_summary_table{SAMPLE_SUFFIX}.tex",
        f"predictor_timeseries{SAMPLE_SUFFIX}.png",
    ]
    missing = [name for name in expected if not (OUTPUT_DIR / name).exists()]
    assert not missing, f"updated-sample outputs missing: {missing}"
