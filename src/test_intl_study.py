"""Tests for the international equities VoC study (issue #16).

Three layers, all skipping until their inputs exist:

1. Pull: expected columns, strictly monthly month-end dates, a span from
   the series' 1990-07 start through at least the paper sample end, and
   DECIMAL units (the French library serves percent; the pull converts).
2. The study input builder: the sample starts one 12-month volatility
   burn-in after the series start (the predictors' 1930 floor never binds
   here), 15 predictor columns, the own-lag same-row convention, all finite.
3. The cached grid: the single target with the full 112-cell structure,
   the fixed seed list, finite statistics.

Deliberately NO numeric anchors: nothing published pins international VoC
numbers. The finding (shape generalizes, level does not, and with ~341 OOS
months a Sharpe below ~0.19 is indistinguishable from zero) is reported on
the chart page, not enforced by tests.
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)

INTL_PATH = DATA_DIR / "intl_equity.parquet"
STD_PATH = DATA_DIR / "kmz_dataset_standardized.parquet"

requires_pull = pytest.mark.skipif(
    not INTL_PATH.exists(),
    reason="intl_equity.parquet missing; run `doit pull:intl_equity`",
)
requires_inputs = pytest.mark.skipif(
    not (INTL_PATH.exists() and STD_PATH.exists()),
    reason="intl pull or standardized dataset missing",
)


def _grid_path():
    from sample_period import SAMPLE_SUFFIX

    return DATA_DIR / f"oos_grid_intl_T{TRAIN_WINDOW}{SAMPLE_SUFFIX}_per_seed.parquet"


@lru_cache(maxsize=1)
def _intl():
    return pd.read_parquet(INTL_PATH)


@requires_pull
def test_pull_columns_span_and_units():
    """Expected columns; monthly month-end dates with no gaps; span from the
    1990-07 series start through at least 2020-12; decimal units (a percent
    slip would inflate the scale a hundredfold)."""
    intl = _intl()
    assert list(intl.columns) == ["date", "intl_mkt_excess", "rf"]
    assert intl["date"].dt.is_month_end.all()
    expected = pd.date_range(intl["date"].min(), intl["date"].max(), freq="ME")
    assert intl["date"].tolist() == list(expected)  # no missing months
    assert intl["date"].min() == pd.Timestamp("1990-07-31")
    assert intl["date"].max() >= pd.Timestamp("2020-12-31")
    ret = intl["intl_mkt_excess"]
    assert ret.abs().mean() < 0.05
    assert ret.std() < 0.15
    assert ret.between(-0.5, 0.5).all()
    assert intl["rf"].between(0.0, 0.02).all()


@requires_inputs
def test_study_inputs_geometry_and_own_lag():
    """The 12-month target-vol burn-in binds (first row 1991-07), 15
    predictor columns (14 shared US predictors + own lag), all finite, and
    the own-lag equals the standardized target on the SAME row (the engine
    applies the single t -> t+1 shift)."""
    from run_intl_study import TARGET_COL, build_intl_study_inputs

    frame = build_intl_study_inputs(sample_end="2020-12")
    assert frame["date"].iloc[0] == pd.Timestamp("1991-07-31")
    assert frame["date"].iloc[-1] == pd.Timestamp("2020-12-31")
    predictor_cols = [c for c in frame.columns if c not in ("date", TARGET_COL)]
    assert len(predictor_cols) == 15
    assert f"lag_{TARGET_COL}" in predictor_cols
    assert np.isfinite(frame.drop(columns="date").to_numpy()).all()
    assert (frame[f"lag_{TARGET_COL}"] == frame[TARGET_COL]).all()


@pytest.mark.skipif(
    not INTL_PATH.exists() or not _grid_path().exists(),
    reason="intl grid cache missing; run `doit intl_study`",
)
def test_intl_grid_structure_and_finiteness():
    """Single target, the full cell structure (14 P levels, 7 ridge z's plus
    ridgeless), the fixed seed list, finite statistics throughout."""
    per_seed = pd.read_parquet(_grid_path())
    assert set(per_seed["target"]) == {"intl_mkt_excess"}
    assert per_seed["P"].nunique() == 14
    assert set(np.log10(per_seed.loc[per_seed["z"] > 0, "z"].unique()).round()) == set(
        range(-3, 4)
    )
    n_seeds = per_seed["seed"].nunique()
    assert set(per_seed["seed"].unique()) == set(range(n_seeds))
    assert len(per_seed) == 14 * 8 * n_seeds
    stats = per_seed[["r2", "sharpe", "alpha", "information_ratio", "alpha_tstat"]]
    assert np.isfinite(stats.to_numpy()).all()
