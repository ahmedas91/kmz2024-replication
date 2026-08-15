"""Tests for the tidy KMZ dataset.

Validates the parquet produced by `clean_goyal_welch.py`: the exact predictor
set, the spread and log-ratio constructions against the raw Goyal-Welch
columns, the no-lookahead convention for the lagged-return predictor, the
excess-return cross-check against the workbook's own S&P 500 series, and a
clean monthly axis with no missing values over the paper's 1926-2020 sample.
The whole module skips (rather than fails) when the parquet has not been
built yet, since data never ships with the repository.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
PARQUET_PATH = DATA_DIR / "kmz_dataset.parquet"

pytestmark = pytest.mark.skipif(
    not PARQUET_PATH.exists(),
    reason="kmz_dataset.parquet not found; run `doit tidy` first",
)


@pytest.fixture(scope="module")
def kmz():
    """Load the saved tidy dataset once for all tests."""
    import clean_goyal_welch

    return clean_goyal_welch.load_kmz_dataset(data_dir=DATA_DIR)


@pytest.fixture(scope="module")
def merged(kmz):
    """Tidy dataset joined with the raw Goyal-Welch columns on date.

    Raw columns whose names collide with tidy ones get a `_gw` suffix.
    """
    import pull_goyal_welch

    gw = pull_goyal_welch.load_goyal_welch(data_dir=DATA_DIR)
    return kmz.merge(gw, on="date", how="inner", suffixes=("", "_gw"))


def test_predictor_columns(kmz):
    """Output schema is exactly date, mkt_excess, and the 15 predictors."""
    from clean_goyal_welch import PREDICTOR_COLUMNS

    assert len(PREDICTOR_COLUMNS) == 15
    assert list(kmz.columns) == ["date", "mkt_excess"] + PREDICTOR_COLUMNS


def test_spread_identities(merged):
    """tms, dfy, dfr equal their defining spreads of raw yield columns."""
    assert (merged["tms"] - (merged["lty_gw"] - merged["tbl_gw"])).abs().max() < 1e-12
    assert (merged["dfy"] - (merged["BAA"] - merged["AAA"])).abs().max() < 1e-12
    assert (merged["dfr"] - (merged["corpr"] - merged["ltr_gw"])).abs().max() < 1e-12


def test_log_ratio_construction(merged):
    """dp, dy, ep, de equal the log of the workbook's plain-ratio columns."""
    pairs = [("dp", "d/p"), ("dy", "d/y"), ("ep", "e/p"), ("de", "d/e")]
    for ours, raw in pairs:
        diff = (merged[ours] - np.log(merged[raw])).dropna().abs().max()
        assert diff < 1e-8, f"{ours} does not match log({raw}): max diff {diff}"


def test_no_lookahead_lag_predictor(kmz):
    """The lagged-return predictor at row t equals mkt_excess at row t.

    Row t holds what is known at the end of month t; the model layer applies
    the single uniform shift when pairing predictors at t with the t+1
    return. Equality here guarantees the column was not pre-shifted.
    """
    pd.testing.assert_series_equal(
        kmz["lag_mkt_excess"], kmz["mkt_excess"], check_names=False
    )


def test_excess_return_cross_check(merged):
    """vwretd-based excess return tracks the workbook's S&P 500 series."""
    sp_excess = merged["ret"] - merged["Rfree"]
    corr = merged["mkt_excess"].corr(sp_excess)
    assert corr > 0.98, f"correlation with CRSP S&P 500 excess return: {corr}"


def test_date_axis_span_and_completeness(kmz):
    """Monthly month-end axis covering 1926-2020 with no missing values.

    Two documented exceptions to completeness at the very start of the
    sample: `dy` needs a prior-month price, which does not exist when the
    raw pull starts exactly in 1926-01, so it is asserted from 1926-02; and
    `ntis` is a trailing 12-month sum of net issuance that only starts in
    1926-12 in the workbook, so it is asserted from there. Both fall inside
    the 36-month standardization burn-in, so the 1930 analysis start is
    unaffected.
    """
    dates = kmz["date"]
    assert (dates == dates + pd.offsets.MonthEnd(0)).all(), "dates not month-end"
    month_ordinals = dates.dt.to_period("M").astype("int64")
    assert not month_ordinals.duplicated().any(), "duplicate months"
    assert (month_ordinals.diff().dropna() == 1).all(), "gaps or disorder in months"
    assert dates.min() <= pd.Timestamp("1926-01-31")
    assert dates.max() >= pd.Timestamp("2020-12-31")
    sample = kmz.set_index("date").loc["1926-02":"2020-12"]
    n_missing = sample.drop(columns="ntis").isna().sum()
    bad = n_missing[n_missing > 0]
    assert bad.empty, f"missing values in paper sample: {bad.to_dict()}"
    assert sample.loc["1926-12":, "ntis"].isna().sum() == 0, "ntis gaps after 1926-12"
