"""Tests for the raw Goyal-Welch pull.

Validates the parquet produced by `pull_goyal_welch.py`: the raw columns the
downstream predictor construction needs, a clean month-end monthly date axis,
and no missing values in the core series over the paper's 1926-2020 sample.
The whole module skips (rather than fails) when the parquet has not been
pulled yet, since raw data never ships with the repository.
"""

from pathlib import Path

import pandas as pd
import pytest

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
PARQUET_PATH = DATA_DIR / "goyal_welch.parquet"

pytestmark = pytest.mark.skipif(
    not PARQUET_PATH.exists(),
    reason="goyal_welch.parquet not found; run `doit pull:goyal_welch` first",
)

# Raw ingredients for the 15 predictors and the forecast target, plus the
# precomputed ratios used for cross-checking the tidy module's construction.
CORE_COLUMNS = [
    "yyyymm",
    "date",
    "price",
    "d12",
    "e12",
    "b/m",
    "tbl",
    "AAA",
    "BAA",
    "lty",
    "ntis",
    "Rfree",
    "infl",
    "ltr",
    "corpr",
    "svar",
    "ret",
    "retx",
    "tms",
    "dfy",
    "dfr",
    "d/p",
    "d/y",
    "e/p",
    "d/e",
]


@pytest.fixture(scope="module")
def gw():
    """Load the saved Goyal-Welch parquet once for all tests."""
    import pull_goyal_welch

    return pull_goyal_welch.load_goyal_welch(data_dir=DATA_DIR)


def test_core_columns_present(gw):
    """Every raw column the tidy module needs is in the saved file."""
    missing = set(CORE_COLUMNS) - set(gw.columns)
    assert not missing, f"missing columns: {sorted(missing)}"


def test_monthly_month_end_date_axis(gw):
    """Dates are month-end, strictly monthly, with no gaps or duplicates."""
    dates = gw["date"]
    assert (dates == dates + pd.offsets.MonthEnd(0)).all(), "dates not month-end"
    month_ordinals = dates.dt.to_period("M").astype("int64")
    assert not month_ordinals.duplicated().any(), "duplicate months"
    assert (month_ordinals.diff().dropna() == 1).all(), "gaps or disorder in months"


def test_spans_paper_sample(gw):
    """The pull covers at least the paper's 1926-01 to 2020-12 sample."""
    assert gw["date"].min() <= pd.Timestamp("1926-01-31")
    assert gw["date"].max() >= pd.Timestamp("2020-12-31")


def test_no_missing_core_values_in_paper_sample(gw):
    """Index level and risk-free rate are complete over 1926-2020."""
    sample = gw.set_index("date").loc["1926-01":"2020-12"]
    for column in ["price", "Rfree"]:
        n_missing = sample[column].isna().sum()
        assert n_missing == 0, f"{column} has {n_missing} missing values"
