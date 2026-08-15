"""Tests for the volatility standardization module.

The synthetic tests pin the windowing conventions (uncentered second moment,
strictly backward-looking scales, expanding window, the 1930 sample start)
on constructed inputs and need no data on disk. The real-file tests validate
the saved standardized parquet and skip (rather than fail) when it has not
been built yet, since data never ships with the repository.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from settings import config
from standardize_kmz import (
    expanding_vol,
    load_standardized_dataset,
    standardize_kmz_dataset,
    trailing_uncentered_vol,
)

DATA_DIR = Path(config("DATA_DIR"))
PARQUET_PATH = DATA_DIR / "kmz_dataset_standardized.parquet"

requires_data = pytest.mark.skipif(
    not PARQUET_PATH.exists(),
    reason="kmz_dataset_standardized.parquet not found; run `doit standardize` first",
)


def _synthetic_tidy(n_months=84):
    """Deterministic tidy-shaped frame starting 1926-01 for convention tests."""
    from clean_goyal_welch import PREDICTOR_COLUMNS

    dates = pd.date_range("1926-01-31", periods=n_months, freq="ME")
    base = pd.Series(np.sin(np.arange(n_months)) * 0.02 + 0.05)
    df = pd.DataFrame({"date": dates, "mkt_excess": base})
    for i, column in enumerate(PREDICTOR_COLUMNS):
        if column == "lag_mkt_excess":
            df[column] = df["mkt_excess"]
        else:
            df[column] = np.cos(np.arange(n_months) / (i + 2)) + 2.0
    return df


def test_uncentered_vol_alternating_series_exact():
    """For an alternating +x/-x series, the uncentered trailing vol is x."""
    x = 0.05
    returns = pd.Series([x, -x] * 12)
    vol = trailing_uncentered_vol(returns, window=12)
    assert np.allclose(vol.dropna(), x)


def test_uncentered_vol_exceeds_centered_for_large_mean():
    """With a large nonzero mean, the uncentered vol exceeds the centered std.

    Pins the footnote-34 convention: sqrt(mean(R^2)) keeps the mean inside
    the scale, so it must exceed the demeaned rolling standard deviation.
    """
    returns = pd.Series(0.10 + 0.01 * np.sin(np.arange(30)))
    uncentered = trailing_uncentered_vol(returns, window=12)
    centered = returns.rolling(12).std().shift(1)
    valid = uncentered.notna() & centered.notna()
    assert (uncentered[valid] > centered[valid]).all()


def test_scales_are_strictly_backward_looking():
    """Perturbing future observations never changes past volatility scales."""
    base = pd.Series(np.sin(np.arange(60)) * 0.02 + 0.03)
    perturbed = base.copy()
    perturbed.iloc[50:] = 9.99
    for vol_func in (trailing_uncentered_vol, expanding_vol):
        original = vol_func(base)
        modified = vol_func(perturbed)
        pd.testing.assert_series_equal(original.iloc[:51], modified.iloc[:51])


def test_expanding_vol_matches_brute_force():
    """expanding_vol equals a hand-rolled loop over prior observations."""
    series = pd.Series([1.0, 2.0, 4.0, 8.0, 3.0, 5.0, 7.0])
    result = expanding_vol(series, min_periods=2)
    for t in range(len(series)):
        prior = series.iloc[:t]
        if len(prior) < 2:
            assert pd.isna(result.iloc[t])
        else:
            assert np.isclose(result.iloc[t], prior.std())


def test_synthetic_input_from_1926_starts_1930():
    """With complete input from 1926-01, the standardized sample starts 1930-01."""
    out = standardize_kmz_dataset(df=_synthetic_tidy(84))
    assert out["date"].iloc[0] == pd.Timestamp("1930-01-31")


def test_interior_nan_raises_instead_of_fragmenting():
    """An interior NaN raises ValueError rather than silently gapping the sample.

    The closing dropna() may only trim leading burn-in rows; a NaN after the
    sample start (here ep in 1931-11, the e12 <= 0 scenario on a future
    vintage) must fail loudly, not shrink the monthly axis by one row.
    """
    df = _synthetic_tidy(84)
    df.loc[70, "ep"] = np.nan
    with pytest.raises(ValueError, match="interior NaN"):
        standardize_kmz_dataset(df=df)


@requires_data
def test_analysis_start_and_paper_window_row_count():
    """The saved dataset starts 1930-01 and the paper window has 1,092 rows."""
    out = load_standardized_dataset(data_dir=DATA_DIR)
    assert out["date"].min() == pd.Timestamp("1930-01-31")
    paper = out.set_index("date").loc["1930-01":"2020-12"]
    assert len(paper) == 1092


@requires_data
def test_schema_matches_tidy_and_complete():
    """Same 17 columns as the tidy dataset, with no missing values."""
    from clean_goyal_welch import PREDICTOR_COLUMNS

    out = load_standardized_dataset(data_dir=DATA_DIR)
    assert list(out.columns) == ["date", "mkt_excess"] + PREDICTOR_COLUMNS
    assert not out.isna().any().any()


@requires_data
def test_lag_identity_and_return_scale():
    """lag_mkt_excess stays equal to mkt_excess, and the scale is sensible.

    The standard deviation of the standardized return should sit near 1; a
    value far outside (0.5, 2) would indicate a misindexed division.
    """
    out = load_standardized_dataset(data_dir=DATA_DIR)
    pd.testing.assert_series_equal(
        out["lag_mkt_excess"], out["mkt_excess"], check_names=False
    )
    scale = out["mkt_excess"].std()
    assert 0.5 < scale < 2.0, f"standardized return scale off: {scale}"
