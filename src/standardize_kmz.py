"""Volatility-standardize the tidy KMZ dataset into the analysis-ready dataset.

Implements the backward-looking volatility standardization of Kelly, Malamud
and Zhou (2024), Section V.A and footnote 34, on top of the tidy dataset from
clean_goyal_welch.py. The output is the dataset every downstream analysis
runs on. The authors defend this standardization explicitly in their 2025
reply (Section 4.1.2), whose timing-weight formula r_{t+1}/sigma_t confirms
that the scale applied to a month's return is known one month earlier.

Exact indexing conventions (nothing dated at or after t enters the scale
applied at t):
 - Returns (`mkt_excess`): standardized value at month t is the raw value
   divided by the trailing volatility sqrt(mean(R_s^2)) over the 12 months
   s = t-12, ..., t-1. The second moment is UNCENTERED (no demeaning),
   per footnote 34, because mean monthly returns are too noisy to estimate
   in 12-month windows.
 - Predictors (the 14 non-return predictors): standardized value at month t
   is the raw value divided by the expanding-window sample standard
   deviation (ddof=1) of all observations strictly before t, requiring at
   least 36 prior observations (the paper's "36 months of data" for initial
   standardization stability). Leading gaps delay the start automatically:
   ntis begins 1926-12, so its scale first exists in 1929-12.
 - `lag_mkt_excess` is a return, not an ordinary predictor: it receives the
   return scheme, so it still EQUALS the standardized `mkt_excess` on the
   same row and the tidy file's no-double-shift convention survives.
 - No demeaning anywhere; standardization divides only.

Sample start: rows with any missing standardized value are dropped, then the
sample is floored at ANALYSIS_START (1930-01). The drop may only trim the
leading burn-in: a month-contiguity guard raises ValueError if it ever
removes an interior month (e.g. ep/de turning NaN because a future data
vintage has e12 <= 0), so the sample fragments loudly instead of silently.
With the current data vintage the natural burn-in ends at 1929-12, one
month earlier; the paper states the
analysis sample "began in 1930", and its out-of-sample recursion over
t in {T, ..., 1,091} implies exactly 1,092 months, 1930-01 through 2020-12,
which the floor reproduces. The floor binds only when the data would allow
an earlier start, so later START_DATE settings are unaffected.
"""

from pathlib import Path

import pandas as pd

from clean_goyal_welch import PREDICTOR_COLUMNS, load_kmz_dataset
from settings import config

DATA_DIR = Path(config("DATA_DIR"))

# First month of the analysis sample when the raw data starts in 1926; see
# the module docstring for why this floor exists.
ANALYSIS_START = "1930-01-31"


def trailing_uncentered_vol(returns, window=12):
    """Trailing volatility from the uncentered second moment, strictly prior.

    The value at row t is sqrt(mean(returns[t-window:t]**2)): the square
    root of the average SQUARED return (no demeaning) over the `window`
    observations strictly before t. The first `window` rows are NaN.
    """
    return returns.pow(2).rolling(window).mean().pow(0.5).shift(1)


def expanding_vol(series, min_periods=36):
    """Expanding-window standard deviation, strictly backward-looking.

    The value at row t is the sample standard deviation (ddof=1) of all
    non-null observations strictly before t, NaN until `min_periods`
    non-null observations have accumulated. Leading NaNs in the input (for
    example ntis before 1926-12) simply delay the start.
    """
    return series.expanding(min_periods=min_periods).std().shift(1)


def standardize_kmz_dataset(df=None, data_dir=DATA_DIR):
    """Apply the paper's volatility standardization to the tidy dataset.

    `df` allows injecting a tidy frame (for tests); by default the saved
    tidy parquet is loaded from `data_dir`. Returns the same 17-column
    schema as the tidy dataset, with `mkt_excess` and `lag_mkt_excess`
    scaled by the trailing 12-month uncentered return volatility and the
    other 14 predictors scaled by their strictly-prior expanding-window
    standard deviations, burn-in rows dropped, and the sample floored at
    ANALYSIS_START. Raises ValueError if dropping missing values removes an
    interior month rather than only leading burn-in rows.
    """
    if df is None:
        df = load_kmz_dataset(data_dir=data_dir)
    df = df.sort_values("date").reset_index(drop=True)

    out = pd.DataFrame({"date": df["date"]})
    out["mkt_excess"] = df["mkt_excess"] / trailing_uncentered_vol(df["mkt_excess"])
    for column in PREDICTOR_COLUMNS:
        if column == "lag_mkt_excess":
            out[column] = out["mkt_excess"]
        else:
            out[column] = df[column] / expanding_vol(df[column])

    out = out.dropna().reset_index(drop=True)
    month_ordinals = out["date"].dt.to_period("M").astype("int64")
    gap_resumes = out["date"].loc[month_ordinals.diff() > 1]
    if not gap_resumes.empty:
        raise ValueError(
            "interior NaN fragmented the standardized sample; dropna() may "
            "only trim leading burn-in rows, but the month axis has gaps "
            f"before {[d.strftime('%Y-%m') for d in gap_resumes]}"
        )
    out = out.loc[out["date"] >= pd.Timestamp(ANALYSIS_START)].reset_index(drop=True)
    return out[["date", "mkt_excess"] + PREDICTOR_COLUMNS]


def load_standardized_dataset(data_dir=DATA_DIR):
    """Load the saved standardized KMZ dataset."""
    path = Path(data_dir) / "kmz_dataset_standardized.parquet"
    return pd.read_parquet(path)


if __name__ == "__main__":
    df = standardize_kmz_dataset(data_dir=DATA_DIR)
    path = Path(DATA_DIR) / "kmz_dataset_standardized.parquet"
    df.to_parquet(path)
