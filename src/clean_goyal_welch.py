"""Construct the tidy KMZ analysis dataset: the 15 predictors plus the target.

Merges the raw Goyal-Welch workbook (pull_goyal_welch.py) with the CRSP
monthly index file (pull_CRSP_stock.py) into the dataset the whole
replication runs on: the monthly excess return of the CRSP value-weighted
index plus the 15 predictors of Kelly, Malamud and Zhou (2024), Section V.A,
footnote 33. Cleaning only: no statistics, no plots, and no volatility
standardization (that lives in the standardization module built on top of
this file).

Conventions:
 - Predictors (Goyal-Welch mnemonics): dfy, infl, svar, de, lty, tms, tbl,
   dfr, dp, dy, ltr, ep, b/m (stored as `bm`), ntis, plus one lag of the
   market excess return (`lag_mkt_excess`), the easy-to-miss 15th predictor.
 - Valuation ratios are LOG ratios as in Goyal and Welch (2008):
   dp = log(d12) - log(price), dy = log(d12) - log(prior-month price),
   ep = log(e12) - log(price), de = log(d12) - log(e12). The workbook's own
   d/p, d/y, e/p, d/e columns are plain ratios; the tests use them as
   independent cross-checks of the log construction.
 - Spreads: tms = lty - tbl, dfy = BAA - AAA, dfr = corpr - ltr.
 - infl passes through AS-IS. The workbook already encodes the Goyal, Welch
   and Zafirov (2024) date convention (inflation for month t counts as
   time-t information), which the paper adopts in footnote 33. Do not add
   another lag on top.
 - Target: mkt_excess = vwretd - Rfree, subtracted on the same row. vwretd
   is the CRSP value-weighted return including dividends; Rfree at row t is
   the workbook's risk-free return over month t. The paper names the target
   ("the monthly excess return of the CRSP value-weighted index") without
   specifying the risk-free series; this is the standard Goyal-Welch
   equity-premium construction.
 - Timing: row t holds values known at the end of month t and forecasts the
   month t+1 return. `lag_mkt_excess` therefore EQUALS `mkt_excess` on the
   same row; the model layer applies the single uniform shift when it pairs
   predictors at t with the return at t+1. Nothing here is pre-shifted, so
   shifting this column again downstream would be a bug.
 - Known head-of-sample gaps, kept as NaN rather than dropped: `ntis` is a
   trailing 12-month sum of net issuance and only starts in 1926-12 in the
   workbook (1926-01 through 1926-11 are NaN); `dy` loses its first month
   when the raw pull starts exactly in 1926-01. Every other column is
   complete over 1926-2024. Both gaps fall inside the 36-month
   standardization burn-in, so the 1930 analysis start is unaffected.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from settings import config
from pull_CRSP_stock import load_CRSP_index_files
from pull_goyal_welch import load_goyal_welch

DATA_DIR = Path(config("DATA_DIR"))

# The 15 predictors in the order of the paper's footnote 33.
PREDICTOR_COLUMNS = [
    "dfy",
    "infl",
    "svar",
    "de",
    "lty",
    "tms",
    "tbl",
    "dfr",
    "dp",
    "dy",
    "ltr",
    "ep",
    "bm",
    "ntis",
    "lag_mkt_excess",
]


def build_kmz_dataset(gw=None, msix=None, data_dir=DATA_DIR):
    """Merge the raw pulls and construct the tidy KMZ dataset.

    `gw` and `msix` allow injecting dataframes (for tests); by default the
    saved parquets are loaded from `data_dir`. Returns one row per month with
    columns [date, mkt_excess] + PREDICTOR_COLUMNS, restricted to months with
    a CRSP value-weighted return (1926-01 onward). The prior-month price for
    `dy` is taken before that restriction, so the first retained month keeps
    its lag whenever the raw pull extends further back.
    """
    if gw is None:
        gw = load_goyal_welch(data_dir=data_dir)
    if msix is None:
        msix = load_CRSP_index_files(data_dir=data_dir)

    gw = gw.sort_values("date").reset_index(drop=True)

    # msix `caldt` is the last trading day of the month; align to month-end.
    index = msix[["caldt", "vwretd"]].copy()
    index["date"] = pd.to_datetime(index["caldt"]) + MonthEnd(0)

    df = gw.merge(
        index[["date", "vwretd"]], on="date", how="left", validate="one_to_one"
    )

    out = pd.DataFrame({"date": df["date"]})
    out["mkt_excess"] = df["vwretd"] - df["Rfree"]
    out["dfy"] = df["BAA"] - df["AAA"]
    out["infl"] = df["infl"]
    out["svar"] = df["svar"]
    out["de"] = np.log(df["d12"]) - np.log(df["e12"])
    out["lty"] = df["lty"]
    out["tms"] = df["lty"] - df["tbl"]
    out["tbl"] = df["tbl"]
    out["dfr"] = df["corpr"] - df["ltr"]
    out["dp"] = np.log(df["d12"]) - np.log(df["price"])
    out["dy"] = np.log(df["d12"]) - np.log(df["price"].shift(1))
    out["ltr"] = df["ltr"]
    out["ep"] = np.log(df["e12"]) - np.log(df["price"])
    out["bm"] = df["b/m"]
    out["ntis"] = df["ntis"]
    out["lag_mkt_excess"] = out["mkt_excess"]

    out = out.loc[df["vwretd"].notna()].reset_index(drop=True)
    return out[["date", "mkt_excess"] + PREDICTOR_COLUMNS]


def load_kmz_dataset(data_dir=DATA_DIR):
    """Load the saved tidy KMZ dataset."""
    path = Path(data_dir) / "kmz_dataset.parquet"
    return pd.read_parquet(path)


if __name__ == "__main__":
    df = build_kmz_dataset(data_dir=DATA_DIR)
    path = Path(DATA_DIR) / "kmz_dataset.parquet"
    df.to_parquet(path)
