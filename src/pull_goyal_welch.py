"""Pull the updated Goyal-Welch predictor dataset from Amit Goyal's website.

Downloads the combined workbook "All data up to 2025" (Goyal and Welch (2008)
extended by Goyal, Welch and Zafirov (2024)) hosted on Amit Goyal's website,
https://sites.google.com/view/agoyal145, and stores the raw Monthly sheet.
This file supplies 14 of the paper's 15 predictors plus the risk-free rate.

Notes on the workbook (checked against its ReadMe sheet):
 - Sheets: ReadMe, Monthly, Quarterly, Annual. Only Monthly is stored here.
 - Monthly spans 1871-01 through 2025-12 (57 columns).
 - Column names are lowercase: the S&P 500 index level is `price` (not
   `Index`), 12-month dividends and earnings are `d12` and `e12`.
 - `ret` and `retx` are CRSP's calculation of the S&P 500 return including
   and excluding dividends, available from 1926-01. Useful as a cross-check
   on the CRSP value-weighted index series pulled from WRDS.
 - The derived ratios (`d/p`, `d/y`, `e/p`, `d/e`, `tms`, `dfy`, `dfr`) come
   precomputed in the file.
 - `infl` follows the Goyal, Welch and Zafirov (2024) date convention
   (inflation for month t is treated as time-t information), which is the
   convention Kelly, Malamud and Zhou (2024) adopt (their footnote 33).

This module stores the RAW sheet only, untrimmed: the full 1871-01 through
2025-12 span is saved regardless of START_DATE/END_DATE, so changing those
settings never requires (or silently misses) a re-pull. Sample bounds are
applied downstream — the tidy module's inner merge with CRSP bounds the start
at 1926-01, and the standardization module floors the analysis sample at
1930-01. Predictor construction and all other transformations likewise happen
in the tidy/cleaning module.
"""

from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from pandas.tseries.offsets import MonthEnd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))

# Export link for the "All data up to 2025" workbook on
# https://sites.google.com/view/agoyal145 (public, no credentials needed).
# Goyal replaces the file with a new one when he posts a new vintage, so if
# this 404s, grab the new spreadsheet id from the site.
GOYAL_WELCH_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "17mw_IpaiLFDrGnrPRQ2o1ugV5nJsZuD1/export?format=xlsx"
)


def pull_goyal_welch(url=GOYAL_WELCH_URL):
    """Download the workbook and return its full, untrimmed Monthly sheet.

    Adds a month-end `date` column parsed from `yyyymm` (month-end to match
    the CRSP `caldt` convention, which the tidy module merges on) and keeps
    every raw column unchanged. No date filtering happens here: sample
    bounds are a downstream concern (see module docstring).
    """
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content), sheet_name="Monthly")
    df["date"] = pd.to_datetime(df["yyyymm"], format="%Y%m") + MonthEnd(0)
    return df.reset_index(drop=True)


def load_goyal_welch(data_dir=DATA_DIR):
    """Load the saved raw Goyal-Welch monthly dataset."""
    path = Path(data_dir) / "goyal_welch.parquet"
    return pd.read_parquet(path)


if __name__ == "__main__":
    df = pull_goyal_welch()
    path = Path(DATA_DIR) / "goyal_welch.parquet"
    df.to_parquet(path)
