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

This module stores the RAW sheet only. Predictor construction and all other
transformations happen in the tidy/cleaning module.
"""

from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from pandas.tseries.offsets import MonthEnd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")

# Export link for the "All data up to 2025" workbook on
# https://sites.google.com/view/agoyal145 (public, no credentials needed).
# Goyal replaces the file with a new one when he posts a new vintage, so if
# this 404s, grab the new spreadsheet id from the site.
GOYAL_WELCH_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "17mw_IpaiLFDrGnrPRQ2o1ugV5nJsZuD1/export?format=xlsx"
)


def pull_goyal_welch(start_date=START_DATE, end_date=END_DATE, url=GOYAL_WELCH_URL):
    """Download the workbook and return its Monthly sheet trimmed to dates.

    Adds a month-end `date` column parsed from `yyyymm` (month-end to match
    the CRSP `caldt` convention, which the tidy module merges on) and keeps
    every raw column unchanged.
    """
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content), sheet_name="Monthly")
    df["date"] = pd.to_datetime(df["yyyymm"], format="%Y%m") + MonthEnd(0)
    mask = (df["date"] >= pd.Timestamp(start_date)) & (
        df["date"] <= pd.Timestamp(end_date)
    )
    return df.loc[mask].reset_index(drop=True)


def load_goyal_welch(data_dir=DATA_DIR):
    """Load the saved raw Goyal-Welch monthly dataset."""
    path = Path(data_dir) / "goyal_welch.parquet"
    return pd.read_parquet(path)


if __name__ == "__main__":
    df = pull_goyal_welch(start_date=START_DATE, end_date=END_DATE)
    path = Path(DATA_DIR) / "goyal_welch.parquet"
    df.to_parquet(path)
