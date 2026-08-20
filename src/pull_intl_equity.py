"""Pull the developed-ex-US market excess return from the Ken French library.

Downloads the monthly ``Developed_ex_US_3_Factors`` dataset from Kenneth
French's data library via ``pandas_datareader`` (public, no credentials —
unlike the WRDS CRSP pull, anyone can rerun this). The series is the
value-weighted return of the developed markets outside the US and starts
1990-07, which caps the international study's sample at roughly a third of
the market study's months; the study pages document that caveat.

Conventions applied here so downstream code never worries about them:

- ``Mkt-RF`` is ALREADY an excess return (the developed-ex-US market return
  minus the US one-month T-bill), so it is stored as ``intl_mkt_excess``
  with no further subtraction; ``RF`` is kept alongside as ``rf`` for
  reference and unit tests.
- The library serves percent units; both columns are converted to decimal.
- The reader's monthly PeriodIndex becomes a month-end ``date`` column,
  matching every other dataset in the pipeline.

The famafrench reader requires explicit start/end dates (its default start
is recent); this pull uses the project-wide ``START_DATE``/``END_DATE``
config, so the saved span follows the same window as the CRSP pull and the
raw file simply starts when the series does.

Writes ``_data/intl_equity.parquet``.
"""

from pathlib import Path

import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")

DATASET = "Developed_ex_US_3_Factors"


def pull_intl_equity(start_date=START_DATE, end_date=END_DATE):
    """Download the monthly developed-ex-US factors; return the tidy frame."""
    import pandas_datareader.data as web

    tables = web.DataReader(DATASET, "famafrench", start=start_date, end=end_date)
    monthly = tables[0]  # table 0 is monthly, table 1 annual
    out = pd.DataFrame(
        {
            "date": monthly.index.to_timestamp(how="end").normalize(),
            "intl_mkt_excess": monthly["Mkt-RF"].to_numpy() / 100.0,
            "rf": monthly["RF"].to_numpy() / 100.0,
        }
    ).reset_index(drop=True)
    return out


def load_intl_equity(data_dir=DATA_DIR):
    """Read the cached international equity parquet from ``data_dir``."""
    return pd.read_parquet(Path(data_dir) / "intl_equity.parquet")


if __name__ == "__main__":
    intl = pull_intl_equity()
    intl.to_parquet(DATA_DIR / "intl_equity.parquet")
    span = f"{intl['date'].min():%Y-%m} to {intl['date'].max():%Y-%m}"
    print(f"[intl] wrote intl_equity.parquet ({len(intl)} months, {span})")
