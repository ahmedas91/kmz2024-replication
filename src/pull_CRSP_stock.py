"""
Pull the monthly CRSP index file (MSIX) from WRDS.

The pipeline needs only the value-weighted index return series (vwretd) as a
cross-check on the workbook market series, so this module pulls just the
index table; the full monthly stock file the template pulled had no
consumers and was removed in the project checkup.

This module uses the CRSP CIZ format (Flat File Format 2.0), which replaced
the legacy SIZ format as of January 2025.

Key resources:
 - Data for indices: https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_a_indexes/
 - Tidy Finance guide: https://www.tidy-finance.org/python/wrds-crsp-and-compustat.html
 - CRSP 2.0 Update: https://www.tidy-finance.org/blog/crsp-v2-update/
 - Transition FAQ: https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/crsp/stocks-and-indices/crsp-stock-and-indexes-version-2/crsp-ciz-faq/
 - Cross-Reference Guide: https://www.crsp.org/wp-content/uploads/guides/CRSP_Cross_Reference_Guide_1.0_to_2.0.pdf

Key changes from SIZ to CIZ format:
 - Monthly stock table: crspm.msf -> crspm.msf_v2
 - Security info: crspm.msenames -> crspm.stksecurityinfohist
 - Delisting returns are now built into mthret (no separate table needed)
 - Column names: date->mthcaldt, ret->mthret, retx->mthretx, prc->mthprc
 - Share code filters (shrcd) replaced with securitytype, securitysubtype, sharetype

Thank you to Tobias Rodriguez del Pozo for his assistance in writing this code.

"""

from pathlib import Path

import pandas as pd
import wrds

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")


def pull_CRSP_index_files(
    start_date=START_DATE, end_date=END_DATE, wrds_username=WRDS_USERNAME
):
    """
    Pulls the CRSP index files from crsp_a_indexes.msix:
    (Monthly) NYSE/AMEX/NASDAQ Capitalization Deciles, Annual Rebalanced (msix)

    Note: The index tables (crsp_a_indexes) were not significantly changed
    in the CIZ transition. The column names and structure remain the same.
    """
    query = f"""
        SELECT *
        FROM crsp_a_indexes.msix
        WHERE caldt BETWEEN '{start_date}' AND '{end_date}'
    """
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=["caldt"])
    db.close()
    return df


def load_CRSP_index_files(data_dir=DATA_DIR):
    """Read the cached CRSP MSIX parquet from ``data_dir``."""
    path = Path(data_dir) / "CRSP_MSIX.parquet"
    df = pd.read_parquet(path)
    return df


if __name__ == "__main__":
    df_msix = pull_CRSP_index_files(start_date=START_DATE, end_date=END_DATE)
    path = Path(DATA_DIR) / "CRSP_MSIX.parquet"
    df_msix.to_parquet(path)
    print(f"[crsp] wrote CRSP_MSIX.parquet ({len(df_msix)} rows)")
