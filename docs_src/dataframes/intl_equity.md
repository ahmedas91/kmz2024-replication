## Description

The monthly developed-ex-US market factors from the Kenneth French data library (`Developed_ex_US_3_Factors`), pulled via `pandas_datareader` — public, no credentials, so unlike the WRDS CRSP pull anyone can rerun it (`doit pull:intl_equity`, `src/pull_intl_equity.py`). The series starts 1990-07 and is stored through the configured `END_DATE` (currently 2024-12), converted from percent to decimal with a month-end `date` column.

`Mkt-RF` is stored as `intl_mkt_excess` and is ALREADY an excess return — the value-weighted developed-ex-US market return minus the US one-month T-bill — so downstream code never subtracts anything; `RF` is kept alongside as `rf` for reference and unit tests. This is the target of the international VoC study (issue #16), whose sample is capped by the 1990-07 start at roughly a third of the market study's months.

## Data Dictionary

One row per month:

- `date`: month-end timestamp.
- `intl_mkt_excess`: developed-ex-US market return in excess of the US one-month T-bill, decimal.
- `rf`: US one-month T-bill return, decimal.
