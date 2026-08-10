## Description

The analysis-ready dataset for the KMZ replication: the tidy dataset (`kmz_dataset`) after the paper's backward-looking volatility standardization (Section V.A plus footnote 34). Same 17 columns; the sample runs from 1930-01 (the paper's stated analysis start, reached after the standardization burn-in) through the configured end date. Built by `src/standardize_kmz.py` (doit task `standardize`).

Two schemes, both strictly backward-looking so nothing dated at or after month t enters the scale applied at t:

- **Returns** (`mkt_excess` and `lag_mkt_excess`): divided by the trailing 12-month standard deviation computed from the uncentered second moment of the 12 months strictly before t.
- **Predictors** (the other 14 columns): divided by an expanding-window sample standard deviation of all observations strictly before t, requiring at least 36 prior observations.

No demeaning is applied anywhere; standardization divides only. Full indexing details are in the module docstring of `src/standardize_kmz.py`.

## Data Dictionary

Columns are identical in name and order to `kmz_dataset` (see that page for the definitions): `date`, `mkt_excess`, then the 15 predictors dfy, infl, svar, de, lty, tms, tbl, dfr, dp, dy, ltr, ep, bm, ntis, lag_mkt_excess. Every value except `date` is the corresponding `kmz_dataset` value divided by its backward-looking volatility scale. `lag_mkt_excess` is standardized as a return and still equals `mkt_excess` on the same row.
