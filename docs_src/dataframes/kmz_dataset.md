## Description

The tidy analysis dataset for the replication of Kelly, Malamud and Zhou (2024), "The Virtue of Complexity in Return Prediction": the monthly excess return of the CRSP value-weighted index plus the paper's 15 predictors (Section V.A, footnote 33, listed in the footnote's order). One row per month from 1926-01. Row t holds values known at the end of month t and is used to forecast the month t+1 return; the model layer applies that single shift, so nothing in this file is pre-shifted.

Built by `src/clean_goyal_welch.py` (doit task `tidy`) from two raw inputs: the updated Goyal-Welch workbook from Amit Goyal's website and the CRSP monthly index file from WRDS. Valuation ratios are log ratios as in Goyal and Welch (2008), reconstructed from the raw levels (the workbook's own `d/p`-style columns are plain ratios and serve only as cross-checks in the tests). Full conventions and their sources are documented in the module docstring of `src/clean_goyal_welch.py`. No volatility standardization is applied here; that happens in a separate downstream module.

## Data Dictionary

- **date**: `datetime64[ns]` month-end timestamp
- **mkt_excess**: `float64` monthly excess return of the CRSP value-weighted index: `vwretd` (includes dividends) minus the Goyal-Welch risk-free return `Rfree` for the same month. Forecast target; also the base of the 15th predictor
- **dfy**: `float64` default yield spread: BAA minus AAA corporate bond yields
- **infl**: `float64` CPI (all urban consumers) inflation; Goyal-Welch-Zafirov dating, under which month-t inflation is time-t information (already encoded in the workbook; no extra lag applied)
- **svar**: `float64` stock variance: sum of squared daily S&P 500 returns within the month
- **de**: `float64` log dividend-payout ratio: log(D12) minus log(E12)
- **lty**: `float64` long-term government bond yield
- **tms**: `float64` term spread: lty minus tbl
- **tbl**: `float64` 3-month Treasury bill rate
- **dfr**: `float64` default return spread: long-term corporate bond return (corpr) minus long-term government bond return (ltr)
- **dp**: `float64` log dividend-price ratio: log(D12) minus log(price)
- **dy**: `float64` log dividend yield: log(D12) minus log(prior-month price); NaN in the first month when the raw pull starts exactly at the sample start
- **ltr**: `float64` long-term government bond return
- **ep**: `float64` log earnings-price ratio: log(E12) minus log(price)
- **bm**: `float64` book-to-market ratio (the workbook's `b/m`)
- **ntis**: `float64` net equity expansion: 12-month moving sum of net equity issues by NYSE-listed stocks over their market capitalization; starts 1926-12 (NaN 1926-01 through 1926-11, inside the 36-month standardization burn-in)
- **lag_mkt_excess**: `float64` the 15th predictor, one lag of the market excess return; equals `mkt_excess` on the same row by the timing convention above
