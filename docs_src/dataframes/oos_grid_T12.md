## Description

The across-seed averaged results of the recursive out-of-sample Virtue-of-Complexity grid (KMZ 2024, Section V.C) at training window T = 12: one row per model, i.e. per (complexity `P`, shrinkage `z`) cell, with every statistic computed within each RFF repetition (seed) and then averaged across seeds (the paper's average-the-statistics convention). Built by `doit estimate` (`src/run_estimation.py` driving `voc/oos_engine.py`) on the volatility-standardized dataset trimmed to the estimation sample (1930-01 through `SAMPLE_END`, default 2020-12, the paper's sample end).

The `z = 0.0` rows are the ridgeless (z -> 0 minimum-norm) limit, computed by its own numerically stable route; the positive-z rows follow the paper's grid log10(z) in {-3, ..., 3}. This is the results dataframe behind the Figure 7 and Figure 8 replications.

## Data Dictionary

One row per (`P`, `z`):

- `P`: number of random Fourier features (even; the paper's grid from 2 to 12,000).
- `z`: ridge shrinkage level; `0.0` denotes the ridgeless minimum-norm limit.
- `c`: model complexity `P / T`.
- `r2`: out-of-sample R^2, centered-variance convention (paper footnote 40).
- `beta_norm`: mean L2 norm of the fitted coefficient vector across training windows (Figure 7 Panel B).
- `mean_return`: mean monthly return of the timing strategy (forecast times realized return).
- `volatility`: monthly standard deviation of the strategy return.
- `sharpe`: annualized Sharpe ratio, sqrt(12) * mean / SD.
- `alpha`: monthly intercept from the OLS of the strategy return on a static position in the volatility-standardized market (Figure 8 caption).
- `information_ratio`: sqrt(12) * alpha / SD(residual).
- `alpha_tstat`: alpha over its OLS intercept standard error.

The number of seeds averaged is the `N_SEEDS` configuration of the `estimate` run (default 50; the paper averages 1,000). The per-seed inputs live in `oos_grid_T12_per_seed`.
