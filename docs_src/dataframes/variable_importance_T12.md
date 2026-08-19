## Description

The per-seed out-of-sample statistics behind the Figure 11 variable-importance replication (KMZ 2024, Section V.E): 16 configurations at the paper's Figure 11 setting (T = 12, P = 12,000, z = 10^3) — the full 15-predictor model plus one leave-one-out model per predictor — each run over the same fixed seed list (`range(FIG11_N_SEEDS)`, default 100; the paper averages 1,000). Built by `doit variable_importance` (`src/run_variable_importance.py` driving `voc/oos_engine.py`) on the volatility-standardized dataset trimmed to the estimation sample (1930-01 through `SAMPLE_END`, default 2020-12).

Random feature weights are redrawn per configuration because the RFF input dimension changes from 15 to 14 when a predictor is excluded; this is inherent to the method. Variable importance itself is computed downstream (`src/figure11.py`, `src/test_figure11.py`) as the across-seed mean statistic of the full model minus that of the leave-one-out model.

## Data Dictionary

One row per (`excluded`, `seed`):

- `excluded`: the predictor left out of this configuration; `"none"` is the full 15-predictor model.
- `seed`: RFF repetition index.
- `P`, `z`, `c`: model settings, constant here (12000, 1000.0, 1000.0).
- Remaining columns: the per-seed metrics of `voc/performance_metrics.py`, as in `oos_grid_T12_per_seed` (`r2`, `beta_norm`, `mean_return`, `volatility`, `sharpe`, `alpha`, `information_ratio`, `alpha_tstat`). Figure 11 uses `r2` and `sharpe`.
