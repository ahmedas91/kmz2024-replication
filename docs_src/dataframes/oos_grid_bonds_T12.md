## Description

The across-seed averaged out-of-sample statistics of the bonds VoC study (issue #15): the full complexity-by-shrinkage grid (the market study's P and z grids at T = 12, ridgeless included) run once per bond target through the generic `voc.run_voc_study` API, on the paper-period sample (1930-01 through `SAMPLE_END`). One row per (`target`, `P`, `z`).

Each target's configuration mirrors the market study: the 14 standardized Goyal-Welch predictors from the shared analysis dataset plus the target's own lag (the trailing-vol-standardized bond excess return on the same row), target standardized by trailing 12-month uncentered volatility via `voc.preprocessing`. Statistics are computed within each RFF seed and averaged across `BONDS_N_SEEDS = 200` seeds (versus 500 for the market study; no published anchors exist for bonds, so the seed budget only needs smooth curves — the across-seed standard error sits well inside line width). The per-seed inputs live in `oos_grid_bonds_T12_per_seed.parquet` alongside.

Built by `doit bonds_study` (`src/run_bonds_study.py`). Columns beyond `target` match `oos_grid_T12` (see its data dictionary): `P`, `z` (0.0 = ridgeless), `c = P/T`, `r2`, `beta_norm`, `mean_return`, `volatility`, `sharpe`, `alpha`, `information_ratio`, `alpha_tstat`.
