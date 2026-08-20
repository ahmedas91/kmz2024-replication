## Description

The across-seed averaged out-of-sample statistics of the international equities VoC study (issue #16): the full complexity-by-shrinkage grid (T = 12, the market study's P and z grids, ridgeless included — no P-grid trim was needed on the short sample) run on the developed-ex-US market excess return through the generic `voc.run_voc_study` API. One row per (`target`, `P`, `z`), with the single target `intl_mkt_excess`; the `target` column keeps the schema identical to the bonds cache.

Configuration mirrors the bonds studies: the 14 standardized US Goyal-Welch predictors plus the target's own lag (same-row convention), target standardized by trailing 12-month uncentered volatility, statistics averaged across `INTL_N_SEEDS = 200` fixed seeds. Sample: 1991-07 (the target's 12-month volatility burn-in on the 1990-07 series start) through `SAMPLE_END` — 354 months on the paper period, about a third of the market study's 1,092, which is one of the study's two stated caveats (the other: US predictors forecasting non-US returns). Per-seed inputs live in `oos_grid_intl_T12_per_seed.parquet`.

Built by `doit intl_study` (`src/run_intl_study.py`). Columns beyond `target` match `oos_grid_T12` (see its data dictionary).
