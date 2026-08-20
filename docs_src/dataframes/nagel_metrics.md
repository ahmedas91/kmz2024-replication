## Description

The Nagel-critique comparison table's tidy input: two rows, the replicated VoC strategy (ridgeless, c = 1000, across-seed mean forecast at the anchor configuration) and the recency-weighted momentum benchmark, scored with IDENTICAL metric definitions (`voc.nagel.benchmark_metrics`, pinned equal to `voc.performance_metrics.compute_metrics` by a parity test). Columns: `strategy`, `r2`, `mean_return`, `volatility`, `sharpe`, `alpha`, `information_ratio`, `alpha_tstat`. Built by `doit nagel` from the anchor forecast export (`doit export_forecasts`); keyed by the sample period.

Note: the VoC row scores the across-seed MEAN forecast path (denoised), so it sits slightly above the average-the-statistics values in `oos_grid_T12`; see the figure_nagel page for the full convention note.
