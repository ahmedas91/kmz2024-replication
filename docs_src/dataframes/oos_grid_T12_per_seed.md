## Description

The per-repetition version of the OOS grid results (`oos_grid_T12`): one row per (`seed`, `P`, `z`), where each row holds the statistics of one RFF repetition before across-seed averaging. Built by the same `doit estimate` run as the averaged frame. Its purposes are seed-noise diagnostics (how much the statistics move across RFF draws, which sets the tolerances of the Figure 8 anchor tests) and any analysis that needs the seed dimension explicitly.

## Data Dictionary

One row per (`seed`, `P`, `z`):

- `seed`: RFF repetition index (`range(N_SEEDS)`; default 50 seeds).
- All other columns are as in `oos_grid_T12` (see that page): `P`, `z`, `c`, `r2`, `beta_norm`, `mean_return`, `volatility`, `sharpe`, `alpha`, `information_ratio`, `alpha_tstat` — computed within this seed only.

Averaging the statistics (not the forecasts) across seeds reproduces `oos_grid_T12` exactly.
