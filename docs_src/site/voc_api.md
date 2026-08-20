# The reusable VoC API

The estimation engine is a small installable package, `voc` (`pip install -e .`
from the repository root), with one generic entry point:

```python
voc.run_voc_study(target, predictors, *, dates=None, T=12, p_grid=..., z_grid=...,
                  seeds=range(50), gamma=2.0, include_ridgeless=True,
                  uncentered=False, n_jobs=1,
                  save_forecasts=False, study_name="market", data_dir=None)
```

It runs a full virtue-of-complexity study — random Fourier features, dual-form
ridge across the complexity-by-shrinkage grid, recursive out-of-sample timing
strategy, statistics averaged across RFF seeds — on ANY `(target, predictors)`
pair, and returns `(per_seed, averaged)` long-format statistics frames. The
KMZ market replication (`doit estimate`) is exactly one configuration of this
call; the bond and international studies are others; `voc.oos_engine.run_grid`
is a thin DataFrame wrapper in the market schema.

Two contracts to know:

1. **Inputs arrive already volatility-standardized.** The engine never
   standardizes your data for you, so no lookahead can be smuggled in on your
   behalf. `voc.preprocessing` provides the KMZ conventions as reusable steps:
   `trailing_uncentered_vol` (targets: trailing 12-month `sqrt(mean(r^2))`,
   strictly prior), `expanding_vol` (predictors: expanding-window SD of
   strictly-prior observations, 36-month burn-in), and `standardize_inputs`,
   which applies both and returns the valid-row mask. These mirror
   `src/standardize_kmz.py` exactly, and a cross-guard test keeps them from
   drifting.
2. **The seed list is the identity of a run.** Statistics are computed within
   each seed and averaged across seeds (the paper's convention); the same
   seed list reproduces a run bit for bit.

A complete synthetic study:

```python
import numpy as np
from voc import run_voc_study, standardize_inputs

rng = np.random.default_rng(0)
raw_target = rng.standard_normal(240)
raw_predictors = rng.standard_normal((240, 4))

t, X, valid = standardize_inputs(raw_target, raw_predictors)
per_seed, averaged = run_voc_study(
    t[valid], X[valid], p_grid=(2, 8, 24), z_grid=(0.1, 1000.0), seeds=range(10)
)
```

Setting `save_forecasts=True` (with a `data_dir`) additionally writes the
per-seed forecast, realized-return, and strategy-return series of every
`(P, z)` cell to `forecasts_<study_name>.parquet` — the input the Nagel
counterfactual needs. The file grows as seeds x cells x months, so keep the
grid small when exporting (the market driver exports only the anchor
configuration, under `SAVE_FORECASTS=1`).
