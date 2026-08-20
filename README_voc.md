# kmz-voc

A small, reusable **Virtue-of-Complexity** return-prediction engine, implementing
the method of Kelly, Malamud & Zhou (2024), *The Virtue of Complexity in Return
Prediction*. Random Fourier features expand a modest predictor set into a large
nonlinear feature space; dual-form (kernel) ridge fits them across a
complexity × shrinkage grid; and the out-of-sample market-timing statistics are
averaged across feature-draw seeds.

The distribution is named `kmz-voc`; the importable package is `voc`.

```bash
pip install kmz-voc
```

## Quick start

`run_voc_study` runs a virtue-of-complexity study on **any** `(target, predictor
set)` pair: the market study is one configuration, and other asset classes or
experiments are others. Inputs are expected already volatility-standardized;
`voc.standardize_inputs` provides the strictly backward-looking standardization.

```python
import numpy as np
from voc import standardize_inputs, run_voc_study

# your own monthly target returns and predictors, row t = month t
target = ...        # (n,)
predictors = ...    # (n, d)

t, X, valid = standardize_inputs(target, predictors)
per_seed, averaged = run_voc_study(
    t[valid], X[valid],
    p_grid=(2, 12, 120, 1200, 12000),   # complexity grid P = c * T
    z_grid=(1e-3, 1e-1, 1e1, 1e3),       # ridge shrinkage levels
    seeds=range(50),
    n_jobs=-1,
    study_name="my_study",
)
# `averaged` has one row per (P, z): out-of-sample R^2, Sharpe, alpha, IR, ...
```

> **Note:** a deeply negative out-of-sample R² is expected here — it deepens as
> complexity rises even while the Sharpe ratio climbs. That's the paper's finding
> (R² tracks forecast scale, profits track forecast direction), not a bug in your run.

## What's inside

- **`voc.rff`** — the random Fourier feature map (paper eq. 20): draw the Gaussian
  weights, project the predictors through the paired sin/cos map, and scale by
  training-window volatility. Nested draws let one maximum-size draw serve every
  smaller model in the complexity grid.
- **`voc.kernel_ridge`** — ridge in its dual (kernel) form, so each fit is a
  T × T solve (12 × 12) even at P = 12,000, plus the ridgeless (minimum-norm)
  limit and prediction. No intercept.
- **`voc.oos_engine`** — the recursive out-of-sample loop (`run_recursive_oos`), the
  generic study runner `run_voc_study` (and the DataFrame convenience `run_grid`),
  and an optional per-seed forecast export. Parallelized across seeds with joblib.
- **`voc.performance_metrics`** — out-of-sample R² against the zero-forecast
  benchmark, the annualized Sharpe ratio, and alpha / information ratio /
  t-statistic versus a static position in the standardized market.
- **`voc.preprocessing`** — the strictly backward-looking volatility standardization
  a new study applies to its inputs (`standardize_inputs`, plus the trailing- and
  expanding-window volatility building blocks).
- **`voc.nagel`** — Nagel's (2025) momentum critique as reusable tests: the
  declining-weight benchmark, the forecast-anatomy regression, and the spanning
  regression.

## Reference

Kelly, B., Malamud, S., & Zhou, K. (2024). *The Virtue of Complexity in Return
Prediction.* The Journal of Finance.

## License

MIT, see [LICENSE](LICENSE).
