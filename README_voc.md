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
set)` pair — the market study is one configuration, and other asset classes or
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

Deeply **negative out-of-sample R²** alongside a **rising Sharpe ratio** as
complexity grows is the paper's central finding, not a bug: R² is dominated by
forecast scale while trading profits load on forecast direction.

## What's inside

- `voc.rff` — the random Fourier feature map (paper eq. 20), with nested draws.
- `voc.kernel_ridge` — dual-form (T × T) ridge and its ridgeless limit; no intercept.
- `voc.oos_engine` — the recursive out-of-sample loop and `run_voc_study`.
- `voc.performance_metrics` — OOS R², Sharpe, alpha / IR / t-stat.
- `voc.preprocessing` — volatility standardization steps for new studies.
- `voc.nagel` — tools to run Nagel's (2025) momentum critique on a study.

## Reference

Kelly, B., Malamud, S., & Zhou, K. (2024). *The Virtue of Complexity in Return
Prediction.* The Journal of Finance.

## License

MIT — see [LICENSE](LICENSE).
