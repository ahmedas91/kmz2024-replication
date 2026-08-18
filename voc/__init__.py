"""voc - a reusable Virtue-of-Complexity return-prediction engine (KMZ 2024).

One generic entry point, :func:`run_voc_study`, runs a virtue-of-complexity study
on ANY ``(target series, predictor set)`` pair: random Fourier features expand the
predictors, dual-form ridge fits them across a complexity x shrinkage grid, and
the out-of-sample market-timing statistics are averaged across RFF seeds. The KMZ
market replication is one configuration; asset-class studies (bonds, international)
and experiments (e.g. the Nagel counterfactual) are other configurations.

Building blocks: :mod:`voc.rff`, :mod:`voc.kernel_ridge`, :mod:`voc.oos_engine`,
:mod:`voc.performance_metrics`, and :mod:`voc.preprocessing` (the standardization
steps callers apply first, since the engine EXPECTS standardized inputs).

Example
-------
>>> import numpy as np
>>> from voc import run_voc_study
>>> rng = np.random.default_rng(0)
>>> target = rng.standard_normal(60)           # standardized target returns
>>> predictors = rng.standard_normal((60, 4))  # standardized predictors
>>> per_seed, averaged = run_voc_study(
...     target, predictors, p_grid=(2, 8), z_grid=(1.0,),
...     seeds=range(2), include_ridgeless=False)
>>> len(averaged)  # one row per (P, z) cell: 2 P x 1 z
2
"""

from voc.oos_engine import run_grid, run_recursive_oos, run_voc_study
from voc.performance_metrics import compute_metrics
from voc.preprocessing import (
    expanding_vol,
    standardize_inputs,
    trailing_uncentered_vol,
)

__all__ = [
    "run_voc_study",
    "run_grid",
    "run_recursive_oos",
    "compute_metrics",
    "standardize_inputs",
    "trailing_uncentered_vol",
    "expanding_vol",
]
