"""Volatility-standardization steps for the reusable VoC engine.

:func:`voc.oos_engine.run_voc_study` expects standardized inputs; these produce
them for ANY ``(target returns, predictors)`` pair using the KMZ conventions, both
strictly backward-looking (no lookahead):

- target returns: trailing ``window``-month UNCENTERED volatility, sqrt(mean(r^2))
  (footnote 34 — mean monthly returns are too noisy to estimate in short windows);
- predictors: expanding-window standard deviation (ddof=1) of all strictly-prior
  observations, with a ``min_periods`` burn-in.

A new asset-class study standardizes its own data with :func:`standardize_inputs`
and feeds the result (dropping the burn-in rows) to the engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trailing_uncentered_vol(returns, window=12):
    """Trailing uncentered volatility ``sqrt(mean(r^2))`` over the prior `window`
    observations (NaN until `window` are available); strictly backward-looking."""
    r = pd.Series(np.asarray(returns, dtype=np.float64))
    return r.pow(2).rolling(window).mean().pow(0.5).shift(1).to_numpy()


def expanding_vol(series, min_periods=36):
    """Expanding-window standard deviation (ddof=1) of all observations strictly
    before each row, NaN until `min_periods` have accumulated."""
    s = pd.Series(np.asarray(series, dtype=np.float64))
    return s.expanding(min_periods=min_periods).std().shift(1).to_numpy()


def standardize_inputs(target, predictors, window=12, min_periods=36):
    """Standardize a (target, predictors) pair for a VoC study.

    Returns ``(target_std, predictors_std, valid)`` where ``valid`` is the boolean
    row mask on which every standardized value is finite (the burn-in rows are
    False). Callers typically keep ``target_std[valid]`` / ``predictors_std[valid]``.
    """
    R = np.asarray(target, dtype=np.float64).ravel()
    G = np.asarray(predictors, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError("predictors must be 2-D (n, d)")
    target_std = R / trailing_uncentered_vol(R, window)
    scales = np.column_stack(
        [expanding_vol(G[:, j], min_periods) for j in range(G.shape[1])]
    )
    predictors_std = G / scales
    valid = np.isfinite(target_std) & np.isfinite(predictors_std).all(axis=1)
    # The documented caller pattern is target_std[valid]; the engine's t -> t+1
    # recursion assumes CONSECUTIVE months, so a mask that goes False
    # mid-sample (a NaN month, or a zero-volatility stretch) would silently
    # splice non-adjacent months together. Refuse rather than mislead.
    if valid.any():
        first = int(np.argmax(valid))
        if not valid[first:].all():
            raise ValueError(
                "standardized values are non-finite mid-sample (a NaN or "
                "zero-volatility stretch after the burn-in); masking with "
                "`valid` would splice non-adjacent months into the engine's "
                "consecutive-month recursion. Clean or truncate the inputs."
            )
    return target_std, predictors_std, valid
