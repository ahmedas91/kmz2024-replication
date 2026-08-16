"""Dual-form (kernel) ridge regression for the KMZ (2024) VoC engine.

Solves the paper's ridge estimator

    beta(z) = (z * I_P + S'S / T)^{-1} (S'R / T)                        (primal)

in its DUAL form, so the only linear system is T x T (T = 12) even when the
feature count P reaches 12,000. This is the computational trick that makes the
full complexity grid feasible. There is NO intercept, ever (footnote 35; defended
in the authors' 2025 reply, Section 4.2.2) — the features already span the
constant direction, so an intercept would double-count it.

Exact normalization (the z levels only line up with the paper if this matches)
------------------------------------------------------------------------------
Clearing the 1/T inside the inverse turns the primal estimator into

    beta(z) = (S'S + z*T*I_P)^{-1} S'R
            = S' (z*T*I_T + S S')^{-1} R,                               (dual)

using the push-through identity (z I_P + S'S/T)^{-1} S' = S' (z I_T + S S'/T)^{-1}.
So the shrinkage enters the T x T system as ``z * T`` (NOT ``z``), and
``sklearn.linear_model.Ridge(fit_intercept=False)`` reproduces this estimator with
``alpha = z * T`` — the mapping the tests encode.

Conventions
-----------
- ``z`` must be > 0. The z -> 0 minimum-norm limit is :func:`ridgeless`; the dual
  form is ill-conditioned at z = 0 once P < T (the Gram matrix is singular).
- :func:`ridge_dual` always returns ``(n_z, P)`` — one row per shrinkage level,
  even for a single ``z`` — so callers never have to branch on the return shape.
- Multi-z reuse: one symmetric eigendecomposition of ``S S'`` serves every z (the
  figure grid sweeps 7 levels per window, a ~7x saving): with
  ``S S' = V diag(lambda) V'``, ``beta(z) = S' V diag(1 / (z*T + lambda)) V' R``,
  evaluated for all z in a single GEMM.
"""

from __future__ import annotations

import numpy as np


def _check_training_arrays(S_train, R_train):
    """Coerce and validate a (design, target) training pair."""
    S = np.asarray(S_train, dtype=np.float64)
    R = np.asarray(R_train, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError("S_train must be 2-D (T, P)")
    if R.shape != (S.shape[0],):
        raise ValueError("R_train must have shape (T,) matching S_train rows")
    return S, R


def ridge_dual(S_train, R_train, z) -> np.ndarray:
    """KMZ ridge coefficients via the dual (T x T) form; no intercept.

    Parameters
    ----------
    S_train : np.ndarray
        ``(T, P)`` training features.
    R_train : np.ndarray
        ``(T,)`` training targets.
    z : float or 1-D array-like of float
        Ridge shrinkage level(s), all strictly positive. A single symmetric
        eigendecomposition of the Gram matrix is reused across every value. For
        the z -> 0 minimum-norm limit use :func:`ridgeless`.

    Returns
    -------
    beta : np.ndarray
        ``(n_z, P)`` — one row per z, always 2-D.
        ``beta(z) = S' (z*T*I_T + S S')^{-1} R``.
    """
    S, R = _check_training_arrays(S_train, R_train)
    z_values = np.atleast_1d(np.asarray(z, dtype=np.float64))
    if z_values.ndim != 1:
        raise ValueError("z must be a scalar or a 1-D sequence")
    if np.any(z_values <= 0.0):
        raise ValueError("z must be > 0; use ridgeless() for the z=0 minimum-norm limit")
    n_obs = S.shape[0]

    gram = S @ S.T  # (T, T), symmetric positive semidefinite
    eigvals, eigvecs = np.linalg.eigh(gram)
    # Clamp PSD round-off: eigh can return tiny negatives, and at P < T the null
    # space should be exactly 0, not ~1e-15 noise that corrupts small z.
    eigvals = np.maximum(eigvals, 0.0)
    projected = eigvecs.T @ R  # (T,)
    # (z*T*I + K)^{-1} R for every z at once, then lift to P-space in one GEMM.
    scaled = projected[:, None] / (z_values[None, :] * n_obs + eigvals[:, None])
    duals = eigvecs @ scaled  # (T, n_z)
    return (S.T @ duals).T  # (n_z, P)


def ridgeless(S_train, R_train) -> np.ndarray:
    """Ridgeless (z -> 0) minimum-norm coefficients.

    Returns ``lstsq(S, R)`` — the minimum-norm least-squares solution: the
    interpolating solution of smallest L2 norm when P > T, and the OLS fit when
    P < T. The Figure 8 anchors are quoted for this ridgeless case, so it gets its
    own numerically stable route rather than :func:`ridge_dual` with a tiny z.
    (``lstsq`` matches ``pinv(S) @ R`` without forming the (P, T) pseudoinverse.)
    """
    S, R = _check_training_arrays(S_train, R_train)
    return np.linalg.lstsq(S, R, rcond=None)[0]


def predict(S_new, beta) -> np.ndarray:
    """Forecast as a plain dot product with the coefficients (no intercept).

    Parameters
    ----------
    S_new : np.ndarray
        ``(P,)`` or ``(m, P)`` out-of-sample features.
    beta : np.ndarray
        ``(P,)`` coefficient vector for a single model. :func:`ridge_dual` returns
        one row per z, so index the row you want before predicting.

    Returns
    -------
    forecast : np.ndarray or float
        Scalar for a single feature row, else ``(m,)``.
    """
    beta = np.asarray(beta, dtype=np.float64)
    if beta.ndim != 1:
        raise ValueError("beta must be 1-D (P,); index one row of a multi-z result")
    return np.asarray(S_new, dtype=np.float64) @ beta
