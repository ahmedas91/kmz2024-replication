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

Multi-z reuse
-------------
:func:`ridge_dual` accepts several ``z`` at once and reuses a SINGLE symmetric
eigendecomposition of the Gram matrix ``S S'`` across all of them (the figure grid
sweeps 7 shrinkage levels per window, so this is a ~7x saving): with
``S S' = V diag(lambda) V'``, ``beta(z) = S' V diag(1 / (z*T + lambda)) V' R``.
"""

from __future__ import annotations

import numpy as np


def ridge_dual(S_train: np.ndarray, R_train: np.ndarray, z) -> np.ndarray:
    """KMZ ridge coefficients via the dual (T x T) form; no intercept.

    Parameters
    ----------
    S_train : np.ndarray
        ``(T, P)`` training features.
    R_train : np.ndarray
        ``(T,)`` training targets.
    z : float or 1-D array-like of float
        Ridge shrinkage level(s). When array-like, one symmetric
        eigendecomposition of the Gram matrix is reused across all values.

    Returns
    -------
    beta : np.ndarray
        ``(P,)`` if ``z`` is scalar, else ``(n_z, P)`` with row i for ``z[i]``.
        ``beta(z) = S' (z*T*I_T + S S')^{-1} R``.
    """
    S = np.asarray(S_train, dtype=np.float64)
    R = np.asarray(R_train, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError("S_train must be 2-D (T, P)")
    if R.shape != (S.shape[0],):
        raise ValueError("R_train must have shape (T,) matching S_train rows")
    n_obs = S.shape[0]
    scalar = np.ndim(z) == 0
    z_values = np.atleast_1d(np.asarray(z, dtype=np.float64))

    # One symmetric eigendecomposition of the Gram matrix, reused across z.
    gram = S @ S.T  # (T, T), symmetric positive semidefinite
    eigvals, eigvecs = np.linalg.eigh(gram)
    projected = eigvecs.T @ R  # (T,)

    betas = np.empty((z_values.size, S.shape[1]), dtype=np.float64)
    for i, z_i in enumerate(z_values):
        # (z*T*I + S S')^{-1} R in the eigenbasis, then lift to P-space.
        dual = eigvecs @ (projected / (z_i * n_obs + eigvals))
        betas[i] = S.T @ dual
    return betas[0] if scalar else betas


def ridgeless(S_train: np.ndarray, R_train: np.ndarray) -> np.ndarray:
    """Ridgeless (z -> 0) minimum-norm coefficients via the pseudoinverse.

    Returns the minimum-norm least-squares solution ``pinv(S) @ R`` — the
    interpolating solution of smallest L2 norm when P > T, and the ordinary
    least-squares fit when P < T. The Figure 8 anchors are quoted for this
    ridgeless case, so it gets its own numerically stable route rather than
    :func:`ridge_dual` with a tiny z (which is ill-conditioned once the Gram
    matrix is singular).
    """
    S = np.asarray(S_train, dtype=np.float64)
    R = np.asarray(R_train, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError("S_train must be 2-D (T, P)")
    if R.shape != (S.shape[0],):
        raise ValueError("R_train must have shape (T,) matching S_train rows")
    return np.linalg.pinv(S) @ R


def predict(S_new: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Forecast as a plain dot product with the coefficients (no intercept).

    Parameters
    ----------
    S_new : np.ndarray
        ``(P,)`` or ``(m, P)`` out-of-sample features.
    beta : np.ndarray
        ``(P,)`` coefficient vector.

    Returns
    -------
    forecast : np.ndarray or float
        Scalar for a single feature row, else ``(m,)``.
    """
    return np.asarray(S_new, dtype=np.float64) @ np.asarray(beta, dtype=np.float64)
