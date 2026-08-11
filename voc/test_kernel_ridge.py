"""Tests for the dual-form kernel ridge module (issue #6).

Synthetic; no data on disk. They pin the estimator's identity: the dual form
equals the primal and matches sklearn's no-intercept ridge under the
``alpha = z*T`` mapping, the ridgeless limit is the minimum-norm least-squares
solution that a tiny-z ridge converges to, the multi-z path matches solving each
z alone, and prediction is an intercept-free dot product.
"""

import numpy as np
import pytest

from voc.kernel_ridge import predict, ridge_dual, ridgeless

RNG = np.random.default_rng(6060)


def _ridge_primal(S, R, z):
    """Reference: the KMZ estimator formed directly in the P x P primal space."""
    n_obs, n_feat = S.shape
    A = z * np.eye(n_feat) + S.T @ S / n_obs
    return np.linalg.solve(A, S.T @ R / n_obs)


@pytest.mark.parametrize("T,P", [(12, 5), (12, 40)])
def test_dual_equals_primal(T, P):
    """Dual (T x T) and primal (P x P) forms agree, both P < T and P > T."""
    S = RNG.standard_normal((T, P))
    R = RNG.standard_normal(T)
    for z in (1e-2, 1.0, 1e2):
        np.testing.assert_allclose(
            ridge_dual(S, R, z), _ridge_primal(S, R, z), rtol=1e-8, atol=1e-10
        )


def test_matches_sklearn_no_intercept():
    """ridge_dual matches sklearn Ridge(fit_intercept=False) with alpha = z*T
    on a P < T problem."""
    linear_model = pytest.importorskip("sklearn.linear_model")
    T, P = 12, 5
    S = RNG.standard_normal((T, P))
    R = RNG.standard_normal(T)
    z = 0.1
    model = linear_model.Ridge(alpha=z * T, fit_intercept=False).fit(S, R)
    np.testing.assert_allclose(
        ridge_dual(S, R, z), model.coef_, rtol=1e-8, atol=1e-10
    )


def test_ridgeless_is_min_norm_least_squares():
    """Ridgeless equals numpy's least-norm solution via an independent route."""
    T, P = 12, 40
    S = RNG.standard_normal((T, P))
    R = RNG.standard_normal(T)
    reference = np.linalg.lstsq(S, R, rcond=None)[0]
    np.testing.assert_allclose(ridgeless(S, R), reference, rtol=1e-8, atol=1e-10)


def test_tiny_z_ridge_converges_to_ridgeless():
    """ridge_dual with a vanishing z approaches the ridgeless solution (P > T)."""
    T, P = 12, 40
    S = RNG.standard_normal((T, P))
    R = RNG.standard_normal(T)
    np.testing.assert_allclose(
        ridge_dual(S, R, 1e-12), ridgeless(S, R), rtol=1e-5, atol=1e-7
    )


def test_multi_z_matches_individual_solves():
    """The eigendecomposition-reuse path returns the same betas, in order, as
    solving each z on its own."""
    T, P = 12, 60
    S = RNG.standard_normal((T, P))
    R = RNG.standard_normal(T)
    zs = [1e-3, 1e-1, 1.0, 1e1, 1e3]
    stacked = ridge_dual(S, R, zs)
    assert stacked.shape == (len(zs), P)
    for i, z in enumerate(zs):
        np.testing.assert_allclose(stacked[i], ridge_dual(S, R, z), rtol=1e-10)


def test_predict_is_intercept_free_dot_product():
    """predict is S_new @ beta, so a zero feature vector forecasts exactly 0."""
    T, P = 12, 20
    S = RNG.standard_normal((T, P))
    R = RNG.standard_normal(T)
    beta = ridge_dual(S, R, 1.0)
    S_new = RNG.standard_normal((3, P))
    np.testing.assert_allclose(predict(S_new, beta), S_new @ beta)
    assert predict(np.zeros(P), beta) == 0.0
