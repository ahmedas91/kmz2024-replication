"""Tests for the dual-form kernel ridge module (issue #6).

Synthetic; no data on disk. They pin the estimator's identity: the dual form
equals the primal and matches sklearn's no-intercept ridge under ``alpha = z*T``;
the ridgeless limit is the minimum-norm least-squares solution that a tiny-z ridge
converges to at BOTH P > T and P < T; the multi-z path matches per-z solves; the
return is always 2-D; z and beta shapes are validated; and prediction is
intercept-free. Each test seeds its own generator.
"""

import numpy as np
import pytest

from voc.kernel_ridge import predict, ridge_dual, ridgeless


def _ridge_primal(S, R, z):
    """Independent primal-space reference for the KMZ estimator."""
    n_obs, n_feat = S.shape
    A = z * np.eye(n_feat) + S.T @ S / n_obs
    return np.linalg.solve(A, S.T @ R / n_obs)


@pytest.mark.parametrize("T,P", [(12, 5), (12, 40)])
def test_dual_equals_primal(T, P):
    """Dual (T x T) and primal (P x P) forms agree, both P < T and P > T."""
    rng = np.random.default_rng(T * 1000 + P)
    S = rng.standard_normal((T, P))
    R = rng.standard_normal(T)
    for z in (1e-2, 1.0, 1e2):
        np.testing.assert_allclose(
            ridge_dual(S, R, z)[0], _ridge_primal(S, R, z), rtol=1e-8, atol=1e-10
        )


def test_matches_sklearn_no_intercept():
    """ridge_dual matches sklearn Ridge(fit_intercept=False) with alpha = z*T."""
    linear_model = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(1)
    T, P, z = 12, 5, 0.1
    S = rng.standard_normal((T, P))
    R = rng.standard_normal(T)
    model = linear_model.Ridge(alpha=z * T, fit_intercept=False).fit(S, R)
    np.testing.assert_allclose(ridge_dual(S, R, z)[0], model.coef_, rtol=1e-8, atol=1e-10)


def test_ridgeless_is_min_norm_least_squares():
    """Ridgeless equals the pseudoinverse least-norm solution (independent route)."""
    rng = np.random.default_rng(2)
    S = rng.standard_normal((12, 40))
    R = rng.standard_normal(12)
    np.testing.assert_allclose(ridgeless(S, R), np.linalg.pinv(S) @ R, rtol=1e-8, atol=1e-10)


def test_tiny_z_converges_to_ridgeless_p_gt_t():
    rng = np.random.default_rng(3)
    S = rng.standard_normal((12, 40))
    R = rng.standard_normal(12)
    np.testing.assert_allclose(
        ridge_dual(S, R, 1e-12)[0], ridgeless(S, R), rtol=1e-5, atol=1e-7
    )


def test_tiny_z_converges_to_ridgeless_p_lt_t():
    """The rank-deficient regime the original tiny-z test missed (P < T)."""
    rng = np.random.default_rng(4)
    S = rng.standard_normal((12, 5))
    R = rng.standard_normal(12)
    np.testing.assert_allclose(
        ridge_dual(S, R, 1e-8)[0], ridgeless(S, R), rtol=1e-4, atol=1e-6
    )


def test_multi_z_matches_individual_solves():
    """The eigendecomposition-reuse path returns the same betas, in order, as
    solving each z on its own."""
    rng = np.random.default_rng(5)
    S = rng.standard_normal((12, 60))
    R = rng.standard_normal(12)
    zs = [1e-3, 1e-1, 1.0, 1e1, 1e3]
    stacked = ridge_dual(S, R, zs)
    assert stacked.shape == (len(zs), 60)
    for i, z in enumerate(zs):
        np.testing.assert_allclose(stacked[i], ridge_dual(S, R, z)[0], rtol=1e-10, atol=1e-12)


def test_return_is_always_two_dimensional():
    """Scalar and array z both return (n_z, P), so callers never branch on shape."""
    rng = np.random.default_rng(6)
    S = rng.standard_normal((12, 8))
    R = rng.standard_normal(12)
    assert ridge_dual(S, R, 1.0).shape == (1, 8)
    assert ridge_dual(S, R, [1.0, 10.0]).shape == (2, 8)


def test_rejects_nonpositive_and_2d_z():
    """z <= 0 and 2-D z are rejected instead of returning silent garbage."""
    rng = np.random.default_rng(7)
    S = rng.standard_normal((12, 8))
    R = rng.standard_normal(12)
    for bad in (0.0, -1.0, [1.0, 0.0], [1.0, -3.0]):
        with pytest.raises(ValueError):
            ridge_dual(S, R, bad)
    with pytest.raises(ValueError):
        ridge_dual(S, R, np.full((2, 12), 0.1))


def test_predict_intercept_free_and_rejects_2d_beta():
    """predict is S_new @ beta (zero features -> 0) and rejects a multi-z matrix."""
    rng = np.random.default_rng(8)
    T, P = 12, 20
    S = rng.standard_normal((T, P))
    R = rng.standard_normal(T)
    beta = ridge_dual(S, R, 1.0)[0]
    S_new = rng.standard_normal((3, P))
    np.testing.assert_allclose(predict(S_new, beta), S_new @ beta)
    assert predict(np.zeros(P), beta) == 0.0
    with pytest.raises(ValueError):
        predict(S_new, ridge_dual(S, R, [1.0, 10.0]))
