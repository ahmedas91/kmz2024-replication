"""Tests for the random Fourier features module (issue #5).

Pure-synthetic and need no data on disk. They pin the feature-map contract the
OOS engine depends on: output shape, seed determinism, the nesting property (by
slicing and by re-draw), the sin/cos Pythagorean identity, gamma wiring, and the
training-only standardization (both conventions, degenerate columns, and shape
guards). Each test seeds its own generator so an isolated rerun is reproducible.
"""

import numpy as np
import pytest

from voc.rff import (
    compute_rff,
    draw_rff_weights,
    standardize_by_training_window,
)


def test_shape_is_two_columns_per_pair():
    """n_pairs weights over a (T, d) block give (T, 2 * n_pairs) features."""
    G = np.random.default_rng(1).standard_normal((30, 15))
    omega = draw_rff_weights(n_pairs=64, n_predictors=15, seed=0)
    assert compute_rff(G, omega).shape == (30, 128)


def test_seed_determinism():
    """The same seed reproduces the weights; different seeds differ."""
    a = draw_rff_weights(50, 15, seed=7)
    b = draw_rff_weights(50, 15, seed=7)
    c = draw_rff_weights(50, 15, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_weights_nest_by_redraw():
    """A smaller draw is the column prefix of a larger draw with the same seed —
    so the natural per-model redraw is safe, not just slicing one max draw."""
    omega_max = draw_rff_weights(100, 15, seed=3)
    for k in (1, 4, 20, 60):
        np.testing.assert_array_equal(draw_rff_weights(k, 15, seed=3), omega_max[:, :k])


def test_features_nest_by_column_slice():
    """The first P feature columns at max size equal the standalone P-model."""
    G = np.random.default_rng(4).standard_normal((24, 15))
    omega_max = draw_rff_weights(n_pairs=100, n_predictors=15, seed=3)
    S_max = compute_rff(G, omega_max)
    for P in (2, 8, 40, 120):
        S_small = compute_rff(G, omega_max[:, : P // 2])
        np.testing.assert_allclose(S_small, S_max[:, :P], rtol=1e-12, atol=1e-12)


def test_sin_cos_pythagorean_pairing():
    """Interleaved columns 2i, 2i+1 are the sin/cos of one projection, so their
    squares sum to 1."""
    G = np.random.default_rng(11).standard_normal((40, 15))
    S = compute_rff(G, draw_rff_weights(60, 15, seed=11))
    np.testing.assert_allclose(S[:, 0::2] ** 2 + S[:, 1::2] ** 2, 1.0, atol=1e-12)


def test_gamma_is_a_parameter():
    """gamma rescales the projection; sin(2x) = 2 sin(x) cos(x) links the two."""
    G = np.random.default_rng(1).standard_normal((15, 15))
    omega = draw_rff_weights(30, 15, seed=1)
    S1 = compute_rff(G, omega, gamma=1.0)
    S2 = compute_rff(G, omega, gamma=2.0)
    np.testing.assert_allclose(S2[:, 0::2], 2.0 * S1[:, 0::2] * S1[:, 1::2], atol=1e-12)


def test_compute_rff_rejects_mismatched_omega():
    """omega's first axis must match the predictor dimension d."""
    G = np.random.default_rng(0).standard_normal((10, 15))
    with pytest.raises(ValueError):
        compute_rff(G, draw_rff_weights(8, n_predictors=14, seed=0))


def test_standardization_uses_training_scale_only():
    """The out-of-sample block is divided by the TRAINING scale, so a wildly
    rescaled test row stays rescaled (no test-period volatility leaks in)."""
    rng = np.random.default_rng(2)
    S_train = compute_rff(rng.standard_normal((12, 15)), draw_rff_weights(20, 15, 2))
    S_test = S_train[0] * 1000.0  # one OOS row at 1000x scale

    train_std, test_std = standardize_by_training_window(
        S_train, S_test, uncentered=True
    )
    # training features sit at unit uncentered scale (uncentered=True) ...
    np.testing.assert_allclose(np.sqrt(np.mean(train_std**2, axis=0)), 1.0, atol=1e-9)
    # ... and the test row is still 1000x its matching training row.
    np.testing.assert_allclose(test_std, train_std[0] * 1000.0, rtol=1e-9)


def test_centered_default_gives_unit_sample_sd():
    """The DEFAULT path is the centered convention pinned by the issue #9
    anchor checks: ddof=1, so each standardized training column has sample
    SD 1 without passing any kwarg."""
    rng = np.random.default_rng(5)
    S_train = compute_rff(rng.standard_normal((40, 15)), draw_rff_weights(25, 15, 5))
    S_test = S_train[:3]
    train_std, _ = standardize_by_training_window(S_train, S_test)
    np.testing.assert_allclose(train_std.std(axis=0, ddof=1), 1.0, atol=1e-9)


def test_degenerate_column_is_left_unscaled():
    """A near-constant column is mapped to scale 1.0, not amplified to infinity,
    on both conventions, and the output stays finite."""
    S_train = np.ones((12, 4))
    S_train[:, 1] = np.linspace(-1.0, 1.0, 12)  # one varying column
    S_test = np.ones((4,))
    for uncentered in (True, False):
        train_std, test_std = standardize_by_training_window(
            S_train, S_test, uncentered=uncentered
        )
        assert np.isfinite(train_std).all() and np.isfinite(test_std).all()
    # constant column on the centered path: SD 0 -> scale 1 -> values unchanged.
    train_std, _ = standardize_by_training_window(S_train, S_test, uncentered=False)
    np.testing.assert_allclose(train_std[:, 0], S_train[:, 0])


def test_standardize_rejects_bad_test_shape():
    """A (P, 1) column vector (the easy reshape mistake) is rejected, not
    silently broadcast into a (P, P) matrix."""
    S_train = np.random.default_rng(0).standard_normal((12, 8))
    with pytest.raises(ValueError):
        standardize_by_training_window(S_train, np.ones((8, 1)))
