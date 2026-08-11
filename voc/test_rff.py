"""Tests for the random Fourier features module (issue #5).

Pure-synthetic and need no data on disk. They pin the feature-map contract the
OOS engine depends on: output shape, seed determinism, the nesting property that
lets one max-size draw serve every smaller model, the sin/cos Pythagorean
identity, that gamma is wired through, and the training-only standardization (no
lookahead).
"""

import numpy as np
import pytest

from voc.rff import (
    compute_rff,
    draw_rff_weights,
    standardize_by_training_window,
)

RNG = np.random.default_rng(20240)


def test_shape_is_two_columns_per_pair():
    """n_pairs weights over a (T, d) block give (T, 2 * n_pairs) features."""
    G = RNG.standard_normal((30, 15))
    omega = draw_rff_weights(n_pairs=64, n_predictors=15, seed=0)
    S = compute_rff(G, omega)
    assert S.shape == (30, 128)


def test_seed_determinism():
    """The same seed reproduces the weights; different seeds differ."""
    a = draw_rff_weights(50, 15, seed=7)
    b = draw_rff_weights(50, 15, seed=7)
    c = draw_rff_weights(50, 15, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_nesting_by_column_slice():
    """The first P columns at max size equal the standalone P-feature model.

    Draw ONCE at the maximum n_pairs and slice: the smaller model must be the
    left block of the larger one. This is the property that lets the grid reuse a
    single draw per seed.
    """
    G = RNG.standard_normal((24, 15))
    omega_max = draw_rff_weights(n_pairs=100, n_predictors=15, seed=3)
    S_max = compute_rff(G, omega_max)
    for P in (2, 8, 40, 120):
        S_small = compute_rff(G, omega_max[:, : P // 2])
        np.testing.assert_allclose(S_small, S_max[:, :P], rtol=1e-12, atol=1e-12)


def test_sin_cos_pythagorean_pairing():
    """Interleaved columns 2i, 2i+1 are the sin/cos of one projection, so their
    squares sum to 1."""
    G = RNG.standard_normal((40, 15))
    omega = draw_rff_weights(60, 15, seed=11)
    S = compute_rff(G, omega)
    np.testing.assert_allclose(S[:, 0::2] ** 2 + S[:, 1::2] ** 2, 1.0, atol=1e-12)


def test_gamma_is_a_parameter():
    """gamma rescales the projection; the identity sin(2x) = 2 sin(x) cos(x)
    links gamma = 2 back to gamma = 1."""
    G = RNG.standard_normal((15, 15))
    omega = draw_rff_weights(30, 15, seed=1)
    S1 = compute_rff(G, omega, gamma=1.0)
    S2 = compute_rff(G, omega, gamma=2.0)
    np.testing.assert_allclose(
        S2[:, 0::2], 2.0 * S1[:, 0::2] * S1[:, 1::2], atol=1e-12
    )


def test_compute_rff_rejects_mismatched_omega():
    """omega's first axis must match the predictor dimension d."""
    G = RNG.standard_normal((10, 15))
    bad = draw_rff_weights(8, n_predictors=14, seed=0)
    with pytest.raises(ValueError):
        compute_rff(G, bad)


def test_standardization_uses_training_scale_only():
    """The out-of-sample block is divided by the TRAINING scale, so a wildly
    rescaled test row stays rescaled — proving no test-period volatility leaks
    into the divisor."""
    G_train = RNG.standard_normal((12, 15))
    omega = draw_rff_weights(20, 15, seed=2)
    S_train = compute_rff(G_train, omega)
    S_test = S_train[0] * 1000.0  # one OOS row at 1000x scale

    S_train_std, S_test_std = standardize_by_training_window(S_train, S_test)

    # Training features now sit at unit uncentered scale ...
    train_scale = np.sqrt(np.mean(S_train_std**2, axis=0))
    np.testing.assert_allclose(train_scale, 1.0, atol=1e-9)
    # ... and the test row is still 1000x its matching training row.
    np.testing.assert_allclose(S_test_std, S_train_std[0] * 1000.0, rtol=1e-9)
