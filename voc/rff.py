"""Random Fourier features for the KMZ (2024) Virtue-of-Complexity engine.

Implements the paper's feature map (equation 20). For an i.i.d. Gaussian weight
vector ``omega_i ~ N(0, I_d)``, the volatility-standardized predictor vector
``G_t`` (the d = 15 predictors at month t) generates the feature pair

    S_{i,t} = [ sin(gamma * omega_i' G_t),  cos(gamma * omega_i' G_t) ],

with ``gamma = 2`` by default. A model of complexity P stacks the first P/2
weight vectors, giving ``P = 2 * n_pairs`` features (each weight contributes one
sin AND one cos column).

Three responsibilities, kept separate so the OOS engine (issue #7) reuses them:

- :func:`draw_rff_weights` — draw the Gaussian weights for one repetition (seed);
- :func:`compute_rff` — map a predictor block through the feature map;
- :func:`standardize_by_training_window` — scale features by TRAINING-window
  volatility only (footnote 39), the no-lookahead convention the engine relies on.

Nesting convention (the key compute saver, paper Section V.B)
------------------------------------------------------------
The paper draws the full-size weight matrix ONCE per repetition and uses "the
first P RFFs" for every smaller model. Reproduce this by drawing ``omega`` at the
MAXIMUM ``n_pairs`` and SLICING columns — do not call :func:`draw_rff_weights`
again with a smaller ``n_pairs``. Because NumPy fills a 2-D normal draw row by
row, ``draw_rff_weights(k, ...)`` is NOT the left block of
``draw_rff_weights(k_max, ...)``; only column-slicing the single max-size draw
yields properly nested feature sets. :func:`compute_rff` interleaves the sin/cos
columns so that ``S[:, :P]`` is exactly the model built from the first P/2 weights.
"""

from __future__ import annotations

import numpy as np

GAMMA_DEFAULT = 2.0
_SD_FLOOR = 1e-12  # guards the division when a feature is ~constant in-window


def draw_rff_weights(n_pairs: int, n_predictors: int, seed: int) -> np.ndarray:
    """Draw the Gaussian RFF weights for one repetition.

    Parameters
    ----------
    n_pairs : int
        Number of (sin, cos) pairs, i.e. P / 2. Draw at the MAXIMUM P/2 you will
        need and slice columns for smaller models (see the module docstring on
        nesting); redrawing at a smaller size does not nest.
    n_predictors : int
        The predictor dimension d (15 for the KMZ market study).
    seed : int
        Repetition index. One ``np.random.default_rng`` per seed keeps runs
        reproducible and embarrassingly parallel across repetitions.

    Returns
    -------
    omega : np.ndarray
        ``(n_predictors, n_pairs)`` array of i.i.d. N(0, 1) weights; column i is
        the weight vector ``omega_i``.
    """
    if n_pairs < 1:
        raise ValueError("n_pairs must be >= 1")
    if n_predictors < 1:
        raise ValueError("n_predictors must be >= 1")
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_predictors, n_pairs))


def compute_rff(
    G: np.ndarray, omega: np.ndarray, gamma: float = GAMMA_DEFAULT
) -> np.ndarray:
    """Map a predictor block through the KMZ feature map (equation 20).

    Parameters
    ----------
    G : np.ndarray
        ``(T, d)`` block of volatility-standardized predictors.
    omega : np.ndarray
        ``(d, n_pairs)`` Gaussian weights from :func:`draw_rff_weights`.
    gamma : float
        Bandwidth in the feature map; paper default 2.0.

    Returns
    -------
    S : np.ndarray
        ``(T, 2 * n_pairs)`` feature block with INTERLEAVED columns
        ``[sin(g w_1'G), cos(g w_1'G), sin(g w_2'G), cos(g w_2'G), ...]``, so that
        ``S[:, :P]`` is precisely the model built from the first P/2 weights.
    """
    G = np.asarray(G, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError("G must be 2-D (T, d)")
    if omega.ndim != 2 or omega.shape[0] != G.shape[1]:
        raise ValueError(
            "omega must have shape (d, n_pairs) with d = G.shape[1] = "
            f"{G.shape[1]}; got {omega.shape}"
        )
    projection = gamma * (G @ omega)  # (T, n_pairs)
    n_rows, n_pairs = projection.shape
    S = np.empty((n_rows, 2 * n_pairs), dtype=np.float64)
    S[:, 0::2] = np.sin(projection)
    S[:, 1::2] = np.cos(projection)
    return S


def standardize_by_training_window(
    S_train: np.ndarray, S_test: np.ndarray, *, uncentered: bool = True
):
    """Scale features by TRAINING-window volatility only (footnote 39).

    The paper standardizes the RFFs by their in-sample volatility before the
    ridge step and applies the SAME training-window scale to the out-of-sample
    feature vector — nothing from the test period enters the scale, which is what
    keeps the forecast free of lookahead.

    Parameters
    ----------
    S_train : np.ndarray
        ``(T, P)`` training features.
    S_test : np.ndarray
        ``(P,)`` or ``(m, P)`` out-of-sample features, scaled by the SAME divisor.
    uncentered : bool
        If True (default), scale by the uncentered volatility
        ``sqrt(mean(S_train**2))``; if False, by the centered standard deviation.
        Footnote 39 is not explicit; we default to the uncentered reading,
        mirroring footnote 34's treatment of returns, because with T as small as
        12 the centered SD of a highly persistent feature is frequently near zero
        and would blow the scale up. (Confirm this is the convention to defend.)

    Returns
    -------
    (S_train_scaled, S_test_scaled) : tuple of np.ndarray
        Same shapes as the inputs.
    """
    S_train = np.asarray(S_train, dtype=np.float64)
    S_test = np.asarray(S_test, dtype=np.float64)
    if S_train.ndim != 2:
        raise ValueError("S_train must be 2-D (T, P)")
    if uncentered:
        scale = np.sqrt(np.mean(S_train**2, axis=0))
    else:
        scale = S_train.std(axis=0)
    scale = np.maximum(scale, _SD_FLOOR)
    return S_train / scale, S_test / scale
