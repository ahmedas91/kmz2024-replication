"""Random Fourier features for the KMZ (2024) Virtue-of-Complexity engine.

Implements the paper's feature map (equation 20). For an i.i.d. Gaussian weight
vector ``omega_i ~ N(0, I_d)``, the volatility-standardized predictor vector
``G_t`` (the d = 15 predictors at month t) generates the feature pair

    S_{i,t} = [ sin(gamma * omega_i' G_t),  cos(gamma * omega_i' G_t) ],

with ``gamma = 2`` by default (footnote 38; no sqrt(2/P) factor, matching eq. 20).
A model of complexity P stacks the first P/2 weight vectors, giving
``P = 2 * n_pairs`` features (each weight contributes one sin AND one cos column).

Three responsibilities, kept separate so the OOS engine (issue #7) reuses them:

- :func:`draw_rff_weights` — draw the Gaussian weights for one repetition (seed);
- :func:`compute_rff` — map a predictor block through the feature map;
- :func:`standardize_by_training_window` — scale features by TRAINING-window
  volatility only (footnote 39), the no-lookahead convention the engine relies on.

Nesting convention (the key compute saver, paper Section V.B)
------------------------------------------------------------
The paper draws the full-size weight matrix once per repetition and uses "the
first P RFFs" for every smaller model. :func:`draw_rff_weights` fills the weights
one vector per row and returns the transpose, so a ``k``-pair draw is exactly the
first ``k`` columns of any larger draw with the same seed: nesting holds whether
you slice one max-size draw (the efficient path — one draw per seed) or redraw at
each size. :func:`compute_rff` interleaves the sin/cos columns so ``S[:, :P]`` is
the model built from the first P/2 weights.

:func:`compute_rff` depends only on ``G`` and ``omega``, not on the OOS window, so
the engine computes ``S`` once per seed and slices rows per window.
"""

from __future__ import annotations

import numpy as np

GAMMA_DEFAULT = 2.0
_SD_FLOOR = 1e-12  # scales at or below this are treated as degenerate (see below)


def draw_rff_weights(n_pairs: int, n_predictors: int, seed: int) -> np.ndarray:
    """Draw the Gaussian RFF weights for one repetition.

    Parameters
    ----------
    n_pairs : int
        Number of (sin, cos) pairs, i.e. P / 2.
    n_predictors : int
        The predictor dimension d (15 for the KMZ market study).
    seed : int
        Repetition index. One ``np.random.default_rng`` per seed keeps runs
        reproducible and embarrassingly parallel across repetitions.

    Returns
    -------
    omega : np.ndarray
        ``(n_predictors, n_pairs)`` array of i.i.d. N(0, 1) weights; column i is
        the weight vector ``omega_i``. Weights are drawn one vector per row and
        transposed, so ``draw_rff_weights(k, d, seed)`` equals
        ``draw_rff_weights(k_max, d, seed)[:, :k]`` — draws of any size nest.
    """
    if n_pairs < 1:
        raise ValueError("n_pairs must be >= 1")
    if n_predictors < 1:
        raise ValueError("n_predictors must be >= 1")
    rng = np.random.default_rng(seed)
    # Fill one weight vector per row so that smaller draws are a prefix of larger
    # ones (row-major fill), then transpose to the (d, n_pairs) layout.
    return rng.standard_normal((n_pairs, n_predictors)).T


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
    # Fold gamma into the (smaller) predictor matrix before the projection.
    projection = (gamma * G) @ omega  # (T, n_pairs)
    n_rows, n_pairs = projection.shape
    S = np.empty((n_rows, 2 * n_pairs), dtype=np.float64)
    np.sin(projection, out=S[:, 0::2])  # write straight into interleaved columns
    np.cos(projection, out=S[:, 1::2])
    return S


def standardize_by_training_window(
    S_train: np.ndarray, S_test: np.ndarray, *, uncentered: bool = False
):
    """Scale features by TRAINING-window volatility only (footnote 39).

    The paper standardizes the RFFs by their in-sample volatility before the ridge
    step and applies the SAME training-window scale to the out-of-sample feature
    vector — nothing from the test period enters the scale, which is what keeps the
    forecast free of lookahead.

    Parameters
    ----------
    S_train : np.ndarray
        ``(T, P)`` training features.
    S_test : np.ndarray
        ``(P,)`` or ``(m, P)`` out-of-sample features, scaled by the SAME divisor.
    uncentered : bool
        If False (default), scale by the centered standard deviation
        (``ddof=1``); if True, by the uncentered volatility
        ``sqrt(mean(S_train**2))``. The two conventions differ materially for
        RFFs because cos columns have means near +1. CENTERED is the pinned
        convention: footnote 39 says "standard deviations" (footnote 34's
        uncentered exception is stated for RETURNS only), and the issue #9
        anchor check confirmed it empirically (with the centered scale the
        Figure 8 ridgeless anchors land within ~1%, e.g. alpha t-stat 2.78 vs
        the paper's 2.81; uncentered leaves it at 1.75-2.2 depending on the
        market series). The kwarg is retained for A/B checks.

    Returns
    -------
    (S_train_scaled, S_test_scaled) : tuple of np.ndarray
        Same shapes as the inputs.
    """
    S_train = np.asarray(S_train, dtype=np.float64)
    S_test = np.asarray(S_test, dtype=np.float64)
    if S_train.ndim != 2:
        raise ValueError("S_train must be 2-D (T, P)")
    if S_test.ndim > 2 or S_test.shape[-1] != S_train.shape[1]:
        raise ValueError(
            "S_test must be (P,) or (m, P) with P = S_train.shape[1] = "
            f"{S_train.shape[1]}; got {S_test.shape}"
        )
    if uncentered:
        scale = np.sqrt(np.mean(S_train**2, axis=0))
    else:
        if S_train.shape[0] < 2:
            # ddof=1 SD is undefined on one row; the degenerate-column guard
            # below would then silently disable standardization wholesale.
            raise ValueError(
                "centered standardization needs at least 2 training rows "
                f"(got {S_train.shape[0]}); use T >= 2 or uncentered=True"
            )
        scale = S_train.std(axis=0, ddof=1)
    # Degenerate columns (a near-constant feature, or a NaN/inf scale) are left
    # unscaled rather than amplified toward infinity (sklearn's convention).
    scale = np.where(np.isfinite(scale) & (scale > _SD_FLOOR), scale, 1.0)
    return S_train / scale, S_test / scale
