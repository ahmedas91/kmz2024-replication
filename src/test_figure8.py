"""Anchor and shape tests for the Figure 8 replication (issue #9).

Quantitative anchors (T = 12, RIDGELESS, c = 1000, paper period 1930-01 to
2020-12; values independently reproduced by Nagel, so they are trustworthy
targets): alpha t-statistic near 2.81, information ratio near 0.296, Sharpe
ratio above 0.4.

TOLERANCE: 10 percent relative on the two point anchors (the issue's
suggested band). Rationale, stated for the record: under the pipeline's
pinned conventions — centered RFF standardization and the Goyal-Welch
workbook market series as the target, both selected by the issue #9 anchor
investigation — the replication lands within about 1-2 percent of each
anchor (alpha t-stat 2.78 vs 2.81, IR 0.297 vs 0.296 at 50 probe seeds), and
across-seed noise at the default N_SEEDS = 500 has a standard error near
0.001 on the Sharpe. Ten percent therefore leaves an order-of-magnitude
margin over both the observed miss and the seed noise without being vacuous.
The seed list is fixed by config (range(N_SEEDS)), so reruns are
deterministic; no seeds were selected on outcomes. The residual 1-2 percent
is attributable to the paper's 1,000-repetition averaging and its one extra
OOS month (its raw data predate the 1930 sample floor).

Shape tests mirror the paper's message: Sharpe, alpha, IR, and the alpha
t-stat all RISE with complexity from the interpolation boundary to extreme
complexity, for every shrinkage level and for ridgeless; the brief Sharpe
dip near c = 1 at low shrinkage is the boundary variance spike and is
allowed, as in the paper.

Real-file tests skip (rather than fail) when the grid parquet has not been
built yet (`doit estimate`), since data never ships with the repository.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)
GRID_PATH = DATA_DIR / f"oos_grid_T{TRAIN_WINDOW}.parquet"

requires_data = pytest.mark.skipif(
    not GRID_PATH.exists(),
    reason=f"{GRID_PATH.name} not found; run `doit estimate` first",
)

# Nagel-reproduced anchors at (T=12, ridgeless, c=1000), and the stated band.
ANCHOR_ALPHA_TSTAT = 2.81
ANCHOR_IR = 0.296
SHARPE_FLOOR = 0.4
RELATIVE_TOLERANCE = 0.10


@lru_cache(maxsize=1)
def _grid():
    return pd.read_parquet(GRID_PATH)


def _line(z, column):
    """One z line of the averaged grid as a Series indexed by c (sorted)."""
    grid = _grid()
    line = grid.loc[grid["z"] == z].sort_values("c")
    return line.set_index("c")[column]


@requires_data
def test_ridgeless_anchors_at_extreme_complexity():
    """The paper's quantitative anchors, within the stated 10 percent band."""
    alpha_tstat = _line(0.0, "alpha_tstat").loc[1000]
    ir = _line(0.0, "information_ratio").loc[1000]
    sharpe = _line(0.0, "sharpe").loc[1000]
    assert abs(alpha_tstat / ANCHOR_ALPHA_TSTAT - 1) < RELATIVE_TOLERANCE
    assert abs(ir / ANCHOR_IR - 1) < RELATIVE_TOLERANCE
    assert sharpe > SHARPE_FLOOR


@requires_data
def test_sharpe_rises_with_complexity():
    """Sharpe at extreme complexity beats the boundary for every line, exceeds
    the paper's 0.4 threshold for ridgeless and weak shrinkage, and increases
    monotonically past the boundary region (c >= 2), where the paper allows
    only the brief low-z dip at c = 1 itself."""
    z_values = [0.0] + [10.0**k for k in range(-3, 4)]
    for z in z_values:
        sharpe = _line(z, "sharpe")
        assert sharpe.loc[1000] > sharpe.loc[1.0]
        assert (np.diff(sharpe.loc[2.0:]) > -0.01).all()
    for z in (0.0, 1e-3, 1e-2):
        assert _line(z, "sharpe").loc[1000] > SHARPE_FLOOR


@requires_data
def test_alpha_ir_and_tstat_rise_with_complexity():
    """Panels B, C, D: alpha, IR, and the alpha t-stat all end far above their
    interpolation-boundary values, for every shrinkage level and ridgeless."""
    z_values = [0.0] + [10.0**k for k in range(-3, 4)]
    for column in ("alpha", "information_ratio", "alpha_tstat"):
        for z in z_values:
            line = _line(z, column)
            assert line.loc[1000] > line.loc[1.0]


@requires_data
def test_ridgeless_tracks_weakest_shrinkage():
    """The ridgeless limit and the z = 10^-3 line are near-identical at
    extreme complexity (the z -> 0 limit), a consistency check tying the
    anchor cell to the plotted grid."""
    for column in ("sharpe", "alpha", "information_ratio", "alpha_tstat"):
        ridgeless = _line(0.0, column).loc[1000]
        weakest = _line(1e-3, column).loc[1000]
        assert abs(ridgeless - weakest) <= 0.02 * max(1.0, abs(ridgeless))
