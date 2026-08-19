"""Shape and sign tests for the Figure 7 replication (issue #8).

They pin the cached OOS grid's qualitative agreement with the paper's
Figure 7 (T = 12) rather than exact levels: with N_SEEDS = 50 repetitions
against the paper's 1,000 the levels carry seed noise, but the SHAPES are
stable, and the quantitative anchors live in the Figure 8 tests (issue #9).
What is pinned, panel by panel:

- Panel A: the low-shrinkage OOS R^2 is negative everywhere short of extreme
  complexity, collapses to its minimum exactly AT the interpolation boundary
  c = 1, and recovers monotonically toward zero beyond it, ending within
  +/- 0.01 of zero at c = 1000. Note the direction: complexity hurts R^2 on
  the way INTO the boundary; past it the R^2 RISES back toward zero (paper
  Panel A) while trading performance keeps improving. That inversion is the
  paper's point; do not "fix" the negative R^2.
- Panel B: ||beta_hat|| spikes at c = 1 for low z and falls by orders of
  magnitude at extreme complexity; heavy shrinkage flattens the spike away
  entirely; at fixed c = 1000 the norm is monotone (non-increasing, with a
  material total drop) in z.
- Panels C/D: timing volatility declines strictly with complexity beyond the
  boundary for low z; expected return rises strictly with complexity under
  heavy shrinkage and severalfold from the smallest model to c = 1000 for
  low z.

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

LOW_Z = (1e-3, 1e-2)  # weak shrinkage: the boundary-explosion lines
HIGH_Z = 1e3


@lru_cache(maxsize=1)
def _grid():
    return pd.read_parquet(GRID_PATH)


def _line(z, column):
    """One z line of the averaged grid as a Series indexed by c (sorted)."""
    grid = _grid()
    line = grid.loc[grid["z"] == z].sort_values("c")
    return line.set_index("c")[column]


@requires_data
def test_grid_has_expected_structure():
    grid = _grid()
    z_values = sorted(grid["z"].unique())
    assert z_values == [0.0] + [10.0**k for k in range(-3, 4)]
    assert grid["P"].nunique() == 14 and grid["P"].max() == 12_000
    assert {"c", "r2", "beta_norm", "mean_return", "volatility"} <= set(grid.columns)


@requires_data
def test_r2_negative_with_collapse_at_boundary_and_recovery():
    """Panel A for low z: minimum exactly at c = 1 and deeply negative there,
    negative through c = 512, monotone recovery beyond the boundary, and
    within 0.01 of zero at c = 1000."""
    for z in LOW_Z:
        r2 = _line(z, "r2")
        assert r2.idxmin() == 1.0
        assert r2.loc[1.0] < -3.0
        assert (r2.loc[:512] < 0).all()
        assert (np.diff(r2.loc[1.0:]) > 0).all()
        assert abs(r2.loc[1000]) < 0.01


@requires_data
def test_beta_norm_spikes_at_boundary_and_shrinkage_flattens_it():
    """Panel B: the interpolation-boundary signature. Low-z norms peak exactly
    at c = 1 and are at least 5x smaller by c = 1000; the z = 10^3 line has no
    boundary spike at all (its maximum sits at extreme complexity and its
    c = 1 value is under 5 percent of the low-z spike)."""
    for z in (*LOW_Z, 1e-1):
        assert _line(z, "beta_norm").idxmax() == 1.0
    for z in LOW_Z:
        norm = _line(z, "beta_norm")
        assert norm.loc[1.0] > 5 * norm.loc[1000]
    flat = _line(HIGH_Z, "beta_norm")
    assert flat.idxmax() == 1000.0
    assert flat.loc[1.0] < 0.05 * _line(1e-3, "beta_norm").loc[1.0]


@requires_data
def test_beta_norm_monotone_in_z_at_extreme_complexity():
    """Panel B, across lines: at fixed c = 1000 heavier shrinkage never raises
    the coefficient norm (tiny-z lines coincide with the ridgeless limit to
    float precision, hence the tolerance) and cuts it materially in total."""
    z_values = [10.0**k for k in range(-3, 4)]
    norms = np.array([_line(z, "beta_norm").loc[1000] for z in z_values])
    assert (np.diff(norms) <= 1e-9).all()
    assert norms[-1] < 0.6 * norms[0]


@requires_data
def test_volatility_declines_beyond_boundary_for_low_z():
    """Panel D: past the boundary spike, more complexity strictly lowers the
    timing strategy's volatility for the weakly shrunk lines."""
    for z in (*LOW_Z, 1e-1):
        vol = _line(z, "volatility")
        assert vol.idxmax() == 1.0
        assert (np.diff(vol.loc[1.0:]) < 0).all()


@requires_data
def test_expected_return_rises_with_complexity():
    """Panel C: under heavy shrinkage the expected return rises strictly with
    complexity across the whole grid; for low z it rises severalfold from the
    smallest model to c = 1000 (net of the boundary dip)."""
    for z in (1e2, HIGH_Z):
        mean_return = _line(z, "mean_return")
        assert (np.diff(mean_return) > 0).all()
    for z in LOW_Z:
        mean_return = _line(z, "mean_return")
        assert mean_return.loc[1000] > 2 * mean_return.iloc[0]
