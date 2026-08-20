"""Tests for the bonds VoC study (issue #15).

Three layers, all skipping until their inputs exist:

1. Construction identities on the tidy dataset: the excess returns equal
   the raw workbook columns' differences exactly.
2. The study input builder: the sample geometry matches the market study
   (the standardized predictors' 36-month burn-in and 1930-01 floor bind),
   every value is finite, and the own-lag column obeys the same-row
   convention (the engine applies the only shift).
3. The cached grid: both targets present with the full cell structure and
   finite statistics, identical fixed seed set per target.

Deliberately NO numeric anchors: nothing published pins bond VoC numbers
(the paper studies the equity market), so asserting levels would be
pinning our own output to itself. Whether the virtue-of-complexity pattern
generalizes to bonds is reported as a finding on the chart page, not
enforced by tests.
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)

RAW_PATH = DATA_DIR / "goyal_welch.parquet"
BONDS_PATH = DATA_DIR / "bond_returns.parquet"
STD_PATH = DATA_DIR / "kmz_dataset_standardized.parquet"

requires_raw = pytest.mark.skipif(
    not (RAW_PATH.exists() and BONDS_PATH.exists()),
    reason="raw or tidy bond parquet missing; run `doit pull:goyal_welch tidy_bonds`",
)
requires_inputs = pytest.mark.skipif(
    not (BONDS_PATH.exists() and STD_PATH.exists()),
    reason="tidy bonds or standardized dataset missing; run `doit tidy_bonds standardize`",
)


def _grid_path():
    from sample_period import SAMPLE_SUFFIX

    return DATA_DIR / f"oos_grid_bonds_T{TRAIN_WINDOW}{SAMPLE_SUFFIX}_per_seed.parquet"


@lru_cache(maxsize=1)
def _per_seed():
    return pd.read_parquet(_grid_path())


@requires_raw
def test_excess_return_identities():
    """ltr_excess and corpr_excess equal the raw column differences exactly."""
    raw = pd.read_parquet(RAW_PATH).set_index("date")
    bonds = pd.read_parquet(BONDS_PATH).set_index("date")
    aligned = raw.loc[bonds.index]
    np.testing.assert_array_equal(
        bonds["ltr_excess"].to_numpy(), (aligned["ltr"] - aligned["Rfree"]).to_numpy()
    )
    np.testing.assert_array_equal(
        bonds["corpr_excess"].to_numpy(),
        (aligned["corpr"] - aligned["Rfree"]).to_numpy(),
    )
    assert bonds.index.min() <= pd.Timestamp("1926-01-31")


@requires_inputs
def test_study_inputs_geometry_and_own_lag():
    """Input frames start at the 1930-01 floor (the standardized predictors'
    36-month burn-in binds, as in the market study), hold 15 predictor
    columns (14 shared + own lag), are fully finite, and the own-lag column
    equals the standardized target on the SAME row — pre-shifting it would
    double the engine's single t -> t+1 shift."""
    from run_bonds_study import build_bond_study_inputs

    for target_col in ("ltr_excess", "corpr_excess"):
        frame = build_bond_study_inputs(target_col, sample_end="2020-12")
        assert frame["date"].iloc[0] == pd.Timestamp("1930-01-31")
        assert frame["date"].iloc[-1] == pd.Timestamp("2020-12-31")
        predictor_cols = [c for c in frame.columns if c not in ("date", target_col)]
        assert len(predictor_cols) == 15
        assert f"lag_{target_col}" in predictor_cols
        assert np.isfinite(frame.drop(columns="date").to_numpy()).all()
        assert (frame[f"lag_{target_col}"] == frame[target_col]).all()


@pytest.mark.skipif(
    not (BONDS_PATH.exists() and STD_PATH.exists()) or not _grid_path().exists(),
    reason="bond grid cache missing; run `doit bonds_study`",
)
def test_bond_grid_structure_and_finiteness():
    """Both targets cached with the market grid's full cell structure (14 P
    levels, 7 ridge z's plus ridgeless), the same fixed seed list, and finite
    statistics throughout."""
    per_seed = _per_seed()
    assert set(per_seed["target"]) == {"ltr_excess", "corpr_excess"}
    for _, rows in per_seed.groupby("target"):
        assert rows["P"].nunique() == 14
        assert set(np.log10(rows.loc[rows["z"] > 0, "z"].unique()).round()) == set(
            range(-3, 4)
        )
        n_seeds = rows["seed"].nunique()
        assert set(rows["seed"].unique()) == set(range(n_seeds))
        assert len(rows) == 14 * 8 * n_seeds
        stats = rows[["r2", "sharpe", "alpha", "information_ratio", "alpha_tstat"]]
        assert np.isfinite(stats.to_numpy()).all()
    seed_sets = per_seed.groupby("target")["seed"].agg(frozenset)
    assert seed_sets.nunique() == 1  # identical seed list across targets
