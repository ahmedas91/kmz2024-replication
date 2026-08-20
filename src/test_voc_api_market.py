"""Market-data guards for the reusable VoC API (issue #14).

Two promises the refactor makes, checked against the real pipeline data
(both skip until the corresponding artifacts exist):

1. **Regression guard**: the market study run through the new public API
   (``voc.run_voc_study``) reproduces the previously cached per-seed
   statistics exactly, for a spot-check of (P, z) cells and seeds. The
   engine is deterministic given a seed, so any drift here means the
   refactor changed behavior, which the issue forbids.
2. **No-drift guard**: ``voc.preprocessing`` re-implements the
   standardization conventions of ``standardize_kmz`` so the package is
   self-contained; the two must produce identical numbers on the real
   dataset, forever.

The synthetic API tests live with the package in ``voc/test_voc_api.py``;
these need the project config and data, so they live in ``src``.
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the top-level `voc` package importable when pytest collects from src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)
STD_PATH = DATA_DIR / "kmz_dataset_standardized.parquet"
# The regression target is always the PAPER-period cache (bare name).
PER_SEED_PATH = DATA_DIR / f"oos_grid_T{TRAIN_WINDOW}_per_seed.parquet"

requires_std_data = pytest.mark.skipif(
    not STD_PATH.exists(),
    reason=f"{STD_PATH.name} not found; run `doit standardize` first",
)
requires_cache = pytest.mark.skipif(
    not (STD_PATH.exists() and PER_SEED_PATH.exists()),
    reason="standardized dataset or cached per-seed grid not found; run `doit estimate`",
)


@lru_cache(maxsize=1)
def _paper_dataset():
    df = pd.read_parquet(STD_PATH)
    cutoff = pd.Timestamp("2020-12") + pd.offsets.MonthEnd(0)
    return df.loc[df["date"] <= cutoff].reset_index(drop=True)


@requires_cache
def test_api_reproduces_cached_market_stats():
    """run_voc_study reproduces the cached per-seed stats for spot-check cells."""
    from voc import run_voc_study

    # The cache under test is the paper-period one (bare filename), so the
    # recomputation trims to the paper period regardless of the configured
    # SAMPLE_END; the guard is meaningful under any .env.
    dataset = _paper_dataset()
    p_grid, z_grid, seeds = (2, 12, 96), (0.1, 1000.0), range(4)
    predictor_cols = [c for c in dataset.columns if c not in ("date", "mkt_excess")]
    fresh, _ = run_voc_study(
        dataset["mkt_excess"],
        dataset[predictor_cols],
        T=TRAIN_WINDOW,
        p_grid=p_grid,
        z_grid=z_grid,
        seeds=seeds,
        n_jobs=1,
    )
    cached = pd.read_parquet(PER_SEED_PATH)
    cached = cached.loc[
        cached["P"].isin(p_grid)
        & cached["z"].isin(list(z_grid) + [0.0])
        & cached["seed"].isin(list(seeds))
    ]
    keys = ["seed", "P", "z"]
    fresh = fresh.set_index(keys).sort_index()
    cached = cached.set_index(keys).sort_index()[fresh.columns]
    assert len(fresh) == len(p_grid) * (len(z_grid) + 1) * len(list(seeds))
    pd.testing.assert_frame_equal(fresh, cached, atol=1e-12, rtol=0.0)


@requires_std_data
def test_preprocessing_matches_standardize_kmz():
    """voc.preprocessing reproduces standardize_kmz's conventions bit for bit
    on the real tidy dataset: trailing uncentered vol for the return,
    expanding vol for a slow and a fast predictor."""
    from clean_goyal_welch import load_kmz_dataset
    from standardize_kmz import expanding_vol, trailing_uncentered_vol
    from voc import preprocessing

    kmz = load_kmz_dataset(data_dir=DATA_DIR)
    ret = kmz["mkt_excess"]
    np.testing.assert_allclose(
        preprocessing.trailing_uncentered_vol(ret.to_numpy(), window=12),
        trailing_uncentered_vol(ret, window=12).to_numpy(),
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )
    for column in ("dp", "svar"):
        np.testing.assert_allclose(
            preprocessing.expanding_vol(kmz[column].to_numpy(), min_periods=36),
            expanding_vol(kmz[column], min_periods=36).to_numpy(),
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
