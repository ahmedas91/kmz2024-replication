"""Structure and ranking tests for the Figure 11 replication (issue #10).

The paper's Figure 11 (printed page 52; T = 12, P = 12,000, z = 10^3): the
three most important variables by out-of-sample R^2 are the lagged market
return (VI 1.9%), the long-term bond return ltr (1.3%), and the default
return dfr (0.8%), with inflation fourth; the persistent valuation ratios
(dp, dy, ep, b/m, d/e) sit near zero. That split is the paper's answer to
"how can 12 months of training data be enough?" — the model feeds on the
fast-moving predictors.

ASSERTION DESIGN, stated for the record: the cache averages FIG11_N_SEEDS =
100 seed draws per configuration versus the paper's 1,000, so the fine
ordering of the near-zero variables is noise; tests therefore assert the
top-group/bottom-group split (the issue's instruction), not an exact
ordering. One deliberate deviation from the issue text: the issue sketch
lists svar among the top fast movers, but the paper's own figure places svar
13th of 15 with VI near zero — the paper's figure is the authority here, so
svar carries no top-group assertion. Magnitude floors are loose (documented
against the paper's values) because the ~25% forecast-scale offset
documented in issues #8/#9 does not cancel exactly in R^2 differences; the
ranking, which is the figure's message, is scale-robust.

The seed list is fixed by config (range(FIG11_N_SEEDS)), so reruns are
deterministic; no seeds were selected on outcomes. Real-file tests skip
(rather than fail) when the cache has not been built yet
(`doit variable_importance`), since data never ships with the repository.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)
VI_PATH = DATA_DIR / "variable_importance_T12.parquet"  # rankings are T=12-specific

requires_data = pytest.mark.skipif(
    not VI_PATH.exists(),
    reason=f"{VI_PATH.name} not found; run `doit variable_importance` first",
)

FULL_MODEL_LABEL = "none"
PREDICTORS = {
    "dfy", "infl", "svar", "de", "lty", "tms", "tbl", "dfr",
    "dp", "dy", "ltr", "ep", "bm", "ntis", "lag_mkt_excess",
}  # fmt: skip
# The paper's top group (its three named leaders plus inflation, fourth) and
# the persistent valuation ratios it places at the bottom.
FAST_MOVERS = {"lag_mkt_excess", "ltr", "dfr", "infl"}
VALUATION_RATIOS = {"dp", "dy", "ep", "bm", "de"}


@lru_cache(maxsize=1)
def _per_seed():
    return pd.read_parquet(VI_PATH)


@lru_cache(maxsize=1)
def _vi():
    """One row per predictor: vi_r2 and vi_sharpe, full model minus leave-one-out."""
    means = _per_seed().groupby("excluded")[["r2", "sharpe"]].mean()
    full = means.loc[FULL_MODEL_LABEL]
    return (full - means.drop(index=FULL_MODEL_LABEL)).rename(
        columns={"r2": "vi_r2", "sharpe": "vi_sharpe"}
    )


@requires_data
def test_cache_structure():
    """16 configurations (full + 15 exclusions), identical fixed seed set in
    each, one (P = 12,000, z = 1000) row per seed, all metrics finite."""
    per_seed = _per_seed()
    assert set(per_seed["excluded"]) == PREDICTORS | {FULL_MODEL_LABEL}
    assert (per_seed["P"] == 12_000).all()
    assert (per_seed["z"] == 1_000.0).all()
    seed_sets = per_seed.groupby("excluded")["seed"].agg(frozenset)
    assert seed_sets.nunique() == 1  # same seeds everywhere
    assert seed_sets.iloc[0] == frozenset(range(len(seed_sets.iloc[0])))
    assert per_seed.groupby(["excluded", "seed"]).size().eq(1).all()
    vi = _vi()
    assert len(vi) == 15
    assert np.isfinite(vi[["vi_r2", "vi_sharpe"]]).all().all()


@requires_data
def test_r2_ranking_matches_paper():
    """The paper's R^2 ranking: lag_mkt first and its top three are
    {lag_mkt, ltr, dfr} as a set; infl in the top five; no valuation ratio
    intrudes into the top three; the fast group out-ranks the ratio group on
    average."""
    vi_r2 = _vi()["vi_r2"].sort_values(ascending=False)
    assert vi_r2.index[0] == "lag_mkt_excess"
    assert set(vi_r2.index[:3]) == {"lag_mkt_excess", "ltr", "dfr"}
    assert "infl" in set(vi_r2.index[:5])
    assert not VALUATION_RATIOS & set(vi_r2.index[:3])
    ranks = pd.Series(range(len(vi_r2)), index=vi_r2.index, dtype=float)
    assert ranks[list(FAST_MOVERS)].mean() < ranks[list(VALUATION_RATIOS)].mean()


@requires_data
def test_r2_magnitudes_in_paper_ballpark():
    """Loose magnitude floors against the paper's values (1.9%, 1.3%, 0.8%
    for the top three; near zero for the rest): the leader clears 0.5% and
    every bottom-group variable stays within 1% of zero in absolute value."""
    vi_r2 = _vi()["vi_r2"].sort_values(ascending=False)
    assert vi_r2.iloc[0] > 0.005
    assert vi_r2.iloc[3:].abs().lt(0.010).all()


@requires_data
def test_sharpe_vi_leader_matches_paper():
    """The paper's Sharpe-VI line peaks at the lagged market return (~0.13),
    with ltr and dfr also positive; infl is NEGATIVE in Sharpe units in the
    paper (its line dips below zero at the fourth bar), so it gets no
    positivity assertion."""
    vi_sharpe = _vi()["vi_sharpe"]
    assert vi_sharpe.idxmax() == "lag_mkt_excess"
    assert vi_sharpe["lag_mkt_excess"] > 0.0
    assert vi_sharpe["ltr"] > 0.0
    assert vi_sharpe["dfr"] > 0.0
