"""Run the international equities VoC study (issue #16) and cache the grid.

External-validity test through the generic ``voc.run_voc_study`` API: can
the same machinery, built on US predictor information, time the developed
equity markets OUTSIDE the US? One study on the Ken French developed-ex-US
market excess return (``pull_intl_equity``), configured exactly like the
bonds studies:

- Predictors: the 14 standardized Goyal-Welch predictors from the shared
  analysis dataset plus the target's own lag (the trailing-vol-standardized
  target copied onto the same row; the engine applies the single t -> t+1
  shift itself).
- Target standardization: trailing 12-month uncentered volatility
  (``voc.preprocessing.trailing_uncentered_vol``).
- Grids: the engine's full default P and z grids at T = TRAIN_WINDOW,
  ridgeless included. No P-grid trim was needed (the issue allows one): the
  short sample makes the full grid cheap, ~1 minute at 200 seeds.

Two design caveats, stated wherever the results appear (the issue's
requirement): the information set is US predictors forecasting non-US
returns, and the sample is short — the French series starts 1990-07, so
after the 12-month target-volatility burn-in the study runs on roughly 350
months (about a third of the market study's 1,092), with correspondingly
wider uncertainty; the predictors' own 1930-01 floor never binds here.

Seed count: INTL_N_SEEDS defaults to 200 (the bonds rationale: no published
anchors exist, so the seed budget only needs smooth curves).

Writes ``_data/oos_grid_intl_T{T}{suffix}.parquet`` (across-seed means) and
``..._per_seed.parquet``, tagged by a ``target`` column for schema
consistency with the bonds cache. ``dodo.py`` imports the path and config
constants from this module. Keep module import light.
"""

import sys
import time
from pathlib import Path

# Make the top-level `voc` package importable when doit runs `python ./src/...`
# (which puts src/, not the repo root, on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sample_period import SAMPLE_SUFFIX, trim_to_sample
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
INTL_N_SEEDS = config("INTL_N_SEEDS", default=200, cast=int)
N_JOBS = config("N_JOBS", default=-1, cast=int)
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)

INTL_AVERAGED_PATH = DATA_DIR / f"oos_grid_intl_T{TRAIN_WINDOW}{SAMPLE_SUFFIX}.parquet"
INTL_PER_SEED_PATH = (
    DATA_DIR / f"oos_grid_intl_T{TRAIN_WINDOW}{SAMPLE_SUFFIX}_per_seed.parquet"
)

TARGET_LABELS = {"intl_mkt_excess": "Developed ex-US equities"}
TARGET_COL = "intl_mkt_excess"


def build_intl_study_inputs(data_dir=DATA_DIR, sample_end=None):
    """The aligned study frame: date, standardized target, own lag, and the
    14 standardized US predictors; inner-joined, finite, trimmed to
    ``sample_end`` (default the configured SAMPLE_END). First row lands
    ~1991-07: the target's 12-month vol burn-in on the 1990-07 series start.
    """
    import pandas as pd

    from pull_intl_equity import load_intl_equity
    from standardize_kmz import load_standardized_dataset
    from voc.preprocessing import trailing_uncentered_vol

    predictors = load_standardized_dataset(data_dir=data_dir).drop(
        columns=["mkt_excess", "lag_mkt_excess"]
    )
    intl = load_intl_equity(data_dir=data_dir)
    raw_target = intl[TARGET_COL].to_numpy()
    # The 12-month volatility window is the KMZ RETURN-standardization
    # convention (footnote 34), fixed at 12 regardless of TRAIN_WINDOW.
    standardized_target = raw_target / trailing_uncentered_vol(raw_target, window=12)
    target = pd.DataFrame(
        {
            "date": intl["date"],
            TARGET_COL: standardized_target,
            f"lag_{TARGET_COL}": standardized_target,
        }
    )
    frame = target.merge(predictors, on="date", how="inner")
    frame = frame.dropna().reset_index(drop=True)
    frame = trim_to_sample(frame, sample_end)
    if frame.empty:
        raise ValueError(f"no rows left for {TARGET_COL} through {sample_end}")
    return frame


def main():
    """Run the international study and cache the per-seed and averaged grids."""

    from voc.oos_engine import run_voc_study

    inputs = build_intl_study_inputs()
    predictor_cols = [c for c in inputs.columns if c not in ("date", TARGET_COL)]
    span = f"{inputs['date'].min():%Y-%m} to {inputs['date'].max():%Y-%m}"
    print(
        f"[intl] {TARGET_LABELS[TARGET_COL]}: {len(inputs)} months ({span}), "
        f"{len(predictor_cols)} predictors, INTL_N_SEEDS={INTL_N_SEEDS}"
    )
    start = time.perf_counter()
    per_seed, averaged = run_voc_study(
        inputs[TARGET_COL],
        inputs[predictor_cols],
        dates=inputs["date"].to_numpy(),
        T=TRAIN_WINDOW,
        seeds=range(INTL_N_SEEDS),
        n_jobs=N_JOBS,
    )
    per_seed.insert(0, "target", TARGET_COL)
    averaged.insert(0, "target", TARGET_COL)
    anchor = averaged.loc[
        (averaged["z"] == 0.0) & (averaged["P"] == averaged["P"].max())
    ].iloc[0]
    print(
        f"[intl] done in {time.perf_counter() - start:.0f}s; ridgeless "
        f"c={anchor.c:g}: r2={anchor.r2:.4f}, sharpe={anchor.sharpe:.3f}"
    )
    averaged.to_parquet(INTL_AVERAGED_PATH)
    per_seed.to_parquet(INTL_PER_SEED_PATH)
    print(f"[intl] wrote {INTL_AVERAGED_PATH.name} and {INTL_PER_SEED_PATH.name}")


if __name__ == "__main__":
    main()
