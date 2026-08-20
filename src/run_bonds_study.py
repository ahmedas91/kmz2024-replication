"""Run the bonds VoC studies (issue #15) and cache the grid statistics.

Two studies through the generic ``voc.run_voc_study`` API, one per bond
excess-return target from ``clean_bond_returns``: long-term government
(``ltr_excess``) and corporate (``corpr_excess``). Each mirrors the market
study's configuration exactly:

- Predictors: the 14 standardized Goyal-Welch predictors from the shared
  analysis dataset (every market-study predictor except ``lag_mkt_excess``),
  PLUS the target's own lag — the trailing-vol-standardized target copied
  onto the same row, never pre-shifted, because the engine applies the
  single t -> t+1 shift itself. For the government target this means raw
  ``ltr`` (expanding-vol scheme) remains among the 14 while the own-lag is
  standardized ``ltr - Rfree`` (trailing scheme): two different transforms
  of closely related series, exactly as the issue's own-lag convention
  implies, and worth knowing when reading variable loadings.
- Target standardization: trailing 12-month uncentered volatility
  (``voc.preprocessing.trailing_uncentered_vol``, the package step).
- Sample: inner-join with the standardized predictor set, whose 36-month
  burn-in and 1930-01 floor bind, then trim to ``SAMPLE_END`` month-end;
  the cache is keyed by the sample period like every estimation artifact.
- Grids: the engine's default P and z grids at T = TRAIN_WINDOW, ridgeless
  included.

Seed count: BONDS_N_SEEDS defaults to 200 versus the market study's 500.
No published anchors exist for bonds, so the seed budget only needs smooth
curves, not anchor precision; 200 seeds put the across-seed standard error
well inside line width, and the two studies together run in ~6 minutes.

Writes ``_data/oos_grid_bonds_T{T}{suffix}.parquet`` (across-seed means)
and ``..._per_seed.parquet``, both with a ``target`` column tagging the two
studies. ``dodo.py`` imports the path and config constants from this module.
Keep module import light (no pandas/engine imports at top level).
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
BONDS_N_SEEDS = config("BONDS_N_SEEDS", default=200, cast=int)
N_JOBS = config("N_JOBS", default=-1, cast=int)
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)

BONDS_AVERAGED_PATH = (
    DATA_DIR / f"oos_grid_bonds_T{TRAIN_WINDOW}{SAMPLE_SUFFIX}.parquet"
)
BONDS_PER_SEED_PATH = (
    DATA_DIR / f"oos_grid_bonds_T{TRAIN_WINDOW}{SAMPLE_SUFFIX}_per_seed.parquet"
)

# Paper display names for figures and pages, keyed by target column.
TARGET_LABELS = {
    "ltr_excess": "Long-term government bonds",
    "corpr_excess": "Corporate bonds",
}


def build_bond_study_inputs(target_col, data_dir=DATA_DIR, sample_end=None):
    """The aligned study frame for one bond target.

    Columns: ``date``, ``target_col`` (the trailing-vol-standardized bond
    excess return, the y variable), ``lag_{target_col}`` (the same values on
    the same row — the own-lag predictor under the engine's single-shift
    convention), and the 14 standardized Goyal-Welch predictors. Rows are the
    inner join of the two sources with every value finite, trimmed to
    ``sample_end`` (default: the configured SAMPLE_END).
    """
    import pandas as pd

    from clean_bond_returns import load_bond_returns
    from standardize_kmz import load_standardized_dataset
    from voc.preprocessing import trailing_uncentered_vol

    predictors = load_standardized_dataset(data_dir=data_dir).drop(
        columns=["mkt_excess", "lag_mkt_excess"]
    )
    bonds = load_bond_returns(data_dir=data_dir)
    raw_target = bonds[target_col].to_numpy()
    # The 12-month volatility window is the KMZ RETURN-standardization
    # convention (footnote 34), fixed at 12 regardless of TRAIN_WINDOW.
    standardized_target = raw_target / trailing_uncentered_vol(raw_target, window=12)
    target = pd.DataFrame(
        {
            "date": bonds["date"],
            target_col: standardized_target,
            f"lag_{target_col}": standardized_target,
        }
    )
    frame = target.merge(predictors, on="date", how="inner")
    frame = frame.dropna().reset_index(drop=True)
    frame = trim_to_sample(frame, sample_end)
    if frame.empty:
        raise ValueError(f"no rows left for {target_col} through {sample_end}")
    return frame


def main():
    """Run both bond studies and cache the tagged per-seed and averaged grids."""
    import pandas as pd

    from voc.oos_engine import run_voc_study

    per_seed_frames, averaged_frames = [], []
    for target_col, label in TARGET_LABELS.items():
        inputs = build_bond_study_inputs(target_col)
        predictor_cols = [c for c in inputs.columns if c not in ("date", target_col)]
        span = f"{inputs['date'].min():%Y-%m} to {inputs['date'].max():%Y-%m}"
        print(
            f"[bonds] {label} ({target_col}): {len(inputs)} months ({span}), "
            f"{len(predictor_cols)} predictors, BONDS_N_SEEDS={BONDS_N_SEEDS}"
        )
        start = time.perf_counter()
        per_seed, averaged = run_voc_study(
            inputs[target_col],
            inputs[predictor_cols],
            dates=inputs["date"].to_numpy(),
            T=TRAIN_WINDOW,
            seeds=range(BONDS_N_SEEDS),
            n_jobs=N_JOBS,
        )
        per_seed.insert(0, "target", target_col)
        averaged.insert(0, "target", target_col)
        per_seed_frames.append(per_seed)
        averaged_frames.append(averaged)
        anchor = averaged.loc[
            (averaged["z"] == 0.0) & (averaged["P"] == averaged["P"].max())
        ].iloc[0]
        print(
            f"[bonds] {target_col} done in {time.perf_counter() - start:.0f}s; "
            f"ridgeless c={anchor.c:g}: r2={anchor.r2:.4f}, "
            f"sharpe={anchor.sharpe:.3f}",
            flush=True,
        )

    pd.concat(averaged_frames, ignore_index=True).to_parquet(BONDS_AVERAGED_PATH)
    pd.concat(per_seed_frames, ignore_index=True).to_parquet(BONDS_PER_SEED_PATH)
    print(f"[bonds] wrote {BONDS_AVERAGED_PATH.name} and {BONDS_PER_SEED_PATH.name}")


if __name__ == "__main__":
    main()
