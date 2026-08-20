"""Run the recursive OOS grid on the standardized KMZ dataset and cache it.

Thin IO/config driver on top of the pure engine: it loads the volatility-
standardized analysis dataset (``standardize_kmz``), trims it to the estimation
sample, runs the reusable VoC engine across seeds, and writes the long-format
grid statistics to ``_data``. All estimation math and conventions live in
``voc/``; the market study is one configuration of the generic
``voc.run_voc_study`` API (issue #14), with the target, predictors, and grids
supplied here.

Config (via ``.env`` / command line, so reruns need zero code edits):
  N_SEEDS       RFF repetitions (default 500; the paper averages 1,000 — the
                interpolation-boundary cells are extremely noisy per seed, so
                low seed counts leave visible wiggles in Figure 7 Panel C)
  N_JOBS        joblib parallelism across seeds (default -1 = all cores)
  TRAIN_WINDOW  rolling training window T (default 12); embedded in the output
                filenames, so different T values cache side by side
  SAMPLE_END    last month (YYYY-MM) of the estimation sample (see
                ``sample_period``). The default, 2020-12, is the paper's
                sample end (1,092 months from 1930-01) and writes the bare
                canonical filenames; any other value (e.g. 2024-12 for the
                updated-sample rerun) suffixes the output filenames with the
                sample end, so both runs coexist. The standardized parquet
                itself extends past the paper period.
  SAVE_FORECASTS  "1"/"true" additionally exports the per-seed forecast,
                realized-return, and strategy-return SERIES at the anchor
                configuration (P = 12,000, z = 10^3 plus ridgeless, same
                seeds) to ``_data/forecasts_market{suffix}.parquet`` — the
                input the Nagel study (issue #17) needs. Deliberately anchor-
                only: exporting the full grid would be seeds x 112 cells x
                months, gigabytes of parquet. Default off; the export run
                adds a minute or two.

``dodo.py`` imports the path and config constants from this module, so the
build graph and the script can never disagree about filenames, and doit reruns
the task when N_SEEDS changes (SAMPLE_END is visible in the filenames). Keep
module import light (no pandas/engine imports at top level) so that stays
cheap.
"""

import sys
import time
from pathlib import Path

# Make the top-level `voc` package importable when doit runs `python ./src/...`
# (which puts src/, not the repo root, on sys.path). Proper packaging is issue #14.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sample_period import SAMPLE_END, SAMPLE_SUFFIX
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
N_SEEDS = config("N_SEEDS", default=500, cast=int)
N_JOBS = config("N_JOBS", default=-1, cast=int)
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)
SAVE_FORECASTS = str(
    config("SAVE_FORECASTS", default="0", cast=str)
).strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
)

AVERAGED_PATH = DATA_DIR / f"oos_grid_T{TRAIN_WINDOW}{SAMPLE_SUFFIX}.parquet"
PER_SEED_PATH = DATA_DIR / f"oos_grid_T{TRAIN_WINDOW}{SAMPLE_SUFFIX}_per_seed.parquet"
# The anchor configuration for the optional forecast export: the paper's
# highest-complexity model at heavy shrinkage, plus its ridgeless limit.
ANCHOR_P = 12_000
ANCHOR_Z = 1_000.0


def main():
    # Heavy imports live here so dodo.py can import the constants above without
    # paying for pandas and the engine at every doit parse.
    import pandas as pd

    from standardize_kmz import load_standardized_dataset
    from voc.oos_engine import run_voc_study

    dataset = load_standardized_dataset(data_dir=DATA_DIR)
    sample_end = pd.Timestamp(SAMPLE_END) + pd.offsets.MonthEnd(0)
    dataset = dataset.loc[dataset["date"] <= sample_end].reset_index(drop=True)
    if dataset.empty:
        raise ValueError(f"SAMPLE_END={SAMPLE_END} leaves no rows in the sample")

    # The market study as one configuration of the generic API.
    predictor_cols = [c for c in dataset.columns if c not in ("date", "mkt_excess")]
    target = dataset["mkt_excess"]
    predictors = dataset[predictor_cols]
    dates = dataset["date"].to_numpy()

    span = f"{dataset['date'].min():%Y-%m} to {dataset['date'].max():%Y-%m}"
    print(f"[estimate] estimation sample: {len(dataset)} months ({span})")
    print(f"[estimate] grid: N_SEEDS={N_SEEDS}, N_JOBS={N_JOBS}, T={TRAIN_WINDOW}")

    start = time.perf_counter()
    per_seed, averaged = run_voc_study(
        target,
        predictors,
        dates=dates,
        T=TRAIN_WINDOW,
        seeds=range(N_SEEDS),
        n_jobs=N_JOBS,
    )
    print(
        f"[estimate] finished in {time.perf_counter() - start:.1f}s - "
        f"{len(per_seed)} per-seed rows, {len(averaged)} (P, z) cells"
    )

    averaged.to_parquet(AVERAGED_PATH)
    per_seed.to_parquet(PER_SEED_PATH)
    print(f"[estimate] wrote {AVERAGED_PATH.name} and {PER_SEED_PATH.name}")

    if SAVE_FORECASTS:
        study_name = f"market{SAMPLE_SUFFIX}"
        print(f"[estimate] exporting anchor forecasts (study_name={study_name})")
        start = time.perf_counter()
        run_voc_study(
            target,
            predictors,
            dates=dates,
            T=TRAIN_WINDOW,
            p_grid=(ANCHOR_P,),
            z_grid=(ANCHOR_Z,),
            include_ridgeless=True,
            seeds=range(N_SEEDS),
            n_jobs=N_JOBS,
            save_forecasts=True,
            study_name=study_name,
            data_dir=DATA_DIR,
        )
        print(
            f"[estimate] wrote forecasts_{study_name}.parquet in "
            f"{time.perf_counter() - start:.1f}s"
        )


if __name__ == "__main__":
    main()
