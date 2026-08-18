"""Run the recursive OOS grid on the standardized KMZ dataset and cache it.

Thin IO/config driver on top of the pure engine: it loads the volatility-
standardized analysis dataset (``standardize_kmz``), trims it to the estimation
sample, runs the reusable VoC engine (``voc.oos_engine.run_grid``) across seeds,
and writes the long-format grid statistics to ``_data``. All estimation math and
conventions live in ``voc/``.

Config (via ``.env`` / command line, so reruns need zero code edits):
  N_SEEDS       RFF repetitions (default 50; the paper averages 1,000)
  N_JOBS        joblib parallelism across seeds (default -1 = all cores)
  TRAIN_WINDOW  rolling training window T (default 12); embedded in the output
                filenames, so different T values cache side by side
  SAMPLE_END    last month (YYYY-MM) of the estimation sample. The default,
                2020-12, is the paper's sample end (1,092 months from 1930-01),
                so the cached statistics are comparable to Figures 7/8/11; set
                e.g. 2024-12 for the updated-sample rerun. The standardized
                parquet itself extends past the paper period.

``dodo.py`` imports the path and config constants from this module, so the
build graph and the script can never disagree about filenames, and doit reruns
the task when N_SEEDS or SAMPLE_END change. Keep module import light (no
pandas/engine imports at top level) so that import stays cheap.
"""

import sys
import time
from pathlib import Path

# Make the top-level `voc` package importable when doit runs `python ./src/...`
# (which puts src/, not the repo root, on sys.path). Proper packaging is issue #14.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
N_SEEDS = config("N_SEEDS", default=50, cast=int)
N_JOBS = config("N_JOBS", default=-1, cast=int)
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)
SAMPLE_END = config("SAMPLE_END", default="2020-12", cast=str)

AVERAGED_PATH = DATA_DIR / f"oos_grid_T{TRAIN_WINDOW}.parquet"
PER_SEED_PATH = DATA_DIR / f"oos_grid_T{TRAIN_WINDOW}_per_seed.parquet"


def main():
    # Heavy imports live here so dodo.py can import the constants above without
    # paying for pandas and the engine at every doit parse.
    import pandas as pd

    from standardize_kmz import load_standardized_dataset
    from voc.oos_engine import run_grid

    dataset = load_standardized_dataset(data_dir=DATA_DIR)
    sample_end = pd.Timestamp(SAMPLE_END) + pd.offsets.MonthEnd(0)
    dataset = dataset.loc[dataset["date"] <= sample_end].reset_index(drop=True)
    if dataset.empty:
        raise ValueError(f"SAMPLE_END={SAMPLE_END} leaves no rows in the sample")

    span = f"{dataset['date'].min():%Y-%m} to {dataset['date'].max():%Y-%m}"
    print(f"[estimate] estimation sample: {len(dataset)} months ({span})")
    print(f"[estimate] grid: N_SEEDS={N_SEEDS}, N_JOBS={N_JOBS}, T={TRAIN_WINDOW}")

    start = time.perf_counter()
    per_seed, averaged = run_grid(
        dataset, T=TRAIN_WINDOW, seeds=range(N_SEEDS), n_jobs=N_JOBS
    )
    print(
        f"[estimate] finished in {time.perf_counter() - start:.1f}s - "
        f"{len(per_seed)} per-seed rows, {len(averaged)} (P, z) cells"
    )

    averaged.to_parquet(AVERAGED_PATH)
    per_seed.to_parquet(PER_SEED_PATH)
    print(f"[estimate] wrote {AVERAGED_PATH.name} and {PER_SEED_PATH.name}")


if __name__ == "__main__":
    main()
