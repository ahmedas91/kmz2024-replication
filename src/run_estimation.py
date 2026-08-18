"""Run the recursive OOS grid on the standardized KMZ dataset and cache it.

Thin IO/config driver on top of the pure engine: it loads the volatility-
standardized analysis dataset (``standardize_kmz``), runs the reusable VoC engine
(``voc.oos_engine.run_grid``) across seeds, and writes the long-format grid
statistics to ``_data``. All estimation math and conventions live in ``voc/``.

Config (via ``.env`` / command line, so reruns need zero code edits):
  N_SEEDS       RFF repetitions (default 50)
  N_JOBS        joblib parallelism across seeds (default -1 = all cores)
  TRAIN_WINDOW  rolling training window T (default 12)
"""

import sys
from datetime import datetime
from pathlib import Path

# Make the top-level `voc` package importable when doit runs `python ./src/...`
# (which puts src/, not the repo root, on sys.path). Proper packaging is issue #14.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from settings import config
from standardize_kmz import load_standardized_dataset

from voc.oos_engine import run_grid

DATA_DIR = Path(config("DATA_DIR"))
N_SEEDS = config("N_SEEDS", default=50, cast=int)
N_JOBS = config("N_JOBS", default=-1, cast=int)
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)

AVERAGED_PATH = DATA_DIR / f"oos_grid_T{TRAIN_WINDOW}.parquet"
PER_SEED_PATH = DATA_DIR / f"oos_grid_T{TRAIN_WINDOW}_per_seed.parquet"


def main():
    dataset = load_standardized_dataset(data_dir=DATA_DIR)
    span = f"{dataset['date'].min():%Y-%m} to {dataset['date'].max():%Y-%m}"
    print(f"[estimate] standardized dataset: {len(dataset)} months ({span})")
    print(f"[estimate] grid: N_SEEDS={N_SEEDS}, N_JOBS={N_JOBS}, T={TRAIN_WINDOW}")

    start = datetime.now()
    per_seed, averaged = run_grid(
        dataset, T=TRAIN_WINDOW, seeds=range(N_SEEDS), n_jobs=N_JOBS
    )
    print(f"[estimate] finished in {datetime.now() - start} - "
          f"{len(per_seed)} per-seed rows, {len(averaged)} (P, z) cells")

    averaged.to_parquet(AVERAGED_PATH)
    per_seed.to_parquet(PER_SEED_PATH)
    print(f"[estimate] wrote {AVERAGED_PATH.name} and {PER_SEED_PATH.name}")


if __name__ == "__main__":
    main()
