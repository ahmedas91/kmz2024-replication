"""Run the Figure 11 variable-importance estimations and cache them.

The paper's Section V.E experiment: re-estimate the machine learning model 16
times at the single setting the paper uses for Figure 11 (T = 12, P = 12,000,
z = 10^3) — once on the full 15-predictor set and once per predictor with that
predictor excluded. Variable importance for predictor i is the change in
performance (OOS R^2, and separately Sharpe ratio) moving from the full model
to the 14-variable model without i; ``figure11.py`` computes those differences
from this cache. Omegas are redrawn per submodel because the RFF input
dimension changes from 15 to 14 (inherent to the method); the seed list is
identical across all 16 configurations.

Thin IO/config driver in the mold of ``run_estimation.py``: loads the
volatility-standardized dataset, trims to the estimation sample, calls the
pure engine (``voc.oos_engine.run_grid``) once per configuration, and writes
the per-seed statistics of all 16 runs — tagged by an ``excluded`` column,
``"none"`` for the full model — to a single parquet in ``_data``.

Config (via ``.env`` / command line):
  FIG11_N_SEEDS  RFF repetitions per configuration (default 100). The paper
                 averages 1,000; at 100 seeds the top-group/bottom-group split
                 is stable but the fine ordering of near-zero variables is not
                 (see ``test_figure11.py``). 16 configurations at P = 12,000
                 make this the most expensive task in the project (~30-60 min
                 on all cores), so the count is deliberately below N_SEEDS.
  N_JOBS         joblib parallelism across seeds (default -1 = all cores)
  TRAIN_WINDOW   rolling training window T (default 12); in the output filename
  SAMPLE_END     last month (YYYY-MM) of the estimation sample (default
                 2020-12, the paper's sample end). A non-paper value suffixes
                 the output filename with the sample end (see
                 ``sample_period``), so both runs coexist.

``dodo.py`` imports VI_PATH and the config constants from this module. Keep
module import light (no pandas/engine imports at top level) so that stays cheap.
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
FIG11_N_SEEDS = config("FIG11_N_SEEDS", default=100, cast=int)
N_JOBS = config("N_JOBS", default=-1, cast=int)
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)

VI_PATH = DATA_DIR / f"variable_importance_T{TRAIN_WINDOW}{SAMPLE_SUFFIX}.parquet"

# The paper's Figure 11 setting: the highest-complexity model on the grid at
# heavy shrinkage (P = 12,000, z = 10^3), ridge only.
FIG11_P = 12_000
FIG11_Z = 1_000.0
TARGET_COL = "mkt_excess"
FULL_MODEL_LABEL = "none"


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

    predictors = [c for c in dataset.columns if c not in ("date", TARGET_COL)]
    configurations = [FULL_MODEL_LABEL] + predictors
    span = f"{dataset['date'].min():%Y-%m} to {dataset['date'].max():%Y-%m}"
    print(f"[vi] estimation sample: {len(dataset)} months ({span})")
    print(
        f"[vi] {len(configurations)} configurations at P={FIG11_P}, z={FIG11_Z:g}, "
        f"T={TRAIN_WINDOW}; FIG11_N_SEEDS={FIG11_N_SEEDS}, N_JOBS={N_JOBS}"
    )

    frames = []
    start = time.perf_counter()
    for i, excluded in enumerate(configurations, start=1):
        cols = [c for c in predictors if c != excluded]
        config_start = time.perf_counter()
        per_seed, _ = run_grid(
            dataset,
            target_col=TARGET_COL,
            predictor_cols=cols,
            T=TRAIN_WINDOW,
            p_grid=(FIG11_P,),
            z_grid=(FIG11_Z,),
            include_ridgeless=False,
            seeds=range(FIG11_N_SEEDS),
            n_jobs=N_JOBS,
        )
        per_seed.insert(0, "excluded", excluded)
        frames.append(per_seed)
        print(
            f"[vi] {i}/{len(configurations)} excluded={excluded}: "
            f"r2={per_seed['r2'].mean():.4f}, sharpe={per_seed['sharpe'].mean():.3f} "
            f"({time.perf_counter() - config_start:.0f}s)",
            flush=True,
        )

    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(VI_PATH)
    print(
        f"[vi] finished in {time.perf_counter() - start:.0f}s - wrote "
        f"{VI_PATH.name} ({len(result)} per-seed rows)"
    )


if __name__ == "__main__":
    main()
