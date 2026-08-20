"""Export the per-seed anchor forecast series the Nagel study consumes.

One small estimation task of its own (``doit export_forecasts``): rerun the
market study at the ANCHOR configuration only — P = 12,000 at z = 10^3 plus
the ridgeless limit, the same seed list as the main grid — with the engine's
``save_forecasts`` export enabled, writing the per-seed forecast, realized-
return, and strategy-return series to
``_data/forecasts_market{suffix}.parquet`` (schema: seed, P, z, obs,
forecast, realized, strategy; ~N_SEEDS x 2 cells x months rows).

Modeling this as a doit task (rather than the earlier ``SAVE_FORECASTS``
switch inside the estimate task) puts the file in the build graph: the Nagel
task's file_dep on the export parquet now resolves by building it, so a
plain ``doit`` run works on a fresh clone. The export is anchor-only by
design — the full grid at 500 seeds would be tens of millions of rows.

``dodo.py`` imports FORECASTS_PATH and the config constants from this
module. Keep module import light.
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
N_SEEDS = config("N_SEEDS", default=500, cast=int)
N_JOBS = config("N_JOBS", default=-1, cast=int)
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)

STUDY_NAME = f"market{SAMPLE_SUFFIX}"
FORECASTS_PATH = DATA_DIR / f"forecasts_{STUDY_NAME}.parquet"
# The anchor configuration: the paper's highest-complexity model at heavy
# shrinkage, plus its ridgeless limit (the Figure 8 anchor case).
ANCHOR_P = 12_000
ANCHOR_Z = 1_000.0


def main():
    """Run the anchor-configuration export and write the forecasts parquet."""
    # Heavy imports live here so dodo.py can import the constants above cheaply.
    from standardize_kmz import load_standardized_dataset
    from voc.oos_engine import run_voc_study

    dataset = trim_to_sample(load_standardized_dataset(data_dir=DATA_DIR))
    predictor_cols = [c for c in dataset.columns if c not in ("date", "mkt_excess")]
    print(
        f"[forecasts] anchor export: {len(dataset)} months, P={ANCHOR_P}, "
        f"z={ANCHOR_Z:g} + ridgeless, N_SEEDS={N_SEEDS}"
    )
    start = time.perf_counter()
    run_voc_study(
        dataset["mkt_excess"],
        dataset[predictor_cols],
        dates=dataset["date"].to_numpy(),
        T=TRAIN_WINDOW,
        p_grid=(ANCHOR_P,),
        z_grid=(ANCHOR_Z,),
        include_ridgeless=True,
        seeds=range(N_SEEDS),
        n_jobs=N_JOBS,
        save_forecasts=True,
        study_name=STUDY_NAME,
        data_dir=DATA_DIR,
    )
    print(
        f"[forecasts] wrote {FORECASTS_PATH.name} in {time.perf_counter() - start:.1f}s"
    )


if __name__ == "__main__":
    main()
