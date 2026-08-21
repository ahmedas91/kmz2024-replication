"""Supplemental Nagel diagnostics in the existing KMZ-standardized workflow.

This driver preserves the project's historical ``nagel_metrics``, ``anatomy``,
and ``spanning`` artifacts while correcting two fidelity problems: the simple
benchmark now includes Nagel's inverse predictor-variance timing, and nonlinear
performance statistics are computed per RFF draw before they are averaged.

These artifacts intentionally retain the KMZ-standardized dependent return so
they remain comparable to Figures 7--8.  The source-faithful raw-return MA(2),
wild-bootstrap, and matched-twin experiments live in ``nagel_experiments.py``.
The anatomy is explicitly a projection diagnostic, not Nagel's interpolation
weight vector.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sample_period import SAMPLE_SUFFIX, trim_to_sample
from settings import config
from standardize_kmz import load_standardized_dataset
from voc.nagel import (
    benchmark_metrics,
    declining_weights,
    forecast_anatomy,
    spanning_regression,
    volatility_timed_momentum,
)

DATA_DIR = Path(config("DATA_DIR"))
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)


def result_paths(data_dir=DATA_DIR):
    """Paths for the forecast input and three supplemental diagnostics."""
    data_dir = Path(data_dir)
    return {
        "forecasts": data_dir / f"forecasts_market{SAMPLE_SUFFIX}.parquet",
        "metrics": data_dir / f"nagel_metrics{SAMPLE_SUFFIX}.parquet",
        "anatomy": data_dir / f"nagel_anatomy{SAMPLE_SUFFIX}.parquet",
        "spanning": data_dir / f"nagel_spanning{SAMPLE_SUFFIX}.parquet",
    }


_DEFAULT_PATHS = result_paths()
FORECASTS_PATH = _DEFAULT_PATHS["forecasts"]
METRICS_PATH = _DEFAULT_PATHS["metrics"]
ANATOMY_PATH = _DEFAULT_PATHS["anatomy"]
SPANNING_PATH = _DEFAULT_PATHS["spanning"]


def load_anchor_forecasts_by_seed(path):
    """Load one ordered ridgeless-anchor forecast path per RFF seed."""
    forecasts = pd.read_parquet(path)
    ridgeless = forecasts.loc[forecasts["z"] == 0.0].copy()
    if ridgeless.empty:
        raise ValueError(f"no ridgeless (z=0) rows in {Path(path).name}")
    if "P" in ridgeless:
        ridgeless = ridgeless.loc[ridgeless["P"] == ridgeless["P"].max()]
    duplicated = ridgeless.duplicated(["seed", "obs"])
    if duplicated.any():
        raise ValueError("anchor export has duplicate (seed, obs) rows")
    return ridgeless.sort_values(["seed", "obs"]).reset_index(drop=True)


def load_anchor_forecasts(path):
    """Backward-compatible across-seed ensemble forecast, one row per month."""
    forecasts = load_anchor_forecasts_by_seed(path)
    return (
        forecasts.groupby("obs", sort=True)[["forecast", "realized", "strategy"]]
        .mean()
        .reset_index()
    )


def _mean_dicts(rows):
    """Average numeric dictionaries without tying aggregation to row order."""
    frame = pd.DataFrame(rows)
    return frame.mean(numeric_only=True).to_dict()


def run(data_dir=DATA_DIR, train_window=TRAIN_WINDOW):
    """Build standardized-target diagnostics and write their three parquets."""
    paths = result_paths(data_dir)
    dataset = trim_to_sample(load_standardized_dataset(data_dir=data_dir))
    returns = dataset["mkt_excess"].to_numpy(dtype=np.float64)
    predictor_columns = [
        column for column in dataset if column not in ("date", "mkt_excess")
    ]
    predictors = dataset[predictor_columns].to_numpy(dtype=np.float64)
    decision_points = np.arange(train_window, returns.size - 1)

    by_seed = load_anchor_forecasts_by_seed(paths["forecasts"])
    seed_groups = list(by_seed.groupby("seed", sort=True))
    if not seed_groups:
        raise ValueError("anchor forecast export contains no seeds")
    expected_rows = decision_points.size
    if any(len(group) != expected_rows for _, group in seed_groups):
        raise ValueError("one or more seed forecast paths do not match the OOS sample")

    benchmark = volatility_timed_momentum(returns, predictors, T=train_window)
    first_realized = seed_groups[0][1]["realized"].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(
        benchmark["realized"],
        first_realized,
        err_msg="benchmark and VoC OOS months differ",
    )

    rff_metric_rows = []
    span_rows = []
    for _, group in seed_groups:
        forecast = group["forecast"].to_numpy(dtype=np.float64)
        realized = group["realized"].to_numpy(dtype=np.float64)
        strategy = forecast * realized
        rff_metric_rows.append(benchmark_metrics(forecast, realized))
        span_rows.append(spanning_regression(strategy, benchmark["strategy"], realized))

    metrics = pd.DataFrame(
        [
            {
                "strategy": "VoC (ridgeless, c=1000)",
                **_mean_dicts(rff_metric_rows),
                "n_seeds": len(seed_groups),
                "target_scale": "KMZ standardized",
            },
            {
                "strategy": "Volatility-timed momentum",
                **benchmark_metrics(benchmark["forecast"], benchmark["realized"]),
                "n_seeds": 1,
                "target_scale": "KMZ standardized",
            },
        ]
    )

    ensemble = load_anchor_forecasts(paths["forecasts"])
    anatomy_fit = forecast_anatomy(
        ensemble["forecast"].to_numpy(dtype=np.float64),
        returns,
        decision_points,
        T=train_window,
    )
    anatomy = pd.DataFrame(
        {
            "lag": np.arange(1, train_window + 1),
            "voc_projection_weight": anatomy_fit["lag_weights"],
            # Backward-compatible alias consumed by older notebooks/tests.
            "voc_lag_weight": anatomy_fit["lag_weights"],
            "benchmark_weight": declining_weights(train_window),
        }
    )

    spanning = pd.DataFrame(
        [
            {
                **_mean_dicts(span_rows),
                "n_seeds": len(seed_groups),
                "aggregation": "mean of per-seed statistics",
                "target_scale": "KMZ standardized",
                "anatomy_r2": anatomy_fit["r2"],
                "anatomy_intercept": anatomy_fit["intercept"],
            }
        ]
    )

    metrics.to_parquet(paths["metrics"], index=False)
    anatomy.to_parquet(paths["anatomy"], index=False)
    spanning.to_parquet(paths["spanning"], index=False)
    return metrics, anatomy, spanning


def main():
    metrics, _, spanning = run()
    voc = metrics.loc[metrics["strategy"].str.startswith("VoC")].iloc[0]
    benchmark = metrics.loc[metrics["strategy"] == "Volatility-timed momentum"].iloc[0]
    span = spanning.iloc[0]
    print(
        f"[nagel:diagnostic] VoC Sharpe {voc.sharpe:.3f} vs "
        f"vol-momentum {benchmark.sharpe:.3f}; spanning t={span.alpha_tstat:.2f}"
    )
    print(
        "[nagel:diagnostic] standardized target; mean of per-seed statistics; "
        f"wrote {METRICS_PATH.name}, {ANATOMY_PATH.name}, {SPANNING_PATH.name}"
    )


if __name__ == "__main__":
    main()
