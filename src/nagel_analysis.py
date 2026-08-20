"""Run Nagel's (2025) critique tests against the replicated VoC market strategy.

Loads the standardized market returns and the anchor forecast export
(``forecasts_market{suffix}.parquet``, P = 12,000 ridgeless, built by
``doit export_forecasts``), then applies the reusable
:mod:`voc.nagel` toolkit: builds the momentum benchmark, regresses the VoC
forecast on the trailing returns (anatomy), and spans the VoC strategy by the
benchmark. Writes tidy results to ``_data`` for the table/figure script. The
market study is one caller of the toolkit; an asset-class study would be another.

Findings are reported as found, whichever side they favor.
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
    momentum_benchmark,
    spanning_regression,
)

DATA_DIR = Path(config("DATA_DIR"))
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)


def result_paths(data_dir=DATA_DIR):
    """The four Nagel artifact paths under ``data_dir`` (input first).

    One place for the names, imported by ``table_nagel`` and ``dodo.py``,
    so the scripts and the build graph can never disagree.
    """
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


def load_anchor_forecasts(path):
    """Across-seed mean VoC forecast/realized/strategy at the ridgeless anchor,
    one row per month, ordered by date."""
    forecasts = pd.read_parquet(path)
    ridgeless = forecasts.loc[forecasts["z"] == 0.0]
    if ridgeless.empty:
        raise ValueError(f"no ridgeless (z=0) rows in {Path(path).name}")
    return (
        ridgeless.groupby("obs", sort=True)[["forecast", "realized", "strategy"]]
        .mean()
        .reset_index()
    )


def run(data_dir=DATA_DIR, train_window=TRAIN_WINDOW):
    """Run all three critique tests; write and return (metrics, anatomy, spanning)."""
    T = train_window
    paths = result_paths(data_dir)
    dataset = trim_to_sample(load_standardized_dataset(data_dir=data_dir))
    returns = dataset["mkt_excess"].to_numpy(dtype=np.float64)
    ts = np.arange(T, returns.shape[0] - 1)  # the engine's OOS decision points

    voc = load_anchor_forecasts(paths["forecasts"])
    if len(voc) != ts.shape[0]:
        raise ValueError(
            f"forecast export has {len(voc)} months but the engine OOS has "
            f"{ts.shape[0]}; sample mismatch"
        )
    voc_forecast = voc["forecast"].to_numpy()
    voc_realized = voc["realized"].to_numpy()
    voc_strategy = voc["strategy"].to_numpy()

    bench = momentum_benchmark(returns, T=T)
    np.testing.assert_allclose(
        bench["realized"], voc_realized, err_msg="benchmark and VoC OOS months differ"
    )

    # (a) same-metrics comparison of the two strategies.
    metrics = pd.DataFrame(
        [
            {
                "strategy": "VoC (ridgeless, c=1000)",
                **benchmark_metrics(voc_forecast, voc_realized),
            },
            {
                "strategy": "Momentum benchmark",
                **benchmark_metrics(bench["forecast"], bench["realized"]),
            },
        ]
    )

    # (b) forecast anatomy: VoC forecast on the trailing T standardized returns.
    anat = forecast_anatomy(voc_forecast, returns, ts, T=T)
    anatomy = pd.DataFrame(
        {
            "lag": np.arange(1, T + 1),
            "voc_lag_weight": anat["lag_weights"],
            "benchmark_weight": declining_weights(T),
        }
    )

    # (c) spanning: VoC strategy on the benchmark strategy + the static market.
    span = spanning_regression(voc_strategy, bench["strategy"], voc_realized)
    spanning = pd.DataFrame(
        [{**span, "anatomy_r2": anat["r2"], "anatomy_intercept": anat["intercept"]}]
    )

    metrics.to_parquet(paths["metrics"])
    anatomy.to_parquet(paths["anatomy"])
    spanning.to_parquet(paths["spanning"])
    return metrics, anatomy, spanning


def main():
    """Run the analysis and print the headline comparison numbers."""
    metrics, _, spanning = run()
    voc = metrics.iloc[0]
    bench = metrics.iloc[1]
    s = spanning.iloc[0]
    print(
        f"[nagel] VoC Sharpe {voc.sharpe:.3f} / alpha_t {voc.alpha_tstat:.2f}  vs  "
        f"benchmark Sharpe {bench.sharpe:.3f} / alpha_t {bench.alpha_tstat:.2f}"
    )
    print(
        f"[nagel] anatomy R^2 {s.anatomy_r2:.3f}; spanning alpha {s.alpha:.4f} "
        f"(t={s.alpha_tstat:.2f}), benchmark beta {s.beta_benchmark:.3f}"
    )
    print(
        f"[nagel] wrote {METRICS_PATH.name}, {ANATOMY_PATH.name}, {SPANNING_PATH.name}"
    )


if __name__ == "__main__":
    main()
