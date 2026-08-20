"""Run Nagel's (2025) critique tests against the replicated VoC market strategy.

Loads the standardized market returns and the anchor forecast export
(``forecasts_market{suffix}.parquet``, P = 12,000 ridgeless, written by
``doit estimate`` with ``SAVE_FORECASTS=1``), then applies the reusable
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

from sample_period import SAMPLE_END, SAMPLE_SUFFIX
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

FORECASTS_PATH = DATA_DIR / f"forecasts_market{SAMPLE_SUFFIX}.parquet"
METRICS_PATH = DATA_DIR / f"nagel_metrics{SAMPLE_SUFFIX}.parquet"
ANATOMY_PATH = DATA_DIR / f"nagel_anatomy{SAMPLE_SUFFIX}.parquet"
SPANNING_PATH = DATA_DIR / f"nagel_spanning{SAMPLE_SUFFIX}.parquet"


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
    T = train_window
    dataset = load_standardized_dataset(data_dir=data_dir)
    sample_end = pd.Timestamp(SAMPLE_END) + pd.offsets.MonthEnd(0)
    dataset = dataset.loc[dataset["date"] <= sample_end].reset_index(drop=True)
    returns = dataset["mkt_excess"].to_numpy(dtype=np.float64)
    ts = np.arange(T, returns.shape[0] - 1)  # the engine's OOS decision points

    voc = load_anchor_forecasts(data_dir / f"forecasts_market{SAMPLE_SUFFIX}.parquet")
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
            {"strategy": "VoC (ridgeless, c=1000)", **benchmark_metrics(voc_forecast, voc_realized)},
            {"strategy": "Momentum benchmark", **benchmark_metrics(bench["forecast"], bench["realized"])},
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

    metrics.to_parquet(METRICS_PATH)
    anatomy.to_parquet(ANATOMY_PATH)
    spanning.to_parquet(SPANNING_PATH)
    return metrics, anatomy, spanning


def main():
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
    print(f"[nagel] wrote {METRICS_PATH.name}, {ANATOMY_PATH.name}, {SPANNING_PATH.name}")


if __name__ == "__main__":
    main()
