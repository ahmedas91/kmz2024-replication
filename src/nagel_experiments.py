"""Run the source-faithful Nagel counterfactuals and matched-twin tests.

The empirical Nagel experiments use raw decimal excess returns as the target,
while retaining the project's expanding-standardized predictors and rolling
RFF standardization.  One fixed MA(2) path is shared across RFF seeds; the wild
bootstrap draws a fresh artificial predictor panel for every RFF seed.  The
twin study pairs random-feature seeds and shocks in the +/- worlds and reports
recovery and world-specific spanning statistics.

Only compact statistics and across-seed recovery paths are cached.  Per-seed
P=12,000 forecast panels are intentionally not written to disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sample_period import SAMPLE_END, SAMPLE_SUFFIX, trim_to_sample
from settings import config
from voc.nagel import (
    add_ma2_reversal,
    benchmark_metrics,
    make_twin_markets,
    recovery_metrics,
    spanning_regression,
    volatility_timed_momentum,
    wild_bootstrap_predictors,
)
from voc.oos_engine import run_recursive_oos
from voc.rff import GAMMA_DEFAULT

DATA_DIR = Path(config("DATA_DIR"))
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)
NAGEL_N_SEEDS = config("NAGEL_N_SEEDS", default=25, cast=int)
NAGEL_N_JOBS = config(
    "NAGEL_N_JOBS", default=config("N_JOBS", default=-1, cast=int), cast=int
)
NAGEL_SIMULATION_SEED = config("NAGEL_SIMULATION_SEED", default=1701, cast=int)
NAGEL_BOOTSTRAP_SEED = config("NAGEL_BOOTSTRAP_SEED", default=2909, cast=int)
NAGEL_TWIN_SHOCK_SEED = config("NAGEL_TWIN_SHOCK_SEED", default=3011, cast=int)
NAGEL_ANCHOR_P = config("NAGEL_ANCHOR_P", default=12_000, cast=int)

TWIN_STRENGTHS = {"easy": 0.20, "realistic": 0.02}
PREDICTOR_14 = (
    "dfy",
    "infl",
    "svar",
    "de",
    "lty",
    "tms",
    "tbl",
    "dfr",
    "dp",
    "dy",
    "ltr",
    "ep",
    "bm",
    "ntis",
)


def result_paths(data_dir=DATA_DIR):
    """Return every compact cache produced by this driver."""
    data_dir = Path(data_dir)
    return {
        "counterfactuals": data_dir / f"nagel_counterfactuals{SAMPLE_SUFFIX}.parquet",
        "counterfactual_spanning": data_dir
        / f"nagel_counterfactual_spanning{SAMPLE_SUFFIX}.parquet",
        "twin_recovery": data_dir / f"nagel_twin_recovery{SAMPLE_SUFFIX}.parquet",
        "twin_spanning": data_dir / f"nagel_twin_spanning{SAMPLE_SUFFIX}.parquet",
        "twin_paths": data_dir / f"nagel_twin_paths{SAMPLE_SUFFIX}.parquet",
    }


_PATHS = result_paths()
COUNTERFACTUALS_PATH = _PATHS["counterfactuals"]
COUNTERFACTUAL_SPANNING_PATH = _PATHS["counterfactual_spanning"]
TWIN_RECOVERY_PATH = _PATHS["twin_recovery"]
TWIN_SPANNING_PATH = _PATHS["twin_spanning"]
TWIN_PATHS_PATH = _PATHS["twin_paths"]


def _anchor_forecast(predictors, returns, seed, *, T, P):
    """Return the single P/ridgeless forecast path for one RFF draw."""
    _, bundle = run_recursive_oos(
        predictors,
        returns,
        seed,
        T=T,
        p_grid=(P,),
        z_grid=(),
        gamma=GAMMA_DEFAULT,
        include_ridgeless=True,
        return_forecasts=True,
    )
    return bundle["forecasts"][(P, 0.0)], bundle["realized"]


def _metric_row(forecast, realized):
    return benchmark_metrics(forecast, realized)


def _direct_experiment_rows(
    seed,
    original_returns,
    original_predictors,
    reversal_returns,
    reversal_predictors,
    *,
    T,
    P,
    bootstrap_seed,
):
    """Per-seed historical, reversal, and wild-bootstrap result rows."""
    bootstrapped_predictors = wild_bootstrap_predictors(
        original_predictors, seed=bootstrap_seed + seed
    )
    designs = (
        ("historical", original_returns, original_predictors),
        ("ma2_reversal", reversal_returns, reversal_predictors),
        ("wild_bootstrap", original_returns, bootstrapped_predictors),
    )
    metric_rows = []
    spanning_rows = []
    for experiment, returns, predictors in designs:
        forecast, realized = _anchor_forecast(predictors, returns, seed, T=T, P=P)
        benchmark = volatility_timed_momentum(returns, predictors, T=T)
        np.testing.assert_allclose(benchmark["realized"], realized)
        metric_rows.extend(
            [
                {
                    "seed": seed,
                    "experiment": experiment,
                    "strategy": "RFF",
                    **_metric_row(forecast, realized),
                },
                {
                    "seed": seed,
                    "experiment": experiment,
                    "strategy": "Volatility-timed momentum",
                    **_metric_row(benchmark["forecast"], realized),
                },
            ]
        )
        spanning_rows.append(
            {
                "seed": seed,
                "experiment": experiment,
                **spanning_regression(
                    forecast * realized, benchmark["strategy"], realized
                ),
            }
        )
    return metric_rows, spanning_rows


def _twin_spanning_row(
    seed,
    predictor_set,
    strength,
    signal_r2,
    market,
    forecast,
    returns,
    predictors,
    *,
    T,
):
    realized = returns[T + 1 :]
    benchmark = volatility_timed_momentum(returns, predictors, T=T)
    np.testing.assert_allclose(benchmark["realized"], realized)
    market_metrics = _metric_row(forecast, realized)
    span = spanning_regression(forecast * realized, benchmark["strategy"], realized)
    return {
        "seed": seed,
        "predictor_set": predictor_set,
        "strength": strength,
        "signal_r2": signal_r2,
        "market": market,
        "sharpe": market_metrics["sharpe"],
        "market_alpha": market_metrics["alpha"],
        "market_alpha_tstat": market_metrics["alpha_tstat"],
        "market_information_ratio": market_metrics["information_ratio"],
        **span,
    }


def _twin_seed_rows(seed, predictors14, twins, *, T, P):
    """Run both twin designs for one paired RFF seed."""
    recovery_rows = []
    spanning_rows = []
    path_rows = []
    decision_points = np.arange(T, predictors14.shape[0] - 1)

    # With an identical 14-column design, ridgeless predictions are linear in
    # y.  Fit shared noise and a unit planted component once, then combine them
    # exactly for both strengths and signs.  This is algebraically identical to
    # fitting every +/- target separately with the same RFF draw.
    reference = next(iter(twins.values()))
    unit_component = np.zeros_like(reference["epsilon"])
    unit_component[1:] = reference["g"][:-1]
    noise_forecast, noise_realized = _anchor_forecast(
        predictors14, reference["epsilon"], seed, T=T, P=P
    )
    signal_forecast, signal_realized = _anchor_forecast(
        predictors14, unit_component, seed, T=T, P=P
    )
    np.testing.assert_allclose(signal_realized, reference["g"][decision_points])

    for strength, twin in twins.items():
        amplitude = twin["amplitude"]
        truth = twin["truth"][decision_points]
        clean_forecasts = {
            "plus": noise_forecast + amplitude * signal_forecast,
            "minus": noise_forecast - amplitude * signal_forecast,
        }
        clean_realized = {
            "plus": noise_realized + truth,
            "minus": noise_realized - truth,
        }
        clean_recovery = recovery_metrics(
            clean_forecasts["plus"], clean_forecasts["minus"], truth
        )
        recovery_rows.append(
            {
                "seed": seed,
                "predictor_set": "x14_shared",
                "strength": strength,
                "signal_r2": twin["signal_r2"],
                **{
                    key: value
                    for key, value in clean_recovery.items()
                    if key != "recovered"
                },
            }
        )
        path_rows.append(
            {
                "seed": seed,
                "predictor_set": "x14_shared",
                "strength": strength,
                "signal_r2": twin["signal_r2"],
                "oos_index": decision_points,
                "truth": truth,
                "recovered": clean_recovery["recovered"],
            }
        )
        for market in ("plus", "minus"):
            # The benchmark uses the return path belonging to its own world.
            spanning_rows.append(
                _twin_spanning_row(
                    seed,
                    "x14_shared",
                    strength,
                    twin["signal_r2"],
                    market,
                    clean_forecasts[market],
                    twin[market],
                    predictors14,
                    T=T,
                )
            )
            np.testing.assert_allclose(clean_realized[market], twin[market][T + 1 :])

        # Robustness: each world receives its own contemporaneously observed
        # return as predictor 15, so the paired designs intentionally differ.
        robust_forecasts = {}
        for market in ("plus", "minus"):
            world_predictors = np.column_stack([predictors14, twin[market]])
            robust_forecasts[market], realized = _anchor_forecast(
                world_predictors, twin[market], seed, T=T, P=P
            )
            np.testing.assert_allclose(realized, twin[market][T + 1 :])
            spanning_rows.append(
                _twin_spanning_row(
                    seed,
                    "x15_world_lag",
                    strength,
                    twin["signal_r2"],
                    market,
                    robust_forecasts[market],
                    twin[market],
                    world_predictors,
                    T=T,
                )
            )
        robust_recovery = recovery_metrics(
            robust_forecasts["plus"], robust_forecasts["minus"], truth
        )
        recovery_rows.append(
            {
                "seed": seed,
                "predictor_set": "x15_world_lag",
                "strength": strength,
                "signal_r2": twin["signal_r2"],
                **{
                    key: value
                    for key, value in robust_recovery.items()
                    if key != "recovered"
                },
            }
        )
        path_rows.append(
            {
                "seed": seed,
                "predictor_set": "x15_world_lag",
                "strength": strength,
                "signal_r2": twin["signal_r2"],
                "oos_index": decision_points,
                "truth": truth,
                "recovered": robust_recovery["recovered"],
            }
        )
    return recovery_rows, spanning_rows, path_rows


def _run_seed(
    seed,
    original_returns,
    original_predictors,
    reversal_returns,
    reversal_predictors,
    predictors14,
    twins,
    *,
    T,
    P,
    bootstrap_seed,
):
    direct_metrics, direct_spanning = _direct_experiment_rows(
        seed,
        original_returns,
        original_predictors,
        reversal_returns,
        reversal_predictors,
        T=T,
        P=P,
        bootstrap_seed=bootstrap_seed,
    )
    twin_recovery, twin_spanning, twin_paths = _twin_seed_rows(
        seed, predictors14, twins, T=T, P=P
    )
    return {
        "counterfactuals": direct_metrics,
        "counterfactual_spanning": direct_spanning,
        "twin_recovery": twin_recovery,
        "twin_spanning": twin_spanning,
        "twin_paths": twin_paths,
    }


def _aggregate(frame, keys):
    """Average completed per-seed statistics and retain the actual seed count."""
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column != "seed"
    ]
    means = frame.groupby(keys, sort=False)[numeric].mean().reset_index()
    counts = frame.groupby(keys, sort=False)["seed"].nunique().rename("n_seeds")
    return means.merge(counts.reset_index(), on=keys, validate="one_to_one")


def _prepare_inputs(data_dir, simulation_seed):
    """Align raw Nagel targets with original and reversal predictor panels."""
    import pandas as pd

    from standardize_kmz import load_standardized_dataset, standardize_kmz_dataset

    raw = pd.read_parquet(Path(data_dir) / "kmz_dataset.parquet").sort_values("date")
    standardized = trim_to_sample(load_standardized_dataset(data_dir=data_dir))
    raw_by_date = raw.set_index("date")
    raw_returns = raw_by_date.loc[standardized["date"], "mkt_excess"].to_numpy()

    artificial_full = add_ma2_reversal(
        raw["mkt_excess"].to_numpy(dtype=np.float64), seed=simulation_seed
    )
    reversal_raw = raw.copy()
    reversal_raw["mkt_excess"] = artificial_full
    reversal_raw["lag_mkt_excess"] = artificial_full
    reversal_standardized = trim_to_sample(standardize_kmz_dataset(df=reversal_raw))
    if not standardized["date"].equals(reversal_standardized["date"]):
        raise ValueError("original and reversal preprocessing produced different dates")
    reversal_by_date = reversal_raw.set_index("date")
    reversal_returns = reversal_by_date.loc[
        reversal_standardized["date"], "mkt_excess"
    ].to_numpy()

    predictor_columns = [
        column for column in standardized if column not in ("date", "mkt_excess")
    ]
    original_predictors = standardized[predictor_columns].to_numpy(dtype=np.float64)
    reversal_predictors = reversal_standardized[predictor_columns].to_numpy(
        dtype=np.float64
    )
    predictors14 = standardized[list(PREDICTOR_14)].to_numpy(dtype=np.float64)
    return {
        "dates": standardized["date"].to_numpy(),
        "original_returns": raw_returns,
        "original_predictors": original_predictors,
        "reversal_returns": reversal_returns,
        "reversal_predictors": reversal_predictors,
        "predictors14": predictors14,
    }


def run(
    data_dir=DATA_DIR,
    *,
    train_window=TRAIN_WINDOW,
    n_seeds=NAGEL_N_SEEDS,
    n_jobs=NAGEL_N_JOBS,
    anchor_p=NAGEL_ANCHOR_P,
    simulation_seed=NAGEL_SIMULATION_SEED,
    bootstrap_seed=NAGEL_BOOTSTRAP_SEED,
    twin_shock_seed=NAGEL_TWIN_SHOCK_SEED,
):
    """Execute every remaining experiment and write five period-keyed caches."""
    import pandas as pd
    from joblib import Parallel, delayed

    if n_seeds < 1:
        raise ValueError("n_seeds must be positive")
    inputs = _prepare_inputs(data_dir, simulation_seed)
    twins = {
        name: make_twin_markets(
            inputs["predictors14"],
            PREDICTOR_14,
            signal_r2=signal_r2,
            shock_seed=twin_shock_seed,
        )
        for name, signal_r2 in TWIN_STRENGTHS.items()
    }
    seed_results = Parallel(n_jobs=n_jobs)(
        delayed(_run_seed)(
            seed,
            inputs["original_returns"],
            inputs["original_predictors"],
            inputs["reversal_returns"],
            inputs["reversal_predictors"],
            inputs["predictors14"],
            twins,
            T=train_window,
            P=anchor_p,
            bootstrap_seed=bootstrap_seed,
        )
        for seed in range(n_seeds)
    )

    def frame_for(key):
        return pd.DataFrame([row for result in seed_results for row in result[key]])

    counterfactuals = _aggregate(
        frame_for("counterfactuals"), ["experiment", "strategy"]
    )
    counterfactual_spanning = _aggregate(
        frame_for("counterfactual_spanning"), ["experiment"]
    )
    twin_recovery = _aggregate(
        frame_for("twin_recovery"), ["predictor_set", "strength"]
    )
    twin_spanning = _aggregate(
        frame_for("twin_spanning"), ["predictor_set", "strength", "market"]
    )

    path_frames = []
    for result in seed_results:
        for row in result["twin_paths"]:
            path_frames.append(
                pd.DataFrame(
                    {
                        "seed": row["seed"],
                        "predictor_set": row["predictor_set"],
                        "strength": row["strength"],
                        "signal_r2": row["signal_r2"],
                        "oos_index": row["oos_index"],
                        "truth": row["truth"],
                        "recovered": row["recovered"],
                    }
                )
            )
    paths_per_seed = pd.concat(path_frames, ignore_index=True)
    twin_paths = (
        paths_per_seed.groupby(
            ["predictor_set", "strength", "signal_r2", "oos_index"],
            sort=False,
        )[["truth", "recovered"]]
        .mean()
        .reset_index()
    )
    twin_paths["obs"] = inputs["dates"][twin_paths["oos_index"].to_numpy(dtype=int) + 1]
    twin_paths["n_seeds"] = n_seeds

    metadata = {
        "T": train_window,
        "P": anchor_p,
        "z": 0.0,
        "gamma": GAMMA_DEFAULT,
        "sample_end": SAMPLE_END,
    }
    for frame in (
        counterfactuals,
        counterfactual_spanning,
        twin_recovery,
        twin_spanning,
        twin_paths,
    ):
        for key, value in metadata.items():
            frame[key] = value
    counterfactuals["target_scale"] = "raw decimal return"
    counterfactuals["simulation_seed"] = simulation_seed
    counterfactuals["bootstrap_seed"] = bootstrap_seed
    counterfactuals["innovation_variance"] = 0.01
    counterfactual_spanning["target_scale"] = "raw decimal return"
    counterfactual_spanning["simulation_seed"] = simulation_seed
    counterfactual_spanning["bootstrap_seed"] = bootstrap_seed
    counterfactual_spanning["innovation_variance"] = 0.01
    for frame in (twin_recovery, twin_spanning, twin_paths):
        frame["shock_seed"] = twin_shock_seed

    paths = result_paths(data_dir)
    for key, frame in (
        ("counterfactuals", counterfactuals),
        ("counterfactual_spanning", counterfactual_spanning),
        ("twin_recovery", twin_recovery),
        ("twin_spanning", twin_spanning),
        ("twin_paths", twin_paths),
    ):
        frame.to_parquet(paths[key], index=False)
    return (
        counterfactuals,
        counterfactual_spanning,
        twin_recovery,
        twin_spanning,
        twin_paths,
    )


def main():
    outputs = run()
    historical_ir = (
        outputs[0]
        .loc[
            (outputs[0]["experiment"] == "historical")
            & (outputs[0]["strategy"] == "RFF"),
            "information_ratio",
        ]
        .iloc[0]
    )
    reversal_ir = (
        outputs[0]
        .loc[
            (outputs[0]["experiment"] == "ma2_reversal")
            & (outputs[0]["strategy"] == "RFF"),
            "information_ratio",
        ]
        .iloc[0]
    )
    print(
        f"[nagel:experiments] RFF IR historical={historical_ir:.3f}, "
        f"MA(2)={reversal_ir:.3f}; {NAGEL_N_SEEDS} paired seeds"
    )
    print(
        "[nagel:experiments] wrote "
        + ", ".join(path.name for path in result_paths().values())
    )


if __name__ == "__main__":
    main()
