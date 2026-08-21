"""Render the Nagel diagnostics, counterfactuals, and matched-twin results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nagel_analysis import ANATOMY_PATH, METRICS_PATH, SPANNING_PATH
from nagel_experiments import (
    COUNTERFACTUAL_SPANNING_PATH,
    COUNTERFACTUALS_PATH,
    TWIN_PATHS_PATH,
    TWIN_RECOVERY_PATH,
    TWIN_SPANNING_PATH,
)
from sample_period import SAMPLE_SUFFIX
from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))

TABLE_PATH = OUTPUT_DIR / f"nagel_comparison_table{SAMPLE_SUFFIX}.tex"
EXPERIMENT_TABLE_PATH = OUTPUT_DIR / f"nagel_experiments_table{SAMPLE_SUFFIX}.tex"
FIGURE_STEM = OUTPUT_DIR / f"figure_nagel{SAMPLE_SUFFIX}"
TWIN_FIGURE_STEM = OUTPUT_DIR / f"figure_nagel_twins{SAMPLE_SUFFIX}"

SERIES_COLOR = "#2a78d6"
BENCH_COLOR = "#c23b22"
DESIGN_COLORS = {"x14_shared": "#2a78d6", "x15_world_lag": "#d17a22"}


def build_table(metrics, spanning):
    """Supplemental standardized-target comparison table."""
    columns = [
        "strategy",
        "sharpe",
        "alpha",
        "information_ratio",
        "alpha_tstat",
        "r2",
    ]
    display = metrics[columns].rename(
        columns={
            "strategy": "Strategy",
            "sharpe": "Sharpe",
            "alpha": "Alpha",
            "information_ratio": "IR",
            "alpha_tstat": "Alpha $t$",
            "r2": "OOS $R^2$",
        }
    )
    body = display.to_latex(
        index=False, escape=False, float_format=lambda value: f"{value:.3f}"
    )
    row = spanning.iloc[0]
    note = (
        "\n\n\\smallskip\\noindent Standardized-target diagnostic. Statistics "
        f"are averaged over {int(row.n_seeds)} RFF draws. Spanning alpha "
        f"$={row.alpha:.4f}$ ($t={row.alpha_tstat:.2f}$), Nagel-payoff "
        f"$\\beta={row.beta_benchmark:.3f}$; projection $R^2={row.anatomy_r2:.3f}$.\n"
    )
    return body + note


def _counterfactual_panel(counterfactuals, spanning):
    labels = {
        "historical": "Historical",
        "ma2_reversal": "MA(2) reversal",
        "wild_bootstrap": "Wild predictors",
    }
    rff = counterfactuals.loc[counterfactuals["strategy"] == "RFF"].set_index(
        "experiment"
    )
    nagel = counterfactuals.loc[
        counterfactuals["strategy"] == "Volatility-timed momentum"
    ].set_index("experiment")
    span = spanning.set_index("experiment")
    order = ["historical", "ma2_reversal", "wild_bootstrap"]
    return pd.DataFrame(
        {
            "Experiment": [labels[key] for key in order],
            "RFF IR": [rff.loc[key, "information_ratio"] for key in order],
            "RFF alpha $t$": [rff.loc[key, "alpha_tstat"] for key in order],
            "Nagel IR": [nagel.loc[key, "information_ratio"] for key in order],
            "Spanned IR": [span.loc[key, "information_ratio"] for key in order],
            "Spanned alpha $t$": [span.loc[key, "alpha_tstat"] for key in order],
        }
    )


def build_experiment_table(counterfactuals, counter_spanning, recovery, twin_spanning):
    """Three compact LaTeX panels corresponding to Slides 16--18."""
    panel_a = _counterfactual_panel(counterfactuals, counter_spanning).to_latex(
        index=False, escape=False, float_format=lambda value: f"{value:.3f}"
    )
    recovery_display = recovery[
        ["predictor_set", "strength", "correlation", "slope", "nrmse"]
    ].copy()
    recovery_display["predictor_set"] = recovery_display["predictor_set"].map(
        {"x14_shared": "14 shared inputs", "x15_world_lag": "15 world inputs"}
    )
    recovery_display = recovery_display.rename(
        columns={
            "predictor_set": "Inputs",
            "strength": "Signal",
            "correlation": "Corr.",
            "slope": "Slope",
            "nrmse": "NRMSE",
        }
    )
    panel_b = recovery_display.to_latex(
        index=False, escape=False, float_format=lambda value: f"{value:.3f}"
    )
    span_display = twin_spanning[
        [
            "predictor_set",
            "strength",
            "market",
            "alpha_tstat",
            "information_ratio",
            "beta_benchmark",
        ]
    ].copy()
    span_display["predictor_set"] = span_display["predictor_set"].map(
        {"x14_shared": "14 shared", "x15_world_lag": "15 world"}
    )
    span_display["market"] = span_display["market"].map({"plus": "$+$", "minus": "$-$"})
    span_display = span_display.rename(
        columns={
            "predictor_set": "Inputs",
            "strength": "Signal",
            "market": "Twin",
            "alpha_tstat": "Alpha $t$",
            "information_ratio": "IR",
            "beta_benchmark": "Nagel $\\beta$",
        }
    )
    panel_c = span_display.to_latex(
        index=False, escape=False, float_format=lambda value: f"{value:.3f}"
    )
    n_seeds = int(counterfactuals["n_seeds"].iloc[0])
    return (
        "\\textit{Panel A: Nagel's direct tests (raw return target)}\\par\n"
        + panel_a
        + "\n\\medskip\\textit{Panel B: twin-signal recovery}\\par\n"
        + panel_b
        + "\n\\medskip\\textit{Panel C: twin payoff spanning}\\par\n"
        + panel_c
        + "\n\\smallskip\\noindent Results average completed statistics over "
        + f"{n_seeds} paired RFF draws at $T=12$, $P=12{{,}}000$, ridgeless.\n"
    )


def plot_lag_weights(anatomy, output_stem):
    """Plot the supplemental forecast projection against declining weights."""
    fig, axis = plt.subplots(figsize=(7.0, 4.0), facecolor="white")
    lag = anatomy["lag"]
    projection_column = (
        "voc_projection_weight"
        if "voc_projection_weight" in anatomy
        else "voc_lag_weight"
    )
    axis.bar(
        lag,
        anatomy[projection_column],
        color=SERIES_COLOR,
        alpha=0.75,
        label="VoC forecast projection",
    )
    axis.plot(
        lag,
        anatomy["benchmark_weight"],
        color=BENCH_COLOR,
        marker="o",
        linewidth=1.6,
        label="Declining momentum weights",
    )
    axis.axhline(0, color="#888888", linewidth=0.7)
    axis.set_xlabel("Return lag (months)")
    axis.set_ylabel("Projection coefficient")
    axis.set_title("Forecast projection on lagged returns (supplemental diagnostic)")
    axis.legend(frameon=False, fontsize=8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{output_stem}.png", dpi=300)
    fig.savefig(f"{output_stem}.pdf")
    plt.close(fig)


def plot_twin_recovery(paths, output_stem):
    """Show recovery for the plotted across-seed ensemble forecast paths."""
    strengths = ["easy", "realistic"]
    designs = ["x14_shared", "x15_world_lag"]
    titles = {"easy": "Easy signal", "realistic": "Realistic signal"}
    design_labels = {
        "x14_shared": "14 identical inputs",
        "x15_world_lag": "15 inputs (own lagged return)",
    }
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), facecolor="white")
    for row_index, design in enumerate(designs):
        for column_index, strength in enumerate(strengths):
            axis = axes[row_index, column_index]
            subset = paths.loc[
                (paths["predictor_set"] == design) & (paths["strength"] == strength)
            ]
            truth = subset["truth"].to_numpy()
            recovered = subset["recovered"].to_numpy()
            intercept, slope = np.linalg.lstsq(
                np.column_stack([np.ones(truth.size), truth]),
                recovered,
                rcond=None,
            )[0]
            correlation = np.corrcoef(recovered, truth)[0, 1]
            nrmse = np.sqrt(np.mean((recovered - truth) ** 2)) / truth.std()
            axis.scatter(
                truth,
                recovered,
                s=7,
                alpha=0.25,
                color=DESIGN_COLORS[design],
                edgecolors="none",
            )
            lower = float(min(truth.min(), recovered.min()))
            upper = float(max(truth.max(), recovered.max()))
            axis.plot([lower, upper], [lower, upper], "--", color="#555555", lw=1)
            grid = np.linspace(lower, upper, 50)
            axis.plot(
                grid,
                intercept + slope * grid,
                color=BENCH_COLOR,
                linewidth=1.3,
            )
            axis.set_title(f"{titles[strength]}: {design_labels[design]}", fontsize=9)
            axis.text(
                0.03,
                0.97,
                f"corr={correlation:.2f}  slope={slope:.2f}\n"
                f"NRMSE={nrmse:.2f}",
                transform=axis.transAxes,
                va="top",
                fontsize=8,
            )
            axis.set_xlabel("Planted $a g(X_t)$")
            axis.set_ylabel("Recovered $(\\hat f^+ - \\hat f^-)/2$")
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
    fig.suptitle("Matched-twin recovery of the planted nonlinear rule", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{output_stem}.png", dpi=300)
    fig.savefig(f"{output_stem}.pdf")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_parquet(METRICS_PATH)
    anatomy = pd.read_parquet(ANATOMY_PATH)
    spanning = pd.read_parquet(SPANNING_PATH)
    counterfactuals = pd.read_parquet(COUNTERFACTUALS_PATH)
    counter_spanning = pd.read_parquet(COUNTERFACTUAL_SPANNING_PATH)
    recovery = pd.read_parquet(TWIN_RECOVERY_PATH)
    twin_spanning = pd.read_parquet(TWIN_SPANNING_PATH)
    twin_paths = pd.read_parquet(TWIN_PATHS_PATH)

    TABLE_PATH.write_text(build_table(metrics, spanning))
    EXPERIMENT_TABLE_PATH.write_text(
        build_experiment_table(
            counterfactuals, counter_spanning, recovery, twin_spanning
        )
    )
    plot_lag_weights(anatomy, FIGURE_STEM)
    plot_twin_recovery(twin_paths, TWIN_FIGURE_STEM)
    print(
        f"[nagel] wrote {TABLE_PATH.name}, {EXPERIMENT_TABLE_PATH.name}, "
        f"{FIGURE_STEM.name}.png, and {TWIN_FIGURE_STEM.name}.png"
    )


if __name__ == "__main__":
    main()
