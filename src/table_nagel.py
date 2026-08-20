"""Comparison table and lag-weight figure for the Nagel critique (issue #17).

Reads the tidy results from ``nagel_analysis`` (``_data/nagel_*.parquet``) and writes:
- ``_output/nagel_comparison_table.tex``: VoC vs the momentum benchmark on the same
  metrics, plus the spanning-regression alpha and t-statistic;
- ``_output/figure_nagel.png`` (+ ``.pdf``): the VoC forecast's estimated lag
  weights (the anatomy) against the benchmark's linearly declining weights - the
  visual test of Nagel's "recency-weighted momentum" claim.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from nagel_analysis import ANATOMY_PATH, METRICS_PATH, SPANNING_PATH
from sample_period import SAMPLE_SUFFIX
from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))

TABLE_PATH = OUTPUT_DIR / f"nagel_comparison_table{SAMPLE_SUFFIX}.tex"
FIGURE_STEM = OUTPUT_DIR / f"figure_nagel{SAMPLE_SUFFIX}"

SERIES_COLOR = "#2a78d6"
BENCH_COLOR = "#c23b22"


def build_table(metrics, spanning):
    """The strategy-metrics LaTeX table plus a one-line spanning-regression note."""
    cols = ["strategy", "sharpe", "alpha", "information_ratio", "alpha_tstat", "r2"]
    display = metrics[cols].rename(
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
        index=False, escape=False, float_format=lambda x: f"{x:.3f}"
    )
    s = spanning.iloc[0]
    note = (
        "\n\n\\smallskip\\noindent Spanning regression (VoC strategy on the momentum "
        f"benchmark and the static market): alpha $={s.alpha:.4f}$ "
        f"($t={s.alpha_tstat:.2f}$), benchmark $\\beta={s.beta_benchmark:.3f}$; "
        f"forecast-anatomy $R^2={s.anatomy_r2:.3f}$.\n"
    )
    return body + note


def plot_lag_weights(anatomy, output_stem):
    """VoC estimated lag weights (bars) against the benchmark's declining weights."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0), facecolor="white")
    lag = anatomy["lag"]
    ax.bar(
        lag,
        anatomy["voc_lag_weight"],
        color=SERIES_COLOR,
        alpha=0.75,
        label="VoC forecast (estimated)",
    )
    ax.plot(
        lag,
        anatomy["benchmark_weight"],
        color=BENCH_COLOR,
        marker="o",
        linewidth=1.6,
        label="Momentum benchmark (declining)",
    )
    ax.axhline(0, color="#888888", linewidth=0.7)
    ax.set_xlabel("Return lag (months)")
    ax.set_ylabel("Weight")
    ax.set_title(
        "Anatomy of the VoC forecast: lag weights vs the momentum benchmark",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{output_stem}.png", dpi=300)
    fig.savefig(f"{output_stem}.pdf")
    plt.close(fig)


def main():
    """Read the tidy Nagel results; write the LaTeX table and the figure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_parquet(METRICS_PATH)
    anatomy = pd.read_parquet(ANATOMY_PATH)
    spanning = pd.read_parquet(SPANNING_PATH)
    TABLE_PATH.write_text(build_table(metrics, spanning))
    plot_lag_weights(anatomy, FIGURE_STEM)
    print(f"[nagel] wrote {TABLE_PATH.name} and {FIGURE_STEM.name}.png/.pdf")


if __name__ == "__main__":
    main()
