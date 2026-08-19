"""Generate the small-multiples time-series overview of the standardized data.

Plots the volatility-standardized market excess return and the 15 predictors
(standardize_kmz) as a 4x4 grid of single-series panels in footnote-33
order, so the persistence split reads at a glance: the valuation ratios and
yields drift slowly while the return-type series (mkt excess, ltr, dfr,
lag mkt) oscillate at high frequency. Panels share the time axis but scale
their own y-axis, because standardization equalizes volatility units, not
levels (dp sits near -10 while lty reaches +20). The sample is trimmed to
the configured ``SAMPLE_END``; a non-paper period writes a suffixed file
alongside the paper one (see ``sample_period``). Writes
``_output/predictor_timeseries{suffix}.png`` and a ``.pdf`` twin for LaTeX.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from sample_period import SAMPLE_SUFFIX
from settings import config
from standardize_kmz import load_standardized_dataset
from table_predictor_summary import DISPLAY_NAMES, SERIES_COLUMNS, trim_to_sample

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

# Single-hue chart: identity lives in the panel titles, not in color.
SERIES_COLOR = "#2a78d6"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
HAIRLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def plot_predictor_grid(df, output_stem):
    """Render the 4x4 standardized-series grid and save png and pdf."""
    fig, axes = plt.subplots(4, 4, figsize=(10.0, 7.0), sharex=True, facecolor="white")
    dates = df["date"]
    for ax, column in zip(axes.flat, SERIES_COLUMNS):
        values = df[column]
        if values.min() < 0 < values.max():
            ax.axhline(0, color=BASELINE, linewidth=0.7, zorder=1)
        ax.plot(dates, values, color=SERIES_COLOR, linewidth=0.6, zorder=2)
        ax.set_title(
            DISPLAY_NAMES[column],
            fontsize=9,
            color=INK_SECONDARY,
            loc="left",
            pad=3,
        )
        ax.tick_params(labelsize=7, colors=INK_MUTED, length=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(HAIRLINE)
        ax.set_facecolor("white")
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    fig.savefig(f"{output_stem}.png", dpi=300)
    fig.savefig(f"{output_stem}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    df = trim_to_sample(load_standardized_dataset(data_dir=DATA_DIR))
    plot_predictor_grid(df, OUTPUT_DIR / f"predictor_timeseries{SAMPLE_SUFFIX}")
