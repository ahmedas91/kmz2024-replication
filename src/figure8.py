"""Replicate Figure 8 of KMZ (2024) from the cached OOS grid (T = 12).

The paper's crux figure: market-timing performance RISES with model
complexity. Reads the across-seed averaged grid (``_data/oos_grid_T{T}.parquet``,
built by ``doit estimate``) and draws the paper's four panels against
complexity c = P/T: Panel A the annualized Sharpe ratio, Panel B the monthly
alpha, Panel C the annualized information ratio, Panel D the alpha
t-statistic. Alphas are versus a static position in the volatility-
standardized market (the paper's Figure 8 caption). One line per ridge
shrinkage level in the paper's legend colors on the paper's broken x-axis
(see ``figure_style``), plus the ridgeless (z -> 0 minimum-norm) limit as a
dashed dark line — the case the paper's quantitative anchors (and the tests
in ``test_figure8.py``) pin at c = 1000.

Panel frames start at 0 and cover at least the paper's plotted range, so the
replication reads side by side against the paper's page 45 figure; the frame
stretches if the data demand it, so nothing is silently clipped.

Writes ``_output/figure8.png`` (plus a ``.pdf`` twin for LaTeX) and the plotted
panel data, ridgeless rows included, to ``_output/figure8_data.parquet``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from figure_style import (
    INK_MUTED,
    PAPER_LINE_COLORS,
    RIDGELESS_STYLE,
    add_lower_right_legend,
    broken_axis_pair,
    finish_broken_pair,
    save_png_pdf,
    z_line_label,
)
from sample_period import SAMPLE_SUFFIX
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)

# SAMPLE_SUFFIX is empty for the paper period; an updated-sample run reads
# and writes its own suffixed artifact set (see sample_period).
GRID_PATH = DATA_DIR / f"oos_grid_T{TRAIN_WINDOW}{SAMPLE_SUFFIX}.parquet"

PANELS = (
    ("sharpe", "Panel A: Sharpe Ratio"),
    ("alpha", "Panel B: Alpha"),
    ("information_ratio", "Panel C: Information Ratio"),
    ("alpha_tstat", "Panel D: Alpha $t$-statistic"),
)
# The paper's plotted ranges and tick units (page 45): top, step; tick labels
# carry two decimals in every panel. The frame stretches beyond the paper's
# top only if the data demand it.
PAPER_YAXES = {
    "sharpe": (0.48, 0.10),
    "alpha": (0.025, 0.01),
    "information_ratio": (0.32, 0.05),
    "alpha_tstat": (3.0, 0.5),
}
YTICK_DECIMALS = 2


def load_panel_data(grid_path=GRID_PATH):
    """All rows of the averaged grid (ridgeless z = 0 included), sorted."""
    grid = pd.read_parquet(grid_path)
    grid = grid.sort_values(["z", "c"]).reset_index(drop=True)
    return grid[["P", "z", "c", "sharpe", "alpha", "information_ratio", "alpha_tstat"]]


def plot_figure8(panel_data, output_stem):
    """Render the 2x2 panel figure with the paper's broken x-axis; save png+pdf."""
    ridge_z = sorted(panel_data.loc[panel_data["z"] > 0.0, "z"].unique())
    ridgeless = panel_data.loc[panel_data["z"] == 0.0]
    fig = plt.figure(figsize=(10.0, 7.4), facecolor="white")
    outer = fig.add_gridspec(
        2, 2, left=0.065, right=0.985, top=0.95, bottom=0.08, hspace=0.36, wspace=0.16
    )
    legend_axis = legend_tail = None
    for cell, (column, title) in enumerate(PANELS):
        ax_main, ax_tail = broken_axis_pair(fig, outer[cell // 2, cell % 2])
        if legend_axis is None:
            legend_axis, legend_tail = ax_main, ax_tail
        for z, color in zip(ridge_z, PAPER_LINE_COLORS):
            line = panel_data.loc[panel_data["z"] == z]
            for ax in (ax_main, ax_tail):
                ax.plot(
                    line["c"],
                    line[column],
                    color=color,
                    linewidth=1.6,
                    label=z_line_label(z) if ax is ax_main else None,
                    zorder=3,
                )
        for ax in (ax_main, ax_tail):
            ax.plot(
                ridgeless["c"],
                ridgeless[column],
                label=r"ridgeless ($z \to 0$)" if ax is ax_main else None,
                **RIDGELESS_STYLE,
            )
        ax_main.axvline(
            1.0,
            color=INK_MUTED,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            zorder=2,
            label="$c = 1$",
        )
        paper_top, ytick_step = PAPER_YAXES[column]
        top = max(paper_top, 1.05 * panel_data[column].max())
        finish_broken_pair(
            ax_main, ax_tail, title, (0.0, top), ytick_step, YTICK_DECIMALS
        )
    # One boxed legend in Panel A's lower-right corner, drawn at figure level
    # so it may sit on top of the axis break and sliver, as in the paper; the
    # curves climb toward the upper right, so that corner stays free of data.
    add_lower_right_legend(fig, legend_axis, legend_tail)
    save_png_pdf(fig, output_stem)


def main():
    """Load the grid, render Figure 8, and write the plotted data parquet."""
    panel_data = load_panel_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_data.to_parquet(OUTPUT_DIR / f"figure8_data{SAMPLE_SUFFIX}.parquet")
    plot_figure8(panel_data, OUTPUT_DIR / f"figure8{SAMPLE_SUFFIX}")
    ridgeless_anchor = panel_data.loc[
        (panel_data["z"] == 0.0) & (panel_data["P"] == panel_data["P"].max())
    ].iloc[0]
    print(
        f"[figure8] wrote figure8{SAMPLE_SUFFIX}.png/.pdf and "
        f"figure8_data{SAMPLE_SUFFIX}.parquet; ridgeless "
        f"c={ridgeless_anchor.c:g}: sharpe={ridgeless_anchor.sharpe:.3f}, "
        f"IR={ridgeless_anchor.information_ratio:.3f}, "
        f"alpha_t={ridgeless_anchor.alpha_tstat:.2f}"
    )


if __name__ == "__main__":
    main()
