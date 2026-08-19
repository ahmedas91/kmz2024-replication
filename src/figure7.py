"""Replicate Figure 7 of KMZ (2024) from the cached OOS grid (T = 12).

Reads the across-seed averaged grid (``_data/oos_grid_T{T}.parquet``, built by
``doit estimate``) and draws the paper's four panels against model complexity
c = P/T: Panel A the OOS R^2, Panel B the mean L2 norm of beta_hat, Panel C the
timing-strategy expected return, Panel D its volatility (monthly units, as in
the paper). One line per ridge shrinkage level, log10(z) in {-3, ..., 3}, in
the paper's own legend colors, on the paper's broken x-axis (see
``figure_style``); the grid's ridgeless z = 0 rows are not drawn (Figure 7
shows the ridge grid; ridgeless is the Figure 8 anchor case).

Panels A, B, and D pin the paper's y-limits (-3..0, 0..3, 0..5), so the
boundary spikes clip exactly as they do in the paper. Expectation management
(the paper's point, not a bug): the OOS R^2 is NEGATIVE and collapses toward
the interpolation boundary for low shrinkage, yet the Figure 8 trading
performance still improves with complexity.

Writes ``_output/figure7.png`` (plus a ``.pdf`` twin for LaTeX) and the plotted
panel data to ``_output/figure7_data.parquet``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from figure_style import (
    INK_MUTED,
    PAPER_LINE_COLORS,
    broken_axis_pair,
    finish_broken_pair,
    z_line_label,
)
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)

GRID_PATH = DATA_DIR / f"oos_grid_T{TRAIN_WINDOW}.parquet"

PANELS = (
    ("r2", "Panel A: $R^2$"),
    ("beta_norm", r"Panel B: $\|\hat{\beta}\|$"),
    ("mean_return", "Panel C: Expected Return"),
    ("volatility", "Panel D: Volatility"),
)
# The paper's y-limits; Panel C scales to the data (its ceiling is not pinned
# in the paper either).
PAPER_YLIMS = {"r2": (-3.0, 0.1), "beta_norm": (0.0, 3.0), "volatility": (0.0, 5.0)}


def load_panel_data(grid_path=GRID_PATH):
    """The ridge rows (z > 0) of the averaged grid, one row per (P, z)."""
    grid = pd.read_parquet(grid_path)
    ridge = grid.loc[grid["z"] > 0.0].sort_values(["z", "c"]).reset_index(drop=True)
    return ridge[["P", "z", "c", "r2", "beta_norm", "mean_return", "volatility"]]


def plot_figure7(panel_data, output_stem):
    """Render the 2x2 panel figure with the paper's broken x-axis; save png+pdf."""
    z_values = sorted(panel_data["z"].unique())
    fig = plt.figure(figsize=(10.0, 7.4), facecolor="white")
    outer = fig.add_gridspec(
        2, 2, left=0.065, right=0.985, top=0.95, bottom=0.08, hspace=0.36, wspace=0.16
    )
    legend_axis = None
    for cell, (column, title) in enumerate(PANELS):
        ax_main, ax_tail = broken_axis_pair(fig, outer[cell // 2, cell % 2])
        if legend_axis is None:
            legend_axis = ax_main
        for z, color in zip(z_values, PAPER_LINE_COLORS):
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
        ax_main.axvline(
            1.0,
            color=INK_MUTED,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            zorder=2,
            label="$c = 1$",
        )
        ylim = PAPER_YLIMS.get(column, (0.0, 1.15 * panel_data[column].max()))
        finish_broken_pair(ax_main, ax_tail, title, ylim)
    # One legend for the whole figure, in Panel A as in the paper.
    legend_axis.legend(fontsize=7, frameon=False, loc="center right")
    fig.savefig(f"{output_stem}.png", dpi=300)
    fig.savefig(f"{output_stem}.pdf")
    plt.close(fig)


def main():
    panel_data = load_panel_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_data.to_parquet(OUTPUT_DIR / "figure7_data.parquet")
    plot_figure7(panel_data, OUTPUT_DIR / "figure7")
    print(
        f"[figure7] wrote figure7.png/.pdf and figure7_data.parquet "
        f"({len(panel_data)} rows, {panel_data['z'].nunique()} z lines)"
    )


if __name__ == "__main__":
    main()
