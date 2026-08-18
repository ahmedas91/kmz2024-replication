"""Replicate Figure 7 of KMZ (2024) from the cached OOS grid (T = 12).

Reads the across-seed averaged grid (``_data/oos_grid_T{T}.parquet``, built by
``doit estimate``) and draws the paper's four panels against model complexity
c = P/T: Panel A the OOS R^2, Panel B the mean L2 norm of beta_hat, Panel C the
timing-strategy expected return, Panel D its volatility (monthly units, as in
the paper). One line per ridge shrinkage level, log10(z) in {-3, ..., 3}, in
the paper's own legend colors (the MATLAB default cycle its figures use), so
the replication reads side by side against the paper's page 43 figure; the
grid's ridgeless z = 0 rows are not drawn (Figure 7 shows the ridge grid;
ridgeless is the Figure 8 anchor case).

The x-axis is broken exactly like the paper's: a linear segment over
c in [0, 50] that emphasizes the interpolation boundary (dashed line at
c = 1), then a break, then a sliver at c in [990, 1000] for the
extreme-complexity end. The grid points at c = 64..512 fall inside the break
and are not visible (the paper hides the same region); they remain in
``_output/figure7_data.parquet``, which carries every plotted line in full.

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
import numpy as np
import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)

GRID_PATH = DATA_DIR / f"oos_grid_T{TRAIN_WINDOW}.parquet"

# Ink tokens shared with plot_predictor_timeseries.py.
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
HAIRLINE = "#e1e0d9"

# The paper's line colors (its figures use the MATLAB default cycle), in
# ascending-z order to match its legend: log10(z) = -3, ..., 3.
PAPER_LINE_COLORS = (
    "#0072BD",  # log10(z) = -3
    "#D95319",  # log10(z) = -2
    "#EDB120",  # log10(z) = -1
    "#7E2F8E",  # log10(z) = 0
    "#77AC30",  # log10(z) = 1
    "#4DBEEE",  # log10(z) = 2
    "#A2142F",  # log10(z) = 3
)

PANELS = (
    ("r2", "Panel A: $R^2$"),
    ("beta_norm", r"Panel B: $\|\hat{\beta}\|$"),
    ("mean_return", "Panel C: Expected Return"),
    ("volatility", "Panel D: Volatility"),
)
# The paper's y-limits; Panel C scales to the data (its ceiling is not pinned
# in the paper either).
PAPER_YLIMS = {"r2": (-3.0, 0.1), "beta_norm": (0.0, 3.0), "volatility": (0.0, 5.0)}

# The paper's broken x-axis: [0, 50], a break, then [990, 1000].
X_MAIN = (0.0, 50.0)
X_TAIL = (990.0, 1000.0)


def load_panel_data(grid_path=GRID_PATH):
    """The ridge rows (z > 0) of the averaged grid, one row per (P, z)."""
    grid = pd.read_parquet(grid_path)
    ridge = grid.loc[grid["z"] > 0.0].sort_values(["z", "c"]).reset_index(drop=True)
    return ridge[["P", "z", "c", "r2", "beta_norm", "mean_return", "volatility"]]


def _style_axis(ax, keep_left):
    ax.tick_params(labelsize=8, colors=INK_MUTED, length=3)
    ax.grid(True, which="major", color=HAIRLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(HAIRLINE)
    if keep_left:
        ax.spines["left"].set_color(HAIRLINE)
    else:
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelleft=False, left=False)


def plot_figure7(panel_data, output_stem):
    """Render the 2x2 panel figure with the paper's broken x-axis; save png+pdf."""
    z_values = sorted(panel_data["z"].unique())
    fig = plt.figure(figsize=(10.0, 7.4), facecolor="white")
    outer = fig.add_gridspec(
        2, 2, left=0.065, right=0.985, top=0.95, bottom=0.08, hspace=0.36, wspace=0.16
    )
    # Slanted double ticks marking the axis break, drawn on the bottom spine.
    break_mark = {
        "marker": [(-1, -0.5), (1, 0.5)],
        "markersize": 7,
        "linestyle": "none",
        "color": INK_SECONDARY,
        "mec": INK_SECONDARY,
        "mew": 1.0,
        "clip_on": False,
    }
    legend_axis = None
    for cell, (column, title) in enumerate(PANELS):
        inner = outer[cell // 2, cell % 2].subgridspec(
            1, 2, width_ratios=(5, 1), wspace=0.15
        )
        ax_main = fig.add_subplot(inner[0, 0])
        ax_tail = fig.add_subplot(inner[0, 1], sharey=ax_main)
        if legend_axis is None:
            legend_axis = ax_main
        for z, color in zip(z_values, PAPER_LINE_COLORS):
            line = panel_data.loc[panel_data["z"] == z]
            label = f"$\\log_{{10}}(z) = {round(float(np.log10(z)))}$"
            ax_main.plot(
                line["c"],
                line[column],
                color=color,
                linewidth=1.6,
                label=label,
                zorder=3,
            )
            ax_tail.plot(line["c"], line[column], color=color, linewidth=1.6, zorder=3)
        ax_main.axvline(
            1.0,
            color=INK_MUTED,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            zorder=2,
            label="$c = 1$",
        )
        ax_main.set_xlim(*X_MAIN)
        ax_main.set_xticks(np.arange(0, 51, 10))
        ax_tail.set_xlim(*X_TAIL)
        ax_tail.set_xticks(X_TAIL)
        if column in PAPER_YLIMS:
            ax_main.set_ylim(*PAPER_YLIMS[column])
        else:
            ax_main.set_ylim(0.0, 1.15 * panel_data[column].max())
        # x = 0.6 of the main axis is the visual center of the axis PAIR.
        ax_main.set_title(title, fontsize=10, color=INK_SECONDARY, pad=6, x=0.6)
        ax_main.set_xlabel("$c$", fontsize=9, color=INK_SECONDARY)
        _style_axis(ax_main, keep_left=True)
        _style_axis(ax_tail, keep_left=False)
        ax_main.plot([1], [0], transform=ax_main.transAxes, **break_mark)
        ax_tail.plot([0], [0], transform=ax_tail.transAxes, **break_mark)
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
