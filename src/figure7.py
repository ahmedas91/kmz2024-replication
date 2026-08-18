"""Replicate Figure 7 of KMZ (2024) from the cached OOS grid (T = 12).

Reads the across-seed averaged grid (``_data/oos_grid_T{T}.parquet``, built by
``doit estimate``) and draws the paper's four panels against model complexity
c = P/T: Panel A the OOS R^2, Panel B the mean L2 norm of beta_hat, Panel C the
timing-strategy expected return, Panel D its volatility (monthly units, as in
the paper). One line per ridge shrinkage level, log10(z) in {-3, ..., 3}, on an
ordered dark-to-light color ramp; the grid's ridgeless z = 0 rows are not drawn
(Figure 7 shows the ridge grid; ridgeless is the Figure 8 anchor case). The
x-axis is log-scale where the paper breaks its axis at an intermediate c; the
dashed vertical line marks the interpolation boundary c = 1.

Panels A, B, and D pin the paper's y-limits (-3..0, 0..3, 0..5) so the output
reads side by side against the paper's page 43 figure, and the c = 1 spikes
clip exactly as they do there. Expectation management (the paper's point, not
a bug): the OOS R^2 is NEGATIVE and collapses toward the interpolation
boundary for low shrinkage, yet the Figure 8 trading performance still
improves with complexity.

Writes ``_output/figure7.png`` (plus a ``.pdf`` twin for LaTeX) and the plotted
panel data to ``_output/figure7_data.parquet``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import to_hex

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)

GRID_PATH = DATA_DIR / f"oos_grid_T{TRAIN_WINDOW}.parquet"

# Ink tokens shared with plot_predictor_timeseries.py.
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
HAIRLINE = "#e1e0d9"

PANELS = (
    ("r2", "Panel A: $R^2$"),
    ("beta_norm", r"Panel B: $\|\hat{\beta}\|$"),
    ("mean_return", "Panel C: Expected Return"),
    ("volatility", "Panel D: Volatility"),
)
# The paper's y-limits; Panel C scales to the data (its ceiling is not pinned
# in the paper either).
PAPER_YLIMS = {"r2": (-3.0, 0.1), "beta_norm": (0.0, 3.0), "volatility": (0.0, 5.0)}


def z_line_colors(n):
    """Ordered dark-to-light viridis stops for the z lines.

    z is an ordered magnitude, so the lines wear a sequential ramp (adjacent
    lines are adjacent shrinkage levels); the ramp is truncated at 0.72 so the
    lightest line keeps enough contrast on the white surface.
    """
    return [to_hex(cm.viridis(x)) for x in np.linspace(0.0, 0.72, n)]


def load_panel_data(grid_path=GRID_PATH):
    """The ridge rows (z > 0) of the averaged grid, one row per (P, z)."""
    grid = pd.read_parquet(grid_path)
    ridge = grid.loc[grid["z"] > 0.0].sort_values(["z", "c"]).reset_index(drop=True)
    return ridge[["P", "z", "c", "r2", "beta_norm", "mean_return", "volatility"]]


def plot_figure7(panel_data, output_stem):
    """Render the 2x2 panel figure and save png and pdf."""
    z_values = sorted(panel_data["z"].unique())
    colors = z_line_colors(len(z_values))
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), facecolor="white")
    for ax, (column, title) in zip(axes.flat, PANELS):
        for z, color in zip(z_values, colors):
            line = panel_data.loc[panel_data["z"] == z]
            ax.plot(
                line["c"],
                line[column],
                color=color,
                linewidth=1.9,
                label=f"$\\log_{{10}}(z) = {round(float(np.log10(z)))}$",
                zorder=3,
            )
        ax.axvline(
            1.0,
            color=INK_MUTED,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            zorder=2,
            label="$c = 1$",
        )
        ax.set_xscale("log")
        ax.set_xlim(0.14, 1_100)
        if column in PAPER_YLIMS:
            ax.set_ylim(*PAPER_YLIMS[column])
        else:
            ax.set_ylim(0.0, 1.15 * panel_data[column].max())
        ax.set_title(title, fontsize=10, color=INK_SECONDARY, pad=6)
        ax.set_xlabel("$c$", fontsize=9, color=INK_SECONDARY)
        ax.grid(True, which="major", color=HAIRLINE, linewidth=0.6, zorder=1)
        ax.tick_params(labelsize=8, colors=INK_MUTED, length=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(HAIRLINE)
        ax.set_facecolor("white")
    # One legend for the whole figure, in Panel A as in the paper; single
    # column so it sits clear of the boundary plunge in the empty lower right.
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
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
