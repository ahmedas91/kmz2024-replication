"""Shared styling for the paper-replication figures (Figures 7 and 8).

One place for the pieces both figure scripts use so they cannot drift: the
house ink tokens, the paper's legend colors (its figures use the MATLAB
default cycle), and the paper's broken x-axis construction — a linear main
segment over c in [0, 50] that emphasizes the interpolation boundary, a break
with slanted marks, then a sliver at c in [990, 1000] for the
extreme-complexity end. Grid points falling inside the break (c = 64..512)
are hidden exactly as in the paper; the figure scripts export them in their
data parquets.
"""

from __future__ import annotations

import numpy as np

# Ink tokens shared with plot_predictor_timeseries.py.
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
HAIRLINE = "#e1e0d9"

# The paper's line colors, in ascending-z order to match its legend:
# log10(z) = -3, ..., 3.
PAPER_LINE_COLORS = (
    "#0072BD",  # log10(z) = -3
    "#D95319",  # log10(z) = -2
    "#EDB120",  # log10(z) = -1
    "#7E2F8E",  # log10(z) = 0
    "#77AC30",  # log10(z) = 1
    "#4DBEEE",  # log10(z) = 2
    "#A2142F",  # log10(z) = 3
)

# The paper's broken x-axis: [0, 50], a break, then [990, 1000].
X_MAIN = (0.0, 50.0)
X_TAIL = (990.0, 1000.0)

# Slanted double ticks marking the axis break, drawn on the bottom spine.
BREAK_MARK = {
    "marker": [(-1, -0.5), (1, 0.5)],
    "markersize": 7,
    "linestyle": "none",
    "color": INK_SECONDARY,
    "mec": INK_SECONDARY,
    "mew": 1.0,
    "clip_on": False,
}


def broken_axis_pair(fig, gridspec_cell):
    """Create one panel's axis pair: the wide main axis plus the tail sliver,
    sharing the y-scale."""
    inner = gridspec_cell.subgridspec(1, 2, width_ratios=(5, 1), wspace=0.15)
    ax_main = fig.add_subplot(inner[0, 0])
    ax_tail = fig.add_subplot(inner[0, 1], sharey=ax_main)
    return ax_main, ax_tail


def style_axis(ax, keep_left):
    """House styling: recessive grid, hairline spines, muted ticks."""
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


def finish_broken_pair(ax_main, ax_tail, title, ylim):
    """Apply the broken-axis limits, ticks, title, styling, and break marks.

    The title centers over the axis PAIR (x = 0.6 of the main axis is the
    pair's visual center given the 5:1 width split).
    """
    ax_main.set_xlim(*X_MAIN)
    ax_main.set_xticks(np.arange(0, 51, 10))
    ax_tail.set_xlim(*X_TAIL)
    ax_tail.set_xticks(X_TAIL)
    ax_main.set_ylim(*ylim)
    ax_main.set_title(title, fontsize=10, color=INK_SECONDARY, pad=6, x=0.6)
    ax_main.set_xlabel("$c$", fontsize=9, color=INK_SECONDARY)
    style_axis(ax_main, keep_left=True)
    style_axis(ax_tail, keep_left=False)
    ax_main.plot([1], [0], transform=ax_main.transAxes, **BREAK_MARK)
    ax_tail.plot([0], [0], transform=ax_tail.transAxes, **BREAK_MARK)


def z_line_label(z):
    """The paper's legend label for a ridge line."""
    return f"$\\log_{{10}}(z) = {round(float(np.log10(z)))}$"
