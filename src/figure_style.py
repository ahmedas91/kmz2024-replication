"""Shared styling for the paper-replication figures (Figures 7 and 8).

One place for the pieces both figure scripts use so they cannot drift: the
house ink tokens, the paper's legend colors (its figures use the MATLAB
default cycle), and the paper's broken x-axis construction — a linear main
segment over c in [0, 50] that emphasizes the interpolation boundary, a break
marked by full-height wavy lines as in the paper, then a sliver at
c in [990, 1000] for the extreme-complexity end. Grid points falling inside the break (c = 64..512)
are hidden exactly as in the paper; the figure scripts export them in their
data parquets.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

# Ink tokens shared with plot_predictor_timeseries.py.
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
HAIRLINE = "#e1e0d9"
# Panel border and break-wave ink; the paper draws plain black boxes.
BORDER = "#262523"

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

# Wavy full-height break lines at the facing edges of an axis pair, the
# paper's style: one serpentine per edge, with the physical gap between the
# axes playing the paper's white band.
_WAVE_Y = np.linspace(0.0, 1.0, 200)
_WAVE_SHAPE = np.sin(_WAVE_Y * 2.0 * np.pi)  # one gentle period, as in the paper


def _draw_break_wave(ax, x_edge, amplitude):
    """One wavy break line at an axes edge, with the strip between the wave
    and the edge whited out so data lines visually stop AT the wave (as in
    the paper). The wave's endpoints land exactly on the panel corners
    (sin(0) = sin(2*pi) = 0), closing the box border."""
    wave_x = x_edge + amplitude * _WAVE_SHAPE
    outside = x_edge + (amplitude + 0.006) * (1.0 if x_edge == 1.0 else -1.0)
    ax.fill_betweenx(
        _WAVE_Y,
        wave_x,
        outside,
        transform=ax.transAxes,
        color="white",
        linewidth=0,
        zorder=4.6,
        clip_on=False,
    )
    ax.plot(
        wave_x,
        _WAVE_Y,
        transform=ax.transAxes,
        color=BORDER,
        linewidth=1.0,
        clip_on=False,
        zorder=5,
    )


def broken_axis_pair(fig, gridspec_cell):
    """Create one panel's axis pair: the wide main axis plus the tail sliver,
    sharing the y-scale."""
    inner = gridspec_cell.subgridspec(1, 2, width_ratios=(5, 1), wspace=0.15)
    ax_main = fig.add_subplot(inner[0, 0])
    ax_tail = fig.add_subplot(inner[0, 1], sharey=ax_main)
    return ax_main, ax_tail


def style_axis(ax, keep_left):
    """Paper styling: boxed panel (open at the break edge, where the wavy
    line completes the border), no grid."""
    ax.tick_params(labelsize=8, colors=INK_SECONDARY, length=3)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.8)
    if keep_left:
        ax.spines["right"].set_visible(False)
    else:
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelleft=False, left=False)


def finish_broken_pair(ax_main, ax_tail, title, ylim, ytick_step=None, decimals=None):
    """Apply the broken-axis limits, ticks, title, styling, and break marks.

    The title centers over the axis PAIR (x = 0.6 of the main axis is the
    pair's visual center given the 5:1 width split). ``ytick_step`` pins the
    y-tick spacing and ``decimals`` the tick-label precision to the paper's
    units for the panel.
    """
    ax_main.set_xlim(*X_MAIN)
    ax_main.set_xticks(np.arange(0, 51, 10))
    ax_tail.set_xlim(*X_TAIL)
    ax_tail.set_xticks(X_TAIL)
    ax_main.set_ylim(*ylim)
    if ytick_step is not None:
        ax_main.yaxis.set_major_locator(MultipleLocator(ytick_step))
    if decimals is not None:
        ax_main.yaxis.set_major_formatter(StrMethodFormatter(f"{{x:.{decimals}f}}"))
    ax_main.set_title(title, fontsize=10, color=INK_SECONDARY, pad=6, x=0.6)
    ax_main.set_xlabel("$c$", fontsize=9, color=INK_SECONDARY)
    style_axis(ax_main, keep_left=True)
    style_axis(ax_tail, keep_left=False)
    # Amplitudes differ because axes coordinates scale with axis width and the
    # main axis is 5x wider than the tail (broken_axis_pair's width ratio).
    _draw_break_wave(ax_main, 1.0, 0.010)
    _draw_break_wave(ax_tail, 0.0, 0.050)


def z_line_label(z):
    """The paper's legend label for a ridge line."""
    return f"$\\log_{{10}}(z) = {round(float(np.log10(z)))}$"


# The dashed dark line for the ridgeless (z -> 0 minimum-norm) limit, shared
# by every figure that draws it.
RIDGELESS_STYLE = {
    "color": "#262523",
    "linewidth": 1.5,
    "linestyle": (0, (5, 2)),
    "zorder": 4,
}


def add_lower_right_legend(fig, legend_axis, legend_tail):
    """One boxed figure-level legend in the lower-right corner of the axis
    pair (legend_axis, legend_tail), drawn at figure level so it may sit on
    top of the axis break and sliver, as in the paper."""
    handles, labels = legend_axis.get_legend_handles_labels()
    anchor = (
        legend_tail.get_position().x1 - 0.004,
        legend_axis.get_position().y0 + 0.008,
    )
    fig.legend(
        handles,
        labels,
        loc="lower right",
        bbox_to_anchor=anchor,
        fontsize=7,
        edgecolor=BORDER,
        facecolor="white",
        framealpha=1.0,
        fancybox=False,
    )


def save_png_pdf(fig, output_stem):
    """Save ``output_stem``.png (300 dpi) and a .pdf twin, then close."""
    fig.savefig(f"{output_stem}.png", dpi=300)
    fig.savefig(f"{output_stem}.pdf")
    plt.close(fig)
