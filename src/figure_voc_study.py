"""Shared renderer for the asset-class VoC study figures (bonds, intl).

One data-driven figure body: :func:`plot_voc_panels` takes a study cache
(rows tagged by a ``target`` column) and renders one ROW of panels per
target found in the data — one, two, or five series lay out without code
changes — with two columns each, the OOS R^2 and the annualized Sharpe
ratio against complexity c = P/T, in the market figures' style: paper
legend colors per shrinkage level, dashed ridgeless line, dashed c = 1
marker, broken x-axis with wavy masked breaks, boxed grid-free panels, one
figure-level boxed legend in the first pair's lower-right corner.

Frames: the R^2 panels pin the market Figure 7 Panel A frame (-3..0.1) so
the interpolation-boundary collapse clips identically across studies; the
Sharpe panels share one frame across all targets in the figure (floor 0.30,
stretched to the data) so the series read against each other.

Extracted verbatim from the issue #15 bonds figure so the intl study (#16)
reuses it; ``figure_bonds.py`` and ``figure_intl.py`` are thin wrappers.
"""

import matplotlib.pyplot as plt

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

# (ylim, tick step, decimals) for the R^2 panels; the Sharpe panels share
# one data-stretched frame across targets.
R2_FRAME = ((-3.0, 0.1), 0.5, 1)
SHARPE_STEP, SHARPE_DECIMALS = 0.05, 2
SHARPE_FLOOR = 0.30


def load_panel_data(grid_path):
    """All cached study rows (ridgeless included), sorted within target.

    The shared loader for the study caches (bonds, intl): reads the averaged
    grid parquet and returns the columns :func:`plot_voc_panels` consumes.
    """
    import pandas as pd

    grid = pd.read_parquet(grid_path)
    grid = grid.sort_values(["target", "z", "c"]).reset_index(drop=True)
    return grid[["target", "P", "z", "c", "r2", "sharpe"]]


def plot_voc_panels(panel_data, target_labels, output_stem):
    """Render len(targets) x 2 panels with the shared broken-axis styling.

    ``panel_data`` needs columns target/P/z/c/r2/sharpe; ``target_labels``
    maps target values to display names (targets appear in this order,
    unknown targets after, alphabetically). Saves ``output_stem``.png/.pdf.
    """
    targets = [t for t in target_labels if t in set(panel_data["target"])]
    targets += sorted(set(panel_data["target"]) - set(targets))  # unknown last
    ridge_z = sorted(panel_data.loc[panel_data["z"] > 0.0, "z"].unique())
    sharpe_top = max(SHARPE_FLOOR, 1.1 * panel_data["sharpe"].max())
    # Stretch the bottom too when data go negative, so nothing clips silently.
    sharpe_bottom = min(0.0, 1.1 * panel_data["sharpe"].min())
    fig = plt.figure(figsize=(10.0, 3.9 * len(targets)), facecolor="white")
    outer = fig.add_gridspec(
        len(targets),
        2,
        left=0.065,
        right=0.985,
        top=1.0 - 0.05 / len(targets),
        bottom=0.16 / len(targets),
        hspace=0.40,
        wspace=0.16,
    )
    legend_axis = legend_tail = None
    for row, target in enumerate(targets):
        rows = panel_data.loc[panel_data["target"] == target]
        label = target_labels.get(target, target)
        for col, (column, metric_label) in enumerate(
            (("r2", "$R^2$"), ("sharpe", "Sharpe Ratio"))
        ):
            ax_main, ax_tail = broken_axis_pair(fig, outer[row, col])
            if legend_axis is None:
                legend_axis, legend_tail = ax_main, ax_tail
            for z, color in zip(ridge_z, PAPER_LINE_COLORS):
                line = rows.loc[rows["z"] == z]
                for ax in (ax_main, ax_tail):
                    ax.plot(
                        line["c"],
                        line[column],
                        color=color,
                        linewidth=1.6,
                        label=z_line_label(z) if ax is legend_axis else None,
                        zorder=3,
                    )
            ridgeless = rows.loc[rows["z"] == 0.0]
            for ax in (ax_main, ax_tail):
                ax.plot(
                    ridgeless["c"],
                    ridgeless[column],
                    label=r"ridgeless ($z \to 0$)" if ax is legend_axis else None,
                    **RIDGELESS_STYLE,
                )
            ax_main.axvline(
                1.0,
                color=INK_MUTED,
                linewidth=1.0,
                linestyle=(0, (4, 3)),
                zorder=2,
                label="$c = 1$" if ax_main is legend_axis else None,
            )
            if column == "r2":
                (bottom, top), step, decimals = R2_FRAME
            else:
                (bottom, top), step, decimals = (
                    (sharpe_bottom, sharpe_top),
                    SHARPE_STEP,
                    SHARPE_DECIMALS,
                )
            finish_broken_pair(
                ax_main,
                ax_tail,
                f"{label}: {metric_label}",
                (bottom, top),
                step,
                decimals,
            )
    add_lower_right_legend(fig, legend_axis, legend_tail)
    save_png_pdf(fig, output_stem)
