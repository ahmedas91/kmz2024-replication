"""Plot the bonds VoC study (issue #15): does the pattern generalize?

Reads the cached bond grid statistics (``run_bonds_study``, tagged by a
``target`` column) and renders one row of panels PER TARGET — the layout is
discovered from the data, so one, two, or five bond series would render
without code changes — with two columns each: the OOS R^2 and the annualized
Sharpe ratio against model complexity c = P/T, in the market figures' style
(paper legend colors per shrinkage level, dashed ridgeless line, dashed
c = 1 marker, broken x-axis with wavy masked breaks, boxed grid-free panels,
one figure-level legend in the first pair's lower-right corner).

Y-frames are chosen from the data (no published frames exist for bonds):
the R^2 panels pin the market Figure 7 Panel A frame (-3..0.1) so the
boundary collapse clips identically, and the Sharpe panels share one frame
across targets so government and corporate read against each other.

Writes ``_output/figure_bonds{suffix}.png`` (plus a ``.pdf`` twin) and the
plotted rows to ``_output/figure_bonds_data{suffix}.parquet``, keyed by the
configured sample period like every estimation artifact.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from figure_style import (
    BORDER,
    INK_MUTED,
    PAPER_LINE_COLORS,
    broken_axis_pair,
    finish_broken_pair,
    z_line_label,
)
from run_bonds_study import BONDS_AVERAGED_PATH, TARGET_LABELS
from sample_period import SAMPLE_SUFFIX
from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))

RIDGELESS_STYLE = {
    "color": "#262523",
    "linewidth": 1.5,
    "linestyle": (0, (5, 2)),
    "zorder": 4,
}
# (ylim, tick step, decimals) per metric column; the Sharpe top stretches to
# the data across all targets so both rows share one frame.
R2_FRAME = ((-3.0, 0.1), 0.5, 1)
SHARPE_STEP, SHARPE_DECIMALS = 0.05, 2


def load_panel_data(grid_path=BONDS_AVERAGED_PATH):
    """All cached rows (ridgeless included), sorted within target."""
    grid = pd.read_parquet(grid_path)
    grid = grid.sort_values(["target", "z", "c"]).reset_index(drop=True)
    return grid[["target", "P", "z", "c", "r2", "sharpe"]]


def plot_figure_bonds(panel_data, output_stem):
    """Render len(targets) x 2 panels with the shared broken-axis styling."""
    targets = [t for t in TARGET_LABELS if t in set(panel_data["target"])]
    targets += sorted(set(panel_data["target"]) - set(targets))  # unknown last
    ridge_z = sorted(panel_data.loc[panel_data["z"] > 0.0, "z"].unique())
    sharpe_top = max(0.30, 1.1 * panel_data["sharpe"].max())
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
        label = TARGET_LABELS.get(target, target)
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
                    (0.0, sharpe_top),
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
    fig.savefig(f"{output_stem}.png", dpi=300)
    fig.savefig(f"{output_stem}.pdf")
    plt.close(fig)


def main():
    panel_data = load_panel_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_data.to_parquet(OUTPUT_DIR / f"figure_bonds_data{SAMPLE_SUFFIX}.parquet")
    plot_figure_bonds(panel_data, OUTPUT_DIR / f"figure_bonds{SAMPLE_SUFFIX}")
    for target in panel_data["target"].unique():
        rows = panel_data.loc[panel_data["target"] == target]
        anchor = rows.loc[(rows["z"] == 0.0) & (rows["P"] == rows["P"].max())].iloc[0]
        print(
            f"[figure_bonds] {target} ridgeless c={anchor.c:g}: "
            f"r2={anchor.r2:.4f}, sharpe={anchor.sharpe:.3f}"
        )
    print(
        f"[figure_bonds] wrote figure_bonds{SAMPLE_SUFFIX}.png/.pdf and "
        f"figure_bonds_data{SAMPLE_SUFFIX}.parquet"
    )


if __name__ == "__main__":
    main()
