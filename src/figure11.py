"""Replicate Figure 11 of KMZ (2024) from the cached variable-importance runs.

Variable importance (VI) for predictor i is the change in performance moving
from the full 15-predictor model to the re-estimated 14-predictor model that
excludes i (paper's caption): positive VI means the model performs worse
without the variable. Performance is measured two ways at the paper's Figure
11 setting (T = 12, P = 12,000, z = 10^3, averaged across seeds): the OOS R^2
and the annualized Sharpe ratio of the timing strategy.

Reads the per-seed cache built by ``run_variable_importance.py``
(``_data/variable_importance_T{T}.parquet``, ``doit variable_importance``),
averages within each configuration across seeds, differences against the full
model, and draws the paper's single panel (printed page 52): blue bars for VI
in R^2 units on the left axis (percent labels), an orange line for VI in
Sharpe units on the right axis, variables sorted descending by R^2 VI. The
paper's story the figure carries: the fast-moving predictors (the lagged
market return, ltr, dfr, infl) dominate; the persistent valuation ratios
contribute little, which is how a 12-month training window can be enough.

Writes ``_output/figure11.png`` (plus a ``.pdf`` twin for LaTeX) and the
plotted VI table to ``_output/figure11_data.parquet``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator, PercentFormatter

from figure_style import (
    BORDER,
    INK_SECONDARY,
    PAPER_LINE_COLORS,
    save_png_pdf,
    style_axis,
)
from sample_period import SAMPLE_SUFFIX
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
TRAIN_WINDOW = config("TRAIN_WINDOW", default=12, cast=int)

# SAMPLE_SUFFIX is empty for the paper period; an updated-sample run reads
# and writes its own suffixed artifact set (see sample_period).
VI_PATH = DATA_DIR / f"variable_importance_T{TRAIN_WINDOW}{SAMPLE_SUFFIX}.parquet"
FULL_MODEL_LABEL = "none"

# The paper's bar (left, R^2) and line (right, Sharpe) colors are the first
# two MATLAB-cycle colors, already in the shared palette.
BAR_COLOR, LINE_COLOR = PAPER_LINE_COLORS[0], PAPER_LINE_COLORS[1]
# The paper's y-frames: (bottom, top), tick step. Tops stretch if data demand.
R2_FRAME = ((-0.005, 0.02), 0.005)
SHARPE_FRAME = ((-0.04, 0.14), 0.02)

# Display names as printed under the paper's bars.
DISPLAY_NAMES = {"lag_mkt_excess": "lag_mkt", "bm": "b/m"}


def load_vi_data(vi_path=VI_PATH):
    """The VI table: one row per predictor, sorted descending by R^2 VI.

    Averages the per-seed cache within each configuration, then differences
    the full model against each leave-one-out model.
    """
    per_seed = pd.read_parquet(vi_path)
    means = per_seed.groupby("excluded")[["r2", "sharpe"]].mean()
    full = means.loc[FULL_MODEL_LABEL]
    vi = (full - means.drop(index=FULL_MODEL_LABEL)).rename(
        columns={"r2": "vi_r2", "sharpe": "vi_sharpe"}
    )
    vi = vi.sort_values("vi_r2", ascending=False).reset_index(names="variable")
    vi["n_seeds"] = per_seed.groupby("excluded")["seed"].nunique().iloc[0]
    return vi


def plot_figure11(vi, output_stem):
    """Render the paper's single dual-axis panel; save png+pdf."""
    fig, ax_r2 = plt.subplots(figsize=(8.6, 5.2), facecolor="white")
    fig.subplots_adjust(left=0.09, right=0.90, top=0.93, bottom=0.14)
    x = range(len(vi))
    ax_r2.bar(
        x,
        vi["vi_r2"],
        width=0.75,
        color=BAR_COLOR,
        edgecolor=BORDER,
        linewidth=0.6,
        zorder=3,
    )
    ax_r2.axhline(0.0, color=BORDER, linewidth=0.8, zorder=2)

    ax_sharpe = ax_r2.twinx()
    ax_sharpe.plot(x, vi["vi_sharpe"], color=LINE_COLOR, linewidth=1.6, zorder=4)

    # The paper's frames, stretched (never shrunk) so nothing is clipped if a
    # config change moves the data.
    (r2_bottom, r2_top), r2_step = R2_FRAME
    (sr_bottom, sr_top), sr_step = SHARPE_FRAME
    ax_r2.set_ylim(
        min(r2_bottom, 1.1 * vi["vi_r2"].min()),
        max(r2_top, 1.1 * vi["vi_r2"].max()),
    )
    ax_sharpe.set_ylim(
        min(sr_bottom, 1.1 * vi["vi_sharpe"].min()),
        max(sr_top, 1.1 * vi["vi_sharpe"].max()),
    )
    ax_r2.yaxis.set_major_locator(MultipleLocator(r2_step))
    ax_r2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax_sharpe.yaxis.set_major_locator(MultipleLocator(sr_step))

    ax_r2.set_xticks(list(x))
    ax_r2.set_xticklabels(
        [DISPLAY_NAMES.get(v, v) for v in vi["variable"]],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax_r2.set_xlim(-0.7, len(vi) - 0.3)
    # Paper styling: boxed panel, no grid; each axis labeled and inked in its
    # series color, as in the paper.
    style_axis(ax_r2, keep_left=True)
    ax_r2.spines["right"].set_visible(True)
    style_axis(ax_sharpe, keep_left=True)
    ax_sharpe.spines["right"].set_visible(True)
    ax_r2.set_ylabel("VI ($R^2$)", fontsize=10, color=BAR_COLOR)
    ax_sharpe.set_ylabel("VI (Sharpe Ratio)", fontsize=10, color=LINE_COLOR)
    ax_r2.tick_params(axis="y", colors=BAR_COLOR, labelsize=8)
    ax_sharpe.tick_params(axis="y", colors=LINE_COLOR, labelsize=8)
    ax_r2.tick_params(axis="x", colors=INK_SECONDARY)

    save_png_pdf(fig, output_stem)


def main():
    """Build the VI table, render the figure, and write the data parquet."""
    vi = load_vi_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vi.to_parquet(OUTPUT_DIR / f"figure11_data{SAMPLE_SUFFIX}.parquet")
    plot_figure11(vi, OUTPUT_DIR / f"figure11{SAMPLE_SUFFIX}")
    top3 = ", ".join(
        f"{row.variable}={row.vi_r2:.4f}" for row in vi.head(3).itertuples()
    )
    print(
        f"[figure11] wrote figure11{SAMPLE_SUFFIX}.png/.pdf and "
        f"figure11_data{SAMPLE_SUFFIX}.parquet "
        f"({len(vi)} variables, {vi['n_seeds'].iloc[0]} seeds); top-3 R2 VI: {top3}"
    )


if __name__ == "__main__":
    main()
