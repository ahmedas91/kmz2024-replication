"""Plot the bonds VoC study (issue #15): does the pattern generalize?

Thin wrapper over the shared study renderer (``figure_voc_study``): loads
the cached bond grid statistics (``run_bonds_study``, rows tagged by a
``target`` column) and renders one row of panels per bond series — OOS R^2
and annualized Sharpe against complexity c = P/T — in the market figures'
broken-axis style. The layout is data-driven, so additional bond series
would render without code changes.

Writes ``_output/figure_bonds{suffix}.png`` (plus a ``.pdf`` twin) and the
plotted rows to ``_output/figure_bonds_data{suffix}.parquet``, keyed by the
configured sample period like every estimation artifact.
"""

from pathlib import Path

from figure_voc_study import load_panel_data, plot_voc_panels
from run_bonds_study import BONDS_AVERAGED_PATH, TARGET_LABELS
from sample_period import SAMPLE_SUFFIX
from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))


def main():
    """Render the bonds panels and write the plotted data parquet."""
    panel_data = load_panel_data(BONDS_AVERAGED_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_data.to_parquet(OUTPUT_DIR / f"figure_bonds_data{SAMPLE_SUFFIX}.parquet")
    plot_voc_panels(
        panel_data, TARGET_LABELS, OUTPUT_DIR / f"figure_bonds{SAMPLE_SUFFIX}"
    )
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
