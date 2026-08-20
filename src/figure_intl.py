"""Plot the international equities VoC study (issue #16).

Thin wrapper over the shared study renderer (``figure_voc_study``): loads
the cached developed-ex-US grid statistics (``run_intl_study``) and renders
the single target's row of panels — OOS R^2 and annualized Sharpe against
complexity c = P/T — in the market figures' broken-axis style. The two
design caveats (US predictors forecasting non-US returns; a ~350-month
sample versus the market study's 1,092) live on the chart page and in the
study driver's docstring.

Writes ``_output/figure_intl{suffix}.png`` (plus a ``.pdf`` twin) and the
plotted rows to ``_output/figure_intl_data{suffix}.parquet``, keyed by the
configured sample period like every estimation artifact.
"""

from pathlib import Path

from figure_voc_study import load_panel_data, plot_voc_panels
from run_intl_study import INTL_AVERAGED_PATH, TARGET_LABELS
from sample_period import SAMPLE_SUFFIX
from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))


def main():
    """Render the international panels and write the plotted data parquet."""
    panel_data = load_panel_data(INTL_AVERAGED_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_data.to_parquet(OUTPUT_DIR / f"figure_intl_data{SAMPLE_SUFFIX}.parquet")
    plot_voc_panels(
        panel_data, TARGET_LABELS, OUTPUT_DIR / f"figure_intl{SAMPLE_SUFFIX}"
    )
    anchor = panel_data.loc[
        (panel_data["z"] == 0.0) & (panel_data["P"] == panel_data["P"].max())
    ].iloc[0]
    print(
        f"[figure_intl] ridgeless c={anchor.c:g}: r2={anchor.r2:.4f}, "
        f"sharpe={anchor.sharpe:.3f}; wrote figure_intl{SAMPLE_SUFFIX}.png/.pdf "
        f"and figure_intl_data{SAMPLE_SUFFIX}.parquet"
    )


if __name__ == "__main__":
    main()
