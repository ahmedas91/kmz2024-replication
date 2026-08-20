"""Generate the predictor summary-statistics LaTeX table.

Builds a two-panel table with a stitched ``\\midrule`` between the panels:
Panel A summarizes the raw tidy dataset
(clean_goyal_welch, 1926 onward) and Panel B the volatility-standardized
analysis dataset (standardize_kmz, 1930 onward). Rows are the market excess
return plus the 15 predictors in the paper's footnote-33 order, labeled with
the paper's display names ("b/m", "lag mkt"). Columns: count, mean, SD, min,
max, and the first-order autocorrelation AC(1), which exposes the
persistence split between the slow valuation ratios and yields and the
fast return-type predictors. Both panels are trimmed to the configured
estimation sample (through ``SAMPLE_END``; the panel headers show the span),
so the default table describes the paper period and an updated-sample run
writes its own suffixed table alongside (see ``sample_period``). Writes
``_output/predictor_summary_table{suffix}.tex``.
"""

from pathlib import Path

import pandas as pd

from clean_goyal_welch import PREDICTOR_COLUMNS, load_kmz_dataset
from sample_period import SAMPLE_SUFFIX, trim_to_sample
from settings import config
from standardize_kmz import load_standardized_dataset

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))


# Paper display names (Figure 11 labels), keyed by our column names.
DISPLAY_NAMES = {
    "mkt_excess": "mkt excess",
    "dfy": "dfy",
    "infl": "infl",
    "svar": "svar",
    "de": "de",
    "lty": "lty",
    "tms": "tms",
    "tbl": "tbl",
    "dfr": "dfr",
    "dp": "dp",
    "dy": "dy",
    "ltr": "ltr",
    "ep": "ep",
    "bm": "b/m",
    "ntis": "ntis",
    "lag_mkt_excess": "lag mkt",
}

# Target first, then the 15 predictors in footnote-33 order.
SERIES_COLUMNS = ["mkt_excess"] + PREDICTOR_COLUMNS


def summary_panel(df, panel_label):
    """One summary panel: count, mean, SD, min, max, AC(1) per series.

    Rows carry the paper's display names; `panel_label` becomes the header
    shown above the panel (the `columns.name` slot in `to_latex`).
    """
    rows = {}
    for column in SERIES_COLUMNS:
        series = df[column].dropna()
        rows[DISPLAY_NAMES[column]] = {
            "count": len(series),
            "mean": series.mean(),
            "SD": series.std(),
            "min": series.min(),
            "max": series.max(),
            "AC(1)": series.autocorr(lag=1),
        }
    panel = pd.DataFrame.from_dict(rows, orient="index").astype(float)
    panel["count"] = panel["count"].astype(int)
    panel.columns.name = panel_label
    return panel


def build_summary_table(data_dir=DATA_DIR):
    """Return the stitched two-panel LaTeX table as a string."""
    raw = trim_to_sample(load_kmz_dataset(data_dir=data_dir))
    standardized = trim_to_sample(load_standardized_dataset(data_dir=data_dir))

    def span(df):
        return f"{df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}"

    def to_latex(panel):
        return panel.to_latex(escape=False, float_format=lambda x: f"{x:.3f}")

    latex_raw = to_latex(summary_panel(raw, f"Panel A: Raw, {span(raw)}"))
    latex_std = to_latex(
        summary_panel(standardized, f"Panel B: Standardized, {span(standardized)}")
    )

    lines = [
        # Keep Panel A up to (not including) its \bottomrule and \end{tabular}.
        *latex_raw.split("\n")[0:-3],
        "\\midrule",
        # Drop Panel B's \begin{tabular} and \toprule; keep its closing lines.
        *latex_std.split("\n")[2:],
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    table = build_summary_table(data_dir=DATA_DIR)
    path = OUTPUT_DIR / f"predictor_summary_table{SAMPLE_SUFFIX}.tex"
    with open(path, "w") as text_file:
        text_file.write(table)
