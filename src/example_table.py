"""Generate an example summary-statistics LaTeX table from CRSP index returns.

Loads the CRSP value- and equal-weighted market index returns pulled by
``pull_CRSP_stock.py`` and writes a stacked full-sample/subsample summary
table to ``_output/example_table.tex``. Demonstrates the pattern of
stitching several ``DataFrame.to_latex`` outputs into one table with
midrules separating sample periods.
"""

from pathlib import Path

import pandas as pd

import pull_CRSP_stock
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

SPLIT_YEAR = 1990

df = pull_CRSP_stock.load_CRSP_index_files(data_dir=DATA_DIR)
df = df.set_index("caldt").sort_index()

columns_for_summary_stats = [
    "vwretd",
    "ewretd",
]

# This maps the column names to their LaTeX descriptions
column_names_map = {
    "vwretd": "Value-weighted return",
    "ewretd": "Equal-weighted return",
}

escape_coverter = {"25%": "25\\%", "50%": "50\\%", "75%": "75\\%"}

# Monthly index returns in percent
df_monthly = 100 * df[columns_for_summary_stats].dropna()

first_year = df_monthly.index.min().year
last_year = df_monthly.index.max().year

## Suppress scientific notation and limit to 3 decimal places
# Sets display, but doesn't affect formatting to LaTeX
pd.set_option("display.float_format", lambda x: "%.2f" % x)
# Sets format for printing to LaTeX
float_format_func = lambda x: "{:.2f}".format(x)

# Pooled summary stats
describe_all = (
    df_monthly[columns_for_summary_stats]
    .describe()
    .T.rename(index=column_names_map, columns=escape_coverter)
)
describe_all["count"] = describe_all["count"].astype(int)
describe_all.columns.name = f"Full Sample: {first_year} - {last_year}"
latex_table_string_all = describe_all.to_latex(
    escape=False, float_format=float_format_func
)

describe1 = (
    df_monthly.loc[: str(SPLIT_YEAR - 1), columns_for_summary_stats]
    .describe()
    .T.rename(index=column_names_map, columns=escape_coverter)
)
describe1["count"] = describe1["count"].astype(int)
describe1.columns.name = f"Subsample: {first_year} - {SPLIT_YEAR - 1}"
latex_table_string1 = describe1.to_latex(escape=False, float_format=float_format_func)

describe2 = (
    df_monthly.loc[str(SPLIT_YEAR) :, columns_for_summary_stats]
    .describe()
    .T.rename(index=column_names_map, columns=escape_coverter)
)
describe2["count"] = describe2["count"].astype(int)
describe2.columns.name = f"Subsample: {SPLIT_YEAR} - {last_year}"
latex_table_string2 = describe2.to_latex(escape=False, float_format=float_format_func)

latex_table_string_split = [
    *latex_table_string_all.split("\n")[
        0:-3
    ],  # Skip the \end{tabular} and \bottomrule lines
    "\\midrule",
    *latex_table_string1.split("\n")[2:-3],  # Skip the \begin and \end lines
    "\\midrule",
    *latex_table_string2.split("\n")[2:],  # Skip the \begin{tabular} and \toprule lines
]
latex_table_string = "\n".join(latex_table_string_split)
# print(latex_table_string)
path = OUTPUT_DIR / "example_table.tex"
with open(path, "w") as text_file:
    text_file.write(latex_table_string)
