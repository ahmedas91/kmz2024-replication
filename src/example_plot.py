"""Generate an example plot from CRSP index returns.

Loads the CRSP value- and equal-weighted market index returns pulled by
``pull_CRSP_stock.py`` and plots the cumulative growth of one dollar
invested in each index, on a log scale, saving the figure to
``_output/example_plot.png``.
"""

from pathlib import Path

import pull_CRSP_stock
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

import seaborn as sns
from matplotlib import pyplot as plt

sns.set()

df = pull_CRSP_stock.load_CRSP_index_files(data_dir=DATA_DIR)
df = df.set_index("caldt").sort_index()

cumulative_growth = (
    (
        1
        + df[["vwretd", "ewretd"]]
        .rename(
            columns={
                "vwretd": "Value-weighted index",
                "ewretd": "Equal-weighted index",
            }
        )
        .dropna()
    ).cumprod()
)

cumulative_growth.plot(logy=True)
plt.title("Cumulative Growth of the CRSP Market Indexes")
plt.ylabel("Growth of one dollar invested (log scale)")
filename = OUTPUT_DIR / "example_plot.png"
plt.savefig(filename)
