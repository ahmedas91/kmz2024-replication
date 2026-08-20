"""Build the tidy bond excess-return dataset from the raw Goyal-Welch pull.

Two monthly excess-return series for the bonds VoC study (issue #15), both
from columns already present in the Goyal-Welch workbook — zero new data
risk:

- ``ltr_excess``  = ``ltr``   - ``Rfree``: long-term U.S. government bond
  return in excess of the T-bill rate (duration / term-premium risk only).
- ``corpr_excess`` = ``corpr`` - ``Rfree``: long-term corporate bond return
  in excess of the T-bill rate (duration plus credit risk).

Timing: ``ltr``, ``corpr``, and ``Rfree`` are all realized simple returns for
month t on row t in the workbook, so the excess returns are same-row
differences — no shifting here, and none downstream either: the study driver
adds each target's own lag by copying the standardized target onto the same
row, exactly like ``lag_mkt_excess`` in the market dataset, because the
engine applies the single t -> t+1 shift itself.

Both series are complete from 1926-01 in the current vintage. Like the other
tidy outputs, the full available history is saved, period-independent; the
study driver trims to the configured ``SAMPLE_END``.

Writes ``_data/bond_returns.parquet`` (date plus the two excess returns).
"""

from pathlib import Path

from pull_goyal_welch import load_goyal_welch
from settings import config

DATA_DIR = Path(config("DATA_DIR"))

BOND_TARGET_COLUMNS = ["ltr_excess", "corpr_excess"]


def build_bond_returns(data_dir=DATA_DIR):
    """The tidy bond excess-return frame: date, ltr_excess, corpr_excess.

    Keeps the rows where every input column exists (1926-01 onward in the
    current vintage; the raw file's 2025 tail rows survive only if all three
    inputs are populated there).
    """
    gw = load_goyal_welch(data_dir=data_dir)
    out = gw[["date"]].copy()
    out["ltr_excess"] = gw["ltr"] - gw["Rfree"]
    out["corpr_excess"] = gw["corpr"] - gw["Rfree"]
    out = out.dropna().reset_index(drop=True)
    return out


def load_bond_returns(data_dir=DATA_DIR):
    """Read the cached tidy bond-return parquet from ``data_dir``."""
    import pandas as pd

    return pd.read_parquet(Path(data_dir) / "bond_returns.parquet")


if __name__ == "__main__":
    bonds = build_bond_returns(data_dir=DATA_DIR)
    bonds.to_parquet(DATA_DIR / "bond_returns.parquet")
    span = f"{bonds['date'].min():%Y-%m} to {bonds['date'].max():%Y-%m}"
    print(f"[bonds] wrote bond_returns.parquet ({len(bonds)} months, {span})")
