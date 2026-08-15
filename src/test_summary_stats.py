"""Tests for the predictor summary-statistics table and overview plot.

Checks that the standardized predictors carry the scale the summary table
advertises, and that the two deliverables (the LaTeX table and the
small-multiples figure) are produced with one row or panel per series.
Data-dependent tests skip (rather than fail) until the pipeline outputs
exist.
"""

from pathlib import Path

import pytest

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
STD_PARQUET = DATA_DIR / "kmz_dataset_standardized.parquet"
TABLE_PATH = OUTPUT_DIR / "predictor_summary_table.tex"
PNG_PATH = OUTPUT_DIR / "predictor_timeseries.png"
PDF_PATH = OUTPUT_DIR / "predictor_timeseries.pdf"

requires_std_data = pytest.mark.skipif(
    not STD_PARQUET.exists(),
    reason="kmz_dataset_standardized.parquet not found; run `doit standardize` first",
)
requires_outputs = pytest.mark.skipif(
    not TABLE_PATH.exists(),
    reason="summary-stats outputs not found; run `doit summary_stats` first",
)


@requires_std_data
def test_standardized_sds_in_loose_band():
    """Full-sample SDs of the standardized series sit in a loose band near 1.

    The band is wide by design: the trailing 12-month scheme puts the return
    near 1 (measured 1.09), while expanding-window standardization leaves
    persistent, level-drifting predictors above it (lty 3.5, dp/dy/ep about
    1.6) because the expanding scale adapts slowly. The test exists to catch
    unit errors (an SD of 0.01 or 100), not to demand exactly 1.
    """
    from standardize_kmz import load_standardized_dataset

    df = load_standardized_dataset(data_dir=DATA_DIR).set_index("date")
    sds = df.std()
    assert ((sds > 0.4) & (sds < 5.0)).all(), f"SDs out of band:\n{sds}"
    for column in ("mkt_excess", "lag_mkt_excess"):
        assert 0.9 < sds[column] < 1.3, f"{column} SD {sds[column]}"


@requires_outputs
def test_table_contains_all_series_rows():
    """The LaTeX table has both panels and one row per series."""
    from table_predictor_summary import DISPLAY_NAMES

    content = TABLE_PATH.read_text()
    assert "Panel A" in content and "Panel B" in content
    for display_name in DISPLAY_NAMES.values():
        assert display_name in content, f"missing table row: {display_name}"


@requires_outputs
def test_plot_files_exist():
    """Both figure files (png for notebooks, pdf for LaTeX) are produced."""
    for path in (PNG_PATH, PDF_PATH):
        assert path.exists() and path.stat().st_size > 0, f"missing {path}"
