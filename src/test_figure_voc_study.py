"""Tests for the shared VoC study renderer and the figure deliverables.

The renderer (``figure_voc_study.plot_voc_panels``) is the one code path
behind the bonds and international figures, with a data-driven layout (one
panel row per target found in the cache). The smoke tests exercise it on
synthetic one- and two-target frames so a rendering regression fails fast
without any pipeline data; the existence test asserts the paper-period
figure files the report and chartbook embed are actually on disk (skipping
until the pipeline has produced them).
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))

PAPER_FIGURES = [
    "figure7.png",
    "figure8.png",
    "figure11.png",
    "figure_bonds.png",
    "figure_intl.png",
    "predictor_timeseries.png",
]


def _synthetic_grid(targets):
    """A tiny averaged-grid frame with the renderer's required columns."""
    rng = np.random.default_rng(0)
    p_values, z_values = (2, 12, 96), (0.1, 1000.0, 0.0)
    rows = [
        {
            "target": target,
            "P": P,
            "z": z,
            "c": P / 12,
            "r2": float(rng.uniform(-1.0, 0.05)),
            "sharpe": float(rng.uniform(-0.05, 0.4)),
        }
        for target, P, z in product(targets, p_values, z_values)
    ]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("targets", [("a",), ("a", "b")])
def test_renderer_writes_png_and_pdf_for_any_target_count(tmp_path, targets):
    """The data-driven layout renders one row per target: both files written
    for one target and for two, from the same code path."""
    from figure_voc_study import plot_voc_panels

    stem = tmp_path / "panels"
    plot_voc_panels(_synthetic_grid(targets), {t: t.upper() for t in targets}, stem)
    for suffix in (".png", ".pdf"):
        path = Path(f"{stem}{suffix}")
        assert path.exists() and path.stat().st_size > 0


@pytest.mark.skipif(
    not (OUTPUT_DIR / "figure8.png").exists(),
    reason="paper-period figures not built yet; run `doit`",
)
def test_paper_period_figures_exist():
    """Every figure the report and chartbook embed exists with content."""
    missing = [
        name
        for name in PAPER_FIGURES
        if not ((OUTPUT_DIR / name).exists() and (OUTPUT_DIR / name).stat().st_size > 0)
    ]
    assert not missing, f"missing paper-period figures: {missing}"
