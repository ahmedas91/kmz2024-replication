"""Wrap static PNG figures in self-contained HTML pages for Chartbook.

Chartbook renders each registered chart in an iframe and therefore requires
the chart manifest's ``path`` to point to an HTML document.  The analysis
pipeline intentionally produces static Matplotlib PNG/PDF pairs, so this
module bridges the two formats without changing the figure-generation code.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

CHART_STEMS = (
    "figure7",
    "figure8",
    "figure11",
    "figure_bonds",
    "figure_intl",
    "figure_nagel",
    "predictor_timeseries",
)


def wrap_png(png_path: Path, html_path: Path) -> None:
    """Write a self-contained HTML page displaying ``png_path``."""
    encoded_png = base64.b64encode(png_path.read_bytes()).decode("ascii")
    title = html.escape(png_path.stem.replace("_", " ").title())
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    html, body {{
      margin: 0;
      min-height: 100%;
      background: #fff;
    }}
    body {{
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
    }}
    img {{
      display: block;
      max-width: 100%;
      max-height: 100vh;
      width: auto;
      height: auto;
    }}
  </style>
</head>
<body>
  <img src="data:image/png;base64,{encoded_png}" alt="{title}">
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(document, encoding="utf-8")


def build_chartbook_images(output_dir: str | Path = "_output") -> None:
    """Create one Chartbook-compatible HTML page for every registered PNG."""
    output_dir = Path(output_dir)
    chartbook_dir = output_dir / "chartbook"
    for stem in CHART_STEMS:
        wrap_png(output_dir / f"{stem}.png", chartbook_dir / f"{stem}.html")


if __name__ == "__main__":
    build_chartbook_images()
