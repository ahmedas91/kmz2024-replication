"""Tests for the static PNG-to-Chartbook adapter."""

from __future__ import annotations

import base64
import re

from build_chartbook_images import wrap_png


def test_wrap_png_creates_self_contained_image_page(tmp_path):
    png_bytes = b"\x89PNG\r\n\x1a\nexample"
    png_path = tmp_path / "sample_chart.png"
    html_path = tmp_path / "chartbook" / "sample_chart.html"
    png_path.write_bytes(png_bytes)

    wrap_png(png_path, html_path)

    document = html_path.read_text(encoding="utf-8")
    encoded = re.search(r"data:image/png;base64,([^\"]+)", document)
    assert encoded is not None
    assert base64.b64decode(encoded.group(1)) == png_bytes
    assert "<img " in document
    assert "<iframe" not in document
    assert 'alt="Sample Chart"' in document
