"""voc — a reusable Virtue-of-Complexity return-prediction engine (KMZ 2024).

This package holds the model engine: random Fourier features (issue #5),
dual-form kernel ridge (issue #6), the recursive out-of-sample loop and
performance metrics (issue #7), and the public ``run_voc_study`` API (issue #14).
The flat ``src/`` scripts are drivers that import from here.

The consolidated public API is assembled in issue #14; until then, import the
modules directly, e.g. ``from voc.rff import compute_rff``.
"""
