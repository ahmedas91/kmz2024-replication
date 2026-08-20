# KMZ 2024 Replication

Replication of Kelly, Malamud, and Zhou (2024), "The Virtue of Complexity in
Return Prediction" (Journal of Finance), for UChicago FINM 32900. The project
replicates the paper's Figures 7, 8, and 11 at the 12-month training window,
validates them against published anchor values with unit tests, reruns every
result on data through 2024, and adds three extensions: a bonds study, an
international equities study, and a test of Nagel's (2025) critique. The
estimation engine is a reusable package, `voc`, installable with
`pip install -e .`

Authors: Ahmed Alahmadi and Jeff Key.

## Pipeline

The project is a `doit` build graph (`dodo.py`). Each stage is a task with
file targets in `_data/` or `_output/`; `doit` runs only what is stale.

| Stage | Tasks | What it produces |
| --- | --- | --- |
| Pull | `pull:goyal_welch`, `pull:crsp_stock`, `pull:intl_equity` | Raw parquets: the Goyal-Welch predictor workbook, the CRSP value-weighted index (WRDS), the Ken French developed-ex-US factors |
| Tidy | `tidy`, `tidy_bonds` | `kmz_dataset.parquet` (market excess return plus the 15 predictors), `bond_returns.parquet`. Cleaning only, no analysis |
| Standardize | `standardize` | `kmz_dataset_standardized.parquet`: backward-looking volatility standardization, the analysis-ready dataset from 1930-01 |
| Estimate | `estimate`, `export_forecasts`, `variable_importance`, `bonds_study`, `intl_study`, `nagel` | Out-of-sample statistics grids per study, the anchor forecast export, the variable-importance runs, the Nagel critique results |
| Present | `figure7`, `figure8`, `figure11`, `figure_bonds`, `figure_intl`, `summary_stats`, `report_values` | Figures, the summary table, and the generated LaTeX macros, all in `_output/` |
| Publish | `compile_latex_docs`, `run_notebooks`, `build_chartbook_site`, `run_pytest` | The report and slide PDFs in `reports/`, the executed tour notebook, the docs site in `docs/` |

Estimation math and conventions live in the `voc` package
(`voc.run_voc_study` runs a study on any target and predictor set); the
timing and standardization conventions are pinned in the module docstrings
of `voc/oos_engine.py`, `src/clean_goyal_welch.py`, and
`src/standardize_kmz.py`, and enforced by tests.

For a guided walk through the data and the estimation mechanics, start with
the tour notebook (`src/01_kmz_tour.ipynb.py`, executed by
`doit run_notebooks` to `_output/01_kmz_tour.html`) or browse the docs site
(`doit build_chartbook_site`, then open `docs/index.html`).

## Setup and run

Requirements: conda (or mamba), a WRDS account, TeXLive with `latexmk` for
the LaTeX outputs.

```bash
conda env create -f environment.yml
conda activate kmz2024_replication
cp .env.example .env   # set WRDS_USERNAME; other keys have defaults
doit                   # full pipeline, paper period
```

The first run pulls data (the CRSP pull needs the WRDS credentials; the
other pulls only need internet) and takes roughly 20 minutes, dominated by
the 500-seed estimation grid. Later runs are incremental.

## Sample periods

Every estimation artifact is keyed by `SAMPLE_END` (`src/sample_period.py`).
The default `2020-12` reproduces the paper's 1930-01 to 2020-12 sample and
writes the bare filenames (`figure8.png`, `_data/oos_grid_T12.parquet`).
Any other value appends `_{SAMPLE_END}` to the filenames, so both runs
coexist:

```bash
doit                     # paper period
SAMPLE_END=2024-12 doit  # updated sample (figure8_2024-12.png, ...)
```

The pulls and tidy datasets are period-independent (full history, bounded
by `START_DATE`/`END_DATE`) and are trimmed downstream, so switching
periods never re-pulls. The report and the presentation deck embed both
periods, so compile them after both runs exist.

Updated-sample vintage: the Goyal-Welch workbook is the "All data up to
2025" file from Amit Goyal's website (complete through 2024-12), and CRSP
is pulled through `END_DATE=2024-12-31`.

## Tests

```bash
pytest                    # from the repo root
pytest --doctest-modules  # include doctests
```

Tests live next to the code (`src/test_*.py`, `voc/test_*.py`). The
replication is validated against published anchor values at the ridgeless
c = 1000 configuration (alpha t-statistic, information ratio, Sharpe
ratio) within tolerances stated in the test docstrings; engine math is
pinned by synthetic tests that need no data. Data-dependent tests skip,
not fail, until the pipeline has produced their inputs.

## Repository layout

- `src/`: pipeline scripts (pulls, cleaning, drivers, figures, tables) and
  their tests
- `voc/`: the reusable estimation package (RFFs, dual-form ridge, the OOS
  engine, preprocessing, the Nagel toolkit) and its tests
- `_data/`, `_output/`: generated, gitignored, fully recreatable
- `data_manual/`: non-recreatable data, tracked (currently none needed)
- `reports/`: the LaTeX report and presentation deck
- `docs_src/`, `docs/`: chartbook site sources and build
- `materials/`: the paper, project instructions, and grading rubric

## Data sources

- Goyal-Welch predictor workbook: public, from Amit Goyal's website
- CRSP value-weighted index: WRDS, account required, used as a cross-check
  on the workbook market series
- Ken French data library: public, the developed-ex-US factors

No raw data is stored in the repository or its history.

## Formatting

```bash
ruff format . && ruff check --select I --fix . && ruff check --fix .
```
