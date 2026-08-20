"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based

"""

#######################################
## Configuration and Helpers for PyDoit
#######################################
## Make sure the src folder is in the path
import sys

sys.path.insert(1, "./src/")

import shutil
from os import environ
from pathlib import Path

from doit.tools import config_changed

# The estimation drivers own their output paths and run config; importing them
# here keeps the build graph and the scripts from ever disagreeing. The modules
# are import-light by design (no pandas/engine imports at top level).
# SAMPLE_SUFFIX names the period-dependent artifacts: empty for the paper
# period, "_{SAMPLE_END}" otherwise, so a paper run and an updated-sample run
# coexist (see src/sample_period.py).
from run_bonds_study import BONDS_AVERAGED_PATH, BONDS_N_SEEDS, BONDS_PER_SEED_PATH
from run_estimation import AVERAGED_PATH, N_SEEDS, PER_SEED_PATH
from run_variable_importance import FIG11_N_SEEDS, VI_PATH
from sample_period import SAMPLE_SUFFIX
from settings import config

DOIT_CONFIG = {"backend": "sqlite3", "dep_file": "./.doit-db.sqlite"}


BASE_DIR = config("BASE_DIR")
DATA_DIR = config("DATA_DIR")
MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
OS_TYPE = config("OS_TYPE")

## Helpers for handling Jupyter Notebook tasks
environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"


# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
def jupyter_execute_notebook(notebook_path):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
def jupyter_to_html(notebook_path, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir={output_dir} {notebook_path}"
def jupyter_to_md(notebook_path, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --output-dir={output_dir} {notebook_path}"
def jupyter_clear_output(notebook_path):
    """Clear the output of a notebook"""
    return f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
# fmt: on


def mv(from_path, to_path):
    """Move a file to a folder"""
    from_path = Path(from_path)
    to_path = Path(to_path)
    to_path.mkdir(parents=True, exist_ok=True)
    if OS_TYPE == "nix":
        command = f"mv {from_path} {to_path}"
    else:
        command = f"move {from_path} {to_path}"
    return command


def copy_file(origin_path, destination_path, mkdir=True):
    """Create a Python action for copying a file."""

    def _copy_file():
        origin = Path(origin_path)
        dest = Path(destination_path)
        if mkdir:
            dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, dest)

    return _copy_file


##################################
## Begin rest of PyDoit tasks here
##################################


def task_config():
    """Create empty directories for data and output if they don't exist"""
    return {
        "actions": ["python ./src/settings.py"],
        "targets": [DATA_DIR, OUTPUT_DIR],
        "file_dep": ["./src/settings.py"],
        "clean": [],
    }


def task_pull():
    """Pull data from external sources"""
    yield {
        "name": "crsp_stock",
        "doc": "Pull CRSP stock data from WRDS",
        "actions": [
            "python ./src/settings.py",
            "python ./src/pull_CRSP_stock.py",
        ],
        "targets": [
            DATA_DIR / "CRSP_monthly_stock.parquet",
            DATA_DIR / "CRSP_MSIX.parquet",
        ],
        "file_dep": ["./src/settings.py", "./src/pull_CRSP_stock.py"],
        "clean": [],
    }
    yield {
        "name": "goyal_welch",
        "doc": "Pull the Goyal-Welch predictor dataset from Amit Goyal's website",
        "actions": [
            "python ./src/settings.py",
            "python ./src/pull_goyal_welch.py",
        ],
        "targets": [DATA_DIR / "goyal_welch.parquet"],
        "file_dep": ["./src/settings.py", "./src/pull_goyal_welch.py"],
        "clean": [],
    }


def task_tidy():
    """Build the tidy KMZ analysis dataset from the raw pulls"""
    return {
        "actions": [
            "python ./src/settings.py",
            "python ./src/clean_goyal_welch.py",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/clean_goyal_welch.py",
            DATA_DIR / "goyal_welch.parquet",
            DATA_DIR / "CRSP_MSIX.parquet",
        ],
        "targets": [DATA_DIR / "kmz_dataset.parquet"],
        "clean": [],
    }


def task_standardize():
    """Volatility-standardize the tidy dataset into the analysis-ready dataset"""
    return {
        "actions": [
            "python ./src/settings.py",
            "python ./src/standardize_kmz.py",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/standardize_kmz.py",
            "./src/clean_goyal_welch.py",
            DATA_DIR / "kmz_dataset.parquet",
        ],
        "targets": [DATA_DIR / "kmz_dataset_standardized.parquet"],
        "clean": [],
    }


def task_tidy_bonds():
    """Build the tidy bond excess-return dataset from the Goyal-Welch pull"""
    return {
        "actions": [
            "python ./src/settings.py",
            "python ./src/clean_bond_returns.py",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/clean_bond_returns.py",
            "./src/pull_goyal_welch.py",
            DATA_DIR / "goyal_welch.parquet",
        ],
        "targets": [DATA_DIR / "bond_returns.parquet"],
        "clean": [],
    }


def task_estimate():
    """Run the recursive OOS grid on the standardized dataset (the voc engine)"""
    return {
        "actions": [
            "python ./src/settings.py",
            "python ./src/run_estimation.py",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/sample_period.py",
            "./src/run_estimation.py",
            "./voc/__init__.py",
            "./voc/oos_engine.py",
            "./voc/performance_metrics.py",
            "./voc/rff.py",
            "./voc/kernel_ridge.py",
            DATA_DIR / "kmz_dataset_standardized.parquet",
        ],
        "targets": [AVERAGED_PATH, PER_SEED_PATH],
        # TRAIN_WINDOW and SAMPLE_END are visible through the target
        # filenames; N_SEEDS is not, so track it explicitly or a changed
        # .env would silently reuse a stale grid.
        "uptodate": [config_changed({"N_SEEDS": str(N_SEEDS)})],
        "clean": [],
    }


def task_figure7():
    """Replicate the paper's Figure 7 from the cached OOS grid"""
    return {
        "actions": [
            "python ./src/settings.py",
            "python ./src/figure7.py",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/sample_period.py",
            "./src/figure7.py",
            "./src/figure_style.py",
            AVERAGED_PATH,
        ],
        "targets": [
            OUTPUT_DIR / f"figure7{SAMPLE_SUFFIX}.png",
            OUTPUT_DIR / f"figure7{SAMPLE_SUFFIX}.pdf",
            OUTPUT_DIR / f"figure7_data{SAMPLE_SUFFIX}.parquet",
        ],
        "clean": True,
    }


def task_figure8():
    """Replicate the paper's Figure 8 from the cached OOS grid"""
    return {
        "actions": [
            "python ./src/settings.py",
            "python ./src/figure8.py",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/sample_period.py",
            "./src/figure8.py",
            "./src/figure_style.py",
            AVERAGED_PATH,
        ],
        "targets": [
            OUTPUT_DIR / f"figure8{SAMPLE_SUFFIX}.png",
            OUTPUT_DIR / f"figure8{SAMPLE_SUFFIX}.pdf",
            OUTPUT_DIR / f"figure8_data{SAMPLE_SUFFIX}.parquet",
        ],
        "clean": True,
    }


def task_nagel():
    """Run Nagel's critique tests, then build the comparison table and figure.

    Consumes the anchor forecast export written by `SAVE_FORECASTS=1 doit estimate`,
    so run that first to create `_data/forecasts_market{suffix}.parquet`.
    """
    return {
        "actions": [
            "python ./src/settings.py",
            "python ./src/nagel_analysis.py",
            "python ./src/table_nagel.py",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/sample_period.py",
            "./src/nagel_analysis.py",
            "./src/table_nagel.py",
            "./voc/nagel.py",
            "./voc/performance_metrics.py",
            DATA_DIR / "kmz_dataset_standardized.parquet",
            DATA_DIR / f"forecasts_market{SAMPLE_SUFFIX}.parquet",
        ],
        "targets": [
            DATA_DIR / f"nagel_metrics{SAMPLE_SUFFIX}.parquet",
            DATA_DIR / f"nagel_anatomy{SAMPLE_SUFFIX}.parquet",
            DATA_DIR / f"nagel_spanning{SAMPLE_SUFFIX}.parquet",
            OUTPUT_DIR / f"nagel_comparison_table{SAMPLE_SUFFIX}.tex",
            OUTPUT_DIR / f"figure_nagel{SAMPLE_SUFFIX}.png",
            OUTPUT_DIR / f"figure_nagel{SAMPLE_SUFFIX}.pdf",
        ],
        "clean": True,
    }


def task_variable_importance():
    """Run the Figure 11 leave-one-out estimations (full model + 15 exclusions)"""
    return {
        "actions": [
            "python ./src/settings.py",
            "python ./src/run_variable_importance.py",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/sample_period.py",
            "./src/run_variable_importance.py",
            "./voc/__init__.py",
            "./voc/oos_engine.py",
            "./voc/performance_metrics.py",
            "./voc/rff.py",
            "./voc/kernel_ridge.py",
            DATA_DIR / "kmz_dataset_standardized.parquet",
        ],
        "targets": [VI_PATH],
        # TRAIN_WINDOW and SAMPLE_END are visible through the target
        # filename; FIG11_N_SEEDS is not, so track it explicitly or a
        # changed .env would silently reuse a stale cache.
        "uptodate": [config_changed({"FIG11_N_SEEDS": str(FIG11_N_SEEDS)})],
        "clean": [],
    }


def task_figure11():
    """Replicate the paper's Figure 11 from the cached variable-importance runs"""
    return {
        "actions": [
            "python ./src/settings.py",
            "python ./src/figure11.py",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/sample_period.py",
            "./src/figure11.py",
            "./src/figure_style.py",
            VI_PATH,
        ],
        "targets": [
            OUTPUT_DIR / f"figure11{SAMPLE_SUFFIX}.png",
            OUTPUT_DIR / f"figure11{SAMPLE_SUFFIX}.pdf",
            OUTPUT_DIR / f"figure11_data{SAMPLE_SUFFIX}.parquet",
        ],
        "clean": True,
    }


def task_bonds_study():
    """Run the bonds VoC studies (government and corporate) through the API"""
    return {
        "actions": [
            "python ./src/settings.py",
            "python ./src/run_bonds_study.py",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/sample_period.py",
            "./src/run_bonds_study.py",
            "./src/clean_bond_returns.py",
            "./voc/__init__.py",
            "./voc/oos_engine.py",
            "./voc/performance_metrics.py",
            "./voc/preprocessing.py",
            "./voc/rff.py",
            "./voc/kernel_ridge.py",
            DATA_DIR / "bond_returns.parquet",
            DATA_DIR / "kmz_dataset_standardized.parquet",
        ],
        "targets": [BONDS_AVERAGED_PATH, BONDS_PER_SEED_PATH],
        # TRAIN_WINDOW and SAMPLE_END are visible through the target
        # filenames; BONDS_N_SEEDS is not, so track it explicitly.
        "uptodate": [config_changed({"BONDS_N_SEEDS": str(BONDS_N_SEEDS)})],
        "clean": [],
    }


def task_figure_bonds():
    """Plot the bonds VoC panels from the cached bond grid"""
    return {
        "actions": [
            "python ./src/settings.py",
            "python ./src/figure_bonds.py",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/sample_period.py",
            "./src/figure_bonds.py",
            "./src/figure_style.py",
            "./src/run_bonds_study.py",
            BONDS_AVERAGED_PATH,
        ],
        "targets": [
            OUTPUT_DIR / f"figure_bonds{SAMPLE_SUFFIX}.png",
            OUTPUT_DIR / f"figure_bonds{SAMPLE_SUFFIX}.pdf",
            OUTPUT_DIR / f"figure_bonds_data{SAMPLE_SUFFIX}.parquet",
        ],
        "clean": True,
    }


def task_template_examples():
    """Run the project template's demo scripts (example LaTeX docs need them)"""
    file_dep = [
        "./src/settings.py",
        "./src/example_table.py",
        "./src/pandas_to_latex_demo.py",
        "./src/example_plot.py",
    ]
    file_output = [
        "example_table.tex",
        "pandas_to_latex_simple_table1.tex",
        "example_plot.png",
    ]
    targets = [OUTPUT_DIR / file for file in file_output]

    return {
        "actions": [
            "python ./src/example_table.py",
            "python ./src/pandas_to_latex_demo.py",
            "python ./src/example_plot.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }


def task_summary_stats():
    """Generate summary statistics tables and plots"""
    file_dep = [
        "./src/settings.py",
        "./src/sample_period.py",
        "./src/table_predictor_summary.py",
        "./src/plot_predictor_timeseries.py",
        "./src/clean_goyal_welch.py",
        "./src/standardize_kmz.py",
        DATA_DIR / "kmz_dataset.parquet",
        DATA_DIR / "kmz_dataset_standardized.parquet",
    ]
    file_output = [
        f"predictor_summary_table{SAMPLE_SUFFIX}.tex",
        f"predictor_timeseries{SAMPLE_SUFFIX}.png",
        f"predictor_timeseries{SAMPLE_SUFFIX}.pdf",
    ]
    targets = [OUTPUT_DIR / file for file in file_output]

    return {
        "actions": [
            "python ./src/table_predictor_summary.py",
            "python ./src/plot_predictor_timeseries.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }


notebook_tasks = {
    "01_example_notebook_interactive.ipynb.py": {
        "path": "./src/01_example_notebook_interactive.ipynb.py",
        "file_dep": [],
        "targets": [],
    },
}


# fmt: off
def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks if the script version of it has been changed.
    """
    for notebook in notebook_tasks:
        pyfile_path = Path(notebook_tasks[notebook]["path"])
        notebook_path = pyfile_path.with_suffix("")  # strips .py, leaves .ipynb
        notebook_name = notebook_path.stem  # e.g. "01_example_notebook_interactive"
        yield {
            "name": notebook,
            "actions": [
                """python -c "import sys; from datetime import datetime; print(f'Start """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
                f"jupytext --to notebook --output {notebook_path} {pyfile_path}",
                jupyter_execute_notebook(notebook_path),
                jupyter_to_html(notebook_path),
                mv(notebook_path, OUTPUT_DIR),
                """python -c "import sys; from datetime import datetime; print(f'End """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
            ],
            "file_dep": [
                pyfile_path,
                *notebook_tasks[notebook]["file_dep"],
            ],
            "targets": [
                OUTPUT_DIR / f"{notebook_name}.html",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
        }
# fmt: on

###############################################################
## Task below is for LaTeX compilation
###############################################################


def task_compile_latex_docs():
    """Compile the LaTeX documents to PDFs"""
    file_dep = [
        "./reports/report_example.tex",
        "./reports/my_article_header.sty",
        "./reports/slides_example.tex",
        "./reports/my_beamer_header.sty",
        "./reports/my_common_header.sty",
        "./reports/report_simple_example.tex",
        "./reports/slides_simple_example.tex",
        # Outputs of task_template_examples that the example docs \input or
        # \includegraphics; depending on the outputs (not the scripts) makes
        # doit order the tasks correctly.
        OUTPUT_DIR / "example_table.tex",
        OUTPUT_DIR / "pandas_to_latex_simple_table1.tex",
        OUTPUT_DIR / "example_plot.png",
    ]
    targets = [
        "./reports/report_example.pdf",
        "./reports/slides_example.pdf",
        "./reports/report_simple_example.pdf",
        "./reports/slides_simple_example.pdf",
    ]

    return {
        "actions": [
            # My custom LaTeX templates
            "latexmk -xelatex -halt-on-error -cd ./reports/report_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/report_example.tex",  # Clean
            "latexmk -xelatex -halt-on-error -cd ./reports/slides_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/slides_example.tex",  # Clean
            # Simple templates based on small adjustments to Overleaf templates
            "latexmk -xelatex -halt-on-error -cd ./reports/report_simple_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/report_simple_example.tex",  # Clean
            "latexmk -xelatex -halt-on-error -cd ./reports/slides_simple_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/slides_simple_example.tex",  # Clean
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }


sphinx_targets = [
    "./docs/index.html",
]


def task_build_chartbook_site():
    """Compile Sphinx Docs"""
    notebook_scripts = [
        Path(notebook_tasks[notebook]["path"]) for notebook in notebook_tasks
    ]
    file_dep = [
        "./README.md",
        "./chartbook.toml",
        *notebook_scripts,
    ]

    return {
        "actions": [
            "chartbook build -f",
        ],  # Use docs as build destination
        "targets": sphinx_targets,
        "file_dep": file_dep,
        "task_dep": [
            "run_notebooks",
        ],
        "clean": True,
    }


def task_run_pytest():
    """Run pytest and save results to OUTPUT_DIR"""
    # Every directory pytest collects from; extend this tuple when a new code
    # package appears so editing its tests retriggers the task.
    tested_py_files = [
        path for folder in ("./src", "./voc") for path in Path(folder).glob("*.py")
    ]
    test_output = OUTPUT_DIR / "pytest_results.xml"

    def run_pytest():
        import subprocess

        result = subprocess.run(
            ["pytest", f"--junitxml={test_output}"],
            check=False,  # the returncode is inspected below
        )
        if result.returncode != 0:
            # Remove the XML so doit won't consider the target up-to-date
            Path(test_output).unlink(missing_ok=True)
            raise RuntimeError(f"pytest failed with exit code {result.returncode}")

    return {
        "actions": [run_pytest],
        "targets": [test_output],
        "file_dep": tested_py_files,
        "clean": True,
        "verbosity": 2,
    }
