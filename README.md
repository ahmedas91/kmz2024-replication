KMZ 2024 Replication
====================

## About this project

Replication of Kelly, Malamud & Zhou (2024), The Virtue of Complexity in Return Prediction

## Quick Start

The quickest way to run code in this repo is to use the following steps.

You must have TexLive (or another LaTeX distribution) installed on your computer and available in your path.
You can do this by downloading and installing it from here ([windows](https://tug.org/texlive/windows.html#install)
and [mac](https://tug.org/mactex/mactex-download.html) installers).


First, you must have the `conda` package manager installed (e.g., via Anaconda). However, I recommend using `mamba`, via [miniforge](https://github.com/conda-forge/miniforge) as it is faster and more lightweight than `conda`.

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate kmz2024_replication
```

Then copy `.env.example` to `.env` and set `WRDS_USERNAME` (a WRDS account
is required for the CRSP pull; the Goyal-Welch and Ken French pulls only
need internet). The other `.env` settings have sensible defaults.

Finally, run the project tasks:
```bash
doit
```
And that's it!


### Paper period vs. updated sample

Every estimation-dependent artifact is driven by the `SAMPLE_END` setting
(`.env` / environment variable / command line; see `src/sample_period.py`).
The default, `SAMPLE_END=2020-12`, reproduces the paper's 1930-01 to 2020-12
estimation sample and writes the bare canonical filenames (`figure7.png`,
`_data/oos_grid_T12.parquet`, ...). Any other value appends `_{SAMPLE_END}`
to those filenames, so both runs coexist side by side. To produce the
updated-sample results through 2024-12 with zero code edits:

```bash
doit                                    # paper period (bare filenames)
SAMPLE_END=2024-12 doit                 # updated sample (figure7_2024-12.png, ...)
```

(or edit `SAMPLE_END` in `.env` and rerun `doit`). The pulls, the tidy
dataset, and the standardized dataset are period-independent: they always
store the full available history (bounded only by `START_DATE`/`END_DATE`)
and are trimmed downstream, so switching `SAMPLE_END` does not re-pull
anything. Once both artifact sets exist, switching back and forth never
recomputes the estimation grids; only the (seconds-fast, deterministic)
figure tasks redraw on a switch, because their grid dependency path changes
with the period.

Data vintage used for the updated sample: the Goyal-Welch predictor workbook
is the "All data up to 2025" file posted on Amit Goyal's website (downloaded
2026-08; monthly rows complete through 2024-12), and the CRSP value-weighted
index comes from WRDS (CIZ format) pulled through `END_DATE=2024-12-31`, so
the updated estimation sample ends 2024-12. Goyal's posted file updates
roughly annually; re-run `doit forget pull:goyal_welch && doit` to refresh
it and raise `END_DATE`/`SAMPLE_END` when a newer vintage appears.

Test gating: the quantitative anchor tests (`src/test_figure7.py`,
`test_figure8.py`, `test_figure11.py`) always read the bare paper-period
artifacts, so they validate the paper's numbers regardless of the configured
`SAMPLE_END` and skip only when those artifacts are missing. When
`SAMPLE_END` is set away from the paper period, `src/test_updated_sample.py`
adds sanity checks on the suffixed artifact set (existence, finite values,
full grid structure); it skips under the paper-period config.


### Other commands

#### Unit Tests and Doc Tests

You can run the unit test, including doctests, with the following command:
```
pytest --doctest-modules
```

You can build the documentation (the chartbook site under `docs/`) with:
```
doit build_chartbook_site
```


#### Setting Environment Variables

You can [export your environment variables](https://stackoverflow.com/questions/43267413/how-to-set-environment-variables-from-env-file)
from your `.env` files like so, if you wish. This can be done easily in a Linux or Mac terminal with the following command:
```bash
set -a  # automatically export all variables
source .env
set +a
```
On Windows (PowerShell):
```powershell
Get-Content .env | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }
```

### Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting Python code.

```bash
# Auto-fix linting issues (e.g., unused imports, undefined names)
ruff check . --fix

# Format code (consistent style, spacing, line length)
ruff format .

# Sort imports, then fix linting issues, then format
ruff format . && ruff check --select I --fix . && ruff check --fix .
```

- `ruff check --fix` applies safe auto-fixes for linting violations
- `ruff format` formats code similar to Black
- `--select I` targets only import sorting rules (isort-compatible)

### General Directory Structure

 - The `assets` folder is used for things like hand-drawn figures or other
   pictures that were not generated from code. These things cannot be easily
   recreated if they are deleted.

 - The `_output` folder, on the other hand, contains dataframes and figures that are
   generated from code. The entire folder should be able to be deleted, because
   the code can be run again, which would again generate all of the contents.

 - The `data_manual` is for data that cannot be easily recreated. This data
   should be version controlled. Anything in the `_data` folder or in
   the `_output` folder should be able to be recreated by running the code
   and can safely be deleted.

 - I'm using the `doit` Python module as a task runner. It works like `make` and
   the associated `Makefile`s. To rerun the code, install `doit`
   (https://pydoit.org/) and execute the command `doit` from the repository
   root (where `dodo.py` lives). Note that doit is very flexible and can be used to run code
   commands from the command prompt, thus making it suitable for projects that
   use scripts written in multiple different programming languages.

 - I'm using the `.env` file as a container for absolute paths that are private
   to each collaborator in the project. You can also use it for private
   credentials, if needed. It should not be tracked in Git.

### Data and Output Storage

I'll often use a separate folder for storing data. Any data in the data folder
can be deleted and recreated by rerunning the PyDoit command (the pulls are in
the dodo.py file). Any data that cannot be automatically recreated should be
stored in the "data_manual" folder. Because of the risk of manually-created data
getting changed or lost, I prefer to keep it under version control if I can.
Thus, data in the "_data" folder is excluded from Git (see the .gitignore file),
while the "data_manual" folder is tracked by Git.

Output is stored in the "_output" directory. This includes dataframes, charts, and
rendered notebooks. When the output is small enough, I'll keep this under
version control. I like this because I can keep track of how dataframes change as my
analysis progresses, for example.

Of course, the _data directory and _output directory can be kept elsewhere on the
machine. To make this easy, I always include the ability to customize these
locations by defining the path to these directories in environment variables,
which I intend to be defined in the `.env` file, though they can also simply be
defined on the command line or elsewhere. The `settings.py` is responsible for
loading these environment variables and doing some preprocessing on them.
The `settings.py` file is the entry point for all other scripts to these
definitions. That is, all code that references these variables and others are
loaded by importing `config`.

### Naming Conventions

 - **`pull_` vs `load_`**: Files or functions that pull data from an external
 data source are prepended with "pull_", as in "pull_fred.py". Functions that
 load data that has been cached in the "_data" folder are prepended with "load_".
 For example, inside of the `pull_goyal_welch.py` file there is both a
 `pull_goyal_welch` function and a `load_goyal_welch` function. The first pulls
 from the web, whereas the other loads cached data from the "_data" directory.


### Dependencies and Virtual Environments

#### Working with `conda` environments

This project uses conda for environment management. The dependencies are stored in `environment.yml`.

To create/update the environment:
```bash
conda env create -f environment.yml
# or to update an existing environment:
conda env update -f environment.yml
```

To activate the environment:
```bash
conda activate kmz2024_replication
```

To export a snapshot of the current environment (to a SEPARATE file, so the
curated `environment.yml` is not overwritten):
```bash
conda env export > environment_snapshot.yml
```

**Tip:** Consider using `mamba` instead of `conda` for faster package resolution. Install via [miniforge](https://github.com/conda-forge/miniforge).

