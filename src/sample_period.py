"""One place for the estimation-sample configuration and the artifact suffix.

``SAMPLE_END`` (YYYY-MM, via ``.env`` / command line) picks the last month of
the estimation sample for every period-dependent artifact: the OOS grid and
variable-importance caches in ``_data`` and the figures and summary outputs
in ``_output``. The paper period (``SAMPLE_END = 2020-12``, the default)
keeps the bare canonical filenames that the chartbook, the reports, and the
anchor tests point at; any OTHER value appends ``_{SAMPLE_END}`` to every
period-dependent artifact stem (``figure7_2024-12.png`` next to
``figure7.png``), so the paper run and an updated-sample run coexist side by
side. Switching the ``.env`` value and rerunning ``doit`` therefore
regenerates the full artifact set for the new period with zero code edits and
without clobbering the paper set.

Period-INDEPENDENT artifacts carry no suffix: the raw pulls, the tidy
dataset, and the standardized dataset always store the full available history
(the pulls are bounded only by ``START_DATE``/``END_DATE``) and are trimmed
downstream by the consumers of ``SAMPLE_END``.

Import-light on purpose (settings only): ``dodo.py`` imports these constants
at every parse to name its targets.
"""

from settings import config

PAPER_SAMPLE_END = "2020-12"
SAMPLE_END = config("SAMPLE_END", default=PAPER_SAMPLE_END, cast=str)
SAMPLE_SUFFIX = "" if SAMPLE_END == PAPER_SAMPLE_END else f"_{SAMPLE_END}"


def trim_to_sample(df, sample_end=None):
    """Rows of ``df`` (with a ``date`` column) up to and including the last
    month of the estimation sample (default: the configured SAMPLE_END).

    The one shared implementation of the sample trim every estimation driver
    and summary script applies; pandas is imported locally so this module
    stays import-light for dodo.py.
    """
    import pandas as pd

    cutoff = pd.Timestamp(sample_end or SAMPLE_END) + pd.offsets.MonthEnd(0)
    return df.loc[df["date"] <= cutoff].reset_index(drop=True)
