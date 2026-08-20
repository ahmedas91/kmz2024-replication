## Description

Monthly bond excess returns, the targets of the bonds VoC study (issue #15), built from columns already present in the raw Goyal-Welch pull — zero new data risk. Both are same-row differences of realized simple returns for month t: `ltr_excess = ltr - Rfree` (long-term U.S. government bonds in excess of the T-bill rate; duration/term-premium risk only) and `corpr_excess = corpr - Rfree` (long-term corporates; duration plus credit risk). Complete from 1926-01 in the current vintage; the full available history is stored, period-independent, and the study driver trims to the configured `SAMPLE_END`.

Built by `doit tidy_bonds` (`src/clean_bond_returns.py`). No shifting happens here or downstream: the study adds each target's own lag by copying the standardized target onto the same row (the `lag_mkt_excess` convention), because the engine applies the single t to t+1 shift itself.

## Data Dictionary

One row per month:

- `date`: month-end timestamp.
- `ltr_excess`: long-term government bond return minus `Rfree`.
- `corpr_excess`: long-term corporate bond return minus `Rfree`.
