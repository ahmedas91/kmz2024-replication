"""Generate the LaTeX macros the project report quotes in prose.

The rubric requires that no statistic in the report is hand-typed. Tables
enter by ``\\input`` and figures by ``\\includegraphics``; this module covers
the remaining channel, numbers quoted in sentences, by writing
``_output/report_values.tex``: one ``\\newcommand`` per number, computed
from the cached pipeline artifacts. The report's preamble inputs the file,
so a rerun of the pipeline reflows every number in the text.

Macro names are letters only (LaTeX cannot digest digits in command names).
``Upd`` suffixes denote the updated sample (through 2024-12); bare names are
the paper period. Requires BOTH period runs to exist (``doit`` and
``SAMPLE_END=2024-12 doit``), like the report itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

VALUES_PATH = OUTPUT_DIR / "report_values.tex"
UPD = "_2024-12"


def _ridgeless_anchor(grid, target=None):
    """The ridgeless c = 1000 row of an averaged grid (optionally per target)."""
    rows = grid if target is None else grid.loc[grid["target"] == target]
    return rows.loc[(rows["z"] == 0.0) & (rows["P"] == rows["P"].max())].iloc[0]


def build_macros(data_dir=DATA_DIR):
    """Compute every reported number; return {macro_name: formatted_value}."""
    import pandas as pd

    from figure11 import load_vi_data
    from sample_period import trim_to_sample
    from standardize_kmz import load_standardized_dataset

    macros = {}

    market = pd.read_parquet(data_dir / "oos_grid_T12.parquet")
    market_upd = pd.read_parquet(data_dir / f"oos_grid_T12{UPD}.parquet")
    for tag, grid in (("", market), ("Upd", market_upd)):
        row = _ridgeless_anchor(grid)
        macros[f"MktSharpe{tag}"] = f"{row.sharpe:.2f}"
        macros[f"MktIR{tag}"] = f"{row.information_ratio:.2f}"
        macros[f"MktAlphaT{tag}"] = f"{row.alpha_tstat:.2f}"

    std = load_standardized_dataset(data_dir=data_dir)
    macros["PaperMonths"] = f"{len(trim_to_sample(std, '2020-12')):,}"
    macros["UpdMonths"] = f"{len(trim_to_sample(std, '2024-12')):,}"

    vi = load_vi_data(data_dir / "variable_importance_T12.parquet")
    leader = vi.iloc[0]
    macros["VILeaderRTwoPct"] = f"{100 * leader.vi_r2:.1f}"

    bonds = pd.read_parquet(data_dir / "oos_grid_bonds_T12.parquet")
    bonds_upd = pd.read_parquet(data_dir / f"oos_grid_bonds_T12{UPD}.parquet")
    for tag, grid in (("", bonds), ("Upd", bonds_upd)):
        macros[f"BondGovSharpe{tag}"] = (
            f"{_ridgeless_anchor(grid, 'ltr_excess').sharpe:.2f}"
        )
        macros[f"BondCorpSharpe{tag}"] = (
            f"{_ridgeless_anchor(grid, 'corpr_excess').sharpe:.2f}"
        )

    intl = pd.read_parquet(data_dir / "oos_grid_intl_T12.parquet")
    intl_upd = pd.read_parquet(data_dir / f"oos_grid_intl_T12{UPD}.parquet")
    macros["IntlSharpe"] = f"{_ridgeless_anchor(intl).sharpe:.2f}"
    best = intl.loc[(intl["z"] == 1000.0) & (intl["P"] == intl["P"].max())].iloc[0]
    macros["IntlSharpeBest"] = f"{best.sharpe:.2f}"
    macros["IntlSharpeUpd"] = f"{_ridgeless_anchor(intl_upd).sharpe:.2f}"

    nagel_metrics = pd.read_parquet(data_dir / "nagel_metrics.parquet")
    nagel_span = pd.read_parquet(data_dir / "nagel_spanning.parquet")
    voc_row = nagel_metrics.loc[
        nagel_metrics["strategy"] == "VoC (ridgeless, c=1000)"
    ].iloc[0]
    bench_row = nagel_metrics.loc[
        nagel_metrics["strategy"] == "Volatility-timed momentum"
    ].iloc[0]
    macros["NagelVoCSharpe"] = f"{voc_row.sharpe:.2f}"
    macros["NagelBenchSharpe"] = f"{bench_row.sharpe:.2f}"
    macros["NagelSpanAlphaT"] = f"{nagel_span.iloc[0].alpha_tstat:.2f}"
    macros["NagelAnatomyRTwo"] = f"{nagel_span.iloc[0].anatomy_r2:.2f}"

    counter = pd.read_parquet(data_dir / "nagel_counterfactuals.parquet")
    counter_span = pd.read_parquet(data_dir / "nagel_counterfactual_spanning.parquet")
    experiment_tags = {
        "historical": "Hist",
        "ma2_reversal": "MA",
        "wild_bootstrap": "Wild",
    }
    for experiment, tag in experiment_tags.items():
        rff = counter.loc[
            (counter["experiment"] == experiment) & (counter["strategy"] == "RFF")
        ].iloc[0]
        span = counter_span.loc[counter_span["experiment"] == experiment].iloc[0]
        macros[f"NagelRaw{tag}IR"] = f"{rff.information_ratio:.2f}"
        macros[f"NagelRaw{tag}AlphaT"] = f"{rff.alpha_tstat:.2f}"
        macros[f"NagelRaw{tag}SpanIR"] = f"{span.information_ratio:.2f}"
        macros[f"NagelRaw{tag}SpanAlphaT"] = f"{span.alpha_tstat:.2f}"
    macros["NagelExperimentSeeds"] = f"{int(counter.n_seeds.iloc[0]):,}"

    recovery = pd.read_parquet(data_dir / "nagel_twin_recovery.parquet")
    twin_span = pd.read_parquet(data_dir / "nagel_twin_spanning.parquet")
    design_tags = {"x14_shared": "Clean", "x15_world_lag": "Lag"}
    strength_tags = {"easy": "Easy", "realistic": "Real"}
    for design, design_tag in design_tags.items():
        for strength, strength_tag in strength_tags.items():
            row = recovery.loc[
                (recovery["predictor_set"] == design)
                & (recovery["strength"] == strength)
            ].iloc[0]
            prefix = f"NagelTwin{strength_tag}{design_tag}"
            macros[f"{prefix}Corr"] = f"{row.correlation:.2f}"
            macros[f"{prefix}Slope"] = f"{row.slope:.2f}"
            macros[f"{prefix}NRMSE"] = f"{row.nrmse:.2f}"
            for market, market_tag in (("plus", "Plus"), ("minus", "Minus")):
                span = twin_span.loc[
                    (twin_span["predictor_set"] == design)
                    & (twin_span["strength"] == strength)
                    & (twin_span["market"] == market)
                ].iloc[0]
                macros[f"{prefix}{market_tag}AlphaT"] = f"{span.alpha_tstat:.2f}"

    return macros


def main():
    """Write the macros file for the report preamble."""
    macros = build_macros()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by src/report_values.py; do not edit by hand."]
    lines += [
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in sorted(macros.items())
    ]
    VALUES_PATH.write_text("\n".join(lines) + "\n")
    print(f"[report_values] wrote {VALUES_PATH.name} ({len(macros)} macros)")


if __name__ == "__main__":
    main()
