# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # A guided tour of the KMZ (2024) replication
#
# This notebook walks through the project's data and analysis in the order the
# pipeline builds them. The project replicates the central out-of-sample
# results of Kelly, Malamud, and Zhou (2024), "The Virtue of Complexity in
# Return Prediction" (*Journal of Finance* 79(1)): a model that predicts the
# monthly market excess return with a huge number of random features and only
# 12 months of training data performs better out of sample as the number of
# features grows, even though its out-of-sample R-squared looks terrible.
#
# The tour has five stops:
#
# 1. **The analysis dataset**: the 15 Goyal-Welch predictors and the market
#    excess return.
# 2. **Volatility standardization**: what it does to a slow and a fast
#    predictor, and why the paper needs it.
# 3. **One estimation window by hand**: random Fourier features, the dual-form
#    ridge solve, a forecast, and a timing-strategy return, at a small feature
#    count so every array fits on screen.
# 4. **The cached complexity grid**: mini versions of the paper's Figures 7
#    and 8 from the pipeline's 500-seed grid.
# 5. **The one-line API**: how the bonds and international extensions reuse
#    the same engine.
#
# The notebook only reads cached pipeline outputs from `_data/`, so run
# `doit standardize estimate` (or a full `doit`) once before executing it.
# `doit run_notebooks` handles the ordering automatically.

# %% [markdown]
# ## 1. The analysis dataset
#
# The pipeline pulls the predictor workbook from Amit Goyal's website and the
# CRSP value-weighted index from WRDS, then tidies them into one monthly
# dataset: the market excess return plus the paper's 15 predictors
# (Goyal-Welch variables, with valuation ratios in logs):
#
# - **Valuation ratios (logs)**: `dp`, `dy`, `ep`, `de`, `bm`
# - **Rates, spreads, and bond returns**: `tbl`, `lty`, `ltr`, `tms`, `dfy`, `dfr`
# - **Macro and issuance**: `infl`, `svar`, `ntis`
# - **The market's own lag**: `lag_mkt_excess`
#
# The timing conventions live in the module docstrings of
# `src/clean_goyal_welch.py` and `src/standardize_kmz.py`. One matters
# everywhere: `lag_mkt_excess` equals `mkt_excess` on the same row, because
# the shift from month t to month t+1 happens exactly once, inside the
# estimation engine. Shifting it again downstream would be a bug.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from sample_period import trim_to_sample
from settings import config

DATA_DIR = config("DATA_DIR")

tidy = pd.read_parquet(DATA_DIR / "kmz_dataset.parquet")
standardized = pd.read_parquet(DATA_DIR / "kmz_dataset_standardized.parquet")

for name, df in [("tidy", tidy), ("standardized", standardized)]:
    print(
        f"{name}: {len(df)} months, "
        f"{df['date'].iloc[0]:%Y-%m} to {df['date'].iloc[-1]:%Y-%m}"
    )
tidy.head()

# %% [markdown]
# ## 2. What volatility standardization does
#
# The paper standardizes everything by *backward-looking* volatility before
# estimation (their footnote 34):
#
# - **Returns** are divided by their trailing 12-month uncentered volatility.
# - **Predictors** are divided by an expanding-window volatility with a
#   36-month burn-in, which is why the standardized dataset starts in 1930-01.
#
# Both windows end strictly in the past, so no future information leaks into
# the scale. The point of the exercise: the random-feature map applies one
# bandwidth (gamma = 2) to all 15 inputs, which only makes sense if a slow
# valuation ratio and a fast return series live on comparable scales. Compare
# one of each, raw versus standardized:

# %%
examples = [
    ("dp", "log dividend-price ratio (persistent)"),
    ("lag_mkt_excess", "lagged market excess return (fast moving)"),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
for row, (col, label) in enumerate(examples):
    axes[row, 0].plot(tidy["date"], tidy[col], lw=0.7)
    axes[row, 0].set_title(f"{label}, raw")
    axes[row, 1].plot(standardized["date"], standardized[col], lw=0.7)
    axes[row, 1].set_title(f"{label}, standardized")
fig.tight_layout()

# %% [markdown]
# The raw panels differ by orders of magnitude in scale and behavior; the
# standardized panels are both O(1). The persistent ratio keeps its slow
# shape, and the return series keeps its high-frequency character, but the
# early-sample volatility spikes (the 1930s) are tamed in both.

# %% [markdown]
# ## 3. One estimation window by hand
#
# Now the mechanics of a single forecast, exactly as the engine does it, but
# with a feature count small enough to inspect. The conventions, each pinned
# to the paper:
#
# - At decision month t, the model trains on the T = 12 most recent pairs:
#   features from rows t-T .. t-1 against returns from rows t-T+1 .. t. This
#   is the single t to t+1 shift.
# - The forecast for month t+1 uses the feature row at t.
# - There is no intercept, ever (their footnote 35).
# - Features are scaled by their training-window standard deviation only
#   (their footnote 39), so nothing from the test month enters the scale.
#
# First, build the random Fourier features. Each Gaussian weight vector
# omega_i contributes a sin and a cos column, so P features come from P/2
# weight vectors:

# %%
from voc.kernel_ridge import predict, ridge_dual, ridgeless
from voc.rff import compute_rff, draw_rff_weights, standardize_by_training_window

sample = trim_to_sample(standardized, "2020-12")  # the paper's sample
predictor_cols = [c for c in sample.columns if c not in ("date", "mkt_excess")]
G = sample[predictor_cols].to_numpy()
R = sample["mkt_excess"].to_numpy()

T = 12  # training window, months
P = 100  # feature count; small here, up to 12,000 in the paper
SEED = 0

omega = draw_rff_weights(n_pairs=P // 2, n_predictors=G.shape[1], seed=SEED)
S = compute_rff(G, omega)  # gamma = 2, the paper's bandwidth

print(f"sample: {len(sample)} months, {G.shape[1]} predictors")
print(f"omega: {omega.shape} (one column per weight vector)")
print(f"S:     {S.shape} (interleaved sin/cos pairs)")

# %% [markdown]
# Pick an arbitrary decision month, standardize the features by the training
# window, and fit. With P = 100 features and only T = 12 observations the
# regression is far past the interpolation point, which is exactly the regime
# the paper studies. `ridge_dual` solves a T x T system (an
# eigendecomposition of the 12 x 12 Gram matrix), never a P x P one; that
# dual-form trick is what makes P = 12,000 across 500 seeds feasible.

# %%
t = 600  # decision month, chosen arbitrarily

S_train, s_oos = standardize_by_training_window(S[t - T : t], S[t])
R_train = R[t - T + 1 : t + 1]  # the single t to t+1 shift

beta_ridge = ridge_dual(S_train, R_train, z=[0.001, 1.0])  # one row per z
beta_minnorm = ridgeless(S_train, R_train)  # the z -> 0 minimum-norm limit

forecast = predict(s_oos, beta_minnorm)
realized = R[t + 1]

print(f"training features: {S_train.shape}, training returns: {R_train.shape}")
print(f"ridge betas: {beta_ridge.shape}, ridgeless beta: {beta_minnorm.shape}")
print(
    f"decision month {sample['date'].iloc[t]:%Y-%m}: "
    f"forecast {forecast:+.4f} for {sample['date'].iloc[t + 1]:%Y-%m}, "
    f"realized {realized:+.4f}"
)
print(f"timing-strategy return (forecast x realized): {forecast * realized:+.4f}")

# %% [markdown]
# The timing strategy holds a position equal to the forecast, so its monthly
# return is forecast times realized return. One window proves nothing; the
# engine repeats this for every month in the sample. `run_recursive_oos` is
# that loop for one seed, returning the paper's statistics per (P, z) model:

# %%
from voc.oos_engine import run_recursive_oos

rows = run_recursive_oos(G, R, seed=SEED, T=T, p_grid=[P], z_grid=[0.001, 1.0])
pd.DataFrame(rows)

# %% [markdown]
# The `z = 0.0` row is the ridgeless (minimum-norm) model. The full pipeline
# repeats this across 500 seeds and 14 model sizes up to P = 12,000, then
# averages the *statistics* across seeds, which is the paper's convention
# (average the statistics, not the forecasts).

# %% [markdown]
# ## 4. The cached complexity grid: mini Figures 7 and 8
#
# The `estimate` task caches that full grid. Each row is the across-seed mean
# of one (P, z) model's out-of-sample statistics; complexity is c = P / T,
# and `z = 0.0` marks the ridgeless limit.

# %%
grid = pd.read_parquet(DATA_DIR / "oos_grid_T12.parquet")
grid.head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
stats = [("r2", "Out-of-sample R-squared"), ("sharpe", "Annualized Sharpe ratio")]
for z_level in [0.001, 1.0, 1000.0]:
    sub = grid[grid["z"] == z_level]
    for ax, (stat, _) in zip(axes, stats):
        ax.plot(sub["c"], sub[stat], marker="o", ms=3, lw=1, label=f"z = {z_level:g}")
ridgeless_rows = grid[grid["z"] == 0.0]
for ax, (stat, title) in zip(axes, stats):
    ax.plot(
        ridgeless_rows["c"],
        ridgeless_rows[stat],
        color="black",
        lw=2,
        label="ridgeless",
    )
    ax.set_xscale("log")
    ax.axhline(0.0, color="gray", lw=0.5)
    ax.set_xlabel("model complexity c = P / T")
    ax.set_title(title)
axes[0].legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# This is the paper's headline contrast in one picture. The out-of-sample
# R-squared is deeply negative through the interpolation zone and climbs back
# only to roughly zero at the highest complexity, yet the Sharpe ratio of the
# timing strategy *rises* with complexity and is highest for the ridgeless
# model at the largest c. A forecast that looks useless by the R-squared
# yardstick still times the market: that is the virtue of complexity.
#
# The test suite validates the ridgeless high-complexity row of this grid
# against the paper's published anchor values (alpha t-statistic, information
# ratio, Sharpe ratio):

# %%
anchor_cols = ["P", "c", "r2", "sharpe", "information_ratio", "alpha_tstat"]
grid.query("z == 0.0 and P == 12000")[anchor_cols]

# %% [markdown]
# ## 5. The same engine on anything: `run_voc_study`
#
# Everything above is wrapped in one reusable entry point. The bonds and
# international-equity extensions in this project are single calls to
# `voc.run_voc_study` with a different standardized target and predictor set;
# `run_grid` is the DataFrame convenience wrapper used for the market study.
# A miniature run (two seeds, two model sizes) takes a few seconds:

# %%
from voc import run_grid

per_seed, averaged = run_grid(
    sample, p_grid=[12, 100], z_grid=[0.001, 1.0], seeds=range(2)
)
averaged[["P", "z", "c", "r2", "sharpe", "alpha_tstat"]]

# %% [markdown]
# `per_seed` holds one row per (seed, P, z); `averaged` is the across-seed
# mean of every statistic. Standardization is deliberately the caller's job
# (see `voc.preprocessing.standardize_inputs`), so the engine can never leak
# future information on your behalf.
#
# ## Where to go next
#
# - The full-size replication figures, the bonds and international studies,
#   and the Nagel-critique results are on the chartbook site built by
#   `doit build_chartbook_site`, with take-away notes per chart.
# - The write-up, with every statistic generated by the pipeline, is
#   `reports/report_kmz.pdf` (`doit compile_latex_docs`).
# - The estimation conventions are pinned in the module docstrings of
#   `voc/oos_engine.py`, `voc/rff.py`, and `voc/kernel_ridge.py`, and
#   enforced by the tests next to them.
