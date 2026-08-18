"""Out-of-sample performance statistics for the KMZ (2024) VoC engine.

Every statistic is computed WITHIN a single RFF repetition (seed); the engine
averages the finished statistics across seeds afterwards. This is the paper's
aggregation convention (2025 reply, Section 4.2.1): never average forecasts
across seeds and then compute one statistic — average the per-seed statistics.

Definitions
-----------
- OOS R^2 (footnote 40): ``1 - Var(forecast errors) / Var(realized returns)``
  with CENTERED variances, so the statistic is invariant to any CONSTANT
  forecast: every constant (zero included) scores exactly 0, and a constant
  bias on top of a good forecast is forgiven entirely. It rewards covariation
  with the realized return, not level accuracy; do not "normalize" it to the
  uncentered zero-benchmark form ``1 - sum(e^2)/sum(R^2)``, which is a
  different statistic. As the paper stresses, a good market-timing strategy
  routinely posts a deeply NEGATIVE R^2 because R^2 is dominated by forecast
  scale while trading profits load on forecast direction.
- Timing strategy: ``return_t = forecast_t * realized_t``; Sharpe =
  ``sqrt(12) * mean / SD`` (annualized; centered SD, footnote 40).
- Alpha / IR / alpha t-stat: OLS of the strategy return on a static position in
  the volatility-standardized market (Figure 8 caption) — intercept ``alpha``,
  slope; ``IR = sqrt(12) * alpha / SD(residual)``; alpha t-stat from the OLS
  standard error of the intercept. Sharpe and IR are annualized by sqrt(12);
  alpha is monthly. The Figure 8 anchor values (issue #9) are the final arbiter
  of these conventions.
"""

from __future__ import annotations

import numpy as np

PERIODS_PER_YEAR = 12


def oos_r2(forecasts, realized):
    """Out-of-sample R^2 with centered variances (footnote 40).

    Invariant to constant forecasts: any constant, zero included, scores
    exactly 0 (see the module docstring before changing this formula).
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    realized = np.asarray(realized, dtype=np.float64)
    return 1.0 - np.var(realized - forecasts) / np.var(realized)


def sharpe_ratio(strategy_returns, periods_per_year=PERIODS_PER_YEAR):
    """Annualized Sharpe ratio of a return series (centered SD, footnote 40)."""
    strategy_returns = np.asarray(strategy_returns, dtype=np.float64)
    return np.sqrt(periods_per_year) * strategy_returns.mean() / strategy_returns.std()


def alpha_ir_tstat(strategy_returns, market_returns, periods_per_year=PERIODS_PER_YEAR):
    """OLS of the strategy on the market: ``(alpha, information_ratio, alpha_tstat)``.

    ``alpha`` and the slope come from intercept-plus-slope OLS; the information
    ratio is ``sqrt(periods) * alpha / SD(residual)``; the alpha t-statistic is
    ``alpha`` over the OLS standard error of the intercept.
    """
    y = np.asarray(strategy_returns, dtype=np.float64)
    x = np.asarray(market_returns, dtype=np.float64)
    n = y.size
    if n < 3:
        # n = 2 fits the two points exactly (resid_var divides by n - 2 = 0) and
        # returns finite garbage; n < 2 degenerates earlier. Fail loudly instead.
        raise ValueError(f"need at least 3 observations for the alpha OLS; got {n}")
    xbar, ybar = x.mean(), y.mean()
    sxx = np.sum((x - xbar) ** 2)
    if sxx == 0.0:
        raise ValueError("market_returns are constant; the alpha OLS is undefined")
    slope = np.sum((x - xbar) * (y - ybar)) / sxx
    alpha = ybar - slope * xbar
    resid = y - alpha - slope * x
    information_ratio = np.sqrt(periods_per_year) * alpha / resid.std()
    resid_var = np.sum(resid**2) / (n - 2)
    alpha_se = np.sqrt(resid_var * (1.0 / n + xbar**2 / sxx))
    alpha_tstat = alpha / alpha_se
    return alpha, information_ratio, alpha_tstat


def compute_metrics(forecasts, realized, beta_norms):
    """Assemble the full statistic set for ONE (seed, P, z) model.

    Annualization is fixed at PERIODS_PER_YEAR (monthly data, the paper's
    convention); the standalone sharpe_ratio/alpha_ir_tstat utilities keep the
    parameter for other frequencies.

    Parameters
    ----------
    forecasts : array-like, shape (n_oos,)
        Out-of-sample forecasts ``beta' S_t``.
    realized : array-like, shape (n_oos,)
        Realized returns ``R_{t+1}`` — both the market leg and the strategy's
        multiplier.
    beta_norms : array-like, shape (n_oos,)
        L2 norm of ``beta_hat`` in each training window (Figure 7 panel B).

    Returns
    -------
    dict
        Keys: r2, beta_norm, mean_return, volatility, sharpe, alpha,
        information_ratio, alpha_tstat.
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    realized = np.asarray(realized, dtype=np.float64)
    strategy = forecasts * realized
    alpha, information_ratio, alpha_tstat = alpha_ir_tstat(strategy, realized)
    return {
        "r2": float(oos_r2(forecasts, realized)),
        "beta_norm": float(np.mean(beta_norms)),
        "mean_return": float(strategy.mean()),
        "volatility": float(strategy.std()),
        "sharpe": float(sharpe_ratio(strategy)),
        "alpha": float(alpha),
        "information_ratio": float(information_ratio),
        "alpha_tstat": float(alpha_tstat),
    }
