"""
Portfolio realized-volatility for the hedge signal's book option.

Computes the trailing realized vol of a *current* portfolio (constituents +
signed weights) to use in place of QQQ's rv20. Consistent with the QQQ
methodology in lib/qqq_hedge.py: annualized (x sqrt(252)) sample std (ddof=1) of
LOG returns, through date t (most recent close included).

Key identity: portfolio SIMPLE return is the weighted sum of constituent SIMPLE
returns (linear in weights), so we aggregate in simple-return space first, then
take ln(1 + r_p) for the vol estimate. (Weighting log returns directly is wrong
for cross-sectional aggregation.)

Pure math, no network -- unit-testable offline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Require at least this fraction of gross weight to be priced, else error.
MIN_COVERAGE = 0.95


def _as_weight_series(weights) -> pd.Series:
    w = weights if isinstance(weights, pd.Series) else pd.Series(weights, dtype=float)
    w.index = [str(s).upper() for s in w.index]
    return w


def portfolio_return_series(
    returns_df: pd.DataFrame,
    weights,
) -> pd.Series:
    """r_p(t) = sum_i w_i * r_i_simple(t) over the priced constituents.

    Inner-aligns to the weight symbols, renormalizes the priced weights to
    sum(|w|)==1 (so dropped names don't change the scale), and drops dates where
    any priced constituent return is NaN (no ffill -- ffill biases vol down).
    """
    w = _as_weight_series(weights)
    cols = [c for c in returns_df.columns if c in w.index]
    if not cols:
        raise ValueError("no priced constituents in returns_df")
    wv = w.reindex(cols).astype(float)
    gross = wv.abs().sum()
    if gross == 0:
        raise ValueError("priced constituent weights sum to zero")
    wv = wv / gross
    sub = returns_df[cols].dropna(how="any")
    return sub.mul(wv, axis=1).sum(axis=1)


def portfolio_value_series(
    returns_df: pd.DataFrame,
    weights,
    base: float = 100.0,
) -> pd.Series:
    """Materialized daily portfolio value (NAV) index.

    V(t) = base * prod_{s<=t} (1 + r_p(s))  where r_p is the (daily-rebalanced)
    weighted simple return of the portfolio. Indexed by the aligned trading days.
    Differencing log(V) recovers the portfolio's daily log returns:
    ln(V_t / V_{t-1}) = ln(1 + r_p(t)).
    """
    r_p = portfolio_return_series(returns_df, weights)
    return base * (1.0 + r_p).cumprod()


def portfolio_realized_vol_asof(
    returns_df: pd.DataFrame,
    weights,
    as_of: Optional[str] = None,
    window: int = 30,
    annualization: float = 252.0,
    min_coverage: float = MIN_COVERAGE,
) -> dict:
    """Scalar trailing portfolio vol for the live single-date signal.

    Builds the explicit daily portfolio value (NAV) series, takes its log
    returns, and annualizes the sample std of the last `window` returns:
      V(t) = base * prod(1 + r_p)          (portfolio_value_series)
      lr(t) = ln(V_t / V_{t-1})            (= ln(1 + r_p(t)))
      sigma = std(lr[last window], ddof=1) * sqrt(annualization)
    `window` daily returns require `window + 1` NAV points (31 values -> 30
    returns). Rolls back to the most recent aligned trading day <= as_of.

    Returns {portfolio_vol, as_of_date, n_aligned_days, n_priced, missing,
    coverage, net_exposure} or {"error": ...}.
    """
    w = _as_weight_series(weights)
    priced = [c for c in w.index if c in returns_df.columns]
    missing = [c for c in w.index if c not in returns_df.columns]
    if not priced:
        return {"error": "no priced constituents"}

    total_gross = w.abs().sum() or 1.0
    coverage = float(w.reindex(priced).abs().sum() / total_gross)
    if coverage < min_coverage:
        return {
            "error": (
                f"only {coverage:.0%} of gross weight priced "
                f"(missing {missing[:10]}); need >= {min_coverage:.0%}"
            ),
            "missing": missing,
            "coverage": coverage,
        }

    # 1) daily portfolio value (NAV) series
    nav = portfolio_value_series(returns_df, w.reindex(priced))
    if as_of is not None:
        nav = nav[nav.index <= pd.Timestamp(as_of)]
    # need window+1 values to produce `window` log returns
    if len(nav) < window + 1:
        return {
            "error": (
                f"insufficient history: {len(nav)} aligned values "
                f"< {window + 1} needed for {window} returns"
            )
        }

    # 2) daily portfolio log returns from the value series
    log_rets = np.log(nav).diff().dropna()
    log_rets = log_rets[np.isfinite(log_rets)]
    if len(log_rets) < window:
        return {"error": "too many non-finite portfolio returns (over-levered book?)"}

    # 3) sample std of the last `window` returns, 4) annualized
    win = log_rets.iloc[-window:]
    vol = float(win.std(ddof=1) * np.sqrt(annualization))
    if not np.isfinite(vol) or vol <= 0:
        return {"error": "portfolio vol is non-finite or zero"}

    return {
        "portfolio_vol": vol,
        "as_of_date": str(nav.index[-1].date()),
        "n_aligned_days": int(len(nav)),
        "window": int(window),
        "n_priced": len(priced),
        "missing": missing,
        "coverage": round(coverage, 4),
        "net_exposure": round(float(w.reindex(priced).sum()), 6),
    }
