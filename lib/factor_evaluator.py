"""
Factor evaluation protocol: standardized metrics for autonomous factor discovery.

Implements the paper's Section 3.3 evaluation framework:
  - Rank IC (Spearman correlation) with t-statistic and ICIR
  - Long-short portfolio (top vs bottom 50%) returns, Sharpe, Sortino, etc.
  - Decile portfolio sorts with monotonicity check
  - Cross-sectional preprocessing (winsorize + z-score per date)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FactorMetrics:
    """Complete evaluation result for a single factor."""

    # Rank IC statistics
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_t_stat: float = 0.0
    icir: float = 0.0

    # Long-short portfolio (top 50% - bottom 50%)
    ls_annual_return: float = 0.0
    ls_annual_vol: float = 0.0
    ls_sharpe: float = 0.0
    ls_sortino: float = 0.0
    ls_max_drawdown: float = 0.0
    ls_calmar: float = 0.0

    # Decile sort diagnostics
    decile_returns: list[float] | None = None
    decile_monotonic: bool = False

    # Turnover (mean daily 2-sided turnover as fraction of portfolio)
    turnover: float = 0.0

    # Signal decay: Sharpe at holding horizons H1..H7
    decay_sharpes: list[float] | None = None

    # Redundancy: max abs correlation with existing promoted factors
    max_corr_with_existing: float = 0.0

    # Meta
    n_days: int = 0
    n_stocks_avg: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_str(self) -> str:
        decay_str = ""
        if self.decay_sharpes:
            decay_str = f" | Decay H1→H7: {self.decay_sharpes[0]:.2f}→{self.decay_sharpes[-1]:.2f}"
        return (
            f"IC: {self.ic_mean:.4f} (t={self.ic_t_stat:.2f}, ICIR={self.icir:.2f}) | "
            f"LS Sharpe: {self.ls_sharpe:.2f}, Ret: {self.ls_annual_return:.2%} | "
            f"MaxDD: {self.ls_max_drawdown:.2%} | TO: {self.turnover:.2f} | "
            f"Mono: {self.decile_monotonic}{decay_str}"
        )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def winsorize_cross_section(
    df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99,
) -> pd.DataFrame:
    """Winsorize each row (date) at given percentiles."""
    def _clip_row(row: pd.Series) -> pd.Series:
        lo = row.quantile(lower)
        hi = row.quantile(upper)
        return row.clip(lo, hi)
    return df.apply(_clip_row, axis=1)


def zscore_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize each row (date)."""
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1).replace(0, np.nan)
    return df.sub(row_mean, axis=0).div(row_std, axis=0)


def preprocess_factor(raw_factor: pd.DataFrame) -> pd.DataFrame:
    """Winsorize then z-score a factor panel (dates x symbols)."""
    winsorized = winsorize_cross_section(raw_factor)
    return zscore_cross_section(winsorized)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def compute_daily_rank_ic(
    factor: pd.DataFrame, forward_returns: pd.DataFrame,
) -> pd.Series:
    """Compute daily Spearman rank IC between factor scores and next-day returns.

    Both inputs: (dates x symbols) DataFrames, aligned.
    Returns: Series indexed by date with daily IC values.
    """
    common_dates = factor.index.intersection(forward_returns.index)
    common_cols = factor.columns.intersection(forward_returns.columns)
    f = factor.loc[common_dates, common_cols]
    r = forward_returns.loc[common_dates, common_cols]

    ics = []
    dates = []
    for dt in common_dates:
        f_row = f.loc[dt].dropna()
        r_row = r.loc[dt].dropna()
        common = f_row.index.intersection(r_row.index)
        if len(common) < 5:
            continue
        corr, _ = stats.spearmanr(f_row[common], r_row[common])
        if not np.isnan(corr):
            ics.append(corr)
            dates.append(dt)

    return pd.Series(ics, index=dates, name="rank_ic")


def compute_ls_portfolio_returns(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    top_pct: float = 0.5,
) -> pd.Series:
    """Compute daily long-short (top - bottom) portfolio returns.

    Sorts stocks by factor score each day. Longs the top_pct, shorts bottom_pct.
    Equal-weighted within each leg. Vectorized for speed.
    """
    common_dates = factor.index.intersection(forward_returns.index)
    common_cols = factor.columns.intersection(forward_returns.columns)
    f = factor.loc[common_dates, common_cols]
    r = forward_returns.loc[common_dates, common_cols]

    # Vectorized: use rank-based approach
    # Mask NaNs in both factor and returns
    valid = f.notna() & r.notna()
    f_masked = f.where(valid)
    r_masked = r.where(valid)

    # Count valid stocks per day
    n_valid = valid.sum(axis=1)
    good_days = n_valid >= 10

    if good_days.sum() == 0:
        return pd.Series([], dtype=float, name="ls_return")

    f_good = f_masked.loc[good_days]
    r_good = r_masked.loc[good_days]
    n_good = n_valid.loc[good_days]

    # Rank factor cross-sectionally (per row)
    ranks = f_good.rank(axis=1, pct=True, na_option="keep")

    # Top bucket: rank > (1 - top_pct), Bottom bucket: rank <= top_pct
    # With top_pct=0.5: top is rank > 0.5, bottom is rank <= 0.5
    top_mask = ranks > (1 - top_pct)
    bottom_mask = ranks <= top_pct

    # Compute mean returns for each bucket
    top_ret = (r_good * top_mask).sum(axis=1) / top_mask.sum(axis=1).replace(0, np.nan)
    bottom_ret = (r_good * bottom_mask).sum(axis=1) / bottom_mask.sum(axis=1).replace(0, np.nan)

    ls = (top_ret - bottom_ret).dropna()
    ls.name = "ls_return"
    return ls


def compute_decile_returns(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_quantiles: int = 10,
) -> list[float]:
    """Compute average annualized return for each decile portfolio.

    Returns list of length n_quantiles (low to high factor score).
    """
    common_dates = factor.index.intersection(forward_returns.index)
    common_cols = factor.columns.intersection(forward_returns.columns)
    f = factor.loc[common_dates, common_cols]
    r = forward_returns.loc[common_dates, common_cols]

    decile_daily: list[list[float]] = [[] for _ in range(n_quantiles)]

    for dt in common_dates:
        f_row = f.loc[dt].dropna()
        r_row = r.loc[dt].dropna()
        common = f_row.index.intersection(r_row.index)
        if len(common) < n_quantiles * 2:
            continue
        f_vals = f_row[common]
        r_vals = r_row[common]

        try:
            labels = pd.qcut(f_vals.rank(method="first"), n_quantiles, labels=False)
        except ValueError:
            continue

        for q in range(n_quantiles):
            mask = labels == q
            if mask.any():
                decile_daily[q].append(r_vals[mask[mask].index].mean())

    # Annualize mean daily returns
    result = []
    for daily_rets in decile_daily:
        if daily_rets:
            mean_daily = np.mean(daily_rets)
            result.append(float(mean_daily * 252))
        else:
            result.append(0.0)
    return result


def _check_monotonic(decile_returns: list[float]) -> bool:
    """Check if decile returns are broadly monotonically increasing."""
    if len(decile_returns) < 3:
        return False
    # Allow 1 violation out of n-1 transitions
    violations = sum(
        1 for i in range(len(decile_returns) - 1)
        if decile_returns[i + 1] < decile_returns[i]
    )
    return violations <= 1


def compute_turnover(
    factor: pd.DataFrame, top_pct: float = 0.5,
) -> float:
    """Compute mean daily 2-sided turnover of the long-short portfolio.

    Turnover = fraction of positions that change each day (0 to 1).
    """
    common_dates = factor.index
    if len(common_dates) < 2:
        return 0.0

    turnovers = []
    prev_top = prev_bottom = None
    for dt in common_dates:
        f_row = factor.loc[dt].dropna()
        if len(f_row) < 10:
            continue
        n = len(f_row)
        cutoff = int(n * top_pct)
        sorted_idx = f_row.sort_values().index
        bottom = set(sorted_idx[:cutoff])
        top = set(sorted_idx[-cutoff:])

        if prev_top is not None:
            long_to = len(top.symmetric_difference(prev_top)) / (2 * len(top))
            short_to = len(bottom.symmetric_difference(prev_bottom)) / (2 * len(bottom))
            turnovers.append((long_to + short_to) / 2)

        prev_top, prev_bottom = top, bottom

    return float(np.mean(turnovers)) if turnovers else 0.0


def compute_signal_decay(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    horizons: list[int] | None = None,
    top_pct: float = 0.5,
) -> list[float]:
    """Compute long-short Sharpe at multiple holding horizons.

    Returns list of Sharpe ratios for each horizon (H1..H7 by default).
    """
    if horizons is None:
        horizons = list(range(1, 8))

    sharpes = []
    for h in horizons:
        fwd_h = returns.rolling(window=h).sum().shift(-h)
        ls_rets = compute_ls_portfolio_returns(factor, fwd_h, top_pct=top_pct)
        if len(ls_rets) > 0:
            mean_d = float(ls_rets.mean())
            std_d = float(ls_rets.std())
            sharpe = float(np.sqrt(252 / h) * mean_d / std_d) if std_d > 0 else 0.0
        else:
            sharpe = 0.0
        sharpes.append(sharpe)
    return sharpes


def compute_redundancy(
    factor: pd.DataFrame,
    existing_panels: dict[str, pd.DataFrame],
) -> float:
    """Compute max absolute rank correlation between factor and existing promoted factors.

    Returns max absolute correlation (0 to 1). 0 means no redundancy.
    """
    if not existing_panels:
        return 0.0

    max_corr = 0.0
    for name, existing in existing_panels.items():
        common_dates = factor.index.intersection(existing.index)
        common_cols = factor.columns.intersection(existing.columns)
        if len(common_dates) < 20 or len(common_cols) < 5:
            continue
        f_flat = factor.loc[common_dates, common_cols].values.flatten()
        e_flat = existing.loc[common_dates, common_cols].values.flatten()
        mask = ~(np.isnan(f_flat) | np.isnan(e_flat))
        if mask.sum() < 100:
            continue
        corr, _ = stats.spearmanr(f_flat[mask], e_flat[mask])
        if not np.isnan(corr):
            max_corr = max(max_corr, abs(corr))
    return max_corr


def _max_drawdown(cumulative: pd.Series) -> float:
    """Compute maximum drawdown from a cumulative return series."""
    if cumulative.empty:
        return 0.0
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max.replace(0, np.nan)
    dd = drawdowns.min()
    return float(dd) if not np.isnan(dd) else 0.0


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_factor(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    preprocess: bool = True,
    raw_returns: pd.DataFrame | None = None,
    existing_panels: dict[str, pd.DataFrame] | None = None,
    fwd_horizon: int = 1,
) -> FactorMetrics:
    """Full evaluation of a single factor.

    Args:
        factor: (dates x symbols) DataFrame of raw or preprocessed factor values.
        forward_returns: (dates x symbols) DataFrame of next-period returns.
        preprocess: if True, apply winsorize + zscore to factor before evaluation.
        raw_returns: un-shifted returns for signal decay analysis. If None, skips decay.
        existing_panels: dict of promoted factor panels for redundancy check.

    Returns:
        FactorMetrics with all evaluation results.
    """
    if preprocess:
        factor = preprocess_factor(factor)

    metrics = FactorMetrics()

    # Rank IC
    daily_ic = compute_daily_rank_ic(factor, forward_returns)
    if len(daily_ic) > 0:
        metrics.ic_mean = float(daily_ic.mean())
        metrics.ic_std = float(daily_ic.std())
        n = len(daily_ic)
        metrics.ic_t_stat = float(
            metrics.ic_mean / (metrics.ic_std / np.sqrt(n))
        ) if metrics.ic_std > 0 else 0.0
        metrics.icir = float(
            metrics.ic_mean / metrics.ic_std
        ) if metrics.ic_std > 0 else 0.0

    # Long-short portfolio
    ls_rets = compute_ls_portfolio_returns(factor, forward_returns)
    periods_per_year = 252 / max(fwd_horizon, 1)
    if len(ls_rets) > 0:
        mean_period = float(ls_rets.mean())
        std_period = float(ls_rets.std())
        metrics.ls_annual_return = mean_period * periods_per_year
        metrics.ls_annual_vol = std_period * np.sqrt(periods_per_year)
        metrics.ls_sharpe = float(
            np.sqrt(periods_per_year) * mean_period / std_period
        ) if std_period > 0 else 0.0

        # Sortino (downside deviation)
        downside = ls_rets[ls_rets < 0]
        downside_std = float(downside.std()) if len(downside) > 0 else 0.0
        metrics.ls_sortino = float(
            np.sqrt(periods_per_year) * mean_period / downside_std
        ) if downside_std > 0 else 0.0

        # Max drawdown
        cum_ret = (1 + ls_rets).cumprod()
        metrics.ls_max_drawdown = _max_drawdown(cum_ret)

        # Calmar
        metrics.ls_calmar = float(
            metrics.ls_annual_return / abs(metrics.ls_max_drawdown)
        ) if metrics.ls_max_drawdown != 0 else 0.0

    # Turnover
    metrics.turnover = compute_turnover(factor)

    # Signal decay (H1..H7)
    if raw_returns is not None:
        metrics.decay_sharpes = compute_signal_decay(factor, raw_returns)

    # Redundancy check against existing promoted factors
    if existing_panels:
        metrics.max_corr_with_existing = compute_redundancy(factor, existing_panels)

    # Decile sorts
    decile_rets = compute_decile_returns(factor, forward_returns)
    metrics.decile_returns = decile_rets
    metrics.decile_monotonic = _check_monotonic(decile_rets)

    # Meta
    metrics.n_days = len(daily_ic) if len(daily_ic) > 0 else 0
    valid_counts = factor.notna().sum(axis=1)
    metrics.n_stocks_avg = float(valid_counts.mean()) if len(valid_counts) > 0 else 0.0

    return metrics
