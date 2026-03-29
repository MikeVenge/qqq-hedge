"""
QQQ Tail Risk Overlay — Usage Examples

Shows three ways to use the hedge signal:
  1. Full backtest on historical data
  2. Live daily signal (e.g. from a cron job or trading bot)
  3. Custom configuration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

import pandas as pd
import numpy as np


# ======================================================================
# EXAMPLE 1: Full backtest
# ======================================================================
def example_backtest():
    """Run a full backtest using Alpha Vantage data."""
    from lib.qqq_hedge import QQQHedgeSignal
    from lib.data import load_ohlcv_alphavantage

    print("=" * 60)
    print(" Example 1: Full Backtest")
    print("=" * 60)

    # Load data
    ohlcv = load_ohlcv_alphavantage(["QQQ"], start="2020-01-01", end="2026-03-29")
    close = ohlcv["close"]["QQQ"]
    returns = ohlcv["returns"]["QQQ"]

    # Run backtest — one line
    result = QQQHedgeSignal.backtest(close, returns)

    # Print results
    s = result["stats"]
    print(f"\n  Hedged:   Sharpe={s['portfolio_sharpe']:.2f}  MaxDD={s['portfolio_max_dd']:.1%}  Return={s['portfolio_total_return']:.1%}")
    print(f"  Buy&Hold: Sharpe={s['buyhold_sharpe']:.2f}  MaxDD={s['buyhold_max_dd']:.1%}  Return={s['buyhold_total_return']:.1%}")
    print(f"  DD saved: {s['dd_reduction']:+.1%}pts")

    # Access the daily positions and returns
    hedge_pos = result["hedge_position"]       # pd.Series: daily hedge size
    port_ret = result["portfolio_returns"]      # pd.Series: daily portfolio returns
    print(f"\n  Today's hedge position: {hedge_pos.iloc[-1]:.0%}")
    print(f"  Last 5 days:")
    for dt, pos in hedge_pos.tail(5).items():
        print(f"    {dt.strftime('%Y-%m-%d')}: {pos:+.0%} short")


# ======================================================================
# EXAMPLE 2: Live daily signal
# ======================================================================
def example_live_signal():
    """Use the signal in live/daily mode — e.g. from a cron job."""
    from lib.qqq_hedge import QQQHedgeSignal, compute_hedge_indicators
    from lib.data import load_ohlcv_alphavantage

    print("\n" + "=" * 60)
    print(" Example 2: Live Daily Signal")
    print("=" * 60)

    # In production, you'd fetch today's close from your broker/data feed.
    # Here we use the last available Alpha Vantage data.
    ohlcv = load_ohlcv_alphavantage(["QQQ"], start="2024-01-01", end="2026-03-29")
    close = ohlcv["close"]["QQQ"]

    # Compute indicators from price history
    indicators = compute_hedge_indicators(close)

    # Create signal object
    signal = QQQHedgeSignal()

    # Get today's values
    today = close.index[-1]
    today_close = close.iloc[-1]
    today_ind = indicators.iloc[-1]

    # Update signal
    position = signal.update(
        close=today_close,
        sma200=today_ind["sma200"],
        sma50=today_ind["sma50"],
        drawdown=today_ind["drawdown"],
        rally_from_trough=today_ind["rally_from_trough"],
        date=today,
    )

    print(f"\n  Date:               {today.strftime('%Y-%m-%d')}")
    print(f"  QQQ Close:          ${today_close:.2f}")
    print(f"  SMA50:              ${today_ind['sma50']:.2f}")
    print(f"  SMA200:             ${today_ind['sma200']:.2f}")
    print(f"  Drawdown:           {today_ind['drawdown']:.1%}")
    print(f"  Rally from trough:  {today_ind['rally_from_trough']:.1%}")
    print(f"  ─────────────────────────────────")
    print(f"  HEDGE POSITION:     {position:+.0%}")
    print(f"  Hedging beyond base: {'YES' if signal.is_hedging else 'NO'}")

    # For a $10M portfolio:
    notional = 10_000_000
    hedge_notional = abs(position) * notional
    print(f"\n  For a ${notional/1e6:.0f}M portfolio:")
    print(f"    Short ${hedge_notional/1e6:.2f}M of QQQ")
    print(f"    ≈ {int(hedge_notional / today_close)} shares")


# ======================================================================
# EXAMPLE 3: Custom configuration
# ======================================================================
def example_custom_config(close=None, returns=None):
    """Customize the hedge parameters for a different risk profile."""
    from lib.qqq_hedge import QQQHedgeSignal, HedgeConfig
    from lib.data import load_ohlcv_alphavantage

    print("\n" + "=" * 60)
    print(" Example 3: Custom Configuration")
    print("=" * 60)

    if close is None:
        ohlcv = load_ohlcv_alphavantage(["QQQ"], start="2020-01-01", end="2026-03-29")
        close = ohlcv["close"]["QQQ"]
        returns = ohlcv["returns"]["QQQ"]

    # Conservative: bigger base hedge, more aggressive scaling
    conservative = HedgeConfig(
        base_hedge=-0.10,          # 10% always short (vs 5% default)
        dd_size_1=-0.20,           # bigger DD tiers
        dd_size_2=-0.40,
        dd_size_3=-0.60,
        below_sma200_size=-0.25,   # more trend protection
    )

    # Aggressive: smaller base, less reactive
    aggressive = HedgeConfig(
        base_hedge=-0.03,          # 3% base (cheaper insurance)
        dd_tier_1=-0.05,           # only react at deeper DD
        dd_size_1=-0.10,
        dd_size_2=-0.20,
        dd_size_3=-0.40,
        below_sma200_size=-0.15,
        fast_exit_rally_thresh=0.08,  # exit faster on recovery
    )

    for name, cfg in [("Default", None), ("Conservative", conservative), ("Aggressive", aggressive)]:
        result = QQQHedgeSignal.backtest(close, returns, config=cfg)
        s = result["stats"]
        print(f"\n  {name}:")
        print(f"    Sharpe={s['portfolio_sharpe']:.2f}  MaxDD={s['portfolio_max_dd']:.1%}  "
              f"Return={s['portfolio_annual_return']:.1%}/yr  Cost={s['return_cost']:+.1%}/yr")


# ======================================================================
if __name__ == "__main__":
    # Load data once, reuse across examples
    from lib.data import load_ohlcv_alphavantage
    ohlcv = load_ohlcv_alphavantage(["QQQ"], start="2020-01-01", end="2026-03-29")
    _close = ohlcv["close"]["QQQ"]
    _returns = ohlcv["returns"]["QQQ"]

    example_backtest()
    example_live_signal()
    example_custom_config(close=_close, returns=_returns)
