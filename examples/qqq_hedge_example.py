"""
QQQ Two-Layer Vol-Target Overlay — Usage Examples

Shows three ways to use the exposure signal:
  1. Full backtest on historical data
  2. Live signal as of a date for a chosen vt level
  3. Compare vt levels (VT15 / VT12 / VT10)
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
def example_backtest(close=None, returns=None):
    """Run a full backtest using Alpha Vantage data."""
    from lib.qqq_hedge import QQQVolTargetSignal, VolTargetConfig
    from lib.data import load_ohlcv_alphavantage

    print("=" * 60)
    print(" Example 1: Full Backtest (VT15)")
    print("=" * 60)

    if close is None:
        ohlcv = load_ohlcv_alphavantage(["QQQ"], start="2020-01-01", end="2026-03-29")
        close = ohlcv["close"]["QQQ"]
        returns = ohlcv["returns"]["QQQ"]

    result = QQQVolTargetSignal.backtest(close, returns, VolTargetConfig.from_vt(15))
    s = result["stats"]
    print(f"\n  Overlay:  Sharpe={s['portfolio_sharpe']:.2f}  MaxDD={s['portfolio_max_dd']:.1%}  "
          f"Calmar={s['portfolio_calmar']:.2f}  Return={s['portfolio_total_return']:.1%}")
    print(f"  Buy&Hold: Sharpe={s['buyhold_sharpe']:.2f}  MaxDD={s['buyhold_max_dd']:.1%}  "
          f"Return={s['buyhold_total_return']:.1%}")
    print(f"  Mean exposure={s['mean_exposure']:.2f}  mean w_vol={s['mean_w_vol']:.2f}  "
          f"%at-cap={s['pct_at_leverage_cap']:.1%}")

    exposure = result["exposure"].dropna()
    print(f"\n  Last 5 exposures:")
    for dt, e in exposure.tail(5).items():
        print(f"    {dt.strftime('%Y-%m-%d')}: {e:.0%} invested")


# ======================================================================
# EXAMPLE 2: Live signal as of a date
# ======================================================================
def example_live_signal(close=None, returns=None):
    """Get hedging parameters as of a date for a chosen vt level."""
    from lib.qqq_hedge import hedge_parameters
    from lib.data import load_ohlcv_alphavantage

    print("\n" + "=" * 60)
    print(" Example 2: Live Signal (as-of date + vt)")
    print("=" * 60)

    if close is None:
        ohlcv = load_ohlcv_alphavantage(["QQQ"], start="2024-01-01", end="2026-03-29")
        close = ohlcv["close"]["QQQ"]
        returns = ohlcv["returns"]["QQQ"]

    params = hedge_parameters(close, returns, as_of=None, vt=15)
    print(f"\n  As-of date:    {params['as_of_date']}")
    print(f"  VT level:      {params['vt']:.0f}  (target_vol {params['target_vol']:.0%})")
    print(f"  QQQ Close:     ${params['close']:.2f}")
    print(f"  SMA100/200:    ${params['sma100']:.2f} / ${params['sma200']:.2f}")
    print(f"  rv20:          {params['rv20_pct']}")
    print(f"  Regime/gate:   {params['regime']} ({params['gate']:.1f})")
    print(f"  w_vol:         {params['w_vol']:.2f}")
    print(f"  ─────────────────────────────────")
    print(f"  EXPOSURE:      {params['exposure_pct']}")
    print(f"  CASH SLEEVE:   {params['cash_pct']}")


# ======================================================================
# EXAMPLE 3: Compare vt levels
# ======================================================================
def example_vt_sweep(close=None, returns=None):
    """Compare VT15 / VT12 / VT10 risk configurations."""
    from lib.qqq_hedge import QQQVolTargetSignal, VolTargetConfig
    from lib.data import load_ohlcv_alphavantage

    print("\n" + "=" * 60)
    print(" Example 3: VT Sweep")
    print("=" * 60)

    if close is None:
        ohlcv = load_ohlcv_alphavantage(["QQQ"], start="2020-01-01", end="2026-03-29")
        close = ohlcv["close"]["QQQ"]
        returns = ohlcv["returns"]["QQQ"]

    print(f"\n  {'Config':<8s} {'Sharpe':>7s} {'MaxDD':>8s} {'Calmar':>7s} "
          f"{'MeanW':>6s} {'%Cap':>6s}")
    print("  " + "─" * 50)
    for vt in (15, 12, 10):
        result = QQQVolTargetSignal.backtest(close, returns, VolTargetConfig.from_vt(vt))
        s = result["stats"]
        print(f"  VT{vt:<6d} {s['portfolio_sharpe']:>7.2f} {s['portfolio_max_dd']:>7.2%} "
              f"{s['portfolio_calmar']:>7.2f} {s['mean_w_vol']:>6.2f} "
              f"{s['pct_at_leverage_cap']:>5.1%}")


# ======================================================================
if __name__ == "__main__":
    from lib.data import load_ohlcv_alphavantage
    ohlcv = load_ohlcv_alphavantage(["QQQ"], start="2020-01-01", end="2026-03-29")
    _close = ohlcv["close"]["QQQ"]
    _returns = ohlcv["returns"]["QQQ"]

    example_backtest(close=_close, returns=_returns)
    example_live_signal(close=_close, returns=_returns)
    example_vt_sweep(close=_close, returns=_returns)
