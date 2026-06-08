"""
QQQ Two-Layer Vol-Target Overlay — production hedge signal module.

Generates a daily *gross long deployment* for QQQ (0% .. 150%), with the
remainder held in a cash sleeve. This replaces the older short-hedge overlay:
the signal is now a long-exposure / cash-allocation model, never a short.

Two multiplicative layers:

  Layer 1 — SMA regime gate (direction):
      close > SMA100 and close > SMA200  -> gate = 1.0   (risk-on)
      close > exactly one of them        -> gate = 0.5   (half)
      close < both                       -> gate = 0.0   (full de-risk to cash)

      Gate confirmation (symmetric): the gate only changes state after the new
      SMA condition holds for `confirm_days` (default 3) consecutive closes --
      both on the way DOWN (closing below) and on the way back UP (closing
      above). Applied per SMA, so e.g. SMA100 can break (1.0 -> 0.5) before
      SMA200 breaks (0.5 -> 0.0), and each side re-risks only after 3 confirmed
      closes back above. Set confirm_days=1 for the instantaneous gate as
      written in the source document.

  Layer 2 — inverse-vol scalar (magnitude):
      w_vol = min( target_vol / rv20, 1.50 )
      rv20  = trailing 20-day realized vol of QQQ THROUGH date t (most recent
              close included), annualized (x sqrt(252)) from daily LOG returns
      target_vol = vt / 100   (e.g. vt=15 -> VT15 -> 0.15)

  Combined:
      exposure = gate * w_vol          (0 .. 1.5)
      cash     = 1 - exposure          (negative = financed leverage)

The "vt" level is an arbitrary positive number (15, 12, 10, 23, 12.5, ...).
Each config is fully invested (0% cash) exactly when QQQ's trailing realized
vol equals its target; above that the inverse-vol scalar moves capital to cash,
below it the scalar finances leverage up to the 1.50x cap.

Usage:
    from lib.qqq_hedge import hedge_parameters
    from lib.data import load_ohlcv_alphavantage

    ohlcv = load_ohlcv_alphavantage(["QQQ"], start="2019-01-01")
    close = ohlcv["close"]["QQQ"]
    returns = ohlcv["returns"]["QQQ"]

    params = hedge_parameters(close, returns, as_of="2026-05-28", vt=23)
    # -> {"as_of_date": "2026-05-28", "exposure": ..., "cash_pct": ..., ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VolTargetConfig:
    """Tunable parameters for the two-layer vol-target overlay."""

    # Layer 2: target (full-deployment) annualized vol. vt=15 -> 0.15.
    target_vol: float = 0.15

    # Realized-vol estimation
    vol_window: int = 20            # trailing window (trading days)
    annualization: float = 252.0    # trading days per year (vol scales by sqrt)

    # Leverage cap on the inverse-vol scalar
    leverage_cap: float = 1.50

    # Layer 1: SMA regime gate windows
    sma_fast: int = 100
    sma_slow: int = 200

    # Gate values for {both above, one above, neither above}
    gate_both: float = 1.0
    gate_one: float = 0.5
    gate_none: float = 0.0

    # Gate confirmation (symmetric): the gate only changes state after the new
    # SMA condition holds for this many consecutive closes -- both on the way
    # DOWN (price closing below) and on the way back UP (price closing above).
    # Applied per SMA. Set to 1 to disable (instantaneous gate, as written in
    # the source document).
    confirm_days: int = 3

    def validate(self) -> None:
        assert self.target_vol > 0, "target_vol must be positive"
        assert self.leverage_cap >= 1.0, "leverage_cap must be >= 1.0"
        assert self.vol_window >= 2, "vol_window must be >= 2"
        assert self.sma_fast > 0 and self.sma_slow > 0, "SMA windows must be positive"
        assert self.confirm_days >= 1, "confirm_days must be >= 1"

    @classmethod
    def from_vt(cls, vt: float, **overrides) -> "VolTargetConfig":
        """Build a config from a vt level (vt=23 -> target_vol=0.23)."""
        return cls(target_vol=float(vt) / 100.0, **overrides)


def config_for_vt(vt: float) -> VolTargetConfig:
    """Convenience: VolTargetConfig for a given vt level."""
    cfg = VolTargetConfig.from_vt(vt)
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def compute_exposure_indicators(
    close: pd.Series,
    returns: Optional[pd.Series] = None,
    config: Optional[VolTargetConfig] = None,
) -> pd.DataFrame:
    """Compute the indicators needed for the exposure signal.

    Returns DataFrame with columns: sma_fast, sma_slow, rv20.
      - sma_fast / sma_slow : simple moving averages (default 100 / 200)
      - rv20                : trailing 20d realized vol through date t -- the most
                              recent close IS included -- annualized x sqrt(252),
                              computed from daily LOG returns ln(P_t / P_{t-1})
                              (sample std, ddof=1).
                              No look-ahead: the as-of-t signal uses only data
                              known at the close of t. Backtest look-ahead is
                              handled separately by lagging the position one day
                              (see backtest()), so the vol is NOT lagged here.

    The `returns` argument is accepted for API compatibility but is not used for
    the vol estimate (which is derived from log returns of `close`).
    """
    cfg = config or VolTargetConfig()

    sma_fast = close.rolling(cfg.sma_fast, min_periods=cfg.sma_fast).mean()
    sma_slow = close.rolling(cfg.sma_slow, min_periods=cfg.sma_slow).mean()

    # Realized vol from LOG returns of close (time-additive; standard for vol).
    log_ret = np.log(close / close.shift(1))
    rv20 = log_ret.rolling(cfg.vol_window, min_periods=cfg.vol_window).std(ddof=1)
    rv20 = rv20 * np.sqrt(cfg.annualization)

    return pd.DataFrame(
        {"sma_fast": sma_fast, "sma_slow": sma_slow, "rv20": rv20},
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

class QQQVolTargetSignal:
    """Two-layer (regime gate x inverse-vol) long-exposure signal for QQQ."""

    @staticmethod
    def _gate(above_fast: bool, above_slow: bool, cfg: VolTargetConfig) -> float:
        n = int(bool(above_fast)) + int(bool(above_slow))
        return {2: cfg.gate_both, 1: cfg.gate_one, 0: cfg.gate_none}[n]

    @staticmethod
    def _debounced_above(close: pd.Series, sma: pd.Series, confirm_days: int) -> pd.Series:
        """Symmetric `confirm_days`-debounced 'price above SMA' state.

        The effective above/below state flips only after the raw condition
        (close > sma) holds for `confirm_days` consecutive sessions, in BOTH
        directions (down and up). Returns 1.0 (above) / 0.0 (below) / NaN
        (sma undefined), aligned to close.index. The initial state is seeded
        with the first valid raw reading (cold start).
        """
        valid = sma.notna()
        raw = (close > sma)[valid]
        out = pd.Series(np.nan, index=close.index)
        if raw.empty:
            return out
        if confirm_days <= 1:
            out.loc[raw.index] = raw.astype(float)
            return out

        raw_f = raw.astype(float)
        run_id = (raw != raw.shift()).cumsum()
        run_len = raw.groupby(run_id).cumcount() + 1
        candidate = raw_f.where(run_len >= confirm_days)
        candidate.iloc[0] = raw_f.iloc[0]   # seed initial state (cold start)
        deb = candidate.ffill()
        out.loc[deb.index] = deb
        return out

    @classmethod
    def compute(
        cls,
        close: float,
        sma_fast: float,
        sma_slow: float,
        rv20: float,
        config: Optional[VolTargetConfig] = None,
    ) -> dict:
        """Compute the exposure for a single point.

        Args:
            close: closing price
            sma_fast / sma_slow: moving averages (default 100 / 200)
            rv20: trailing annualized realized vol used to size (t-1)
        Returns dict: gate, w_vol, exposure, cash, leverage_capped.
        """
        cfg = config or VolTargetConfig()
        gate = cls._gate(close > sma_fast, close > sma_slow, cfg)
        raw = cfg.target_vol / rv20
        capped = raw >= cfg.leverage_cap
        w_vol = min(raw, cfg.leverage_cap)
        exposure = gate * w_vol
        return {
            "gate": gate,
            "w_vol": w_vol,
            "exposure": exposure,
            "cash": 1.0 - exposure,
            "leverage_capped": bool(capped),
        }

    @classmethod
    def from_series(
        cls,
        close: pd.Series,
        returns: Optional[pd.Series] = None,
        config: Optional[VolTargetConfig] = None,
    ) -> pd.DataFrame:
        """Vectorized exposure for an entire price series.

        Returns DataFrame (same index as close) with columns:
            gate, w_vol, exposure, cash, leverage_capped,
            rv20, close, sma_fast, sma_slow
        The gate applies the symmetric `confirm_days` debounce (per SMA): it
        only changes state after the new SMA condition holds that many
        consecutive closes, both down and up.
        Rows lacking enough history (NaN SMA or rv20) yield NaN exposure.
        """
        cfg = config or VolTargetConfig()
        cfg.validate()
        ind = compute_exposure_indicators(close, returns, cfg)

        sma_fast = ind["sma_fast"]
        sma_slow = ind["sma_slow"]
        rv20 = ind["rv20"]

        cd = max(1, int(cfg.confirm_days))
        above_fast = cls._debounced_above(close, sma_fast, cd)
        above_slow = cls._debounced_above(close, sma_slow, cd)
        valid_gate = sma_fast.notna() & sma_slow.notna()
        n_above = above_fast + above_slow
        gate = pd.Series(
            np.select(
                [n_above == 2, n_above == 1, n_above == 0],
                [cfg.gate_both, cfg.gate_one, cfg.gate_none],
                default=np.nan,
            ),
            index=close.index,
        ).where(valid_gate)

        raw = cfg.target_vol / rv20
        leverage_capped = raw >= cfg.leverage_cap
        w_vol = raw.clip(upper=cfg.leverage_cap)

        exposure = gate * w_vol
        cash = 1.0 - exposure

        return pd.DataFrame(
            {
                "gate": gate,
                "w_vol": w_vol,
                "exposure": exposure,
                "cash": cash,
                "leverage_capped": leverage_capped.where(rv20.notna()),
                "rv20": rv20,
                "close": close,
                "sma_fast": sma_fast,
                "sma_slow": sma_slow,
            },
            index=close.index,
        )

    @classmethod
    def backtest(
        cls,
        close: pd.Series,
        returns: Optional[pd.Series] = None,
        config: Optional[VolTargetConfig] = None,
        fed_funds_rate: float = 0.0,
    ) -> dict:
        """Backtest the overlay on long QQQ with a cash sleeve.

        Position decided at close t earns the t -> t+1 return; the cash sleeve
        (1 - exposure) earns the (annual) fed_funds_rate. Lookahead-free.

        Returns dict:
            exposure (Series), portfolio_returns, buyhold_returns, stats
        """
        cfg = config or VolTargetConfig()
        if returns is None:
            returns = close.pct_change()
        returns = returns.reindex(close.index)

        df = cls.from_series(close, returns, cfg)
        exposure = df["exposure"]
        w_vol = df["w_vol"]

        # exposure decided at t earns r_{t+1}  ->  exposure.shift(1) * r_t
        e_lag = exposure.shift(1)
        ff_daily = fed_funds_rate / cfg.annualization
        portfolio_ret = e_lag * returns + (1.0 - e_lag) * ff_daily
        buyhold_ret = returns.copy()

        def compute_stats(rets: pd.Series, label: str) -> dict:
            rets = rets.dropna()
            if len(rets) < 10:
                return {}
            ann_ret = rets.mean() * cfg.annualization
            ann_vol = rets.std() * np.sqrt(cfg.annualization)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
            cum = (1 + rets).cumprod()
            max_dd = (cum / cum.cummax() - 1).min()
            calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
            total_ret = cum.iloc[-1] - 1
            return {
                f"{label}_annual_return": ann_ret,
                f"{label}_annual_vol": ann_vol,
                f"{label}_sharpe": sharpe,
                f"{label}_max_dd": max_dd,
                f"{label}_calmar": calmar,
                f"{label}_total_return": total_ret,
            }

        p_stats = compute_stats(portfolio_ret, "portfolio")
        b_stats = compute_stats(buyhold_ret, "buyhold")

        exp_valid = exposure.dropna()
        wv_valid = w_vol.dropna()
        capped = df["leverage_capped"].dropna()

        stats = {
            **p_stats,
            **b_stats,
            "dd_reduction": b_stats.get("buyhold_max_dd", 0) - p_stats.get("portfolio_max_dd", 0),
            "return_cost": p_stats.get("portfolio_annual_return", 0) - b_stats.get("buyhold_annual_return", 0),
            "mean_exposure": float(exp_valid.mean()) if len(exp_valid) else 0.0,
            "mean_w_vol": float(wv_valid.mean()) if len(wv_valid) else 0.0,
            "pct_in_cash": float((exp_valid < 1.0).mean()) if len(exp_valid) else 0.0,
            "pct_levered": float((exp_valid > 1.0).mean()) if len(exp_valid) else 0.0,
            "pct_at_leverage_cap": float(capped.mean()) if len(capped) else 0.0,
            "target_vol": cfg.target_vol,
        }

        return {
            "exposure": exposure,
            "portfolio_returns": portfolio_ret,
            "buyhold_returns": buyhold_ret,
            "stats": stats,
        }


# ---------------------------------------------------------------------------
# Public helper: build the "hedging parameters" output dict
# ---------------------------------------------------------------------------

_REGIME_LABELS = {1.0: "risk-on", 0.5: "half", 0.0: "cash"}


def hedge_parameters(
    close: pd.Series,
    returns: Optional[pd.Series] = None,
    as_of: Optional[str] = None,
    vt: float = 15.0,
    config: Optional[VolTargetConfig] = None,
) -> dict:
    """Compute the QQQ hedging parameters as of a date for a given vt level.

    Args:
        close: QQQ close prices (DatetimeIndex)
        returns: daily returns (computed from close if None)
        as_of: ISO date string; rolls back to the most recent trading day on or
               before it. None -> latest available trading day.
        vt: vol-target level (15 -> VT15 -> target_vol 0.15). Any positive number.

    Returns a dict of hedging parameters, or {"error": ...} on failure.
    """
    if config is None:
        config = VolTargetConfig.from_vt(vt)
    config.validate()

    df = QQQVolTargetSignal.from_series(close, returns, config)

    if as_of is None:
        target = df.index[-1]
    else:
        ts = pd.Timestamp(as_of)
        prior = df.index[df.index <= ts]
        if len(prior) == 0:
            return {"error": f"No QQQ trading day on or before {as_of}"}
        target = prior[-1]

    row = df.loc[target]
    if pd.isna(row["exposure"]):
        return {
            "error": (
                f"Insufficient history to compute the signal as of "
                f"{target.date()} (need >= {config.sma_slow} sessions of data)."
            )
        }

    exposure = float(row["exposure"])
    cash = 1.0 - exposure
    gate = float(row["gate"])
    regime = _REGIME_LABELS.get(round(gate, 3), f"gate={gate:.2f}")
    cash_desc = (
        f"{cash * 100:.1f}% cash" if cash >= 0 else f"{abs(cash) * 100:.1f}% leverage"
    )

    return {
        "as_of_date": str(target.date()),
        "requested_date": str(pd.Timestamp(as_of).date()) if as_of else str(target.date()),
        "vt": float(vt),
        "target_vol": round(config.target_vol, 4),
        "exposure": round(exposure, 4),
        "exposure_pct": f"{exposure * 100:.1f}% invested",
        "cash": round(cash, 4),
        "cash_pct": cash_desc,
        "gate": gate,
        "regime": regime,
        "w_vol": round(float(row["w_vol"]), 4),
        "leverage_capped": bool(row["leverage_capped"]),
        "rv20": round(float(row["rv20"]), 4),
        "rv20_pct": f"{float(row['rv20']) * 100:.1f}%",
        "close": round(float(row["close"]), 2),
        "sma100": round(float(row["sma_fast"]), 2),
        "sma200": round(float(row["sma_slow"]), 2),
    }
