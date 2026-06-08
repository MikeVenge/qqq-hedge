"""
Unit tests for the QQQ two-layer vol-target overlay (lib.qqq_hedge).

These test the library directly with no MCP dependency, so they run with a
plain Python interpreter as well as under pytest.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from lib.qqq_hedge import (
    VolTargetConfig,
    QQQVolTargetSignal,
    hedge_parameters,
)


def _series(values, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_default_confirm_days_is_3():
    assert VolTargetConfig().confirm_days == 3


def test_debounce_symmetric_explicit():
    """Gate flips only after 3 consecutive closes in the new direction,
    in BOTH directions (down and up)."""
    sma = _series([10.0] * 15)
    close = _series([11, 11, 11,           # above (confirmed)
                     9, 9, 9,              # 3 closes below -> flip down on 3rd
                     11, 11,               # 2 closes above -> NOT enough, stays below
                     9, 9, 9, 9,           # back below
                     11, 11, 11])          # 3 closes above -> flip up on 3rd
    deb = QQQVolTargetSignal._debounced_above(close, sma, confirm_days=3)
    expected = [1, 1, 1,   # held above
                1, 1, 0,   # 2 days below held above; 3rd day flips down
                0, 0,      # 2 days above NOT enough -> stays below (symmetric!)
                0, 0, 0, 0,
                0, 0, 1]   # 2 days above held; 3rd day flips up
    assert deb.tolist() == [float(x) for x in expected]


def test_confirm_days_1_is_instantaneous():
    sma = _series([10.0] * 6)
    close = _series([11, 9, 11, 9, 11, 9])
    deb = QQQVolTargetSignal._debounced_above(close, sma, confirm_days=1)
    assert deb.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_confirmed_gate_never_below_instantaneous():
    """Symmetric confirmation only delays transitions; it must never push the
    gate strictly below the instantaneous gate AND never strictly above it by
    more than a held prior state. We assert the gate stays in {0,0.5,1.0}."""
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2018-01-01", periods=900)
    rets = pd.Series(rng.normal(0.0004, 0.013, 900), index=dates)
    close = (1 + rets).cumprod() * 300
    df3 = QQQVolTargetSignal.from_series(close, rets, VolTargetConfig.from_vt(15))
    g = df3["gate"].dropna()
    assert set(g.unique()).issubset({0.0, 0.5, 1.0})
    # exposure still reconciles to gate * w_vol
    row = df3.dropna().iloc[-1]
    assert abs(row["exposure"] - row["gate"] * row["w_vol"]) < 1e-9


def test_confirm_changes_are_fewer_than_instantaneous():
    """Debouncing should reduce the number of gate transitions vs instantaneous."""
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2018-01-01", periods=900)
    rets = pd.Series(rng.normal(0.0003, 0.016, 900), index=dates)
    close = (1 + rets).cumprod() * 300
    g3 = QQQVolTargetSignal.from_series(close, rets, VolTargetConfig.from_vt(15))["gate"].dropna()
    g1 = QQQVolTargetSignal.from_series(close, rets, VolTargetConfig.from_vt(15, confirm_days=1))["gate"].dropna()
    flips3 = (g3 != g3.shift()).sum()
    flips1 = (g1 != g1.shift()).sum()
    assert flips3 <= flips1


def test_rv20_is_log_returns_same_day():
    """rv20 at date t = trailing-20 vol of LOG returns THROUGH t (most recent
    close included), annualized -- not simple returns, and not lagged to t-1."""
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2019-01-01", periods=300)
    rets = pd.Series(rng.normal(0.0004, 0.013, 300), index=dates)
    close = (1 + rets).cumprod() * 300
    df = QQQVolTargetSignal.from_series(close, rets, VolTargetConfig.from_vt(15))

    log_ret = np.log(close / close.shift(1))
    expected_log_t = log_ret.iloc[-20:].std(ddof=1) * np.sqrt(252)   # log, through t
    lagged_log_t1 = log_ret.iloc[-21:-1].std(ddof=1) * np.sqrt(252)  # log, through t-1
    simple_t = close.pct_change().iloc[-20:].std(ddof=1) * np.sqrt(252)

    got = df["rv20"].iloc[-1]
    assert abs(got - expected_log_t) < 1e-12, "rv20 should be log-return vol through t"
    assert abs(got - lagged_log_t1) > 1e-9, "rv20 must not be lagged to t-1"
    assert abs(got - simple_t) > 1e-9, "rv20 must use log returns, not simple returns"


def test_hedge_parameters_reconciles():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2019-01-01", periods=600)
    rets = pd.Series(rng.normal(0.0004, 0.013, 600), index=dates)
    close = (1 + rets).cumprod() * 300
    p = hedge_parameters(close, rets, as_of=None, vt=23)
    assert p["vt"] == 23
    assert 0.0 <= p["exposure"] <= 1.5
    assert abs(p["exposure"] - p["gate"] * p["w_vol"]) < 1e-6
    assert p["gate"] in (0.0, 0.5, 1.0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
