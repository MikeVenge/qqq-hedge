"""
Unit tests for the book portfolio-vol option (lib/mango parsing + lib/portfolio_vol).
No network, no MCP -- runs with a plain interpreter or pytest.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from lib.mango import _constituents_from_positions
from lib.portfolio_vol import (
    portfolio_return_series,
    portfolio_value_series,
    portfolio_realized_vol_asof,
)


def _payload():
    """Synthetic list_positions payload: 2 longs (one duplicated), 1 short,
    1 option (dropped), 1 closed (qty 0)."""
    return {
        "positions": [
            {"symbol": "AAA", "quantity": 5, "market_value": "500",
             "weight_of_gross": "25", "asset_class": "equity", "long_short": "L",
             "book_name": "Test Book"},
            {"symbol": "AAA", "quantity": 5, "market_value": "500",
             "weight_of_gross": "25", "asset_class": "equity", "long_short": "L"},
            {"symbol": "BBB", "quantity": 3, "market_value": "600",
             "weight_of_gross": "30", "asset_class": "equity", "long_short": "L"},
            {"symbol": "CCC", "quantity": -2, "market_value": "-400",
             "weight_of_gross": "20", "asset_class": "equity", "long_short": "S"},
            {"symbol": "OPT", "quantity": 1, "market_value": "100",
             "weight_of_gross": "10", "asset_class": "option", "long_short": "L"},
            {"symbol": "OLD", "quantity": 0, "market_value": "0",
             "weight_of_gross": "0", "asset_class": "equity", "long_short": "L"},
        ]
    }


def test_constituents_parsing_gross():
    r = _constituents_from_positions(_payload(), book_id=99, weighting="gross")
    assert "error" not in r, r
    assert r["book_name"] == "Test Book"
    assert r["symbols"] == ["AAA", "BBB", "CCC"]          # option + closed dropped
    assert r["n_dropped_options"] == 1
    assert r["weighting"] == "gross"
    w = r["weights"]
    # AAA duplicated (25+25=50%), BBB 30%, CCC -20%; gross 100% -> normalized
    assert abs(w["AAA"] - 0.5) < 1e-9
    assert abs(w["BBB"] - 0.3) < 1e-9
    assert abs(w["CCC"] - (-0.2)) < 1e-9               # short -> negative
    assert abs(sum(abs(v) for v in w.values()) - 1.0) < 1e-9
    assert abs(r["net_exposure"] - 0.6) < 1e-9


def test_constituents_equal_weight_default():
    """Default weighting is equal: each risk name +/- 1/N (signed by long/short)."""
    r = _constituents_from_positions(_payload(), book_id=99)
    assert r["weighting"] == "equal"
    w = r["weights"]
    assert r["symbols"] == ["AAA", "BBB", "CCC"]
    assert abs(w["AAA"] - 1/3) < 1e-9
    assert abs(w["BBB"] - 1/3) < 1e-9
    assert abs(w["CCC"] - (-1/3)) < 1e-9               # short stays negative
    assert abs(sum(abs(v) for v in w.values()) - 1.0) < 1e-9


def test_constituents_empty_book():
    r = _constituents_from_positions({"positions": [
        {"symbol": "X", "quantity": 0, "asset_class": "equity"}]}, book_id=1)
    assert "error" in r


def test_constituents_drops_cash():
    """Cash/T-bill ETFs (e.g. BIL) are excluded from the risk basket and the
    remaining risk weights are renormalized."""
    payload = {"positions": [
        {"symbol": "AAA", "quantity": 1, "market_value": "500",
         "weight_of_gross": "50", "asset_class": "equity", "long_short": "L"},
        {"symbol": "BBB", "quantity": 1, "market_value": "300",
         "weight_of_gross": "30", "asset_class": "equity", "long_short": "L"},
        {"symbol": "BIL", "quantity": 1, "market_value": "200",
         "weight_of_gross": "20", "asset_class": "equity", "long_short": "L"},
    ]}
    # gross weighting: renormalized over AAA(0.5)+BBB(0.3) -> 0.625 / 0.375
    r = _constituents_from_positions(payload, book_id=132, weighting="gross")
    assert r["symbols"] == ["AAA", "BBB"], r        # BIL dropped
    assert r["dropped_cash"] == ["BIL"]
    assert abs(r["cash_weight"] - 0.20) < 1e-9      # 20% of gross was cash
    assert abs(r["weights"]["AAA"] - 0.625) < 1e-9
    assert abs(r["weights"]["BBB"] - 0.375) < 1e-9
    # equal weighting (default): the two risk names -> 0.5 / 0.5
    re = _constituents_from_positions(payload, book_id=132)
    assert abs(re["weights"]["AAA"] - 0.5) < 1e-9
    assert abs(re["weights"]["BBB"] - 0.5) < 1e-9
    assert re["dropped_cash"] == ["BIL"]            # cash still excluded
    # include_cash=True keeps it
    r2 = _constituents_from_positions(payload, book_id=132, include_cash=True)
    assert "BIL" in r2["symbols"]


def test_constituents_market_value_fallback():
    # No weight_of_gross -> derive from |market_value|
    payload = {"positions": [
        {"symbol": "AAA", "quantity": 1, "market_value": "300", "asset_class": "equity"},
        {"symbol": "BBB", "quantity": 1, "market_value": "100", "asset_class": "equity"},
    ]}
    r = _constituents_from_positions(payload, book_id=2, weighting="gross")
    assert abs(r["weights"]["AAA"] - 0.75) < 1e-9
    assert abs(r["weights"]["BBB"] - 0.25) < 1e-9


def _returns(cols, n=80, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({c: rng.normal(0.0004, 0.012 + 0.004 * i, n)
                         for i, c in enumerate(cols)}, index=idx)


def test_portfolio_value_series():
    """NAV index = base * cumprod(1 + r_p); log-diff recovers the daily returns."""
    df = _returns(["AAA", "BBB"])
    w = {"AAA": 0.6, "BBB": 0.4}
    r_p = portfolio_return_series(df, w)
    nav = portfolio_value_series(df, w, base=100.0)
    assert len(nav) == len(r_p)
    assert abs(nav.iloc[0] - 100.0 * (1 + r_p.iloc[0])) < 1e-9
    assert (nav - 100.0 * (1 + r_p).cumprod()).abs().max() < 1e-9
    # log-diff of the value series == log1p of the portfolio return
    lr = np.log(nav).diff().dropna()
    assert (lr - np.log1p(r_p).iloc[1:]).abs().max() < 1e-12


def test_value_series_vol_equals_log1p_formula():
    """The NAV-based vol must equal the prior std(ln(1+r_p)) formula (no numeric
    change from materializing the value series)."""
    df = _returns(["AAA", "BBB", "CCC"], n=120, seed=4)
    w = {"AAA": 0.5, "BBB": 0.3, "CCC": -0.2}
    out = portfolio_realized_vol_asof(df, w, window=30)
    r_p = portfolio_return_series(df, w)
    prior = float(np.log1p(r_p).iloc[-30:].std(ddof=1) * np.sqrt(252))
    assert abs(out["portfolio_vol"] - prior) < 1e-12


def test_single_asset_matches_its_own_logvol():
    df = _returns(["AAA", "BBB"])
    out = portfolio_realized_vol_asof(df, {"AAA": 1.0}, window=30)
    expected = np.log1p(df["AAA"]).iloc[-30:].std(ddof=1) * np.sqrt(252)
    assert abs(out["portfolio_vol"] - expected) < 1e-9
    assert out["n_priced"] == 1


def test_weighted_portfolio_return_identity():
    df = _returns(["AAA", "BBB"])
    w = {"AAA": 0.6, "BBB": 0.4}
    r_p = portfolio_return_series(df, w)
    manual = 0.6 * df["AAA"] + 0.4 * df["BBB"]
    assert (r_p - manual).abs().max() < 1e-12


def test_coverage_guard():
    df = _returns(["AAA"])  # ZZZ missing -> only 50% priced
    out = portfolio_realized_vol_asof(df, {"AAA": 0.5, "ZZZ": 0.5}, window=30)
    assert "error" in out and "ZZZ" in out.get("missing", [])


def test_insufficient_history():
    df = _returns(["AAA"], n=10)
    out = portfolio_realized_vol_asof(df, {"AAA": 1.0}, window=30)
    assert "error" in out


def test_asof_rolls_back():
    df = _returns(["AAA", "BBB"], n=80)
    out = portfolio_realized_vol_asof(df, {"AAA": 0.5, "BBB": 0.5},
                                      as_of="2024-03-01", window=30)
    assert out["as_of_date"] <= "2024-03-01"
    assert out["portfolio_vol"] > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1; print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
