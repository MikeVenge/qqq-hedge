"""
Tests for the MCP server tool functions.

Tests call tool functions directly (no HTTP transport) with mocked data loading.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_panel(n_days: int = 200, n_assets: int = 20) -> dict[str, pd.DataFrame]:
    """Build a synthetic data panel matching the structure of build_data_panel."""
    dates = pd.bdate_range("2020-01-01", periods=n_days, freq="B")
    symbols = [f"S{i}" for i in range(n_assets)]
    rng = np.random.default_rng(42)

    returns = pd.DataFrame(rng.normal(0, 0.02, (n_days, n_assets)), index=dates, columns=symbols)
    price = (1 + returns).cumprod() * 100
    volume = pd.DataFrame(rng.lognormal(15, 1, (n_days, n_assets)), index=dates, columns=symbols)
    mkt_ret = returns.mean(axis=1)
    market_return = pd.DataFrame(
        np.tile(mkt_ret.values[:, None], (1, n_assets)),
        index=dates, columns=symbols,
    )

    panel = {
        "return": returns,
        "price": price,
        "volume": volume,
        "market_return": market_return,
    }

    # Derived primitives needed by common formulas
    vol_ma20 = volume.rolling(20, min_periods=5).mean()
    panel["volume_ratio"] = volume / vol_ma20.replace(0, np.nan)
    panel["realized_vol"] = returns.rolling(20, min_periods=5).std()
    panel["price_to_ma"] = price / price.rolling(20, min_periods=5).mean().replace(0, np.nan)
    panel["turnover"] = volume / volume.rolling(60, min_periods=20).mean().replace(0, np.nan)
    panel["market_vol"] = market_return.rolling(20, min_periods=5).std()

    # SMAs
    panel["sma_50"] = price.rolling(50, min_periods=30).mean()
    panel["sma_100"] = price.rolling(100, min_periods=60).mean()
    panel["sma_200"] = price.rolling(200, min_periods=120).mean()
    panel["price_to_sma50"] = price / panel["sma_50"].replace(0, np.nan)
    panel["price_to_sma100"] = price / panel["sma_100"].replace(0, np.nan)
    panel["price_to_sma200"] = price / panel["sma_200"].replace(0, np.nan)
    panel["sma50_to_sma200"] = panel["sma_50"] / panel["sma_200"].replace(0, np.nan)

    # RSI
    delta = returns.copy()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14, min_periods=10).mean()
    avg_loss = loss.rolling(14, min_periods=10).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    panel["rsi_14"] = 100 - (100 / (1 + rs))

    # Momentum
    panel["return_10d"] = returns.rolling(10, min_periods=5).sum()
    panel["return_20d"] = returns.rolling(20, min_periods=10).sum()
    panel["return_60d"] = returns.rolling(60, min_periods=30).sum()

    # Beta and correlation
    mkt_series = market_return.iloc[:, 0]
    mkt_var = mkt_series.rolling(60, min_periods=30).var()
    betas = {}
    corrs = {}
    for col in returns.columns:
        cov = returns[col].rolling(60, min_periods=30).cov(mkt_series)
        betas[col] = cov / mkt_var.replace(0, np.nan)
        corrs[col] = returns[col].rolling(60, min_periods=30).corr(mkt_series)
    panel["beta_60d"] = pd.DataFrame(betas, index=dates)
    panel["corr_market_60d"] = pd.DataFrame(corrs, index=dates)

    # Extremes
    panel["high_low_range"] = panel["realized_vol"] * np.sqrt(1 / 252)
    rolling_high = price.rolling(252, min_periods=60).max()
    panel["distance_from_52w_high"] = (price - rolling_high) / rolling_high.replace(0, np.nan)

    return panel


@pytest.fixture
def mock_panel():
    """Provide a synthetic panel and patch _get_or_build_panel."""
    panel = _make_synthetic_panel()
    with patch("mcp_server._get_or_build_panel", return_value=panel):
        yield panel


# ---------------------------------------------------------------------------
# Group A: Grammar tools
# ---------------------------------------------------------------------------

class TestParseFactorExpression:

    def test_valid_primitive(self):
        from mcp_server import parse_factor_expression
        result = parse_factor_expression("return")
        assert result["valid"] is True
        assert result["formula"] == "return"
        assert result["depth"] == 0
        assert result["error"] is None

    def test_valid_nested(self):
        from mcp_server import parse_factor_expression
        result = parse_factor_expression("rank(rolling_mean(return, 20))")
        assert result["valid"] is True
        assert result["depth"] == 2

    def test_invalid_operator(self):
        from mcp_server import parse_factor_expression
        result = parse_factor_expression("foobar(return)")
        assert result["valid"] is False
        assert result["error"] is not None
        assert "Unknown operator" in result["error"]

    def test_depth_exceeded(self):
        from mcp_server import parse_factor_expression
        deep = "rank(rank(rank(rank(rank(return)))))"
        result = parse_factor_expression(deep)
        assert result["valid"] is False
        assert "depth" in result["error"].lower()

    def test_two_arg_operator(self):
        from mcp_server import parse_factor_expression
        result = parse_factor_expression("ratio(volume, price)")
        assert result["valid"] is True
        assert result["depth"] == 1


class TestListGrammar:

    def test_returns_all_keys(self):
        from mcp_server import list_grammar
        result = list_grammar()
        assert "grammar_text" in result
        assert "primitives" in result
        assert "operators" in result
        assert "max_depth" in result

    def test_primitives_present(self):
        from mcp_server import list_grammar
        result = list_grammar()
        assert "return" in result["primitives"]
        assert "volume" in result["primitives"]
        assert "price" in result["primitives"]

    def test_operators_present(self):
        from mcp_server import list_grammar
        result = list_grammar()
        assert "rank" in result["operators"]
        assert "rolling_mean" in result["operators"]
        assert "ratio" in result["operators"]

    def test_max_depth_positive(self):
        from mcp_server import list_grammar
        result = list_grammar()
        assert result["max_depth"] > 0


# ---------------------------------------------------------------------------
# Group B: Evaluation tools
# ---------------------------------------------------------------------------

class TestEvaluateFactor:

    def test_valid_formula(self, mock_panel):
        from mcp_server import evaluate_factor
        result = evaluate_factor("rank(return)")
        assert "error" not in result
        assert "metrics" in result
        assert "summary" in result
        assert result["formula"] == "rank(return)"

    def test_invalid_formula(self, mock_panel):
        from mcp_server import evaluate_factor
        result = evaluate_factor("unknown_op(return)")
        assert "error" in result

    def test_metrics_structure(self, mock_panel):
        from mcp_server import evaluate_factor
        result = evaluate_factor("rank(return)")
        metrics = result["metrics"]
        assert "ic_mean" in metrics
        assert "ic_t_stat" in metrics
        assert "ls_sharpe" in metrics
        assert "turnover" in metrics
        assert "n_days" in metrics


class TestGateFactor:

    def test_returns_decision(self, mock_panel):
        from mcp_server import gate_factor
        result = gate_factor("rank(return)")
        assert "error" not in result
        assert "decision" in result
        assert result["decision"] in ("promote", "retire", "hold")
        assert "reason" in result
        assert "metrics" in result

    def test_invalid_formula(self, mock_panel):
        from mcp_server import gate_factor
        result = gate_factor("bad_op(x)")
        assert "error" in result


# ---------------------------------------------------------------------------
# Group C: QQQ Hedge tools
# ---------------------------------------------------------------------------

class TestQQQHedgeSignal:

    def _mock_ohlcv(self):
        """Build mock QQQ OHLCV data."""
        dates = pd.bdate_range("2019-01-01", periods=600, freq="B")
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0005, 0.015, 600), index=dates, name="QQQ")
        close = (1 + returns).cumprod() * 300
        return {
            "close": pd.DataFrame({"QQQ": close}),
            "volume": pd.DataFrame({"QQQ": rng.lognormal(20, 1, 600)}, index=dates),
            "returns": pd.DataFrame({"QQQ": returns}),
        }

    def test_latest(self):
        from mcp_server import qqq_hedge_signal
        mock = self._mock_ohlcv()
        with patch("lib.data.load_ohlcv_yfinance", return_value=mock):
            result = qqq_hedge_signal()
        assert "error" not in result
        assert "position" in result
        assert "date" in result
        assert result["position"] < 0

    def test_specific_date(self):
        from mcp_server import qqq_hedge_signal
        mock = self._mock_ohlcv()
        target_date = str(mock["close"].index[400].date())
        with patch("lib.data.load_ohlcv_yfinance", return_value=mock):
            result = qqq_hedge_signal(date=target_date)
        assert "error" not in result
        assert result["date"] == target_date

    def test_date_not_trading_day(self):
        from mcp_server import qqq_hedge_signal
        mock = self._mock_ohlcv()
        # Use a Saturday — should snap to previous trading day
        with patch("lib.data.load_ohlcv_yfinance", return_value=mock):
            result = qqq_hedge_signal(date="2020-06-06")
        assert "error" not in result
        assert result["date"] <= "2020-06-06"


class TestQQQHedgeBacktest:

    def test_with_mocked_data(self):
        """Test backtest with mocked yfinance data."""
        from mcp_server import qqq_hedge_backtest

        dates = pd.bdate_range("2020-01-01", periods=500, freq="B")
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0005, 0.015, 500), index=dates, name="QQQ")
        close = (1 + returns).cumprod() * 300

        mock_ohlcv = {
            "close": pd.DataFrame({"QQQ": close}),
            "volume": pd.DataFrame({"QQQ": rng.lognormal(20, 1, 500)}, index=dates),
            "returns": pd.DataFrame({"QQQ": returns}),
        }

        with patch("lib.data.load_ohlcv_yfinance", return_value=mock_ohlcv):
            result = qqq_hedge_backtest()

        assert "error" not in result
        assert "stats" in result
        assert "recent_positions" in result
        assert len(result["recent_positions"]) <= 10


# ---------------------------------------------------------------------------
# Group D: Aggregation tools
# ---------------------------------------------------------------------------

class TestAggregateFactors:

    def test_linear_two_factors(self, mock_panel):
        from mcp_server import aggregate_factors
        result = aggregate_factors(
            formulas=["rank(return)", "rank(volume_ratio)"],
            method="linear",
        )
        assert "error" not in result
        assert result["n_factors"] == 2
        assert result["method"] == "linear"
        assert "composite_summary" in result

    def test_empty_formulas(self, mock_panel):
        from mcp_server import aggregate_factors
        result = aggregate_factors(formulas=[])
        assert "error" in result

    def test_invalid_formula_in_list(self, mock_panel):
        from mcp_server import aggregate_factors
        result = aggregate_factors(formulas=["rank(return)", "bad_op(x)"])
        assert "error" in result


# ---------------------------------------------------------------------------
# Import sanity
# ---------------------------------------------------------------------------

class TestServerImport:

    def test_mcp_instance_exists(self):
        from mcp_server import mcp
        assert mcp is not None
        assert mcp.name == "System Factors"
