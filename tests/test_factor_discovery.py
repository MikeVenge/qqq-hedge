"""
Tests for the agentic factor discovery pipeline.

Covers:
  - Factor grammar: parsing, serialization, evaluation
  - Factor evaluator: metrics computation on synthetic data
  - Promotion gate: decision logic
  - Factor aggregator: linear combination
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Factor Grammar tests
# ---------------------------------------------------------------------------

class TestFactorGrammar:

    def test_parse_primitive(self):
        from lib.factor_grammar import parse_expression
        expr = parse_expression("return")
        assert expr.op is None
        assert expr.name == "return"
        assert expr.to_string() == "return"

    def test_parse_simple_operator(self):
        from lib.factor_grammar import parse_expression
        expr = parse_expression("rank(return)")
        assert expr.op == "rank"
        assert len(expr.children) == 1
        assert expr.children[0].name == "return"
        assert expr.to_string() == "rank(return)"

    def test_parse_operator_with_param(self):
        from lib.factor_grammar import parse_expression
        expr = parse_expression("rolling_mean(price, 20)")
        assert expr.op == "rolling_mean"
        assert expr.params == {"n": 20}
        assert expr.children[0].name == "price"

    def test_parse_nested(self):
        from lib.factor_grammar import parse_expression
        expr = parse_expression("rank(rolling_mean(return, 5))")
        assert expr.op == "rank"
        assert expr.depth() == 2
        inner = expr.children[0]
        assert inner.op == "rolling_mean"
        assert inner.params == {"n": 5}

    def test_parse_two_arg_operator(self):
        from lib.factor_grammar import parse_expression
        expr = parse_expression("ratio(volume, rolling_mean(volume, 20))")
        assert expr.op == "ratio"
        assert len(expr.children) == 2
        assert expr.children[0].name == "volume"
        assert expr.children[1].op == "rolling_mean"

    def test_depth_limit(self):
        from lib.factor_grammar import parse_expression, MAX_EXPRESSION_DEPTH
        # Depth MAX+1 should fail
        # Build a nested expression deeper than the limit
        formula = "return"
        for i in range(MAX_EXPRESSION_DEPTH + 1):
            formula = f"rank({formula})"
        with pytest.raises(ValueError, match="depth"):
            parse_expression(formula)

    def test_unknown_operator(self):
        from lib.factor_grammar import parse_expression
        with pytest.raises(ValueError, match="Unknown operator"):
            parse_expression("foobar(return)")

    def test_roundtrip_serialization(self):
        from lib.factor_grammar import parse_expression
        formula = "ratio(rolling_mean(volume, 5), rolling_mean(volume, 20))"
        expr = parse_expression(formula)
        assert expr.to_string() == formula

    def test_evaluate_primitives(self):
        from lib.factor_grammar import parse_expression
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        symbols = ["A", "B", "C"]
        rng = np.random.default_rng(42)
        data = {
            "return": pd.DataFrame(rng.normal(0, 0.02, (50, 3)), index=dates, columns=symbols),
            "price": pd.DataFrame(rng.uniform(10, 100, (50, 3)), index=dates, columns=symbols),
            "volume": pd.DataFrame(rng.lognormal(15, 1, (50, 3)), index=dates, columns=symbols),
            "market_return": pd.DataFrame(rng.normal(0, 0.01, (50, 3)), index=dates, columns=symbols),
        }

        expr = parse_expression("return")
        result = expr.evaluate(data)
        assert result.shape == (50, 3)
        pd.testing.assert_frame_equal(result, data["return"])

    def test_evaluate_rolling_mean(self):
        from lib.factor_grammar import parse_expression
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        symbols = ["A", "B"]
        data = {
            "return": pd.DataFrame(
                np.ones((50, 2)) * 0.01, index=dates, columns=symbols
            ),
        }
        expr = parse_expression("rolling_mean(return, 5)")
        result = expr.evaluate(data)
        # All values are 0.01, so rolling mean should also be 0.01
        assert abs(result.iloc[-1, 0] - 0.01) < 1e-10

    def test_evaluate_rank(self):
        from lib.factor_grammar import parse_expression
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        symbols = ["A", "B", "C"]
        data = {
            "return": pd.DataFrame(
                [[1, 2, 3]] * 10, index=dates, columns=symbols, dtype=float
            ),
        }
        expr = parse_expression("rank(return)")
        result = expr.evaluate(data)
        # Ranks should be percentile: A=0.33, B=0.67, C=1.0
        assert result.iloc[0, 0] < result.iloc[0, 1] < result.iloc[0, 2]

    def test_grammar_description(self):
        from lib.factor_grammar import grammar_description
        desc = grammar_description()
        assert "return" in desc
        assert "rolling_mean" in desc
        assert "rank" in desc


# ---------------------------------------------------------------------------
# Factor Evaluator tests
# ---------------------------------------------------------------------------

class TestFactorEvaluator:

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic factor and return data with known signal."""
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        symbols = [f"S{i}" for i in range(50)]
        rng = np.random.default_rng(42)

        # Create a factor with known predictive power
        factor = pd.DataFrame(
            rng.normal(0, 1, (500, 50)), index=dates, columns=symbols
        )
        # Forward returns = factor * signal_strength + noise
        noise = rng.normal(0, 0.02, (500, 50))
        fwd_returns = pd.DataFrame(
            factor.values * 0.001 + noise, index=dates, columns=symbols
        )
        return factor, fwd_returns

    def test_winsorize(self):
        from lib.factor_evaluator import winsorize_cross_section
        df = pd.DataFrame({"A": [1, 100], "B": [2, 200], "C": [3, 300]})
        result = winsorize_cross_section(df, lower=0.01, upper=0.99)
        # Extreme values should be clipped
        assert result.max().max() <= 300
        assert result.min().min() >= 1

    def test_zscore(self):
        from lib.factor_evaluator import zscore_cross_section
        df = pd.DataFrame({"A": [1.0, 4.0], "B": [2.0, 5.0], "C": [3.0, 6.0]})
        result = zscore_cross_section(df)
        # Each row should have mean ~0
        assert abs(result.iloc[0].mean()) < 1e-10
        assert abs(result.iloc[1].mean()) < 1e-10

    def test_rank_ic_positive_signal(self, synthetic_data):
        from lib.factor_evaluator import compute_daily_rank_ic
        factor, fwd_returns = synthetic_data
        ic = compute_daily_rank_ic(factor, fwd_returns)
        # With positive signal, average IC should be positive
        assert ic.mean() > 0

    def test_ls_portfolio_returns(self, synthetic_data):
        from lib.factor_evaluator import compute_ls_portfolio_returns
        factor, fwd_returns = synthetic_data
        ls_rets = compute_ls_portfolio_returns(factor, fwd_returns)
        assert len(ls_rets) > 0
        # With positive signal, LS returns should be positive on average
        assert ls_rets.mean() > 0

    def test_evaluate_factor_full(self, synthetic_data):
        from lib.factor_evaluator import evaluate_factor
        factor, fwd_returns = synthetic_data
        metrics = evaluate_factor(factor, fwd_returns, preprocess=True)
        assert metrics.ic_mean > 0
        assert metrics.n_days > 0
        assert metrics.decile_returns is not None
        assert len(metrics.decile_returns) == 10

    def test_evaluate_factor_summary(self, synthetic_data):
        from lib.factor_evaluator import evaluate_factor
        factor, fwd_returns = synthetic_data
        metrics = evaluate_factor(factor, fwd_returns, preprocess=True)
        summary = metrics.summary_str()
        assert "IC:" in summary
        assert "Sharpe:" in summary


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------

class TestGate:

    def test_promote(self):
        from lib.factor_evaluator import FactorMetrics
        from lib.factor_gate import gate_decision, GateThresholds
        m = FactorMetrics(ic_t_stat=3.0, ls_sharpe=1.5)
        assert gate_decision(m) == "promote"

    def test_retire(self):
        from lib.factor_evaluator import FactorMetrics
        from lib.factor_gate import gate_decision, GateThresholds
        m = FactorMetrics(ic_t_stat=0.5, ls_sharpe=0.3)
        assert gate_decision(m) == "retire"

    def test_hold(self):
        from lib.factor_evaluator import FactorMetrics
        from lib.factor_gate import gate_decision, GateThresholds
        # t_IC above retire threshold but below promote, or Sharpe too low
        m = FactorMetrics(ic_t_stat=1.5, ls_sharpe=0.5)
        assert gate_decision(m) == "hold"

    def test_custom_thresholds(self):
        from lib.factor_evaluator import FactorMetrics
        from lib.factor_gate import gate_decision, GateThresholds
        thresholds = GateThresholds(t_ic_promote=3.0, sharpe_promote=2.0, t_ic_retire=0.5)
        m = FactorMetrics(ic_t_stat=2.5, ls_sharpe=1.8)
        assert gate_decision(m, thresholds) == "hold"  # Below custom promote thresholds

    def test_gate_reason(self):
        from lib.factor_evaluator import FactorMetrics
        from lib.factor_gate import gate_reason
        m = FactorMetrics(ic_t_stat=3.0, ls_sharpe=1.5)
        reason = gate_reason(m, "promote")
        assert "Promoted" in reason


# ---------------------------------------------------------------------------
# Aggregator tests
# ---------------------------------------------------------------------------

class TestAggregator:

    def test_linear_aggregation(self):
        from lib.factor_aggregator import aggregate_linear
        dates = pd.date_range("2020-01-01", periods=20, freq="B")
        symbols = ["A", "B", "C"]
        rng = np.random.default_rng(42)

        panels = {
            "f1": pd.DataFrame(rng.normal(0, 1, (20, 3)), index=dates, columns=symbols),
            "f2": pd.DataFrame(rng.normal(0, 1, (20, 3)), index=dates, columns=symbols),
        }

        composite = aggregate_linear(panels)
        assert composite.shape == (20, 3)
        assert not composite.isna().all().all()

    def test_linear_aggregation_single_factor(self):
        from lib.factor_aggregator import aggregate_linear
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        symbols = ["X", "Y"]
        panel = pd.DataFrame([[1.0, 2.0]] * 10, index=dates, columns=symbols)
        composite = aggregate_linear({"only": panel})
        assert composite.shape == (10, 2)

    def test_empty_raises(self):
        from lib.factor_aggregator import aggregate_linear
        with pytest.raises(ValueError):
            aggregate_linear({})


# ---------------------------------------------------------------------------
# Agent memory tests (no LLM calls)
# ---------------------------------------------------------------------------

class TestAgentMemory:

    def test_memory_tracking(self):
        from lib.factor_agent import AgentMemory, RoundRecord
        memory = AgentMemory()

        r1 = RoundRecord(round=1, hypothesis="test", formula="rank(return)",
                         decision="promote", reason="passed")
        r2 = RoundRecord(round=2, hypothesis="test2", formula="diff(price, 1)",
                         decision="retire", reason="failed")
        memory.add_round(r1)
        memory.add_round(r2)

        assert len(memory.rounds) == 2
        assert len(memory.promoted_formulas) == 1
        assert len(memory.retired_formulas) == 1
        assert "rank(return)" in memory.promoted_formulas

    def test_memory_prompt_context(self):
        from lib.factor_agent import AgentMemory, RoundRecord
        memory = AgentMemory()
        # Empty memory
        ctx = memory.to_prompt_context()
        assert "first iteration" in ctx.lower()

        # With a round
        r = RoundRecord(round=1, hypothesis="momentum signal",
                        formula="rolling_mean(return, 20)",
                        metrics={"ic_t_stat": 2.5, "ls_sharpe": 1.3, "ls_annual_return": 0.15},
                        decision="promote", reason="passed")
        memory.add_round(r)
        ctx = memory.to_prompt_context()
        assert "Promoted" in ctx
        assert "rolling_mean" in ctx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
