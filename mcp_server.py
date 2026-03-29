"""
MCP Server for System Factors — quantitative finance factor discovery platform.

Exposes factor grammar, evaluation, gate decisions, QQQ hedge signals,
factor aggregation, and the full discovery pipeline as MCP tools via
Streamable HTTP transport.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from collections import OrderedDict
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# MCP Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "System Factors",
    instructions=(
        "Quantitative finance factor discovery and QQQ hedge platform. "
        "Use `list_grammar` to see available primitives and operators, "
        "then `parse_factor_expression` to validate formulas before evaluation."
    ),
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 8000)),
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data panel cache (avoid re-downloading on every tool call)
# ---------------------------------------------------------------------------

_MAX_CACHE = 3
_panel_cache: OrderedDict[tuple, dict] = OrderedDict()


def _get_or_build_panel(
    assets: list[str] | None = None,
    benchmark: str = "XLF",
    start: str = "2020-01-01",
    end: str = "2026-12-31",
) -> dict:
    """Return cached data panel or build a new one."""
    from apps.factor_discovery.run import XLF_COMPONENTS, build_data_panel

    assets = assets or XLF_COMPONENTS
    if len(assets) > 100:
        assets = assets[:100]

    key = (tuple(sorted(assets)), benchmark, start, end)
    if key in _panel_cache:
        _panel_cache.move_to_end(key)
        return _panel_cache[key]

    panel = build_data_panel(assets, benchmark=benchmark, start=start, end=end)

    if len(_panel_cache) >= _MAX_CACHE:
        _panel_cache.popitem(last=False)
    _panel_cache[key] = panel
    return panel


# ===========================================================================
# Group A: Factor Grammar (stateless, fast)
# ===========================================================================


@mcp.tool()
def parse_factor_expression(formula: str) -> dict:
    """Parse and validate a factor formula string.

    Returns the parsed AST, expression depth, and any errors.
    Use `list_grammar` first to see available primitives and operators.

    Args:
        formula: Factor formula string, e.g. "rank(rolling_mean(return, 20))"
    """
    from lib.factor_grammar import parse_expression

    try:
        expr = parse_expression(formula)
        return {
            "valid": True,
            "formula": expr.to_string(),
            "depth": expr.depth(),
            "error": None,
        }
    except (ValueError, Exception) as e:
        return {
            "valid": False,
            "formula": formula,
            "depth": None,
            "error": str(e),
        }


@mcp.tool()
def list_grammar() -> dict:
    """Return the full factor grammar: available primitives, operators, and constraints.

    Use this to understand what building blocks are available for constructing
    factor formulas before calling `parse_factor_expression` or `evaluate_factor`.
    """
    from lib.factor_grammar import (
        MAX_EXPRESSION_DEPTH,
        OPERATORS,
        PRIMITIVES,
        grammar_description,
    )

    return {
        "grammar_text": grammar_description(),
        "primitives": PRIMITIVES,
        "operators": {
            name: {"args": info["args"], "description": info["desc"]}
            for name, info in OPERATORS.items()
        },
        "max_depth": MAX_EXPRESSION_DEPTH,
    }


# ===========================================================================
# Group B: Factor Evaluation (data-dependent)
# ===========================================================================


@mcp.tool()
def evaluate_factor(
    formula: str,
    start: str = "2020-01-01",
    end: str = "2026-12-31",
    fwd_horizon: int = 10,
) -> dict:
    """Evaluate a factor formula on market data and return full performance metrics.

    Computes rank IC, long-short Sharpe, decile returns, turnover, signal decay,
    and more. Uses XLF financial sector stocks by default.

    Args:
        formula: Factor formula string, e.g. "rank(mul(return_20d, volume_ratio))"
        start: Start date for data (ISO format, default "2020-01-01")
        end: End date for data (ISO format, default "2026-12-31")
        fwd_horizon: Forward return horizon in days (default 10)
    """
    from lib.factor_evaluator import evaluate_factor as _evaluate_factor
    from lib.factor_grammar import parse_expression

    try:
        expr = parse_expression(formula)
    except ValueError as e:
        return {"error": f"Invalid formula: {e}"}

    try:
        panel = _get_or_build_panel(start=start, end=end)
        returns = panel["return"]

        # Compute forward returns
        fwd_returns = returns.rolling(window=fwd_horizon).sum().shift(-fwd_horizon)

        # Evaluate factor expression
        factor_values = expr.evaluate(panel)

        # Compute metrics
        metrics = _evaluate_factor(
            factor_values,
            fwd_returns,
            preprocess=True,
            raw_returns=returns,
            fwd_horizon=fwd_horizon,
        )

        return {
            "formula": expr.to_string(),
            "metrics": metrics.to_dict(),
            "summary": metrics.summary_str(),
        }
    except Exception as e:
        logger.error(f"evaluate_factor failed: {traceback.format_exc()}")
        return {"error": str(e)}


@mcp.tool()
def gate_factor(
    formula: str,
    start: str = "2020-01-01",
    end: str = "2026-12-31",
    fwd_horizon: int = 10,
    t_ic_promote: float = 3.0,
    sharpe_promote: float = 1.0,
) -> dict:
    """Evaluate a factor and apply promotion gate logic.

    Returns whether the factor should be promoted, retired, or held,
    along with the reasoning and full metrics.

    Args:
        formula: Factor formula string
        start: Start date for data (ISO format)
        end: End date for data (ISO format)
        fwd_horizon: Forward return horizon in days (default 10)
        t_ic_promote: Minimum t-statistic for IC to promote (default 3.0)
        sharpe_promote: Minimum long-short Sharpe to promote (default 1.0)
    """
    from lib.factor_evaluator import evaluate_factor as _evaluate_factor
    from lib.factor_gate import GateThresholds, gate_decision, gate_reason
    from lib.factor_grammar import parse_expression

    try:
        expr = parse_expression(formula)
    except ValueError as e:
        return {"error": f"Invalid formula: {e}"}

    try:
        panel = _get_or_build_panel(start=start, end=end)
        returns = panel["return"]
        fwd_returns = returns.rolling(window=fwd_horizon).sum().shift(-fwd_horizon)

        factor_values = expr.evaluate(panel)
        metrics = _evaluate_factor(
            factor_values,
            fwd_returns,
            preprocess=True,
            raw_returns=returns,
            fwd_horizon=fwd_horizon,
        )

        thresholds = GateThresholds(
            t_ic_promote=t_ic_promote,
            sharpe_promote=sharpe_promote,
        )

        decision = gate_decision(metrics, thresholds)
        reason = gate_reason(metrics, decision, thresholds)

        return {
            "formula": expr.to_string(),
            "decision": decision,
            "reason": reason,
            "metrics": metrics.to_dict(),
            "summary": metrics.summary_str(),
        }
    except Exception as e:
        logger.error(f"gate_factor failed: {traceback.format_exc()}")
        return {"error": str(e)}


# ===========================================================================
# Group C: QQQ Hedge
# ===========================================================================


@mcp.tool()
def qqq_hedge_signal(
    from_date: str | None = None,
    to_date: str | None = None,
) -> dict:
    """Get the QQQ hedge position for today, or for each day in a date range.

    Fetches QQQ price data, computes all indicators (SMA50, SMA200, drawdown,
    rally from trough), and returns the hedge position(s).

    - Called with no arguments: returns today's (latest) hedge position.
    - Called with from_date and to_date: returns the hedge position for every
      trading day in that range.

    Args:
        from_date: Start date (ISO format, e.g. "2025-12-01"). Omit for latest only.
        to_date: End date (ISO format, e.g. "2026-03-28"). Omit for latest only.
    """
    import pandas as pd

    from lib.data import load_ohlcv_yfinance
    from lib.qqq_hedge import QQQHedgeSignal, compute_hedge_indicators

    try:
        ohlcv = load_ohlcv_yfinance(["QQQ"], start="2019-01-01")
        if ohlcv is None:
            from lib.data import load_ohlcv_alphavantage
            ohlcv = load_ohlcv_alphavantage(["QQQ"], start="2019-01-01")

        if ohlcv is None:
            return {"error": "Could not load QQQ data from any source"}

        close = ohlcv["close"]["QQQ"]
        indicators = compute_hedge_indicators(close)

        def _compute_for_date(target, signal):
            row = indicators.loc[target]
            close_val = float(close.loc[target])
            position = signal.update(
                close=close_val,
                sma200=float(row["sma200"]),
                sma50=float(row["sma50"]),
                drawdown=float(row["drawdown"]),
                rally_from_trough=float(row["rally_from_trough"]),
            )
            return {
                "date": str(target.date()),
                "position": position,
                "position_pct": f"{abs(position) * 100:.1f}% short",
                "close": round(close_val, 2),
                "sma200": round(float(row["sma200"]), 2),
                "drawdown": round(float(row["drawdown"]), 4),
            }

        # --- Date range mode ---
        if from_date is not None and to_date is not None:
            start_ts = pd.Timestamp(from_date)
            end_ts = pd.Timestamp(to_date)
            mask = (indicators.index >= start_ts) & (indicators.index <= end_ts)
            dates_in_range = indicators.index[mask]

            if dates_in_range.empty:
                return {"error": f"No trading days found between {from_date} and {to_date}"}

            signal = QQQHedgeSignal()
            daily = [_compute_for_date(dt, signal) for dt in dates_in_range]

            return {
                "from_date": str(dates_in_range[0].date()),
                "to_date": str(dates_in_range[-1].date()),
                "n_trading_days": len(daily),
                "daily": daily,
            }

        # --- Single date mode (default: latest) ---
        target = indicators.index[-1]
        signal = QQQHedgeSignal()
        return _compute_for_date(target, signal)

    except Exception as e:
        logger.error(f"qqq_hedge_signal failed: {traceback.format_exc()}")
        return {"error": str(e)}


@mcp.tool()
def qqq_hedge_backtest(
    start: str = "2020-01-01",
    end: str = "2026-12-31",
    base_hedge: float = -0.05,
    dd_tier_1: float = -0.03,
    dd_tier_2: float = -0.07,
    dd_tier_3: float = -0.12,
) -> dict:
    """Run a full historical backtest of the QQQ tail risk hedge overlay.

    Compares a hedged portfolio (100% long QQQ + hedge) against buy-and-hold.
    Returns performance stats including Sharpe, max drawdown reduction, and return cost.

    Args:
        start: Backtest start date (ISO format, default "2020-01-01")
        end: Backtest end date (ISO format, default "2026-12-31")
        base_hedge: Always-on base short position (default -0.05 = 5% short)
        dd_tier_1: Mild drawdown threshold (default -0.03)
        dd_tier_2: Moderate drawdown threshold (default -0.07)
        dd_tier_3: Severe drawdown threshold (default -0.12)
    """
    from lib.data import load_ohlcv_yfinance
    from lib.qqq_hedge import HedgeConfig, QQQHedgeSignal

    try:
        ohlcv = load_ohlcv_yfinance(["QQQ"], start=start, end=end)
        if ohlcv is None:
            from lib.data import load_ohlcv_alphavantage

            ohlcv = load_ohlcv_alphavantage(["QQQ"], start=start, end=end)

        if ohlcv is None:
            return {"error": "Could not load QQQ data from any source"}

        close = ohlcv["close"]["QQQ"]
        returns = ohlcv["returns"]["QQQ"]

        config = HedgeConfig(
            base_hedge=base_hedge,
            dd_tier_1=dd_tier_1,
            dd_tier_2=dd_tier_2,
            dd_tier_3=dd_tier_3,
        )

        result = QQQHedgeSignal.backtest(close, returns, config)
        stats = result["stats"]

        # Format stats for readability
        formatted_stats = {}
        for k, v in stats.items():
            if isinstance(v, float):
                formatted_stats[k] = round(v, 4)
            else:
                formatted_stats[k] = v

        # Last 10 positions
        hedge_pos = result["hedge_position"]
        recent = [
            {"date": str(d.date()), "position": round(float(p), 4)}
            for d, p in hedge_pos.tail(10).items()
        ]

        return {
            "stats": formatted_stats,
            "recent_positions": recent,
            "data_range": f"{close.index.min().date()} to {close.index.max().date()}",
            "n_trading_days": len(close),
        }
    except Exception as e:
        logger.error(f"qqq_hedge_backtest failed: {traceback.format_exc()}")
        return {"error": str(e)}


# ===========================================================================
# Group D: Factor Aggregation
# ===========================================================================


@mcp.tool()
def aggregate_factors(
    formulas: list[str],
    method: str = "linear",
    start: str = "2020-01-01",
    end: str = "2026-12-31",
) -> dict:
    """Combine multiple factor formulas into a single composite score.

    Evaluates each formula on market data, then aggregates using either
    equal-weighted z-score average (linear) or LightGBM (non-linear).

    Args:
        formulas: List of factor formula strings to aggregate
        method: Aggregation method - "linear" (default) or "lgbm"
        start: Start date for data (ISO format)
        end: End date for data (ISO format)
    """
    from lib.factor_aggregator import aggregate_lgbm, aggregate_linear
    from lib.factor_evaluator import preprocess_factor
    from lib.factor_grammar import parse_expression

    if not formulas:
        return {"error": "No formulas provided"}

    try:
        panel = _get_or_build_panel(start=start, end=end)
        returns = panel["return"]

        factor_panels = {}
        for i, formula in enumerate(formulas):
            try:
                expr = parse_expression(formula)
                raw = expr.evaluate(panel)
                factor_panels[f"factor_{i + 1}"] = preprocess_factor(raw)
            except ValueError as e:
                return {"error": f"Invalid formula '{formula}': {e}"}

        if method == "lgbm" and len(factor_panels) >= 2:
            fwd_returns = returns.rolling(window=10).sum().shift(-10)
            composite = aggregate_lgbm(
                factor_panels, fwd_returns, train_end="2023-12-31"
            )
        else:
            composite = aggregate_linear(factor_panels)

        # Summary stats
        latest = composite.iloc[-1].dropna()
        top_5 = latest.nlargest(5)
        bottom_5 = latest.nsmallest(5)

        return {
            "method": method if method != "lgbm" or len(factor_panels) >= 2 else "linear",
            "n_factors": len(factor_panels),
            "composite_summary": {
                "mean": round(float(composite.mean().mean()), 6),
                "std": round(float(composite.std().mean()), 6),
                "n_dates": len(composite),
                "n_symbols": len(composite.columns),
                "top_5_latest": {
                    str(k): round(float(v), 4) for k, v in top_5.items()
                },
                "bottom_5_latest": {
                    str(k): round(float(v), 4) for k, v in bottom_5.items()
                },
            },
        }
    except Exception as e:
        logger.error(f"aggregate_factors failed: {traceback.format_exc()}")
        return {"error": str(e)}


# ===========================================================================
# Group E: Discovery Pipeline
# ===========================================================================


@mcp.tool()
def run_discovery(
    max_rounds: int = 5,
    llm_backend: str = "anthropic",
    fwd_horizon: int = 10,
    is_end: str = "2023-12-31",
) -> dict:
    """Run the full LLM-guided factor discovery pipeline.

    This is a long-running operation that uses an LLM to generate factor
    hypotheses, evaluates them via backtesting, and promotes the best ones.
    May take several minutes depending on the number of rounds.

    Args:
        max_rounds: Number of discovery rounds (default 5, max 10)
        llm_backend: LLM backend - "anthropic" or "moonshot" (default "anthropic")
        fwd_horizon: Forward return horizon in days (default 10)
        is_end: In-sample end date (ISO format, default "2023-12-31")
    """
    from apps.factor_discovery.run import run

    max_rounds = min(max_rounds, 10)

    try:
        result = run(
            max_rounds=max_rounds,
            llm_backend=llm_backend,
            fwd_horizon=fwd_horizon,
            is_end=is_end,
        )

        promoted = result.get("promoted_factors", {})
        audit_log = result.get("audit_log", [])

        # Serialize for MCP response
        serialized_promoted = {}
        for name, info in promoted.items():
            serialized_promoted[name] = {
                "formula": info["formula"],
                "rationale": info["rationale"],
                "metrics": info["metrics"],
                "round": info["round"],
            }

        return {
            "n_promoted": len(promoted),
            "promoted_factors": serialized_promoted,
            "audit_log": audit_log,
            "total_rounds": len(audit_log),
        }
    except Exception as e:
        logger.error(f"run_discovery failed: {traceback.format_exc()}")
        return {"error": str(e)}


# ===========================================================================
# MCP Resources
# ===========================================================================


@mcp.resource("factor-grammar://description")
def grammar_resource() -> str:
    """Full factor grammar reference including primitives, operators, and constraints."""
    from lib.factor_grammar import grammar_description

    return grammar_description()


@mcp.resource("factor-library://current")
def library_resource() -> str:
    """Current promoted factor library (output/factor_library.json)."""
    path = ROOT / "output" / "factor_library.json"
    if path.exists():
        return path.read_text()
    return json.dumps({}, indent=2)


@mcp.resource("discovery-log://latest")
def log_resource() -> str:
    """Latest factor discovery audit log (output/factor_discovery_log.json)."""
    path = ROOT / "output" / "factor_discovery_log.json"
    if path.exists():
        return path.read_text()
    return json.dumps([], indent=2)


# ===========================================================================
# Health check route (for Railway)
# ===========================================================================

from starlette.requests import Request
from starlette.responses import JSONResponse


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "server": "System Factors MCP"})


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info(f"Starting System Factors MCP server on port {mcp.settings.port}")

    mcp.run(transport="streamable-http")
