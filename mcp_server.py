"""
MCP Server for System Factors — quantitative finance factor discovery platform.

Exposes factor grammar, evaluation, gate decisions, QQQ hedge signals,
factor aggregation, and the full discovery pipeline as MCP tools via
Streamable HTTP transport.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import traceback
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
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


def _compute_hedge_signal(
    date: str | None = None,
    vt: float = 15,
    book_id: int | None = None,
) -> dict:
    """Shared core for the QQQ hedge signal (used by the MCP tool and REST API).

    Blocking (Alpha Vantage fetch + pandas + optional Mango call); call via a
    threadpool from async contexts. Returns the hedging-parameters dict, or
    {"error": ...}.

    When `book_id` is set, the inverse-vol scalar uses the 30-day realized
    volatility of that Mango book's current portfolio instead of QQQ's; the SMA
    regime gate still uses QQQ.
    """
    from lib.data import load_ohlcv_alphavantage
    from lib.qqq_hedge import hedge_parameters

    ohlcv = load_ohlcv_alphavantage(["QQQ"], start="2019-01-01")
    if ohlcv is None:
        return {"error": "Could not load QQQ data from Alpha Vantage"}
    close = ohlcv["close"]["QQQ"]
    returns = ohlcv["returns"]["QQQ"]

    if book_id is None:
        return hedge_parameters(close, returns, as_of=date, vt=vt)

    # --- Book portfolio-vol path ---
    from lib.mango import resolve_book_constituents
    from lib.portfolio_vol import portfolio_realized_vol_asof

    book = resolve_book_constituents(int(book_id))
    if "error" in book:
        return {"error": f"book {book_id}: {book['error']}"}
    if not book["symbols"]:
        return {"error": f"book {book_id}: no priced constituents"}

    panel = load_ohlcv_alphavantage(book["symbols"], start="2019-01-01")
    if panel is None:
        return {"error": "Could not load book constituent prices from Alpha Vantage"}

    volinfo = portfolio_realized_vol_asof(
        panel["returns"], book["weights"], as_of=date, window=30
    )
    if "error" in volinfo:
        return {"error": f"book {book_id} vol: {volinfo['error']}"}

    return hedge_parameters(
        close, returns, as_of=date, vt=vt,
        rv_override=volinfo["portfolio_vol"], vol_source="portfolio",
        book_meta={
            "book_id": book["book_id"], "book_name": book["book_name"],
            "n_constituents": book["n_constituents"],
        },
    )


@mcp.tool()
def qqq_hedge_signal(
    date: str | None = None,
    vt: float = 15,
    book_id: int | None = None,
) -> dict:
    """Get the QQQ hedging parameters as of a date for a chosen vol-target level.

    Two-layer overlay: an SMA100/SMA200 regime gate (1.0 / 0.5 / 0.0) times an
    inverse-vol scalar min(target_vol / rv20, 1.5). Returns the gross long
    deployment (exposure) and the cash sleeve (= 1 - exposure; negative = financed
    leverage). Fetches QQQ historical closing prices from Alpha Vantage.

    Args:
        date: As-of date (ISO format, e.g. "2026-05-28"). Rolls back to the most
              recent trading day on/before it. Omit for the latest trading day.
        vt: Vol-target level (any positive number). 15 -> VT15 -> target_vol 0.15;
            23 -> VT23 -> target_vol 0.23. Default 15.
        book_id: Optional Mango trading-book ID. When set, the inverse-vol scalar
            uses the book's 30-day realized portfolio volatility instead of QQQ's
            (the SMA regime gate still uses QQQ).
    """
    try:
        return _compute_hedge_signal(date, vt, book_id)
    except Exception as e:
        logger.error(f"qqq_hedge_signal failed: {traceback.format_exc()}")
        return {"error": str(e)}


@mcp.tool()
def qqq_hedge_backtest(
    start: str = "2020-01-01",
    end: str = "2026-12-31",
    vt: float = 15,
    leverage_cap: float = 1.5,
    fed_funds_rate: float = 0.0,
) -> dict:
    """Backtest the QQQ two-layer vol-target overlay vs buy-and-hold.

    The overlay sizes gross long deployment as gate(SMA100/200) x
    min(target_vol / rv20, leverage_cap); the cash sleeve (1 - exposure) earns
    fed_funds_rate. Returns performance stats (Sharpe, max DD, Calmar, mean
    exposure / w_vol, % at leverage cap). Fetches QQQ closes from Alpha Vantage.

    Args:
        start: Backtest start date (ISO format, default "2020-01-01")
        end: Backtest end date (ISO format, default "2026-12-31")
        vt: Vol-target level (e.g. 15 -> VT15 -> target_vol 0.15). Default 15.
        leverage_cap: Cap on the inverse-vol scalar (default 1.5).
        fed_funds_rate: Annual rate earned by the cash sleeve (default 0.0).
    """
    from lib.data import load_ohlcv_alphavantage
    from lib.qqq_hedge import QQQVolTargetSignal, VolTargetConfig

    try:
        ohlcv = load_ohlcv_alphavantage(["QQQ"], start=start, end=end)

        if ohlcv is None:
            return {"error": "Could not load QQQ data from Alpha Vantage"}

        close = ohlcv["close"]["QQQ"]
        returns = ohlcv["returns"]["QQQ"]

        config = VolTargetConfig.from_vt(vt, leverage_cap=leverage_cap)

        result = QQQVolTargetSignal.backtest(
            close, returns, config, fed_funds_rate=fed_funds_rate
        )
        stats = result["stats"]

        # Format stats for readability
        formatted_stats = {}
        for k, v in stats.items():
            if isinstance(v, float):
                formatted_stats[k] = round(v, 4)
            else:
                formatted_stats[k] = v

        # Last 10 exposures
        exposure = result["exposure"].dropna()
        recent = [
            {"date": str(d.date()), "exposure": round(float(p), 4)}
            for d, p in exposure.tail(10).items()
        ]

        return {
            "vt": float(vt),
            "target_vol": round(config.target_vol, 4),
            "stats": formatted_stats,
            "recent_exposures": recent,
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
# Async REST API (submit + poll) for the QQQ hedge signal
# ===========================================================================
#
#   POST /api/hedge            -> 202 { job_id, status: "pending", poll_url }
#       body or query: { "date": "2026-05-28"(optional), "vt": 23(optional, def 15) }
#   GET  /api/hedge/{job_id}   -> 202 while pending/running; 200 when done/error
#       done  -> { status: "done",  result: {...hedging params...} }
#       error -> { status: "error", error: "..." }
#
# Jobs are kept in memory (single Railway container); they are lost on restart
# and the store is capped at _HEDGE_JOBS_MAX (oldest evicted).

_HEDGE_JOBS: "OrderedDict[str, dict]" = OrderedDict()
_HEDGE_JOBS_MAX = 1000
_HEDGE_BG_TASKS: set = set()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _store_job(job: dict) -> None:
    _HEDGE_JOBS[job["job_id"]] = job
    while len(_HEDGE_JOBS) > _HEDGE_JOBS_MAX:
        _HEDGE_JOBS.popitem(last=False)


async def _process_hedge_job(
    job_id: str, date: str | None, vt: float, book_id: int | None = None
) -> None:
    job = _HEDGE_JOBS.get(job_id)
    if job is None:
        return
    job["status"] = "running"
    try:
        # Offload the blocking fetch + compute so the event loop stays free.
        result = await asyncio.to_thread(_compute_hedge_signal, date, vt, book_id)
        if isinstance(result, dict) and "error" in result:
            job["status"] = "error"
            job["error"] = result["error"]
        else:
            job["status"] = "done"
            job["result"] = result
    except Exception as e:
        logger.error(f"hedge job {job_id} failed: {traceback.format_exc()}")
        job["status"] = "error"
        job["error"] = str(e)
    job["completed_at"] = _now_iso()


@mcp.custom_route("/api/hedge", methods=["POST"])
async def submit_hedge_job(request: Request) -> JSONResponse:
    """Submit an async QQQ hedge-signal job; returns 202 + job_id to poll."""
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    date = body.get("date")
    if date in (None, ""):
        date = request.query_params.get("date") or None

    vt_raw = body.get("vt", request.query_params.get("vt", 15))
    try:
        vt = float(vt_raw)
    except (TypeError, ValueError):
        return JSONResponse({"error": f"invalid vt: {vt_raw!r}"}, status_code=400)
    if vt <= 0:
        return JSONResponse({"error": "vt must be a positive number"}, status_code=400)

    book_id_raw = body.get("book_id", request.query_params.get("book_id"))
    book_id = None
    if book_id_raw not in (None, ""):
        try:
            book_id = int(book_id_raw)
        except (TypeError, ValueError):
            return JSONResponse({"error": f"invalid book_id: {book_id_raw!r}"}, status_code=400)

    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id,
        "status": "pending",
        "request": {"date": date, "vt": vt, "book_id": book_id},
        "result": None,
        "error": None,
        "created_at": _now_iso(),
        "completed_at": None,
    }
    _store_job(job)

    task = asyncio.create_task(_process_hedge_job(job_id, date, vt, book_id))
    _HEDGE_BG_TASKS.add(task)
    task.add_done_callback(_HEDGE_BG_TASKS.discard)

    return JSONResponse(
        {
            "job_id": job_id,
            "status": "pending",
            "request": job["request"],
            "poll_url": f"/api/hedge/{job_id}",
        },
        status_code=202,
    )


@mcp.custom_route("/api/hedge/{job_id}", methods=["GET"])
async def get_hedge_job(request: Request) -> JSONResponse:
    """Poll an async hedge-signal job. 202 while pending/running; 200 when done/error."""
    job_id = request.path_params.get("job_id")
    job = _HEDGE_JOBS.get(job_id)
    if job is None:
        return JSONResponse({"error": "unknown job_id", "job_id": job_id}, status_code=404)

    resp = {
        "job_id": job["job_id"],
        "status": job["status"],
        "request": job["request"],
        "created_at": job["created_at"],
        "completed_at": job["completed_at"],
    }
    if job["status"] == "done":
        resp["result"] = job["result"]
    elif job["status"] == "error":
        resp["error"] = job["error"]

    code = 200 if job["status"] in ("done", "error") else 202
    return JSONResponse(resp, status_code=code)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info(f"Starting System Factors MCP server on port {mcp.settings.port}")

    mcp.run(transport="streamable-http")
