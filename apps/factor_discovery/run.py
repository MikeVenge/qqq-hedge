"""
Agentic factor discovery app.

Runs the closed-loop factor discovery pipeline:
  1. Load daily return data (Alpha Vantage or synthetic)
  2. Build primitive data panel (return, price, volume, market_return)
  3. Run K rounds of LLM-guided hypothesis → backtest → gate
  4. Save promoted factor library and audit log

Usage:
  python run.py factor_discovery
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from lib.config import get_settings
from lib.data import get_returns, DEFAULT_ASSETS, load_returns_from_alphavantage, load_ohlcv_alphavantage

# XLF ETF components (Financial Select Sector SPDR) — major holdings
XLF_COMPONENTS = [
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "AXP",
    "BLK", "C", "SCHW", "CB", "MMC", "PGR", "ICE", "CME", "AON", "MCO",
    "MET", "AIG", "TFC", "PNC", "USB", "AJG", "AFL", "TRV", "ALL", "MSCI",
    "FIS", "FITB", "STT", "MTB", "COF", "BK", "HBAN", "RJF", "CFG", "KEY",
    "CINF", "RF", "NTRS", "WRB", "TROW", "L", "DFS", "SYF", "RE", "ERIE",
    "NDAQ", "CBOE", "MKTX", "GL", "AIZ", "BEN", "IVZ", "ZION", "CMA",
    "BRO", "FDS", "EG", "WTW", "HIG", "LNC", "ACGL", "RNR", "ALLY",
]
from lib.factor_grammar import parse_expression
from lib.factor_evaluator import evaluate_factor, preprocess_factor
from lib.factor_gate import gate_decision, gate_reason, GateThresholds
from lib.factor_agent import FactorAgent, RoundRecord
from lib.factor_aggregator import aggregate_linear, aggregate_lgbm

logger = logging.getLogger(__name__)


def build_data_panel(
    assets: list[str],
    benchmark: str = "XLF",
    start: str = "2020-01-01",
    end: str = "2026-12-31",
) -> dict[str, pd.DataFrame]:
    """Build the primitive data panel from market data.

    Tries Alpha Vantage first (real OHLCV), then synthetic fallback.
    Returns dict mapping primitive names to (dates x symbols) DataFrames.
    """
    # --- Try Alpha Vantage (preferred: real price + volume) ---
    all_symbols = list(dict.fromkeys(assets + [benchmark]))
    print(f"  Loading {len(all_symbols)} symbols from Alpha Vantage...")
    ohlcv = load_ohlcv_alphavantage(all_symbols, start=start, end=end)

    if ohlcv is not None:
        returns_df = ohlcv["returns"]
        close_df = ohlcv["close"]
        volume_df = ohlcv["volume"]

        # Separate stock columns from benchmark
        stock_cols = [c for c in returns_df.columns if c != benchmark]
        stock_returns = returns_df[stock_cols]
        price_df = close_df[stock_cols]
        vol_df = volume_df[stock_cols]

        # Market return: broadcast benchmark return
        if benchmark in returns_df.columns:
            mkt_ret = returns_df[benchmark]
        else:
            mkt_ret = stock_returns.mean(axis=1)

        market_return_df = pd.DataFrame(
            np.tile(mkt_ret.values[:, None], (1, stock_returns.shape[1])),
            index=stock_returns.index,
            columns=stock_returns.columns,
        )

        return _derive_primitives(stock_returns, price_df, vol_df, market_return_df)

    # --- Fallback: Synthetic ---
    logger.warning("No market data source available, using synthetic data")
    from lib.data import make_synthetic_returns
    n_days = 252 * 5
    returns_df = make_synthetic_returns(n_assets=len(assets), n_days=n_days)
    returns_df.columns = assets[:returns_df.shape[1]]
    returns_df.index = pd.bdate_range(start, periods=n_days, freq="B")
    market_ret = returns_df.mean(axis=1)
    price_df = (1 + returns_df).cumprod() * 100
    rng = np.random.default_rng(42)
    volume_df = pd.DataFrame(
        rng.lognormal(15, 1, returns_df.shape),
        index=returns_df.index,
        columns=returns_df.columns,
    )
    market_return_df = pd.DataFrame(
        np.tile(market_ret.values[:, None], (1, returns_df.shape[1])),
        index=returns_df.index,
        columns=returns_df.columns,
    )
    return _derive_primitives(returns_df, price_df, volume_df, market_return_df)


def _derive_primitives(
    returns_df: pd.DataFrame,
    price_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    market_return_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build the full primitive panel including derived fields."""
    panel = {
        "return": returns_df,
        "price": price_df,
        "volume": volume_df,
        "market_return": market_return_df,
    }

    # Turnover proxy: volume / rolling 60-day mean volume
    vol_ma60 = volume_df.rolling(window=60, min_periods=20).mean()
    panel["turnover"] = volume_df / vol_ma60.replace(0, np.nan)

    # Volume ratio: volume / 20-day MA
    vol_ma20 = volume_df.rolling(window=20, min_periods=5).mean()
    panel["volume_ratio"] = volume_df / vol_ma20.replace(0, np.nan)

    # 20-day realized volatility
    panel["realized_vol"] = returns_df.rolling(window=20, min_periods=5).std()

    # Price-to-MA ratio (20-day)
    price_ma20 = price_df.rolling(window=20, min_periods=5).mean()
    panel["price_to_ma"] = price_df / price_ma20.replace(0, np.nan)

    # Market volatility (broadcast)
    mkt_vol = market_return_df.iloc[:, 0].rolling(window=20, min_periods=5).std()
    panel["market_vol"] = pd.DataFrame(
        np.tile(mkt_vol.values[:, None], (1, returns_df.shape[1])),
        index=returns_df.index,
        columns=returns_df.columns,
    )

    # --- Technical signal primitives ---

    # SMAs
    panel["sma_50"] = price_df.rolling(window=50, min_periods=30).mean()
    panel["sma_100"] = price_df.rolling(window=100, min_periods=60).mean()
    panel["sma_200"] = price_df.rolling(window=200, min_periods=120).mean()

    # Price-to-SMA ratios
    panel["price_to_sma50"] = price_df / panel["sma_50"].replace(0, np.nan)
    panel["price_to_sma100"] = price_df / panel["sma_100"].replace(0, np.nan)
    panel["price_to_sma200"] = price_df / panel["sma_200"].replace(0, np.nan)

    # Golden/death cross signal
    panel["sma50_to_sma200"] = panel["sma_50"] / panel["sma_200"].replace(0, np.nan)

    # RSI (14-day)
    delta = returns_df.copy()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=14, min_periods=10).mean()
    avg_loss = loss.rolling(window=14, min_periods=10).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    panel["rsi_14"] = 100 - (100 / (1 + rs))

    # Multi-horizon cumulative returns
    panel["return_10d"] = returns_df.rolling(window=10, min_periods=5).sum()
    panel["return_20d"] = returns_df.rolling(window=20, min_periods=10).sum()
    panel["return_60d"] = returns_df.rolling(window=60, min_periods=30).sum()

    # 60-day rolling beta and correlation with market
    mkt_series = market_return_df.iloc[:, 0]
    mkt_var = mkt_series.rolling(window=60, min_periods=30).var()
    betas = {}
    corrs = {}
    for col in returns_df.columns:
        cov = returns_df[col].rolling(window=60, min_periods=30).cov(mkt_series)
        betas[col] = cov / mkt_var.replace(0, np.nan)
        corrs[col] = returns_df[col].rolling(window=60, min_periods=30).corr(mkt_series)
    panel["beta_60d"] = pd.DataFrame(betas, index=returns_df.index)
    panel["corr_market_60d"] = pd.DataFrame(corrs, index=returns_df.index)

    # High-low range proxy (from realized vol)
    panel["high_low_range"] = panel["realized_vol"] * np.sqrt(1.0 / 252)

    # Distance from 52-week high
    rolling_high_252 = price_df.rolling(window=252, min_periods=60).max()
    panel["distance_from_52w_high"] = (price_df - rolling_high_252) / rolling_high_252.replace(0, np.nan)

    return panel


def run(config_path: str | None = None, **overrides) -> dict:
    """Run the agentic factor discovery pipeline.

    Returns dict with promoted factor library and audit log.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load config
    opts = get_settings("factor_discovery", **overrides)
    max_rounds = opts.get("max_rounds", 10)
    is_end = opts.get("is_end", "2023-12-31")
    llm_backend = opts.get("llm_backend", "moonshot")
    llm_model = opts.get("llm_model", None)  # None => use backend default
    aggregation_method = opts.get("aggregation_method", "linear")
    fwd_horizon = opts.get("fwd_horizon", 10)

    thresholds = GateThresholds.from_config(opts)

    print("=" * 60)
    print(" Agentic Factor Discovery")
    print("=" * 60)
    display_model = llm_model or FactorAgent.BACKENDS[llm_backend]["model"]
    print(f"  Backend: {llm_backend} ({display_model})")
    print(f"  Rounds: {max_rounds}")
    print(f"  IS end: {is_end}")
    print(f"  Forward horizon: {fwd_horizon} days")
    print(f"  Gate: t_IC >= {thresholds.t_ic_promote}, Sharpe >= {thresholds.sharpe_promote}")
    print()

    # 1. Build data panel
    assets = opts.get("assets", XLF_COMPONENTS)
    benchmark = opts.get("benchmark", "XLF")
    data_panel = build_data_panel(assets, benchmark=benchmark)
    returns = data_panel["return"]

    # 2. Split IS/OOS
    is_mask = returns.index <= pd.Timestamp(is_end)
    is_returns = returns[is_mask]
    oos_returns = returns[~is_mask]

    # Forward returns (10-day cumulative)
    fwd_returns = returns.rolling(window=fwd_horizon).sum().shift(-fwd_horizon)
    is_fwd = fwd_returns[is_mask]

    is_panel = {k: v[is_mask] for k, v in data_panel.items()}

    print(f"  IS period: {is_returns.index.min().date()} to {is_returns.index.max().date()} ({len(is_returns)} days)")
    print(f"  OOS period: {oos_returns.index.min().date() if len(oos_returns) > 0 else 'N/A'} to {oos_returns.index.max().date() if len(oos_returns) > 0 else 'N/A'} ({len(oos_returns)} days)")
    print(f"  Universe: {len(returns.columns)} stocks")
    print()

    # 3. Initialize agent
    agent_kwargs: dict = {"backend": llm_backend}
    if llm_model:
        agent_kwargs["model"] = llm_model
    agent = FactorAgent(**agent_kwargs)

    # 4. Run discovery loop
    promoted_factors: dict[str, dict] = {}  # name -> {formula, rationale, metrics}
    promoted_panels: dict[str, pd.DataFrame] = {}  # name -> factor values panel

    for k in range(1, max_rounds + 1):
        print(f"--- Round {k}/{max_rounds} ---")

        try:
            # Step 1: Generate hypothesis
            hypothesis = agent.generate_hypothesis(k)
            print(f"  Rationale: {hypothesis.economic_rationale[:100]}...")
            print(f"  Formula: {hypothesis.formula}")

            # Step 2: Compute factor values on IS data
            factor_values = hypothesis.expression.evaluate(is_panel)

            # Step 3: Evaluate (with turnover, decay, and redundancy)
            metrics = evaluate_factor(
                factor_values, is_fwd,
                preprocess=True,
                raw_returns=is_returns,
                existing_panels=promoted_panels,
                fwd_horizon=fwd_horizon,
            )
            print(f"  {metrics.summary_str()}")

            # Step 4: Gate decision (with turnover + redundancy awareness)
            decision = gate_decision(metrics, thresholds, existing_factors=promoted_panels)
            reason = gate_reason(metrics, decision, thresholds)
            print(f"  Decision: {decision.upper()} — {reason}")

            # Record
            record = RoundRecord(
                round=k,
                hypothesis=hypothesis.economic_rationale,
                formula=hypothesis.formula,
                metrics=metrics.to_dict(),
                decision=decision,
                reason=reason,
            )
            agent.record_round(record)

            # If promoted, add to library
            if decision == "promote":
                factor_name = f"factor_{len(promoted_factors) + 1}"
                promoted_factors[factor_name] = {
                    "formula": hypothesis.formula,
                    "rationale": hypothesis.economic_rationale,
                    "metrics": metrics.to_dict(),
                    "round": k,
                }
                # Compute on full panel for aggregation
                full_values = hypothesis.expression.evaluate(data_panel)
                promoted_panels[factor_name] = preprocess_factor(full_values)

        except Exception as e:
            logger.error(f"Round {k} failed: {e}")
            traceback.print_exc()
            record = RoundRecord(
                round=k,
                hypothesis="",
                formula="",
                error=str(e),
            )
            agent.record_round(record)

        print()

    # 5. Summary
    print("=" * 60)
    print(f" Discovery Complete: {len(promoted_factors)} factors promoted")
    print("=" * 60)

    for name, info in promoted_factors.items():
        m = info["metrics"]
        print(f"  {name}: {info['formula']}")
        print(f"    IC t-stat={m['ic_t_stat']:.2f}, Sharpe={m['ls_sharpe']:.2f}, Ret={m['ls_annual_return']:.2%}")

    # 6. Aggregate if we have promoted factors
    composite_scores = None
    if promoted_panels:
        print(f"\nAggregating {len(promoted_panels)} factors ({aggregation_method})...")
        if aggregation_method == "lgbm" and len(promoted_panels) >= 2:
            try:
                composite_scores = aggregate_lgbm(
                    promoted_panels, fwd_returns, train_end=is_end,
                )
                print("  LightGBM composite scores computed.")
            except Exception as e:
                logger.warning(f"LightGBM aggregation failed ({e}), falling back to linear")
                composite_scores = aggregate_linear(promoted_panels)
                print("  Linear composite scores computed (LightGBM fallback).")
        else:
            composite_scores = aggregate_linear(promoted_panels)
            print("  Linear composite scores computed.")

    # 7. Save outputs
    out_dir = Path(opts.get("out_dir") or "output")
    out_dir.mkdir(exist_ok=True)

    library_path = out_dir / "factor_library.json"
    with open(library_path, "w") as f:
        json.dump(promoted_factors, f, indent=2, default=str)
    print(f"\nFactor library saved to {library_path}")

    log_path = out_dir / "factor_discovery_log.json"
    with open(log_path, "w") as f:
        json.dump(agent.get_audit_log(), f, indent=2, default=str)
    print(f"Audit log saved to {log_path}")

    if composite_scores is not None:
        scores_path = out_dir / "composite_scores.csv"
        composite_scores.to_csv(scores_path)
        print(f"Composite scores saved to {scores_path}")

    return {
        "promoted_factors": promoted_factors,
        "audit_log": agent.get_audit_log(),
        "composite_scores": composite_scores,
    }


if __name__ == "__main__":
    run()
