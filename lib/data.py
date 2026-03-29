"""
Shared data loading for all Riskfolio-Lib apps.
Supports Alpha Vantage (live) and synthetic returns (offline/demo).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Default config used across apps when not overridden
DEFAULT_ASSETS = [
    "JCI", "TGT", "CMCSA", "CPB", "MO", "APA", "MMC", "JPM",
    "ZION", "PSA", "BAX", "BMY", "LUV", "PCAR", "TXT", "TMO",
]
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2026-12-31"


def load_returns_from_alphavantage(
    assets: list[str],
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame | None:
    """Load historical adjusted-close returns via Alpha Vantage. Returns None on failure."""
    ohlcv = load_ohlcv_alphavantage(assets, start=start, end=end)
    if ohlcv is None:
        return None
    return ohlcv.get("returns")


def load_ohlcv_alphavantage(
    assets: list[str],
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    calls_per_minute: int = 75,
) -> dict[str, pd.DataFrame] | None:
    """Load historical OHLCV data via Alpha Vantage (premium).

    Returns dict with keys: 'close', 'volume', 'returns', or None on failure.
    """
    import requests
    import time

    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return None

    delay = 60.0 / calls_per_minute  # seconds between calls
    all_closes: dict[str, dict[str, float]] = {}
    all_volumes: dict[str, dict[str, float]] = {}
    failed: list[str] = []

    for i, symbol in enumerate(assets):
        if i > 0:
            time.sleep(delay)

        url = (
            "https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_DAILY_ADJUSTED"
            f"&symbol={symbol}&outputsize=full&apikey={api_key}"
        )
        try:
            resp = requests.get(url, timeout=30)
            data = resp.json()

            # Check for rate limit / error messages
            if "Note" in data or "Information" in data:
                msg = data.get("Note") or data.get("Information", "")
                print(f"  AV rate limit hit at symbol {i+1}/{len(assets)}: {msg[:80]}")
                time.sleep(15)  # back off
                resp = requests.get(url, timeout=30)
                data = resp.json()

            ts = data.get("Time Series (Daily)")
            if not ts:
                failed.append(symbol)
                continue

            closes = {dt: float(vals["5. adjusted close"]) for dt, vals in ts.items()}
            volumes = {dt: float(vals["6. volume"]) for dt, vals in ts.items()}
            all_closes[symbol] = closes
            all_volumes[symbol] = volumes

            if (i + 1) % 10 == 0:
                print(f"  Loaded {i+1}/{len(assets)} symbols...")

        except Exception as e:
            failed.append(symbol)
            continue

    if not all_closes:
        return None

    if failed:
        print(f"  Warning: {len(failed)} symbols failed: {failed[:10]}...")

    closes_df = pd.DataFrame(all_closes)
    closes_df.index = pd.to_datetime(closes_df.index)
    closes_df = closes_df.sort_index()
    closes_df = closes_df.loc[start:end]

    volume_df = pd.DataFrame(all_volumes)
    volume_df.index = pd.to_datetime(volume_df.index)
    volume_df = volume_df.sort_index()
    volume_df = volume_df.loc[start:end]

    if closes_df.empty or closes_df.shape[1] == 0:
        return None

    # Align columns
    common_cols = closes_df.columns.intersection(volume_df.columns)
    closes_df = closes_df[common_cols]
    volume_df = volume_df[common_cols]

    returns_df = closes_df.pct_change().dropna()
    volume_df = volume_df.reindex(returns_df.index)

    print(f"  Alpha Vantage: {len(common_cols)} symbols, {len(returns_df)} days ({returns_df.index.min().date()} to {returns_df.index.max().date()})")

    return {
        "close": closes_df.reindex(returns_df.index),
        "volume": volume_df,
        "returns": returns_df,
    }


def load_ohlcv_yfinance(
    assets: list[str],
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> dict[str, pd.DataFrame] | None:
    """Load OHLCV data via yfinance. Returns dict with 'close', 'volume', 'returns' DataFrames.

    Returns None on failure.
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed")
        return None

    logger.info(f"Downloading {len(assets)} symbols from yfinance ({start} to {end})...")

    # Download in one batch for efficiency
    try:
        data = yf.download(
            assets,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        logger.error(f"yfinance download failed: {e}")
        return None

    if data is None or data.empty:
        return None

    # Handle single vs multi-ticker column structure
    if isinstance(data.columns, pd.MultiIndex):
        close_df = data["Close"]
        volume_df = data["Volume"]
    else:
        # Single ticker
        close_df = data[["Close"]].rename(columns={"Close": assets[0]})
        volume_df = data[["Volume"]].rename(columns={"Volume": assets[0]})

    # Drop tickers with insufficient data (< 50% of dates)
    min_obs = len(close_df) * 0.5
    valid_cols = close_df.columns[close_df.notna().sum() >= min_obs]
    close_df = close_df[valid_cols]
    volume_df = volume_df[valid_cols]

    if close_df.empty:
        return None

    # Forward-fill small gaps, then drop remaining NaN rows at start
    close_df = close_df.ffill(limit=5)
    volume_df = volume_df.ffill(limit=5).fillna(0)

    returns_df = close_df.pct_change().iloc[1:]  # drop first NaN row
    close_df = close_df.iloc[1:]
    volume_df = volume_df.iloc[1:]

    # Align all to same index
    common_idx = returns_df.index
    close_df = close_df.loc[common_idx]
    volume_df = volume_df.loc[common_idx]

    dropped = set(assets) - set(close_df.columns)
    if dropped:
        logger.warning(f"Dropped {len(dropped)} tickers with insufficient data: {sorted(dropped)[:10]}...")

    logger.info(f"Loaded {len(close_df.columns)} tickers, {len(common_idx)} trading days")

    return {
        "close": close_df,
        "volume": volume_df,
        "returns": returns_df,
    }


def make_synthetic_returns(
    n_assets: int = 10,
    n_days: int = 252 * 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic returns for demo when no API/data available."""
    rng = np.random.default_rng(seed)
    cov = rng.uniform(0.05, 0.35, (n_assets, n_assets))
    cov = (cov + cov.T) / 2 + np.eye(n_assets) * 0.3
    cov = np.clip(cov, 0.01, None)
    mu = rng.uniform(-0.02, 0.12, n_assets)
    returns = rng.multivariate_normal(mu, cov, size=n_days)
    assets = [f"Asset_{i+1}" for i in range(n_assets)]
    return pd.DataFrame(returns, columns=assets)


def get_returns(
    assets: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    min_assets: int = 3,
    use_synthetic_if_fail: bool = True,
) -> tuple[pd.DataFrame, bool]:
    """
    Load or generate returns for portfolio apps.

    Returns:
        (returns_df, from_live_data): from_live_data is True if Alpha Vantage succeeded.
    """
    assets = assets or DEFAULT_ASSETS
    returns = load_returns_from_alphavantage(assets, start=start, end=end)
    if returns is not None and not returns.empty:
        returns = returns.dropna(axis=1, thresh=returns.shape[0] // 2)
    if returns is None or returns.empty or returns.shape[1] < min_assets:
        if use_synthetic_if_fail:
            returns = make_synthetic_returns(n_assets=max(min_assets, 10), n_days=252 * 4)
            return returns, False
        raise ValueError("Insufficient return data and use_synthetic_if_fail=False")
    return returns, True
