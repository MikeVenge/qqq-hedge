"""
Mango client — resolve a trading book's current constituents and weights.

Mango exposes an MCP (JSON-RPC over HTTP) endpoint at MANGO_BASE_URL
(default https://mango.alphax.inc/mcp/). Access is open (no token), but the host
sits behind Cloudflare bot protection that rejects non-browser User-Agents
(Error 1010) -- so a realistic browser UA is required. The endpoint is stateless
(no mcp-session-id) and returns plain JSON-RPC (not SSE).

This module does I/O only; the volatility math lives in lib/portfolio_vol.py.
The payload->constituents parsing is split into a pure function
(`_constituents_from_positions`) so it can be unit-tested offline.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_MANGO_URL = "https://mango.alphax.inc/mcp/"
# Cloudflare Error 1010 blocks library UAs; a browser UA is required.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_DEFAULT_TIMEOUT = 25.0


class MangoError(Exception):
    """Network / protocol / RPC failure talking to Mango."""


def _endpoint() -> str:
    return os.environ.get("MANGO_BASE_URL", DEFAULT_MANGO_URL)


def _max_constituents() -> int:
    try:
        return int(os.environ.get("MAX_BOOK_CONSTITUENTS", "60"))
    except (TypeError, ValueError):
        return 60


# Cash / T-bill / money-market holdings: excluded from the risk basket when
# computing portfolio volatility (they are cash, not risk -- counting them
# deflates the vol and over-deploys the hedge). Extend via env CASH_TICKERS
# (comma-separated). The hedge manages its own cash sleeve separately.
_CASH_EQUIVALENTS = frozenset({
    "BIL", "BILS", "SGOV", "SHV", "USFR", "TBIL", "GBIL", "ICSH", "CLIP",
    "XHLF", "TFLO", "CSHI", "BILZ", "OBIL", "JPST", "MINT", "NEAR", "GSY",
    "CASH", "USD",
})

# Non-cash holdings also excluded from the risk basket: sector-hedge / defensive
# overlays that shouldn't drive the portfolio-vol target. Reported under the same
# `excluded_cash` field as the cash equivalents above.
_EXCLUDED_HEDGES = frozenset({"XLU", "XLV"})


def _cash_tickers() -> frozenset:
    """Tickers excluded from the risk basket (cash equivalents + hedge overlays).
    Extend at runtime via env CASH_TICKERS (comma-separated)."""
    extra = os.environ.get("CASH_TICKERS", "")
    extras = {t.strip().upper() for t in extra.split(",") if t.strip()}
    return _CASH_EQUIVALENTS | _EXCLUDED_HEDGES | extras


def _parse_rpc(raw: str) -> dict:
    """Parse a JSON-RPC response that may be plain JSON or SSE-framed."""
    raw = raw.strip()
    if raw.startswith("data:") or "\ndata:" in raw:
        for line in raw.split("\n"):
            if line.startswith("data: "):
                return json.loads(line[6:])
        raise MangoError("no data line in SSE response")
    return json.loads(raw)


def _post(payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        _endpoint(),
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": _BROWSER_UA,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _parse_rpc(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        raise MangoError(f"Mango HTTP {e.code}: {body}")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise MangoError(f"Mango request failed: {e}")


def call_tool(name: str, arguments: dict, timeout: float = _DEFAULT_TIMEOUT) -> dict:
    """Invoke a Mango MCP tool and return its decoded JSON result payload."""
    # Stateless server: a fresh initialize before the call keeps it happy.
    _post(
        {
            "jsonrpc": "2.0", "method": "initialize", "id": 1,
            "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "qqq-hedge", "version": "1.0"},
            },
        },
        timeout,
    )
    resp = _post(
        {
            "jsonrpc": "2.0", "method": "tools/call", "id": 2,
            "params": {"name": name, "arguments": arguments},
        },
        timeout,
    )
    if "error" in resp:
        raise MangoError(f"Mango RPC error: {resp['error']}")
    result = resp.get("result") or {}
    if result.get("isError"):
        raise MangoError(f"Mango tool error: {str(result.get('content'))[:200]}")
    content = result.get("content") or []
    if not content:
        raise MangoError("Mango returned empty content")
    return json.loads(content[0]["text"])


def list_positions(book_id: int, timeout: float = _DEFAULT_TIMEOUT) -> dict:
    """Raw `list_positions` payload for a book (root has `positions` list)."""
    return call_tool("list_positions", {"body": {"book_id": int(book_id)}}, timeout)


def _constituents_from_positions(
    data: dict,
    book_id: int,
    *,
    include_options: bool = False,
    include_cash: bool = False,
    weighting: str = "equal",
    max_constituents: int | None = None,
) -> dict:
    """Pure: turn a `list_positions` payload into signed, normalized weights.

    - keep `quantity != 0` (active); drop `asset_class == 'option'` unless asked
    - drop cash/T-bill/MMF holdings (see _cash_tickers) unless include_cash --
      they are cash, not risk; including them deflates the portfolio vol
    - sign: 'S' / negative quantity -> short (negative weight)
    - aggregate duplicate symbols, cap to top `max_constituents` by gross |weight|
    - `weighting`: "equal" (default) -> each risk name gets +/- 1/N;
      "gross" -> weight_of_gross/100 (falls back to |market_value| share),
      renormalized so sum(|w|) == 1 (vol per unit of risk capital)

    Returns {book_id, book_name, symbols, weights, weighting, n_constituents,
    n_dropped_options, n_dropped_cash, dropped_cash, cash_weight, n_truncated,
    gross_market_value, net_exposure} or {error}.
    """
    if max_constituents is None:
        max_constituents = _max_constituents()
    weighting = weighting if weighting in ("equal", "gross") else "equal"
    positions = data.get("positions") or []
    cash_set = _cash_tickers()

    book_name = None
    active = []
    n_dropped_options = 0
    for p in positions:
        try:
            qty = float(p.get("quantity") or 0)
        except (TypeError, ValueError):
            qty = 0.0
        if qty == 0:
            continue
        if book_name is None:
            book_name = p.get("book_name")
        if (p.get("asset_class") or "equity").lower() == "option" and not include_options:
            n_dropped_options += 1
            continue
        active.append(p)

    if not active:
        kind = "" if include_options else " equity"
        return {"error": f"book {book_id} has no active{kind} constituents"}

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    have_wog = all(p.get("weight_of_gross") not in (None, "") for p in active)
    mv_total = sum(abs(_f(p.get("market_value"))) for p in active) or 1.0

    weights: dict[str, float] = {}
    dropped_cash: dict[str, float] = {}   # symbol -> |weight| as fraction of gross
    for p in active:
        sym = str(p.get("symbol") or "").upper()
        if not sym:
            continue
        mag = abs(_f(p.get("weight_of_gross"))) / 100.0 if have_wog \
            else abs(_f(p.get("market_value"))) / mv_total
        is_short = (str(p.get("long_short")).upper() == "S") or (_f(p.get("quantity")) < 0)
        if not include_cash and sym in cash_set:
            dropped_cash[sym] = dropped_cash.get(sym, 0.0) + mag
            continue
        weights[sym] = weights.get(sym, 0.0) + (-mag if is_short else mag)

    weights = {k: v for k, v in weights.items() if v != 0.0}
    if not weights:
        return {
            "error": (
                f"book {book_id}: no risk constituents after excluding cash "
                f"{sorted(dropped_cash)} ({sum(dropped_cash.values()) * 100:.0f}% of gross)"
            )
        }

    n_truncated = 0
    if len(weights) > max_constituents:
        top = sorted(weights.items(), key=lambda kv: -abs(kv[1]))[:max_constituents]
        n_truncated = len(weights) - max_constituents
        weights = dict(top)

    if weighting == "equal":
        n = len(weights)
        weights = {k: (1.0 / n if v > 0 else -1.0 / n) for k, v in weights.items()}
    else:  # "gross"
        gross = sum(abs(w) for w in weights.values()) or 1.0
        weights = {k: v / gross for k, v in weights.items()}

    return {
        "book_id": int(book_id),
        "book_name": book_name,
        "symbols": sorted(weights.keys()),
        "weights": weights,
        "weighting": weighting,
        "n_constituents": len(weights),
        "n_dropped_options": n_dropped_options,
        "n_dropped_cash": len(dropped_cash),
        "dropped_cash": sorted(dropped_cash.keys()),
        "cash_weight": round(sum(dropped_cash.values()), 4),
        "n_truncated": n_truncated,
        "gross_market_value": mv_total,
        "net_exposure": round(sum(weights.values()), 6),
    }


def constituents_from_tickers(
    tickers,
    *,
    include_cash: bool = False,
    max_constituents: int | None = None,
) -> dict:
    """Pure: turn a user-supplied ticker list into signed equal weights.

    Same output contract as `_constituents_from_positions`, so the hedge
    signal's portfolio-vol path can consume either source. No Mango I/O.

    - accepts a list of symbols or one comma/space-separated string
    - a "-" prefix marks a short (e.g. "-IWM" -> weight -1/N); default long
    - duplicates collapse; the same symbol both long and short is an error
    - cash/T-bill/MMF + hedge-overlay tickers (see _cash_tickers) are dropped
      unless include_cash, reported via dropped_cash; cash_weight is their
      would-have-been equal-weight share
    - equal weighting only (an ad-hoc list has no market values): +/- 1/N
    - more than `max_constituents` names is an error (no silent truncation --
      the caller typed the list)
    """
    if max_constituents is None:
        max_constituents = _max_constituents()
    if isinstance(tickers, str):
        tickers = [t for t in tickers.replace(",", " ").split() if t]

    signs: dict[str, int] = {}
    order: list[str] = []
    for raw in tickers or []:
        t = str(raw).strip().upper()
        sign = 1
        if t.startswith("-"):
            sign, t = -1, t[1:].strip()
        if not t:
            continue
        if not all(c.isalnum() or c in ".-" for c in t):
            return {"error": f"invalid ticker: {raw!r}"}
        prev = signs.get(t)
        if prev is not None and prev != sign:
            return {"error": f"ticker {t} appears both long and short"}
        if prev is None:
            order.append(t)
        signs[t] = sign

    if not signs:
        return {"error": "no tickers provided"}
    if len(signs) > max_constituents:
        return {"error": f"{len(signs)} tickers exceeds the max of {max_constituents}"}

    cash_set = _cash_tickers()
    dropped_cash = [t for t in order if not include_cash and t in cash_set]
    risk = [t for t in order if t not in dropped_cash]
    if not risk:
        return {
            "error": (
                f"no risk tickers after excluding cash/hedge overlays "
                f"{sorted(dropped_cash)}"
            )
        }

    n = len(risk)
    weights = {t: signs[t] * (1.0 / n) for t in risk}
    return {
        "book_id": None,
        "book_name": None,
        "tickers": [("-" + t) if signs[t] < 0 else t for t in risk],
        "symbols": sorted(weights.keys()),
        "weights": weights,
        "weighting": "equal",
        "n_constituents": n,
        "n_dropped_options": 0,
        "n_dropped_cash": len(dropped_cash),
        "dropped_cash": sorted(dropped_cash),
        "cash_weight": round(len(dropped_cash) / (n + len(dropped_cash)), 4),
        "n_truncated": 0,
        "net_exposure": round(sum(weights.values()), 6),
    }


def resolve_book_constituents(
    book_id: int,
    *,
    include_options: bool = False,
    include_cash: bool = False,
    weighting: str = "equal",
    max_constituents: int | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict:
    """Fetch a book and resolve it to signed, normalized constituents+weights."""
    data = list_positions(book_id, timeout=timeout)
    return _constituents_from_positions(
        data, book_id, include_options=include_options,
        include_cash=include_cash, weighting=weighting,
        max_constituents=max_constituents,
    )
