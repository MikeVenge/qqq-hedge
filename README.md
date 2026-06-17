# QQQ Hedge Service — API Manual

A hosted service that computes a **QQQ vol-target hedge signal**: given a date and
a volatility-target level, it returns the gross long deployment (how much of the
book to be invested) and the cash sleeve (the rest), for a long QQQ / tech book.

It exposes the same logic two ways:

| Surface | URL | For |
|---|---|---|
| **REST API** (async job) | `https://qqq-hedge-production.up.railway.app/api/hedge` | plain HTTP clients / coding agents |
| **MCP** (JSON-RPC, streamable-http) | `https://qqq-hedge-production.up.railway.app/mcp` | MCP clients (Claude, etc.) |
| Health | `https://qqq-hedge-production.up.railway.app/health` | liveness check |

> Both surfaces call one shared core, so identical inputs return identical results.
> No authentication is required. Data source: Alpha Vantage QQQ daily closes.

---

## The signal in 30 seconds

Two multiplicative layers:

1. **Regime gate** (direction): based on price vs SMA100 / SMA200.
   `1.0` if above both, `0.5` if above one, `0.0` if below both. The gate uses a
   **symmetric 3-day confirmation** — it only changes state after price holds the
   new condition for 3 consecutive closes (both down and up).
2. **Inverse-vol scalar** (magnitude): `w_vol = min(target_vol / rv20, 1.5)`
   where `rv20` is QQQ's trailing 20-day annualized realized vol and
   `target_vol = vt / 100`.

```
exposure = gate × w_vol            # 0.0 .. 1.5  (gross long deployment)
cash     = 1 − exposure            # negative cash = financed leverage
```

`vt` (the vol target) is any positive number: `vt=15` → target 15% (default),
`vt=23` → 23%, etc. Lower `vt` = more defensive (more cash); higher `vt` = more
aggressive (can lever up to 1.5×).

---

## REST API (recommended for coding agents)

Asynchronous **submit → poll** pattern: you `POST` a job, get a `job_id`, then
`GET` that job until it finishes (usually ~2–3 s).

### 1) Submit a job

```
POST /api/hedge
Content-Type: application/json

{ "date": "2026-05-28", "vt": 23 }
```

All fields are **optional**:
- `date` — ISO `YYYY-MM-DD`. Omit (or `null`) for the latest trading day. A
  non-trading day rolls back to the most recent trading day on/before it.
- `vt` — vol-target level, any positive number. Default `15`.
- `book_id` — a Mango trading-book ID (integer). When set, the inverse-vol
  scalar uses the book's **30-day realized portfolio volatility** instead of
  QQQ's; the SMA regime gate still uses QQQ. The response adds
  `vol_source:"portfolio"`, `portfolio_vol`, `book_name`, `n_constituents`.
  (Slower: the server fetches each constituent's prices — use the async poll.)

You may also pass them as query params: `POST /api/hedge?date=2026-05-28&vt=23&book_id=31`.

**Response — `202 Accepted`:**

```json
{
  "job_id": "c3e928676fc04fdf866d001bbda765be",
  "status": "pending",
  "request": { "date": "2026-05-28", "vt": 23.0 },
  "poll_url": "/api/hedge/c3e928676fc04fdf866d001bbda765be"
}
```

**Errors on submit — `400 Bad Request`:** `vt` not a number, or `vt <= 0`.

### 2) Poll the job

```
GET /api/hedge/{job_id}
```

| HTTP | `status` | Meaning |
|---|---|---|
| `202` | `pending` / `running` | not done yet — poll again |
| `200` | `done` | finished — read `result` |
| `200` | `error` | the job ran but computation failed — read `error` |
| `404` | — | unknown `job_id` |

**Response when done — `200 OK`:**

```json
{
  "job_id": "c3e928676fc04fdf866d001bbda765be",
  "status": "done",
  "request": { "date": "2026-05-28", "vt": 23.0 },
  "created_at": "2026-06-04T06:48:43.683546+00:00",
  "completed_at": "2026-06-04T06:48:46.332717+00:00",
  "result": {
    "as_of_date": "2026-05-28",
    "requested_date": "2026-05-28",
    "vt": 23.0,
    "target_vol": 0.23,
    "exposure": 1.4271,
    "exposure_pct": "142.7% invested",
    "cash": -0.4271,
    "cash_pct": "42.7% leverage",
    "gate": 1.0,
    "regime": "risk-on",
    "w_vol": 1.4271,
    "leverage_capped": false,
    "rv20": 0.1612,
    "rv20_pct": "16.1%",
    "close": 735.6,
    "sma100": 631.06,
    "sma200": 616.03
  }
}
```

**Response when error — `200 OK`:**

```json
{
  "job_id": "...",
  "status": "error",
  "request": { "date": "2015-01-01", "vt": 15.0 },
  "created_at": "...",
  "completed_at": "...",
  "error": "No QQQ trading day on or before 2015-01-01"
}
```

### `result` field reference

| Field | Type | Meaning |
|---|---|---|
| `as_of_date` | string `YYYY-MM-DD` | Trading day actually used (rolled back from request) |
| `requested_date` | string | The date you asked for (= `as_of_date` if omitted) |
| `vt` | number | Vol-target level you passed |
| `target_vol` | number | `vt / 100` |
| `exposure` | number `0.0–1.5` | **Gross long deployment** as a fraction of the book |
| `exposure_pct` | string | Human label, e.g. `"142.7% invested"` |
| `cash` | number | `1 − exposure` (negative = financed leverage) |
| `cash_pct` | string | Human label, e.g. `"42.7% leverage"` or `"6.8% cash"` |
| `gate` | `0.0` / `0.5` / `1.0` | SMA regime gate (3-day symmetric confirmation applied) |
| `regime` | string | `"risk-on"` (1.0) / `"half"` (0.5) / `"cash"` (0.0) |
| `w_vol` | number `0.0–1.5` | Inverse-vol scalar `min(target_vol / rv20, 1.5)` |
| `leverage_capped` | bool | `true` if `w_vol` hit the 1.5× cap |
| `rv20` | number | Trailing 20-day annualized realized vol through `as_of_date` (most recent close included) |
| `rv20_pct` | string | Human label, e.g. `"16.1%"` |
| `close` | number | QQQ close on `as_of_date` |
| `sma100` / `sma200` | number | 100- / 200-day simple moving averages |

Invariant for sanity checks: `exposure == round(gate * w_vol, 4)`.

### Examples

**curl**

```bash
BASE=https://qqq-hedge-production.up.railway.app

# submit
RESP=$(curl -s -X POST "$BASE/api/hedge" \
  -H 'Content-Type: application/json' \
  -d '{"date":"2026-05-28","vt":23}')
JOB=$(echo "$RESP" | python3 -c 'import sys,json; print(json.load(sys.stdin)["job_id"])')

# poll
curl -s "$BASE/api/hedge/$JOB"
```

**Python (stdlib only)**

```python
import json, time, urllib.request

BASE = "https://qqq-hedge-production.up.railway.app"

def get_hedge(date=None, vt=15, timeout_s=60):
    body = json.dumps({"date": date, "vt": vt}).encode()
    req = urllib.request.Request(
        f"{BASE}/api/hedge", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    job = json.load(urllib.request.urlopen(req))          # 202 -> {job_id, ...}
    url = f"{BASE}{job['poll_url']}"

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        j = json.load(urllib.request.urlopen(url))
        if j["status"] == "done":
            return j["result"]
        if j["status"] == "error":
            raise RuntimeError(j["error"])
        time.sleep(1)
    raise TimeoutError("hedge job did not finish")

print(get_hedge(date="2026-05-28", vt=23))   # latest if date=None
```

> Tip: `GET /api/hedge/{job_id}` returns `202` until the job is done, so you can
> branch on either the HTTP status code or the `status` field.

---

## MCP interface

- **URL:** `https://qqq-hedge-production.up.railway.app/mcp`
- **Transport:** streamable-http (JSON-RPC: `initialize`, then `tools/call`)

### Tools

**`qqq_hedge_signal(date: str | None = None, vt: float = 15, book_id: int | None = None) -> dict`**
Same inputs and same `result` object as the REST API (returned synchronously,
not as a job). With `book_id`, the vol input becomes the Mango book's 30-day
portfolio volatility (gate stays on QQQ).

**`qqq_hedge_backtest(start="2020-01-01", end="2026-12-31", vt=15, leverage_cap=1.5, fed_funds_rate=0.0) -> dict`**
Backtests the overlay vs buy-and-hold. Returns `stats` (Sharpe, max drawdown,
Calmar, mean exposure / w_vol, % at leverage cap), `recent_exposures`, and the
data range. **MCP-only** — not exposed over REST.

### Example `tools/call` payload

```json
{
  "jsonrpc": "2.0", "method": "tools/call", "id": 2,
  "params": { "name": "qqq_hedge_signal", "arguments": { "date": "2026-05-28", "vt": 23 } }
}
```

(Send `initialize` first; reuse the returned `Mcp-Session-Id` header. Responses
are Server-Sent Events: parse the `data:` line, then
`result.content[0].text` is the JSON string of the `result` object.)

---

## Notes & limitations

- **No auth.** All endpoints are open.
- **Jobs are in-memory** on a single container: lost on restart, store capped
  (oldest evicted). Fine for submit-and-poll within seconds; not durable.
- **As-of is lookahead-free.** Every indicator at date *t* uses only data ≤ *t*
  (rolling windows over closes through *t*; `rv20` includes day *t*). In the
  backtest, look-ahead is avoided by lagging the *position* one day, not the vol.
- **`exposure > 1.0` means leverage** (financed); `cash < 0` is the borrow.
- The signal is **long-only + cash** — it never goes short.

## Testing

```bash
# REST integration smoke test (hits the live server)
python3 tests/check_rest_api.py                       # prod (default)
python3 tests/check_rest_api.py http://localhost:8000 # local

# library unit tests (no network, no MCP)
python3 tests/test_qqq_hedge.py
```
